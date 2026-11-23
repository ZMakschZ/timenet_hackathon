"""Build matched-question SCADA QA pairs with different observed outcomes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROMPTS = {
    "power": (
        "How closely does active power track its reference in the latest interval, and what should I check?",
        ("tracking", "shortfall"),
    ),
    "operation": (
        "Do the latest speed, power and running-time readings show generation, and what should I check?",
        ("generating", "idle"),
    ),
    "temperature": (
        "Did all three generator windings warm or cool over the last hour, and what should I check?",
        ("warming", "cooling"),
    ),
    "electrical": (
        "Is the latest phase-current spread below 3 A or above 8 A, and what should I check?",
        ("small spread", "wide spread"),
    ),
    "telemetry": (
        "Are power, wind, speed and running time all present at cutoff, or are all selected readings missing? What should I check?",
        ("reporting", "missing"),
    ),
    "maintenance": (
        "Was turbine error active at cutoff and in at least one earlier interval, or absent throughout the window? What should I hand over?",
        ("error present", "no error recorded"),
    ),
}
QUOTAS = {"train": 1000, "validation": 10}
COLS = ("P", "C", "W", "R", "ON", "ERR", "U", "V", "Z", "IR", "IS", "IT")


def examples(signal, windows, rng, candidate_limit):
    """Find clear cases using only values available by each cutoff."""
    steps = np.asarray([row["stop_step"] - 1 for row in windows])
    if not len(steps):
        return
    raw = {name: signal[name].to_numpy() for name in COLS}
    p, c, _, rpm, on, error = (raw[name][steps] for name in COLS[:6])
    on_valid = (on >= 0) & (on <= 600)
    producing = (p >= 50) & (rpm > 300) & (on > 0) & on_valid
    idle = (np.abs(p) <= 30) & (rpm >= 0) & (rpm < 100) & (on == 0)
    temp = np.asarray([raw[name] for name in ("U", "V", "Z")])
    temp_diff = temp[:, steps] - temp[:, steps - 6]
    current = np.asarray([raw[name][steps] for name in ("IR", "IS", "IT")])
    spread = np.ptp(current, axis=0)
    empty = np.logical_and.reduce(
        [
            signal[name].is_null().to_numpy(zero_copy_only=False)[steps]
            for name in signal.column_names
            if name != "station_id"
        ]
    )
    present = producing | idle
    present |= np.isfinite(raw["P"][steps]) & np.isfinite(raw["R"][steps]) & on_valid
    present &= np.isfinite(raw["W"][steps])
    flags = np.lib.stride_tricks.sliding_window_view(raw["ERR"], 36)[steps - 35]
    flag_valid = ((flags >= 0) & (flags <= 600)).all(axis=1)
    conditions = {
        "power": (
            producing & (c >= 400) & (np.abs(c - p) <= 120),
            producing & (c >= 400) & (c - p >= 300),
        ),
        "operation": (producing, idle),
        "temperature": (
            np.all(temp_diff >= 2, axis=0),
            np.all(temp_diff <= -2, axis=0),
        ),
        "electrical": (
            producing & (p >= 500) & (spread < 3),
            producing & (p >= 500) & (spread > 8),
        ),
        "telemetry": (present, empty),
        "maintenance": (
            flag_valid & (error > 0) & ((flags > 0).sum(axis=1) >= 2),
            flag_valid & (flags == 0).all(axis=1),
        ),
    }
    labels = {
        "power": ("tracking", "shortfall"),
        "operation": ("generating", "idle"),
        "temperature": ("warming", "cooling"),
        "electrical": ("small spread", "wide spread"),
        "telemetry": ("reporting", "missing"),
        "maintenance": ("error present", "no error recorded"),
    }
    for family, masks in conditions.items():
        for index, mask in enumerate(masks):
            hits = np.flatnonzero(mask)
            for i in rng.choice(
                hits, size=min(candidate_limit, len(hits)), replace=False
            ):
                outcome = labels[family][index]
                if family == "power":
                    evidence = f"The latest mean power was {p[i]:.0f} kW against a {c[i]:.0f} kW interval-end reference."
                    action = (
                        "Watch whether the gap opens up."
                        if index == 0
                        else "Check wind availability, dispatch limits and controller state."
                    )
                elif family == "operation":
                    evidence = f"Power was {p[i]:.0f} kW, speed was {rpm[i]:.0f} rpm and running time was {on[i]:.0f} s."
                    action = (
                        "Confirm live availability before scheduling output."
                        if index == 0
                        else "Check the live controller state for the reason it is idle."
                    )
                elif family == "temperature":
                    evidence = (
                        "Changes across the three windings were "
                        + ", ".join(f"{value:+.0f}" for value in temp_diff[:, i])
                        + " °C."
                    )
                    action = (
                        "Check load and cooling if warming continues."
                        if index == 0
                        else "Check recent load and live limits before changing output."
                    )
                elif family == "electrical":
                    evidence = f"The difference between the highest and lowest phase-current means was {spread[i]:.1f} A."
                    action = (
                        "Check live protection if it flags an issue."
                        if index == 0
                        else "Compare live phase measurements with independent metering."
                    )
                elif family == "telemetry":
                    evidence = (
                        "All four operating channels reported at cutoff."
                        if index == 0
                        else "None of the 20 selected channels reported at cutoff."
                    )
                    action = (
                        "Use the readings to assess operation."
                        if index == 0
                        else "Restore visibility before assigning downtime."
                    )
                else:
                    n = int((flags[i] > 0).sum())
                    evidence = f"The error flag was active in {n} of 36 intervals."
                    action = (
                        "Retrieve the first-out alarm and event sequence."
                        if index == 0
                        else "Check live status if a fault remains suspected."
                    )
                yield (
                    windows[i],
                    family,
                    outcome,
                    f"Answer: {outcome}. {evidence} {action}",
                )


def build(
    corpus,
    output,
    *,
    train_pairs=6000,
    exclude=None,
    test_pairs=None,
    validation_pairs=None,
):
    """Balance outcomes per family and spread samples across months."""
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2401)
    manifest = json.loads((corpus / "manifest.json").read_text())
    turbines = {row["Station ID"]: row for row in manifest["turbines"]}
    candidates = defaultdict(lambda: defaultdict(list))
    excluded = set()
    held_out = test_pairs is not None or validation_pairs is not None
    for path in [exclude] if isinstance(exclude, Path) else exclude or []:
        for row in pq.read_table(
            path, columns=["month", "station_id", "stop_step"]
        ).to_pylist():
            month, station, stop = row["month"], row["station_id"], row["stop_step"]
            # Keep full windows disjoint across evaluation panels.
            stops = range(stop - 35, stop + 36) if held_out else (stop,)
            excluded.update((month, station, step) for step in stops)
    if test_pairs is not None:
        splits = ("test",)
    elif validation_pairs is not None:
        splits = ("validation",)
    else:
        splits = ("train",) if exclude is not None else tuple(QUOTAS)
    candidate_limit = 20 if train_pairs > 6000 else 8
    for item in manifest["months"]:
        month = item["month"]
        if test_pairs is not None and month < "2025_01":
            continue
        if validation_pairs is not None and month < "2024_10":
            continue
        if test_pairs is None and month >= "2025_01":
            break
        folder = corpus / month
        by_station = defaultdict(list)
        for row in pq.read_table(
            folder / "windows.parquet",
            columns=[
                "station_id",
                "split",
                "cohort",
                "start_step",
                "stop_step",
                "cutoff_utc",
            ],
        ).to_pylist():
            if (
                row["split"] in splits
                and (month, row["station_id"], row["stop_step"]) not in excluded
            ):
                by_station[row["station_id"]].append(row)
        file = pq.ParquetFile(folder / "signals.parquet")
        for group in range(file.num_row_groups):
            signal = file.read_row_group(group)
            station = signal["station_id"][0].as_py()
            meta = turbines[station]
            for row, family, outcome, target in examples(
                signal, by_station[station], rng, candidate_limit
            ):
                candidates[row["split"], family, outcome][month].append(
                    {
                        **row,
                        "month": month,
                        "family": family,
                        "expected_outcome": outcome,
                        "turbine_id": meta["Turbine Name"],
                        "rated_capacity_kw": meta["Rated power (kW)"],
                        "target": target,
                    }
                )
    for split in splits:
        result = []
        requested = {
            "train": train_pairs,
            "validation": validation_pairs or QUOTAS["validation"] * len(PROMPTS),
            "test": test_pairs,
        }[split]
        capacities = {
            family: min(
                sum(map(len, candidates[split, family, label].values()))
                for label in labels
            )
            for family, (_, labels) in PROMPTS.items()
        }
        quotas = dict.fromkeys(PROMPTS, 0)
        remaining = requested
        while remaining:
            active = [
                family for family in PROMPTS if quotas[family] < capacities[family]
            ]
            if not active:
                raise RuntimeError(
                    f"Only {requested - remaining} of {requested} {split} pairs available"
                )
            share = max(1, remaining // len(active))
            for family in active:
                take = min(remaining, share, capacities[family] - quotas[family])
                quotas[family] += take
                remaining -= take
        for family, (question, labels) in PROMPTS.items():
            quota = quotas[family]
            selections = []
            for label in labels:
                pools = candidates[split, family, label]
                for sample in pools.values():
                    rng.shuffle(sample)
                chosen = []
                while len(chosen) < quota:
                    active = [pool for pool in pools.values() if pool]
                    if not active:
                        raise RuntimeError(
                            f"{split}/{family}/{label}: only {len(chosen)} candidates"
                        )
                    rng.shuffle(active)
                    chosen.extend(pool.pop() for pool in active[: quota - len(chosen)])
                selections.append(chosen)
            question += f" Begin the answer with Answer: {labels[0]} or Answer: {labels[1]}, then give brief evidence and a check."
            for i, pair in enumerate(zip(*selections, strict=True)):
                result.extend(
                    {
                        **row,
                        "pair_id": f"{split}-{family}-{i:04d}",
                        "question": question,
                    }
                    for row in pair
                )
        result.sort(
            key=lambda row: (
                row["month"],
                row["station_id"],
                row["stop_step"],
                row["family"],
            )
        )
        name = "qa" if split == "train" else split
        pq.write_table(
            pa.Table.from_pylist(result),
            output / f"{name}.parquet",
            compression="zstd",
            row_group_size=512,
        )
        print(
            split,
            len(result),
            "examples across",
            len({row["month"] for row in result}),
            "months",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-pairs", type=int, default=6000)
    parser.add_argument("--exclude-previous", type=Path, action="append")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument(
        "--test-pairs", type=int, help="Build only fresh test pairs"
    )
    evaluation.add_argument(
        "--validation-pairs", type=int, help="Build only fresh validation pairs"
    )
    args = parser.parse_args()
    if args.train_pairs < 1:
        parser.error("--train-pairs must be positive")
    if args.test_pairs is not None and args.test_pairs < 1:
        parser.error("--test-pairs must be positive")
    if args.validation_pairs is not None and args.validation_pairs < 1:
        parser.error("--validation-pairs must be positive")
    build(
        args.corpus,
        args.output,
        train_pairs=args.train_pairs,
        exclude=args.exclude_previous,
        test_pairs=args.test_pairs,
        validation_pairs=args.validation_pairs,
    )


if __name__ == "__main__":
    main()
