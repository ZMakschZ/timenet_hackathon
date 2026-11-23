"""Prepare operator QA continuation data from observed Hill of Towie windows."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_evaluation import QUESTION
from build_examples import CHANNELS

PROMPTS = (
    (
        "Give me a shift handover for this turbine. What is it doing, what can we verify, "
        "and what evidence would change the assessment?"
    ),
    (
        "Summarize the turbine's operating state from this window. Explain the evidence, "
        "its limitations, and the next check."
    ),
    (
        "What should the operator know about this turbine right now? Use the SCADA window "
        "to explain what you see and when you would revise the assessment."
    ),
)
STATES = (
    "generating_in_latest_interval",
    "idle_or_stopped_at_cutoff",
    "mixed_or_transitional",
    "visibility_unavailable",
    "state_unknown",
)
TARGETS = {
    STATES[0]: (
        "The turbine generated in the latest interval. Power, generator speed and logged "
        "running time all show operation. Compare subsequent output with the power reference "
        "and check live limits if a shortfall persists."
    ),
    STATES[1]: (
        "The latest readings are consistent with idle or stopped operation. Power and "
        "generator speed were low and no running time was logged. Check the live controller "
        "and dispatch state for the reason. Reassess if speed, power and running time return."
    ),
    STATES[2]: (
        "The operating state is unresolved. Power, speed and logged running time do not "
        "jointly indicate steady generation or idle operation. Check the live controller "
        "and preceding readings. Use the next consistent set of readings to classify the state."
    ),
    STATES[4]: (
        "Generation state is uncertain. At least one of power, generator speed or running "
        "time is missing or invalid. Verify those channels and the live controller before "
        "assigning a state."
    ),
}
GAP = (
    "Operating state at cutoff is unknown. The selected telemetry is missing at cutoff. "
    "Verify live communications or an independent source before assigning downtime."
)
EMPTY = (
    "Operating state is unknown. None of the selected SCADA channels reported during the "
    "window. Restore telemetry or obtain an independent status before classifying the turbine."
)
SHORTFALL = (
    "The turbine generated in the latest interval, but mean output was below its interval-end "
    "reference. Verify wind availability, dispatch limits and controller status. Check whether "
    "the shortfall persists in the following readings before attributing a cause."
)


def examples(corpus, manifest, split, rng):
    """Return diverse examples per month without augmenting or replacing signals."""
    turbines = {row["Station ID"]: row for row in manifest["turbines"]}
    for item in manifest["months"]:
        month = item["month"]
        if (split == "train" and month >= "2024_10") or (
            split == "validation" and not "2024_10" <= month < "2025_01"
        ):
            continue
        folder = corpus / month
        windows = defaultdict(list)
        columns = [
            "station_id",
            "split",
            "cohort",
            "start_step",
            "stop_step",
            "cutoff_utc",
        ]
        for row in pq.read_table(
            folder / "windows.parquet", columns=columns, filters=[("split", "=", split)]
        ).to_pylist():
            windows[row["station_id"]].append(row)
        signal = pq.ParquetFile(folder / "signals.parquet")
        for group in range(signal.num_row_groups):
            table = signal.read_row_group(group, columns=["station_id", *CHANNELS])
            station = table["station_id"][0].as_py()
            rows = windows[station]
            if not rows:
                continue
            steps = np.asarray([row["stop_step"] - 1 for row in rows])
            raw = {name: table[name].to_numpy() for name in CHANNELS}
            power, reference, rpm, running = (
                raw[name][steps] for name in ("P", "C", "R", "ON")
            )
            observed = np.logical_or.reduce(
                [np.isfinite(values) for values in raw.values()]
            )
            last = np.maximum.accumulate(np.where(observed, np.arange(len(table)), -1))[
                steps
            ]
            valid = (
                np.isfinite(power)
                & np.isfinite(rpm)
                & (rpm >= 0)
                & (running >= 0)
                & (running <= 600)
            )
            live = observed[steps]
            generating = valid & live & (power >= 50) & (rpm > 300) & (running > 0)
            idle = valid & live & (np.abs(power) <= 30) & (rpm < 100) & (running == 0)
            mixed = valid & live & ~generating & ~idle
            masks = (generating, idle, mixed, ~live, live & ~valid)
            for state, mask in zip(STATES, masks, strict=True):
                hits = np.flatnonzero(mask)
                if split == "train":
                    # Samples at most one cutoff per six-hour block per state and turbine.
                    rng.shuffle(hits)
                    _, first = np.unique(steps[hits] // 36, return_index=True)
                    hits = rng.permutation(hits[first])[:85]
                for i in hits:
                    row = rows[i]
                    if state == STATES[3]:
                        target = EMPTY if last[i] < row["start_step"] else GAP
                    elif (
                        state == STATES[0]
                        and reference[i] >= 400
                        and reference[i] - power[i] >= 300
                    ):
                        target = SHORTFALL
                    else:
                        target = TARGETS[state]
                    yield {
                        **row,
                        "month": month,
                        "expected_state": state,
                        "target": target,
                        "question": str(rng.choice(PROMPTS))
                        if split == "train"
                        else QUESTION,
                        "rated_capacity_kw": turbines[station]["Rated power (kW)"],
                        "turbine_id": turbines[station]["Turbine Name"],
                    }


def write(path, rows):
    rows.sort(key=lambda row: (row["month"], row["station_id"], row["stop_step"]))
    pq.write_table(
        pa.Table.from_pylist(rows), path, compression="zstd", row_group_size=1024
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("paired", type=Path)
    parser.add_argument("--broad-rows", type=int, default=60000)
    parser.add_argument("--paired-rows", type=int, default=40000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261120)
    manifest = json.loads((args.corpus / "manifest.json").read_text())

    groups = defaultdict(list)
    for row in examples(args.corpus, manifest, "train", rng):
        groups[row["expected_state"]].append(row)
    counts = {state: len(groups[state]) for state in STATES}
    goals = (18000, 12000, 15000, 10000, 5000)
    rare = groups[STATES[4]]
    groups[STATES[4]] = [
        {**row, "question": PROMPTS[(repeat + i) % len(PROMPTS)]}
        for repeat in range(12)
        for i, row in enumerate(rare)
    ]
    broad = []
    for state, goal in zip(STATES, goals, strict=True):
        rng.shuffle(groups[state])
        quota = round(args.broad_rows * goal / sum(goals))
        broad.extend(groups[state][:quota])
        groups[state] = groups[state][quota:]
    # Keep rare real states wherever available, filling unused quota from other real cases.
    while len(broad) < args.broad_rows:
        active = [
            groups[state]
            for state in (STATES[0], STATES[1], STATES[2], STATES[3], STATES[4])
            if groups[state]
        ]
        if not active:
            raise ValueError("Insufficient distinct training windows")
        broad.extend(pool.pop() for pool in active[: args.broad_rows - len(broad)])
    write(args.output / "operator-train.parquet", broad)

    outcomes = defaultdict(list)
    for row in examples(args.corpus, manifest, "validation", rng):
        outcomes[row["expected_state"]].append(row)
    held_out = []
    for state in STATES:
        rng.shuffle(outcomes[state])
        used = defaultdict(list)
        chosen = []
        for row in outcomes[state]:
            key = (row["month"], row["station_id"])
            if any(abs(row["stop_step"] - past) < 36 for past in used[key]):
                continue
            used[key].append(row["stop_step"])
            chosen.append(row)
            if len(chosen) == 16:
                break
        held_out.extend(chosen)
    write(args.output / "operator-validation.parquet", held_out)

    pair_groups = defaultdict(list)
    for row in pq.read_table(args.paired).to_pylist():
        if row["split"] != "train":
            raise ValueError("Replay source contains held-out data")
        pair_groups[row["pair_id"]].append(row)
    ids = list(pair_groups)
    rng.shuffle(ids)
    paired = [row for key in ids[: args.paired_rows // 2] for row in pair_groups[key]]
    if len(paired) != args.paired_rows:
        raise ValueError("Expected an even number of training rows in complete pairs")
    write(args.output / "paired-train.parquet", paired)
    audit = {
        "available_training_windows_by_state": counts,
        "unknown_state_replays_per_window": 12,
        "operator_training": dict(Counter(row["expected_state"] for row in broad)),
        "operator_validation": dict(Counter(row["expected_state"] for row in held_out)),
        "paired_training": dict(Counter(row["family"] for row in paired)),
        "operator_training_months": len({row["month"] for row in broad}),
        "operator_training_turbines": len({row["station_id"] for row in broad}),
    }
    (args.output / "data.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
