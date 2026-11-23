"""Sample the Hill of Towie TimeF curriculum for the Nebius training pilot."""

import argparse
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from aws_scada_inputs import example_from_task, load_aws
from timenet.dataset import Record, Signal, Source
from timenet.types import AnswerTask, TimeOrigin, read_value
from timenet_connectors.datasets.res.hill_of_towie_scada.windows import (
    STEP,
    month_periods,
)
from timenet_connectors.datasets.res.hill_of_towie_scada_curriculum.connector import (
    _months,
    _read_month,
    _read_turbines,
    _signal_values,
)
from timenet_connectors.datasets.res.hill_of_towie_scada_curriculum.tasks import (
    AXIS,
    SPECS,
    curriculum_tasks,
)


def load_months(source, months):
    """Yield tasks at the source cadence and within the connector's split boundaries."""
    metadata = source / "Hill_of_Towie_turbine_metadata.csv"
    turbines = _read_turbines(metadata)
    stations = np.array([turbine.station for turbine in turbines])
    for requested in months:
        date = datetime.strptime(requested, "%Y-%m")
        month = next(month for month in _months(date.year) if month.start.month == date.month)
        archive = source / f"{date.year}.zip"
        values, alarms, lubrication = data = _read_month(archive, month, stations)

        def use_month(*_, month_data=data):
            return month_data

        for period in month_periods(month.start, month.end):
            left = (period.start - month.start) // STEP
            right = (period.end - month.start) // STEP
            steps = slice(left, right)
            for station, turbine in enumerate(turbines):
                signals = tuple(
                    Signal.from_loader(
                        spec=spec,
                        name=spec.name,
                        time_axis=AXIS,
                        n_values=right - left,
                        loader=partial(_signal_values, use_month, (archive, month), channel, station, steps),
                    )
                    for channel, spec in enumerate(SPECS)
                )
                record = Record(
                    start_time=TimeOrigin(period.start + STEP),
                    sources=(Source(name="SCADA", signals=signals),),
                    metadata={"station_id": turbine.station, "turbine_id": turbine.name},
                )
                yield from curriculum_tasks(
                    record, period, values[:, station, steps], alarms[station, steps], lubrication[station, steps]
                )
        print(f"Scanned {requested}", flush=True)


def sample(tasks, aws, limits, caps, seed, train_limit=None):
    """Keep bounded per-label samples with separate pools for each chronological split."""
    rng = random.Random(seed)
    pools, counts = defaultdict(list), Counter()
    for task in tasks:
        split = str(task.split)
        if split not in caps:
            continue
        stage = str(task.metadata["stage"])
        category = str(read_value(task.annotations[0].value))
        label = "caption" if isinstance(task, AnswerTask) else str(read_value(task.targets[0]))
        key = split, stage, category, label
        counts[key] += 1
        pool = pools[key]
        limit = caps[split]
        position = len(pool) if len(pool) < limit else rng.randrange(counts[key])
        if position >= limit:
            continue
        example = example_from_task(aws, limits, task)
        if example is None:
            counts[key] -= 1
            continue
        payload = asdict(example)
        for series in payload["series"]:
            series["values"] = [None if math.isnan(value) else value for value in series["values"]]
        row = {
            "example": payload,
            "stage": stage,
            "cutoff": task.metadata["cutoff_utc"],
            "turbine": task.inputs[0].metadata["turbine_id"],
            "steps": len(example.series[0].values),
        }
        if position == len(pool):
            pool.append(row)
        else:
            pool[position] = row

    for split in caps:
        groups = [pool for key, pool in pools.items() if key[0] == split and pool]
        if split == "train" and train_limit is not None:
            # Truncate by sampling evenly across groups so rare labels survive the cap.
            for group in groups:
                rng.shuffle(group)
            rng.shuffle(groups)
            rows = []
            while groups and len(rows) < train_limit:
                rows.extend(group.pop() for group in groups[: train_limit - len(rows)])
                groups = [group for group in groups if group]
        else:
            rows = [row for group in groups for row in group]
        rng.shuffle(rows)
        if {row["stage"] for row in rows} != {"trend_mcq", "caption", "domain_qa", "forecast"}:
            raise ValueError(f"{split} must cover all four curriculum stages")
        if {row["steps"] for row in rows} != {36, 1008}:
            raise ValueError(f"{split} must cover six-hour and seven-day inputs")
        yield split, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--utils", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--months", nargs="+", default=["2018-01", "2018-07", "2024-03", "2024-07", "2024-10"])
    parser.add_argument("--train-per-label", type=int, default=128)
    parser.add_argument("--validation-per-label", type=int, default=2)
    parser.add_argument("--test-per-label", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.train_per_label, args.validation_per_label) < 1:
        parser.error("label quotas must be positive")
    if (args.test_per_label is not None and args.test_per_label < 1) or (
        args.train_limit is not None and args.train_limit < 1
    ):
        parser.error("test quota and training limit must be positive")
    aws = load_aws(args.utils)
    config = json.loads(args.config.read_text())
    encoder = config["encoder_config"]
    limits = SimpleNamespace(
        max_values=encoder["patch_size"] * encoder["max_patches"], max_media=config["resampler_config"]["max_media"]
    )
    args.output.mkdir(parents=True, exist_ok=False)
    caps = {"train": args.train_per_label, "validation": args.validation_per_label}
    if args.test_per_label is not None:
        caps["test"] = args.test_per_label
    manifest = {"months": args.months, "caps": caps, "train_limit": args.train_limit, "seed": args.seed, "splits": {}}
    for split, rows in sample(load_months(args.source, args.months), aws, limits, caps, args.seed, args.train_limit):
        with gzip.open(args.output / f"{split}.jsonl.gz", "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        manifest["splits"][split] = {
            "rows": len(rows),
            "stages": dict(Counter(row["stage"] for row in rows)),
            "steps": dict(Counter(row["steps"] for row in rows)),
            "categories": dict(Counter(row["example"]["category"] for row in rows)),
        }
        print(split, manifest["splits"][split], flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
