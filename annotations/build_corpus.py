"""Index every supported Hill of Towie annotation across the pinned release."""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from build_examples import CHANNELS, read_month, select
from scenario_answers import ANSWERS

MONTHS = tuple(
    t.strftime("%Y_%m") for t in pd.date_range("2016-01-01", "2026-04-01", freq="MS")
)
PERIODS = (
    ("2016-01-01", "train", "historical_train"),
    ("2024-10-01", "validation", "post_tuneup_validation"),
    ("2025-01-01", "test", "pre_commissioning"),
    ("2025-10-28T13:00", "test", "dynamic_yaw_commissioning"),
    ("2026-01-07T13:00", "test", "dynamic_yaw_west_trial"),
    ("2026-02-03T18:00", "test", "dynamic_yaw_west_and_east_trial"),
    ("2026-05-01", None, None),
)
INDEX_SCHEMA = pa.schema(
    [
        ("station_id", pa.int32()),
        ("split", pa.string()),
        ("cohort", pa.string()),
        ("start_step", pa.int32()),
        ("stop_step", pa.int32()),
        ("cutoff_utc", pa.timestamp("us", tz="UTC")),
        ("scenarios", pa.list_(pa.string())),
    ]
)
SIGNAL_SCHEMA = pa.schema(
    [("station_id", pa.int32()), *[(key, pa.float64()) for key in CHANNELS]]
)


def segments(month):
    """Yield slices of a month using the forecasting release's cohort boundaries."""
    begin = pd.Timestamp(month.replace("_", "-") + "-01", tz="UTC")
    end = begin + pd.offsets.MonthBegin()
    for (start, split, cohort), (stop, _, _) in itertools.pairwise(PERIODS):
        left = max(begin, pd.Timestamp(start, tz="UTC"))
        right = min(end, pd.Timestamp(stop, tz="UTC"))
        if left < right:
            yield (
                int((left - begin) / pd.Timedelta(minutes=10)),
                int((right - begin) / pd.Timedelta(minutes=10)),
                split,
                cohort,
            )


def build_month(job):
    archives, stations, root, month = job
    folder = root / month
    folder.mkdir(exist_ok=True)
    frames = read_month(archives, month, stations=stations)
    index = []
    with pq.ParquetWriter(
        folder / "signals.parquet", SIGNAL_SCHEMA, compression="zstd"
    ) as writer:
        for station, frame in frames.items():
            writer.write_table(
                pa.Table.from_pandas(
                    frame.assign(station_id=station),
                    schema=SIGNAL_SCHEMA,
                    preserve_index=False,
                )
            )
            for left, right, split, cohort in segments(month):
                segment = frame.iloc[left:right]
                labels = {}
                for label, scores in select(segment):
                    if label not in ANSWERS:
                        raise ValueError(f"Unrecognized scenario {label}")
                    for cutoff in scores.index:
                        labels.setdefault(cutoff, []).append(label)
                for step in range(left + 36, right + 1, 6):
                    cutoff = frame.index[step - 1]
                    index.append(
                        {
                            "station_id": station,
                            "split": split,
                            "cohort": cohort,
                            "start_step": step - 36,
                            "stop_step": step,
                            "cutoff_utc": cutoff,
                            "scenarios": labels.get(cutoff, []),
                        }
                    )
    pq.write_table(
        pa.Table.from_pylist(index, schema=INDEX_SCHEMA),
        folder / "windows.parquet",
        compression="zstd",
    )
    matched = sum(bool(row["scenarios"]) for row in index)
    pairs = sum(len(row["scenarios"]) for row in index)
    return month, len(index), matched, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    turbines = pd.read_csv(args.metadata).sort_values("Station ID")
    stations = tuple(int(i) for i in turbines["Station ID"])
    jobs = [(args.archives, stations, args.output, month) for month in MONTHS]
    summary = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for month, candidates, matched, pairs in pool.map(build_month, jobs):
            print(month, candidates, matched, pairs, flush=True)
            summary.append(
                {
                    "month": month,
                    "candidates": candidates,
                    "matched": matched,
                    "scenario_pairs": pairs,
                }
            )
    manifest = {
        "source": "https://zenodo.org/records/22662930",
        "channels": CHANNELS,
        "turbines": turbines[
            ["Station ID", "Turbine Name", "Rated power (kW)"]
        ].to_dict("records"),
        "months": summary,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
