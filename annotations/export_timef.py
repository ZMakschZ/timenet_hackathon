"""Export the generated Hill of Towie corpus as a local, scoped TimeF QA dataset."""

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from build_corpus import segments
from build_examples import CHANNELS
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.types import (
    AnswerTask,
    DatasetMetadata,
    MissingReason,
    MissingValues,
    TimeInterval,
    TimeOrigin,
    TimeSeriesSpec,
    ureg,
)

from timenet import TimeNet

STEP_US = 600_000_000
AXIS = RegularAxis.from_rate_hz(Fraction(1, 600))
DATASET_ID = "res/hill-of-towie-scada-qa"
MISSING = MissingValues(
    reason=MissingReason.NOT_RECORDED,
    description="Missing SCADA reading in the pinned release.",
)
UNIT_NAMES = {
    "kW": "kilowatt",
    "rpm": "rpm",
    "m/s": "meter / second",
    "s": "second",
    "deg": "degree",
    "°C": "degC",
    "V": "volt",
    "A": "ampere",
    "kvar": "kilovolt * ampere",
}


def add_records(dataset, folder, turbines):
    """Add one nullable 20-channel record per station and segment."""
    monthly = pq.ParquetFile(folder / "signals.parquet")
    start = pd.Timestamp(
        folder.name.replace("_", "-") + "-01", tz="UTC"
    ).to_pydatetime()
    step = pd.Timedelta(minutes=10).to_pytimedelta()
    records = {}
    specs = {
        alias: TimeSeriesSpec(
            spec_type=field,
            name=field,
            unit_value=ureg.parse_units(UNIT_NAMES[unit]),
            dtype="float64",
            missing=MISSING,
        )
        for alias, (_, field, unit) in CHANNELS.items()
    }
    for group in range(monthly.num_row_groups):
        raw = monthly.read_row_group(group)
        station = raw["station_id"][0].as_py()
        for left, right, _, cohort in segments(folder.name):
            signals = tuple(
                Signal.from_values(
                    raw[alias].slice(left, right - left),
                    spec=spec,
                    name=spec.name,
                    time_axis=AXIS,
                )
                for alias, spec in specs.items()
            )
            turbine = turbines[station]
            record = Record(
                start_time=TimeOrigin(start + (left + 1) * step),
                sources=(Source(name="SCADA", signals=signals),),
                metadata={
                    "station_id": station,
                    "turbine_id": turbine["Turbine Name"],
                    "rated_power_kw": turbine["Rated power (kW)"],
                    "cohort": cohort,
                    "source_month": folder.name,
                },
            )
            dataset.add_record(record=record)
            records[station, cohort] = record, left
    return records


def tasks_from_month(folder, records):
    """Stream the paired text targets against a shared scope."""
    for batch in pq.ParquetFile(folder / "answers.parquet").iter_batches(
        batch_size=4096
    ):
        for row in batch.to_pylist():
            cutoff = row["cutoff_utc"].isoformat()
            record, left = records[row["station_id"], row["cohort"]]
            window_id = f"{record.metadata['turbine_id']}_{cutoff}"
            scope = TimeInterval(
                start_us=(row["start_step"] - left) * STEP_US,
                end_us=(row["stop_step"] - left) * STEP_US,
            )
            for stage in ("reasoning", "qa"):
                yield AnswerTask(
                    inputs=(record,),
                    scope=scope,
                    split=row["split"],
                    prompt=row[f"{stage}_question"],
                    targets=(row[f"{stage}_answer"],),
                    metadata={
                        "stage": stage,
                        "scenario": row["scenario"],
                        "family": row["family"],
                        "window_id": window_id,
                        "cutoff_utc": cutoff,
                        "cohort": row["cohort"],
                    },
                )


def export(root, registry, months, version):
    """Write a local dataset version, either full or restricted for a smoke test."""
    manifest = json.loads((root / "manifest.json").read_text())
    turbines = {row["Station ID"]: row for row in manifest["turbines"]}
    dataset = TimeFDataset(
        metadata=DatasetMetadata(
            dataset_id=DATASET_ID,
            dataset_version=version,
            source_revision="2.1.0",
            name="Hill of Towie SCADA Operator QA",
            description="Two-stage operator QA on 20-channel, six-hour Hill of Towie SCADA windows.",
            source_url=manifest["source"],
            license="CC-BY-4.0",
            domains=("energy",),
            tags=("wind-energy", "scada", "qa"),
            citation="Clerc, A., and Lingkan, E. (2026). Hill of Towie wind farm open dataset (v2.1.0). Zenodo.",
        )
    )
    registered = [
        (root / month, add_records(dataset, root / month, turbines)) for month in months
    ]

    def stream():
        for folder, records in registered:
            yield from tasks_from_month(folder, records)

    dataset.set_task_stream([AnswerTask], stream)
    path = dataset.write(path=registry)
    counts = Counter()
    with TimeNet(sources=[str(registry)]).open(DATASET_ID, version=version) as reader:
        for task in reader.iter_tasks():
            counts[task.split.value, task.metadata["stage"]] += 1
    print(path, dict(counts))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    months = args.months or [row["month"] for row in manifest["months"]]
    export(args.corpus, args.registry, months, args.version)


if __name__ == "__main__":
    main()
