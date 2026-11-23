"""Create one neutral-prompt evaluation example for every held-out SCADA window."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from build_examples import CHANNELS

QUESTION = (
    "What is this turbine doing now, what should I check next, "
    "and what would change the decision? Use the available telemetry and state any uncertainty."
)
SPLITS = ("validation", "test")
KEY = ("station_id", "start_step", "stop_step")
SCHEMA = pa.schema(
    [
        ("month", pa.string()),
        ("station_id", pa.int32()),
        ("turbine_id", pa.string()),
        ("rated_capacity_kw", pa.int32()),
        ("split", pa.string()),
        ("cohort", pa.string()),
        ("start_step", pa.int32()),
        ("stop_step", pa.int32()),
        ("cutoff_utc", pa.timestamp("us", tz="UTC")),
        ("question", pa.string()),
        ("expected_state", pa.string()),
        ("latest_power_kw", pa.float64()),
        ("latest_reference_kw", pa.float64()),
        ("latest_wind_mps", pa.float64()),
        ("latest_generator_rpm", pa.float64()),
        ("latest_running_seconds", pa.float64()),
        ("minutes_since_last_observation", pa.int16()),
        ("reference_answer", pa.string()),
        ("scenarios", pa.list_(pa.string())),
        (
            "scenario_references",
            pa.list_(pa.struct([("scenario", pa.string()), ("answer", pa.string())])),
        ),
    ]
)


def state_and_reference(latest, age):
    """Describe the observable state and give a verification step."""
    power, reference, wind, rpm, running = latest
    if age is None:
        return (
            "visibility_unavailable",
            (
                "Operating state is unknown. None of the 20 SCADA channels reported within this six-hour "
                "window. Check live communications or an independent source before recording downtime."
            ),
        )
    if age > 0:
        return (
            "visibility_unavailable",
            (
                f"Operating state at cutoff is unknown. The last interval with any selected reading ended "
                f"{age} minutes earlier. Confirm live telemetry or independently verify turbine state."
            ),
        )
    evidence = []
    for label, reading, unit in (
        ("power", power, "kW"),
        ("power reference", reference, "kW"),
        ("wind", wind, "m/s"),
        ("generator speed", rpm, "rpm"),
        ("in-operation time", running, "s"),
    ):
        if reading is not None:
            evidence.append(
                f"{label} {reading:.3f}".rstrip("0").rstrip(".") + f" {unit}"
            )
    observed = (
        "The latest interval reports " + ", ".join(evidence) + "."
        if evidence
        else (
            "Some telemetry is present at cutoff, but the power, reference, wind, speed and running time are unavailable."
        )
    )
    if (
        power is None
        or rpm is None
        or running is None
        or rpm < 0
        or not 0 <= running <= 600
    ):
        return (
            "state_unknown",
            (
                f"Generation state is uncertain. {observed} Verify power, speed and the live controller "
                "state before assigning an operating condition."
            ),
        )
    if power >= 50 and rpm > 300 and running > 0:
        return (
            "generating_in_latest_interval",
            (
                f"The turbine generated in the latest interval. {observed} Check the trend against the "
                "requested output and review live alarms or limits if output departs from the request."
            ),
        )
    if abs(power) <= 30 and rpm < 100 and running == 0:
        return (
            "idle_or_stopped_at_cutoff",
            (
                f"The latest readings are consistent with idle or stopped operation. {observed} "
                "Check current controller state, dispatch and any start restrictions for the reason."
            ),
        )
    return (
        "mixed_or_transitional",
        (
            f"The latest readings leave the operating state unresolved. {observed} "
            "Check the live controller state and preceding intervals before classifying the event."
        ),
    )


def build_month(folder, turbines, writers, counts):
    """Resolve held-out offsets onto shared monthly signals and attach rule references."""
    windows = pq.read_table(
        folder / "windows.parquet", filters=[("split", "in", SPLITS)]
    )
    if not windows.num_rows:
        return
    scenarios = pq.read_table(
        folder / "answers.parquet",
        columns=[*KEY, "scenario", "qa_answer"],
        filters=[("split", "in", SPLITS)],
    ).to_pylist()
    references = {}
    for row in scenarios:
        references.setdefault(tuple(row[field] for field in KEY), []).append(
            {"scenario": row["scenario"], "answer": row["qa_answer"]}
        )
    signal_file = pq.ParquetFile(folder / "signals.parquet")
    readings = {}
    for group in range(signal_file.num_row_groups):
        table = signal_file.read_row_group(group)
        station = table["station_id"][0].as_py()
        any_observed = np.zeros(len(table), dtype=bool)
        for channel in CHANNELS:
            any_observed |= table[channel].is_valid().to_numpy(zero_copy_only=False)
        latest_index = np.maximum.accumulate(
            np.where(any_observed, np.arange(len(table)), -1)
        )
        readings[station] = table, latest_index
    parts = {split: [] for split in SPLITS}
    for row in windows.to_pylist():
        station = row["station_id"]
        end = row["stop_step"] - 1
        table, latest_index = readings[station]
        last_seen = int(latest_index[end])
        age = (end - last_seen) * 10 if last_seen >= row["start_step"] else None
        latest = tuple(
            table[channel][end].as_py() for channel in ("P", "C", "W", "R", "ON")
        )
        state, reference = state_and_reference(latest, age)
        refs = references.get(tuple(row[field] for field in KEY), [])
        assert [ref["scenario"] for ref in refs] == row["scenarios"]
        turbine = turbines[station]
        parts[row["split"]].append(
            {
                **row,
                "month": folder.name,
                "turbine_id": turbine["Turbine Name"],
                "rated_capacity_kw": turbine["Rated power (kW)"],
                "question": QUESTION,
                "expected_state": state,
                "latest_power_kw": latest[0],
                "latest_reference_kw": latest[1],
                "latest_wind_mps": latest[2],
                "latest_generator_rpm": latest[3],
                "latest_running_seconds": latest[4],
                "minutes_since_last_observation": age,
                "reference_answer": reference,
                "scenario_references": refs,
            }
        )
        counts[row["split"]][state] += 1
    for split in SPLITS:
        if parts[split]:
            writers[split].write_table(
                pa.Table.from_pylist(parts[split], schema=SCHEMA)
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    input_manifest = json.loads((args.corpus / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    turbines = {row["Station ID"]: row for row in input_manifest["turbines"]}
    counts = {split: Counter() for split in SPLITS}
    with (
        pq.ParquetWriter(
            args.output / "validation.parquet", SCHEMA, compression="zstd"
        ) as validation,
        pq.ParquetWriter(
            args.output / "test.parquet", SCHEMA, compression="zstd"
        ) as test,
    ):
        for month in input_manifest["months"]:
            build_month(
                args.corpus / month["month"],
                turbines,
                {"validation": validation, "test": test},
                counts,
            )
    manifest = {
        "question": QUESTION,
        "signals": "Join to corpus/<month>/signals.parquet by station_id and [start_step, stop_step).",
        "source": input_manifest["source"],
        "counts": {
            split: {
                "windows": sum(values.values()),
                "state_counts": dict(sorted(values.items())),
            }
            for split, values in counts.items()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
