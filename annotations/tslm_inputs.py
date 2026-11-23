"""Load the same masked TSLM input from TimeF tasks or held-out Parquet windows."""

import json
from pathlib import Path

import pyarrow.parquet as pq
from build_examples import CHANNELS


def task_example(task):
    """Keep metadata labels out of the input and recover Arrow validity before tensor conversion."""
    from timenet.types import AnswerTask, TimeInterval

    if not isinstance(task, AnswerTask) or not isinstance(task.scope, TimeInterval):
        raise TypeError("Expected a scoped SCADA answer task")
    if len(task.inputs) != 1 or task.prompt is None or len(task.targets or ()) != 1:
        raise ValueError(
            "Expected one source record, one question, and one text answer"
        )
    record = task.inputs[0]
    channels = {signal.spec.spec_type: signal for signal in record.walk_signals()}
    step_us = 600_000_000
    start, start_rest = divmod(task.scope.start_us, step_us)
    stop, stop_rest = divmod(task.scope.end_us, step_us)
    if start_rest or stop_rest or stop - start != 36:
        raise ValueError("Expected a scope covering 36 ten-minute readings")
    arrays = [
        channels[column].read_steps(start, stop) for _, column, _ in CHANNELS.values()
    ]
    output = _model_input(
        task.prompt.read(),
        record.metadata["turbine_id"],
        task.metadata["cutoff_utc"],
        record.metadata["rated_power_kw"],
        arrays,
    )
    return output, task.targets[0].read()


def _model_input(question, turbine, cutoff, rating, arrays):
    return {
        "question": question,
        "context": {
            "turbine_id": turbine,
            "cutoff_utc": cutoff,
            "rated_capacity_kw": rating,
            "alarm_status": None,
            "controller_status": None,
            "operating_rules": None,
            "matched_baseline": None,
            "independent_evidence": None,
        },
        "values": [array.to_pylist() for array in arrays],
        "valid": [array.is_valid().to_pylist() for array in arrays],
    }


def iter_evaluation(
    corpus: Path, examples: Path, *, stride: int = 1, row_groups=None, question_column="question"
):
    """Yield model input and separate references using one turbine's readings at a time."""
    manifest = json.loads((corpus / "manifest.json").read_text())
    groups = {
        station["Station ID"]: index
        for index, station in enumerate(manifest["turbines"])
    }
    source_month = None
    source_station = None
    index = 0
    for batch in pq.ParquetFile(examples).iter_batches(
        batch_size=2048, row_groups=row_groups
    ):
        for row in batch.to_pylist():
            index += 1
            if (index - 1) % stride:
                continue
            if row["month"] != source_month:
                source_month = row["month"]
                source = pq.ParquetFile(corpus / source_month / "signals.parquet")
                source_station = None
            if row["station_id"] != source_station:
                source_station = row["station_id"]
                table = source.read_row_group(
                    groups[source_station], columns=["station_id", *CHANNELS]
                )
                if table["station_id"][0].as_py() != source_station:
                    raise ValueError(f"Unexpected station in {source_month}")
            start, stop = row["start_step"], row["stop_step"]
            if stop - start != 36 or start < 0 or stop > table.num_rows:
                raise ValueError(f"Invalid evaluation scope at {row['cutoff_utc']}")
            arrays = [table[channel].slice(start, stop - start) for channel in CHANNELS]
            model_input = _model_input(
                row[question_column],
                row["turbine_id"],
                row["cutoff_utc"].isoformat(),
                row["rated_capacity_kw"],
                arrays,
            )
            yield model_input, row
