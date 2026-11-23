"""Read scoped SCADA windows as AWS OpenTSLM Example objects."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from build_examples import CHANNELS
from tslm_inputs import iter_evaluation

AWS_REVISION = "b83a9901d401e9e872ee9048fe61e6c9df9a5b78"
AWS_SHA256 = "c19db31d2ce6e84a4ce02c9a0fdfe0869c53775f77b04dd4be5c893afd8335e2"
CHECKPOINT_REVISION = "5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983"
CONFIG_SHA256 = "4cc0f5236517b6288ecf14ce54270acf9cb931232b399494caa0e1919faf7348"
DURATION_CHANNELS = {"ON", "READY", "ERR"}
NAMES = dict(zip(CHANNELS, (
    "mean active power", "mean wind speed", "mean generator speed",
    "interval-end active power reference", "mean main shaft speed",
    "running time", "ready time", "turbine error active time",
    "mean blade A pitch", "mean ambient temperature",
    "mean winding 1U1 temperature", "mean winding 1V1 temperature",
    "mean winding 1W1 temperature", "mean phase R voltage", "mean phase S voltage",
    "mean phase T voltage", "mean phase R current", "mean phase S current",
    "mean phase T current", "mean reactive power",
), strict=True))
FULL_OUTAGE_RESPONSE = "No usable SCADA readings are available in this window. Verify live telemetry and controller status."


def load_aws(path):
    """Load the pinned AWS sample through its native Python API."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != AWS_SHA256:
        raise ValueError(f"Expected AWS utils.py at revision {AWS_REVISION}")
    spec = importlib.util.spec_from_file_location("aws_opentslm_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def example_from_window(aws, window, row):
    """Return one Example, or None if the selected channels are entirely unobserved."""
    selected = set(row["channels"])
    series, absent = [], []
    for channel, values, mask in zip(CHANNELS, window["values"], window["valid"], strict=True):
        if channel not in selected:
            continue
        readings = [
            float(value)
            if present and value is not None and math.isfinite(value)
            and (channel not in DURATION_CHANNELS or 0 <= value <= 600)
            else math.nan
            for value, present in zip(values, mask, strict=True)
        ]
        name = f"{NAMES[channel]} ({CHANNELS[channel][2]})"
        if all(math.isnan(value) for value in readings):
            absent.append(name)
        else:
            series.append(aws.SeriesInput(values=readings, name=name))
    if not series:
        return None
    prompt = row["prompt"]
    if absent:
        prompt += f" No observations are available for {', '.join(absent)} in this window."
    return aws.Example(
        series=series, prompt=prompt, options=row["options"], answer=row["answer"], category=row["category"]
    )


def iter_examples(aws, corpus, path):
    """Yield native examples with their audit rows, including the no-media route."""
    for window, row in iter_evaluation(corpus, path, question_column="prompt"):
        yield example_from_window(aws, window, row), row


def model_examples(aws, corpus, path):
    """Yield the examples accepted by the stock AWS train and evaluation functions."""
    for example, _ in iter_examples(aws, corpus, path):
        if example is not None:
            yield example


def example_from_task(aws, model, task):
    """Adapt a scoped TimeF curriculum task to a native AWS example."""
    from timenet.types import AnswerTask, ClassificationTask, TimeInterval, read_value

    if not isinstance(task, (AnswerTask, ClassificationTask)) or not isinstance(task.scope, TimeInterval):
        raise ValueError("Expected a scoped classification or caption task")
    if not task.scope.time_series or task.prompt is None or len(task.targets or ()) != 1:
        raise ValueError("Expected selected channels, a prompt, and one text target")
    start, start_rest = divmod(task.scope.start_us, 600_000_000)
    stop, stop_rest = divmod(task.scope.end_us, 600_000_000)
    if start_rest or stop_rest or start < 0 or stop <= start:
        raise ValueError("Expected a scope on the ten-minute SCADA grid")
    if stop - start > model.max_values:
        raise ValueError("The task exceeds the AWS checkpoint's series length")

    series, absent = [], []
    for signal in task.scope.time_series:
        if stop > signal.n_values:
            raise ValueError("Task scope extends past the source signal")
        key = signal.spec.spec_type
        name = f"{key}: {signal.spec.name} ({signal.spec.unit_value:~})"
        readings = [
            float(value)
            if value is not None and math.isfinite(value)
            and (key not in DURATION_CHANNELS or 0 <= value <= 600)
            else math.nan
            for value in signal.read_steps(start, stop).to_pylist()
        ]
        if all(math.isnan(value) for value in readings):
            absent.append(name)
        else:
            series.append(aws.SeriesInput(values=readings, name=name))
    if not series:
        return None
    if len(series) > min(32, model.max_media):
        raise ValueError("The task exceeds the AWS series limit")

    annotations = {note.key: read_value(note.value) for note in task.annotations}
    options = [label.name for label in task.label_set.labels] if isinstance(task, ClassificationTask) else []
    prompt = "The SCADA readings are spaced ten minutes apart. " + task.prompt.read()
    if absent:
        prompt += f" No observations are available for {', '.join(absent)} in this window."
    answer = read_value(task.targets[0])
    if not isinstance(answer, str):
        raise ValueError("The AWS curriculum requires a text target")
    return aws.Example(series, prompt, options, answer, str(annotations["task_category"]))


def verify_example(aws, config, example):
    prompt, values, observed = aws.OpenTSLMFlamingo.prepare(
        config, example.series, example.evaluated_question()
    )
    if not (
        len(example.series)
        == values.shape[0]
        == observed.shape[0]
        == prompt.count(aws.MEDIA_TOKEN)
        == prompt.count(aws.END_OF_CHUNK_TOKEN)
    ):
        raise ValueError("AWS media markers and padded tensor rows disagree")
    for index, series in enumerate(example.series):
        mask = [not math.isnan(value) for value in series.values]
        actual = observed[index].tolist()
        if len(mask) != 36 or actual[:len(mask)] != mask or any(actual[len(mask):]):
            raise ValueError("AWS observed mask disagrees with the source window")
    if aws.chosen_option(example.answer, example.options or [example.answer]) != example.answer:
        raise ValueError("The target cannot round-trip through the AWS scorer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--aws-utils", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    aws = load_aws(args.aws_utils)
    config_bytes = args.checkpoint_config.read_bytes()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if config_digest != CONFIG_SHA256:
        raise ValueError(f"Expected tslm-nov26 config at revision {CHECKPOINT_REVISION}")
    config = json.loads(config_bytes)
    encoder = config["encoder_config"]
    limits = SimpleNamespace(
        patch_size=encoder["patch_size"],
        max_values=encoder["patch_size"] * encoder["max_patches"],
        max_media=config["resampler_config"]["max_media"],
    )
    views, smoke = {}, {}
    for path in sorted(args.dataset.glob("*.parquet")):
        counts, labels, covered = Counter(), Counter(), set()
        for example, row in iter_examples(aws, args.corpus, path):
            if example is None:
                if (row["stage"], row["category"], row["gold_label"]) != ("trend", "telemetry", "missing"):
                    raise ValueError(f"Unexpected no-media task: {row['task_id']}")
                counts["availability_check"] += 1
                smoke.setdefault("full_outage", {
                    "task_id": row["task_id"], "split": row["split"], "operator_answer": FULL_OUTAGE_RESPONSE,
                })
                continue
            counts["model_examples"] += 1
            labels[row["gold_label"]] += 1
            if len(example.series) > min(32, limits.max_media) or any(
                len(series.values) > limits.max_values for series in example.series
            ):
                raise ValueError(f"Out-of-bounds media: {row['task_id']}")
            partial = len(example.series) < len(row["channels"]) or any(
                math.isnan(value) for series in example.series for value in series.values
            )
            zero = any(value == 0 for series in example.series for value in series.values)
            counts["examples_with_missingness"] += partial
            counts["examples_with_measured_zero"] += zero
            signature = (row["category"], row["gold_label"], partial, zero, len(example.series))
            if signature not in covered:
                verify_example(aws, limits, example)
                covered.add(signature)
                counts["aws_prepare_checked"] += 1
                for kind, present in (("partial_gap", partial), ("measured_zero", zero)):
                    if present:
                        smoke.setdefault(kind, {
                            "task_id": row["task_id"], "split": row["split"], "media_items": len(example.series),
                        })
        views[path.stem] = {**counts, "model_labels": dict(labels)}
        print(path.stem, dict(counts), flush=True)
    args.report.write_text(json.dumps({
        "aws_revision": AWS_REVISION, "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_config_sha256": config_digest,
        "checkpoint_limits": vars(limits), "views": views, "smoke": smoke,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
