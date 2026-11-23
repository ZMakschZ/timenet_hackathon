"""Build an operator QA supplement from an existing, split SCADA curriculum."""

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from aws_scada_inputs import example_from_task, load_aws
from build_nebius_pilot import load_months
from expanded_operator_targets import TELEMETRY, expanded_target, power_reference_target
from run_nebius_pilot import examples
from timenet.types import read_value

_HELDOUT_POWER_QUOTA = 8


def write_target(output, row, target):
    """Write a native example and its exact final answer label."""
    label = target.answer.rsplit("\nAnswer: ", 1)[1]
    if label not in target.options:
        raise ValueError(f"Unexpected label {label!r} for {target.category}")
    fields = asdict(target)
    for channel in fields["series"]:
        channel["values"] = [value if math.isfinite(value) else None for value in channel["values"]]
    record = row | {"example": fields, "target_format": "operator_response", "label": label}
    output.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    return target.category, label


def heldout_tracking(aws, archive, months):
    """Find normal tracking controls on distinct turbine-days in each held-out split."""
    matches = defaultdict(list)
    days = set()
    # The selected telemetry task has 36 readings and at most 26 channels.
    limits = SimpleNamespace(max_media=26, max_values=36)
    for task in load_months(archive, months):
        split = str(task.split)
        if split not in {"validation", "test"} or len(matches[split]) >= _HELDOUT_POWER_QUOTA:
            continue
        if str(read_value(task.annotations[0].value)) != TELEMETRY:
            continue
        cutoff = task.metadata["cutoff_utc"]
        turbine = task.inputs[0].metadata["turbine_id"]
        day = turbine, cutoff[:10]
        if day in days:
            continue
        item = example_from_task(aws, limits, task)
        target = power_reference_target(item) if item is not None else None
        if target is not None and target.answer.endswith("\nAnswer: tracking reference"):
            row = {"stage": "domain_qa", "cutoff": cutoff, "turbine": turbine, "steps": 36}
            matches[split].append((row, target))
            days.add(day)
            if all(len(matches[name]) >= _HELDOUT_POWER_QUOTA for name in ("validation", "test")):
                break
    return matches


def build_split(aws, source_path, output_path, controls):
    """Render full targets and retain each source window's split and metadata."""
    counts = Counter()
    power_windows = set()
    with gzip.open(output_path, "wt") as output:
        for row in examples(aws, source_path):
            item = row.pop("example")
            operator = expanded_target(item)
            if operator is not None:
                counts[write_target(output, row, operator)] += 1
            power = power_reference_target(item)
            if power is not None:
                counts[write_target(output, row, power)] += 1
                power_windows.add((row["turbine"], row["cutoff"]))
        for row, target in controls:
            if (row["turbine"], row["cutoff"]) not in power_windows:
                counts[write_target(output, row, target)] += 1
    categories = {}
    for (category, label), count in sorted(counts.items()):
        categories.setdefault(category, {})[label] = count
    return {"rows": counts.total(), "by_category_and_label": categories}


def main():
    """Export the supplement and report outcome coverage by split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--utils", required=True, type=Path)
    parser.add_argument("--archive", type=Path, help="Raw SCADA directory for held-out tracking controls")
    parser.add_argument("--months", nargs="+", help="Chronological months to scan for held-out controls")
    args = parser.parse_args()
    if bool(args.archive) != bool(args.months):
        parser.error("--archive and --months must be used together")
    aws = load_aws(args.utils)
    controls = heldout_tracking(aws, args.archive, args.months) if args.archive else {}
    args.output.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split in ("train", "validation", "test"):
        filename = f"{split}.jsonl.gz"
        splits[split] = build_split(aws, args.source / filename, args.output / filename, controls.get(split, ()))
    summary = {"source": str(args.source.resolve()), "heldout_control_months": args.months or [], "splits": splits}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
