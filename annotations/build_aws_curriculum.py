"""Build the Hill of Towie task views for the AWS OpenTSLM example."""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_curriculum import FIELDS, OPTIONS
from build_examples import CHANNELS
from build_paired_qa import PROMPTS
from build_qualitative_qa import PAIRED_EVIDENCE
from prepare_trend_stage import OBSERVATION_QUESTIONS
from tslm_inputs import iter_evaluation

SPLITS = ("train", "validation", "test")
STAGES = ("mcq", "caption", "trend")
CADENCE = "The SCADA readings are spaced ten minutes apart."
LOCATOR = ("month", "station_id", "start_step", "stop_step", "cutoff_utc", "turbine_id", "rated_capacity_kw", "split", "cohort")


def task(row, stage, category, prompt, answer, *, gold_label, options=(), channels=None, evidence=None):
    key = row["month"], row["station_id"], row["stop_step"]
    return {
        **{field: row[field] for field in LOCATOR},
        "task_id": ":".join(map(str, (stage, *key, category))),
        "pair_id": f"{row.get('training_source', row['split'])}:{row['pair_id']}" if stage == "trend" else None,
        "stage": stage,
        "category": category,
        "channels": list(CHANNELS if channels is None else channels),
        "prompt": f"{CADENCE} {prompt}",
        "options": list(options),
        "answer": answer,
        "gold_label": gold_label,
        "evidence": evidence,
    }


def warmup_task(row, stage, channel, direction):
    name, unit, tolerance = FIELDS[channel]
    comparison = (
        f"Compare {name} in the first six readings with the last six. "
        f"Treat absolute changes below {tolerance:g} {unit} as steady."
    )
    if stage == "mcq":
        return task(row, stage, channel, comparison, direction,
                    gold_label=direction, options=OPTIONS, channels=(channel,))
    return task(row, stage, channel, comparison + " Describe the trend in one qualitative sentence.",
                f"The {name} {direction}.", gold_label=direction, channels=(channel,))


def warmup_train(root, stage):
    for row in pq.read_table(root / f"{stage}.parquet").to_pylist():
        channel, = [key for key, (name, _, _) in FIELDS.items() if name in row["question"]]
        direction, = [value for value in OPTIONS if value in row["target"]]
        yield warmup_task(row, stage, channel, direction)


def holdout_warmup(corpus, path):
    """Label warmup questions from real held-out windows with complete observations."""
    seen = set()
    for example, row in iter_evaluation(corpus, path):
        window = row["month"], row["station_id"], row["stop_step"]
        if window in seen:
            continue
        seen.add(window)
        for channel, values in zip(CHANNELS, example["values"], strict=True):
            if channel not in FIELDS:
                continue
            array = np.asarray(values, dtype=float)
            if not np.isfinite(array).all():
                continue
            delta = array[-6:].mean() - array[:6].mean()
            tolerance = FIELDS[channel][2]
            direction = OPTIONS[0] if delta >= tolerance else OPTIONS[1] if delta <= -tolerance else OPTIONS[2]
            yield row, channel, direction


def trend_tasks(path):
    for batch in pq.ParquetFile(path).iter_batches():
        for row in batch.to_pylist():
            family, label = row["family"], row["expected_outcome"]
            yield task(
                row, "trend", family, OBSERVATION_QUESTIONS[family], label,
                gold_label=label, options=PROMPTS[family][1],
                evidence=PAIRED_EVIDENCE[family, label],
            )


def write_view(output, stage, split, rows):
    rows = list(rows)
    ids = {row["task_id"] for row in rows}
    if len(ids) != len(rows) or not rows or any(row["split"] != split for row in rows):
        raise ValueError(f"{stage}/{split}: empty, duplicate, or out-of-split task")
    pq.write_table(pa.Table.from_pylist(rows), output / f"{stage}-{split}.parquet", compression="zstd")
    return {
        "tasks": len(rows),
        "labels": dict(Counter(row["gold_label"] for row in rows)),
        "start": min(row["cutoff_utc"] for row in rows).isoformat(),
        "end": max(row["cutoff_utc"] for row in rows).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--warmup", type=Path, default=Path("curriculum-20k"))
    parser.add_argument("--trend", type=Path, default=Path("large-trend-data"))
    parser.add_argument("--test", type=Path, default=Path("evidence-continuation-data/focused/test.parquet"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = {"train": args.trend / "qa.parquet", "validation": args.trend / "validation.parquet", "test": args.test}
    counts = {}
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        for split, source in sources.items():
            counts[f"trend-{split}"] = write_view(staged, "trend", split, trend_tasks(source))
            held_out = list(holdout_warmup(args.corpus, source)) if split != "train" else []
            for stage in ("mcq", "caption"):
                rows = (
                    warmup_train(args.warmup, stage)
                    if split == "train"
                    else (warmup_task(row, stage, channel, direction) for row, channel, direction in held_out)
                )
                counts[f"{stage}-{split}"] = write_view(staged, stage, split, rows)
        manifest = {
            "training_order": STAGES,
            "corpus_source": json.loads((args.corpus / "manifest.json").read_text())["source"],
            "views": counts,
            "trend_sources": {split: str(path) for split, path in sources.items()},
            "window_steps": 36,
            "cadence_seconds": 600,
            "full_outage_route": "availability_check",
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(args.output)
    print({key: value["tasks"] for key, value in counts.items()})


if __name__ == "__main__":
    main()
