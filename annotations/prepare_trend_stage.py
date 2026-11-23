"""Separate SCADA observation supervision from downstream operator checks."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from build_paired_qa import PROMPTS
from build_qualitative_qa import PAIRED_EVIDENCE

OBSERVATION_QUESTIONS = {
    "power": "How closely does the latest mean power track its interval-end reference?",
    "operation": "Do the latest speed, power and running-time readings show generation or idle operation?",
    "temperature": "Did the generator windings warm or cool over the latest hour?",
    "electrical": "Are the latest phase-current averages close together or widely separated?",
    "telemetry": "Did the selected operating channels report at cutoff, or were the selected readings missing?",
    "maintenance": "Was turbine error active at cutoff and earlier in the window, or absent throughout?",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    guidance = {}
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "data"
        staged.mkdir()
        for filename in ("qa.parquet", "validation.parquet", "test.parquet"):
            source = pq.ParquetFile(args.source / filename)
            with pq.ParquetWriter(staged / filename, source.schema_arrow, compression="zstd") as writer:
                for batch in source.iter_batches(batch_size=8192):
                    questions, targets = [], []
                    for row in batch.to_pylist():
                        family, label = row["family"], row["expected_outcome"]
                        observation = PAIRED_EVIDENCE[family, label]
                        answer = f"\nAnswer: {label}."
                        full = row["target"]
                        prefix = f"{observation} "
                        if not (full.startswith(prefix) and full.endswith(answer)):
                            raise ValueError(f"Unexpected paired target for {row['pair_id']}")
                        action = full[len(prefix):-len(answer)]
                        key = family, label
                        if not action or guidance.setdefault(key, action) != action:
                            raise ValueError(f"Inconsistent operator check for {key}")
                        options = " or ".join(PROMPTS[family][1])
                        questions.append(
                            f"{OBSERVATION_QUESTIONS[family]} Briefly describe the observed evidence. "
                            f"End on a new line with Answer: followed by {options}."
                        )
                        targets.append(observation + answer)
                    table = pa.Table.from_batches([batch])
                    for field, values in (("question", questions), ("target", targets)):
                        index = table.schema.get_field_index(field)
                        table = table.set_column(index, field, pa.array(values))
                    writer.write_table(table, row_group_size=512)
            print(f"{filename}: {source.metadata.num_rows:,} trend-only examples", flush=True)
        if guidance.keys() != PAIRED_EVIDENCE.keys():
            raise ValueError("Incomplete operator guidance")
        operators = {
            family: {
                label: {"observation": PAIRED_EVIDENCE[family, label], "check": guidance[family, label]}
                for label in labels
            }
            for family, (_, labels) in PROMPTS.items()
        }
        (staged / "operator-actions.json").write_text(json.dumps(operators, indent=2) + "\n")
        shutil.copyfile(args.source / "scaling.json", staged / "scaling.json")
        staged.rename(args.output)


if __name__ == "__main__":
    main()
