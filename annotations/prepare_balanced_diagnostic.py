"""Prepare one balanced replay pass for evidence-first SCADA QA."""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from build_paired_qa import PROMPTS

PAIRS_PER_SOURCE = 250


def sample_pairs(path, seed):
    grouped = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        if row["split"] != "train":
            raise ValueError("Diagnostic training expects training rows only")
        grouped[row["pair_id"]].append(row)
    pools = defaultdict(list)
    for left, right in grouped.values():
        family = left["family"]
        if right["family"] != family or {
            left["expected_outcome"], right["expected_outcome"]
        } != set(PROMPTS[family][1]):
            raise ValueError(f"Invalid training pair: {left['pair_id']}")
        pools[family].append((left, right))
    rng = Random(seed)
    return {
        family: rng.sample(pools[family], PAIRS_PER_SOURCE)
        for family in PROMPTS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay", type=Path, default=Path("paired-qa-evidence-first"))
    parser.add_argument("--new", type=Path, default=Path("evidence-continuation-data/focused"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    old = sample_pairs(args.replay / "qa.parquet", 24121)
    new = sample_pairs(args.new / "qa.parquet", 24122)
    rng = Random(24123)
    families = list(PROMPTS)
    rows = []
    for cycle in range(2 * PAIRS_PER_SOURCE):
        # Each optimizer batch has an opposing-outcome pair from every family.
        # Alternating provenance across families and cycles keeps replay at half.
        block = []
        for index, family in enumerate(families):
            replay = (cycle + index) % 2 == 0
            pair = (old if replay else new)[family][cycle // 2]
            block.extend({**row, "training_source": "replay" if replay else "new"} for row in pair)
        rng.shuffle(block)
        for row in block:
            rows.append({**row, "training_order": len(rows)})
    rows.sort(key=lambda row: (row["month"], row["station_id"], row["stop_step"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "data"
        staged.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), staged / "qa.parquet", compression="zstd")
        for filename in ("validation.parquet", "scaling.json", "test.parquet"):
            shutil.copyfile(args.new / filename, staged / filename)
        staged.rename(args.output)
    print(f"Prepared {len(rows):,} rows in balanced optimizer batches", flush=True)


if __name__ == "__main__":
    main()
