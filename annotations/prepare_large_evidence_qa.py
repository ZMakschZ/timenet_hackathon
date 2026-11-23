"""Mix the evidence-first replay and unused pools for the time-capped run."""

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from build_paired_qa import PROMPTS


def load_pairs(dataset, stage, claimed_windows):
    pairs = defaultdict(list)
    current_windows = set()
    for row in pq.read_table(dataset / "qa.parquet").to_pylist():
        window = row["month"], row["station_id"], row["stop_step"]
        if row["split"] != "train" or window in claimed_windows:
            raise ValueError(f"{stage} contains a held-out or reused window: {window}")
        current_windows.add(window)
        pairs[row["pair_id"]].append({**row, "training_source": stage})
    claimed_windows.update(current_windows)
    strata = defaultdict(list)
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("Expected exactly two windows per pair")
        family = pair[0]["family"]
        if pair[1]["family"] != family or {
            row["expected_outcome"] for row in pair
        } != set(PROMPTS[family][1]):
            raise ValueError(f"Expected opposing outcomes in {pair[0]['pair_id']}")
        strata[family].append(pair)
    return strata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay", type=Path, default=Path("paired-qa-evidence-first"))
    parser.add_argument("--continuation", type=Path, default=Path("evidence-continuation-data"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")

    mixed = []
    claimed_windows = set()
    rng = Random(24124)
    phases = {
        "replay": args.replay,
        "focused": args.continuation / "focused",
        "scale": args.continuation / "scale",
    }
    for stage, dataset in phases.items():
        for bucket in load_pairs(dataset, stage, claimed_windows).values():
            rng.shuffle(bucket)
            # Spread every family and source across the run even if the cap cuts it short.
            for position, pair in enumerate(bucket):
                rng.shuffle(pair)
                mixed.append(((position + rng.random()) / len(bucket), pair))
    mixed.sort(key=lambda item: item[0])

    rows = []
    for _, pair in mixed:
        for row in pair:
            rows.append({**row, "training_order": len(rows)})
    rows.sort(key=lambda row: (row["month"], row["station_id"], row["stop_step"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "data"
        staged.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), staged / "qa.parquet", compression="zstd")
        for filename in ("validation.parquet", "scaling.json", "test.parquet"):
            shutil.copyfile(phases["scale"] / filename, staged / filename)
        staged.rename(args.output)
    print(f"{len(rows):,} training windows by family", dict(Counter(row["family"] for row in rows)))
    print("Training sources", dict(Counter(row["training_source"] for row in rows)))


if __name__ == "__main__":
    main()
