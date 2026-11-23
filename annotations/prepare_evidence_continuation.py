"""Prepare disjoint training stages and held-out panels for evidence-first QA."""

import argparse
import shutil
from collections import Counter, defaultdict, deque
from itertools import chain
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from build_paired_qa import PROMPTS, build
from build_qualitative_qa import convert


def key(row):
    return row["month"], row["station_id"], row["stop_step"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--original", type=Path, default=Path("paired-qa"))
    parser.add_argument("--pool", type=Path, default=Path("paired-qa-100k"))
    parser.add_argument("--previous", type=Path, default=Path("paired-qa-evidence-first"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")

    previous = pq.read_table(args.previous / "qa.parquet").to_pylist()
    used = {key(row) for row in previous}
    by_pair = defaultdict(list)
    pool = pq.read_table(args.pool / "qa.parquet")
    for row in pool.to_pylist():
        by_pair[row["pair_id"]].append(row)
    families = defaultdict(list)
    for pair in by_pair.values():
        left, right = pair
        if left["family"] != right["family"] or left["expected_outcome"] == right["expected_outcome"]:
            raise ValueError("Expected opposing outcomes in each training pair")
        if any(row["split"] != "train" for row in pair):
            raise ValueError("Training pool contains a held-out window")
        if all(key(row) not in used for row in pair):
            families[left["family"]].append(pair)

    rng = Random(24120)
    focus = []
    for family in PROMPTS:
        pairs = families[family]
        rng.shuffle(pairs)
        quota = 2000 if family in ("temperature", "maintenance") else 500
        if len(pairs) < quota:
            raise ValueError(f"Insufficient new {family} pairs")
        focus.extend(pairs[:quota])
        families[family] = pairs[quota:]
    used.update(key(row) for pair in focus for row in pair)

    remaining = {
        family: deque(pair for pair in pairs if all(key(row) not in used for row in pair))
        for family, pairs in families.items()
    }
    scale = []
    needed = (100_000 - len(previous) - 2 * len(focus)) // 2
    if sum(map(len, remaining.values())) < needed:
        raise ValueError("Insufficient non-replayed training pairs for the larger stage")
    while len(scale) < needed:
        for pairs in remaining.values():
            if pairs and len(scale) < needed:
                scale.append(pairs.popleft())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        root = Path(temporary)
        gate = root / "gate-raw"
        final = root / "final-raw"
        previous_test = args.previous / "test.parquet"
        build(args.corpus, gate, test_pairs=120, exclude=[previous_test])
        build(args.corpus, final, test_pairs=120, exclude=[previous_test, gate / "test.parquet"])
        staged = root / args.output.name
        staged.mkdir()
        for name, pairs, test in (("focused", focus, gate), ("scale", scale, final)):
            raw = root / f"{name}-raw"
            raw.mkdir()
            rows = sorted(chain.from_iterable(pairs), key=lambda row: (*key(row), row["family"]))
            pq.write_table(pa.Table.from_pylist(rows, schema=pool.schema), raw / "qa.parquet")
            for filename in ("validation.parquet", "scaling.json"):
                shutil.copyfile(args.original / filename, raw / filename)
            convert(raw, staged / name, evidence_first=True, test=test / "test.parquet")
            print(name, dict(Counter(row["family"] for row in rows)), flush=True)
        staged.rename(args.output)


if __name__ == "__main__":
    main()
