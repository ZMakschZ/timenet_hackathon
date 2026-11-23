"""Check every full-corpus window, answer pair and monthly source layout."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from build_corpus import MONTHS, segments
from build_examples import CHANNELS
from scenario_answers import ANSWERS


def check_month(folder, stations):
    """Reconcile physical rows, calendar ownership, scenario index and text pair tables."""
    first = pd.Timestamp(folder.name.replace("_", "-") + "-01", tz="UTC")
    month_steps = (first + pd.offsets.MonthBegin() - first) // pd.Timedelta(minutes=10)
    signals = pq.ParquetFile(folder / "signals.parquet")
    assert signals.schema_arrow.names == ["station_id", *CHANNELS]
    assert signals.num_row_groups == len(stations)
    for row_group, station in enumerate(stations):
        group = signals.read_row_group(row_group, columns=["station_id"]).column(0)
        assert len(group) == month_steps and group.unique().to_pylist() == [station]

    windows = pq.read_table(folder / "windows.parquet").to_pandas()
    candidate_rows = []
    for station in stations:
        for left, right, split, cohort in segments(folder.name):
            candidate_rows.extend(
                (station, stop - 36, stop, split, cohort)
                for stop in range(left + 36, right + 1, 6)
            )
    keys = ["station_id", "start_step", "stop_step", "split", "cohort"]
    assert list(windows[keys].itertuples(index=False, name=None)) == candidate_rows
    assert np.array_equal(
        windows.cutoff_utc, first + pd.to_timedelta(windows.stop_step * 10, unit="min")
    )
    assert windows.scenarios.map(lambda labels: len(set(labels)) == len(labels)).all()
    assert windows.scenarios.map(
        lambda labels: all(label in ANSWERS for label in labels)
    ).all()

    matched = windows.scenarios.map(len).gt(0)
    expected = (
        windows.loc[matched]
        .explode("scenarios")
        .rename(columns={"scenarios": "scenario"})
    )
    answers = pq.read_table(folder / "answers.parquet").to_pandas()
    keys.extend(["cutoff_utc", "scenario"])
    assert list(expected[keys].itertuples(index=False, name=None)) == list(
        answers[keys].itertuples(index=False, name=None)
    )
    assert answers.family.eq(
        answers.scenario.map(lambda label: ANSWERS[label][0])
    ).all()
    for field in ("reasoning_question", "reasoning_answer", "qa_question", "qa_answer"):
        assert answers[field].str.strip().str.len().gt(0).all()
        assert (
            not answers[field]
            .str.contains(
                r"(?<![\w.])(?:nan|inf)(?![\w.])|\[[A-Z][A-Z0-9 /−]*\]",
                case=not field.endswith("answer"),
                regex=True,
            )
            .any()
        )
    assert all(
        qa.startswith(reason)
        for reason, qa in zip(answers.reasoning_answer, answers.qa_answer, strict=True)
    )
    return (
        len(windows),
        int(matched.sum()),
        len(answers),
        answers.groupby(["split", "cohort", "family", "scenario"]).size().to_dict(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    stations = [row["Station ID"] for row in manifest["turbines"]]
    assert [row["month"] for row in manifest["months"]] == list(MONTHS)
    counts = Counter()
    totals = Counter()
    for month in manifest["months"]:
        candidates, matched, pairs, month_counts = check_month(
            args.corpus / month["month"], stations
        )
        assert (candidates, matched, pairs) == (
            month["candidates"],
            month["matched"],
            month["scenario_pairs"],
        )
        counts.update(month_counts)
        totals.update(candidates=candidates, matched=matched, pairs=pairs)
    logged = json.loads((args.corpus / "answer_counts.json").read_text())
    assert counts == {
        (r["split"], r["cohort"], r["family"], r["scenario"]): r["pairs"]
        for r in logged
    }
    print(f"Verified {dict(totals)} across {len(MONTHS)} months")
    for split in ("train", "validation", "test"):
        print(split, sum(count for key, count in counts.items() if key[0] == split))


if __name__ == "__main__":
    main()
