"""Render all matching scenario pairs from the full-corpus index into Parquet."""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from build_corpus import MONTHS
from build_examples import CHANNELS
from scenario_answers import ANSWERS

SCHEMA = pa.schema(
    [
        ("station_id", pa.int32()),
        ("start_step", pa.int32()),
        ("stop_step", pa.int32()),
        ("cutoff_utc", pa.timestamp("us", tz="UTC")),
        ("split", pa.string()),
        ("cohort", pa.string()),
        ("family", pa.string()),
        ("scenario", pa.string()),
        ("reasoning_question", pa.string()),
        ("reasoning_answer", pa.string()),
        ("qa_question", pa.string()),
        ("qa_answer", pa.string()),
    ]
)


def render_month(folder):
    index_file = folder / "windows.parquet"
    index = pq.read_table(index_file).to_pandas()
    index = index[index.scenarios.map(len).gt(0)]
    counts = Counter()
    temp = folder / "answers.tmp"
    start = pd.Timestamp(folder.name.replace("_", "-") + "-01", tz="UTC")
    with pq.ParquetWriter(temp, SCHEMA, compression="zstd") as writer:
        values = pq.ParquetFile(folder / "signals.parquet")
        stations = pq.read_table(
            folder / "signals.parquet", columns=["station_id"]
        ).column(0)
        station_groups = {int(v.as_py()): i for i, v in enumerate(stations.unique())}
        for station, group in index.groupby("station_id"):
            frame = values.read_row_group(
                station_groups[station], columns=list(CHANNELS)
            ).to_pandas()
            frame.index = pd.date_range(
                start + pd.Timedelta(minutes=10), periods=len(frame), freq="10min"
            )
            rows = []
            for row in group.itertuples(index=False):
                window = frame.iloc[row.start_step : row.stop_step]
                for label in row.scenarios:
                    family, _, render = ANSWERS[label]
                    (rq, ra), (qq, qa) = render(window)
                    if not all((rq, ra, qq, qa)) or "nan" in ra.lower().split():
                        raise ValueError(
                            f"Invalid answer for {station}, {row.cutoff_utc}, {label}"
                        )
                    rows.append(
                        {
                            "station_id": station,
                            "start_step": row.start_step,
                            "stop_step": row.stop_step,
                            "cutoff_utc": row.cutoff_utc,
                            "split": row.split,
                            "cohort": row.cohort,
                            "family": family,
                            "scenario": label,
                            "reasoning_question": rq,
                            "reasoning_answer": ra,
                            "qa_question": qq,
                            "qa_answer": qa,
                        }
                    )
                    counts[row.split, row.cohort, family, label] += 1
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    temp.replace(folder / "answers.parquet")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for month, result in zip(
            MONTHS,
            pool.map(render_month, (args.corpus / m for m in MONTHS)),
            strict=True,
        ):
            counts.update(result)
            print(month, sum(result.values()), flush=True)
    summary = [
        {
            "split": split,
            "cohort": cohort,
            "family": family,
            "scenario": label,
            "pairs": count,
        }
        for (split, cohort, family, label), count in sorted(counts.items())
    ]
    (args.corpus / "answer_counts.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
