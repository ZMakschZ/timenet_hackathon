"""Measure operator QA coverage, prompt shortcuts, overlap, and SCADA checks."""

import argparse
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq


def report_tables(connection):
    """Aggregate questions, windows, and answer pairs without duplicating signals."""
    queries = {
        "question vocabulary and verbatim reasoning prefix": """
            SELECT count(DISTINCT qa_question), count(DISTINCT reasoning_question),
                   count(DISTINCT scenario),
                   count(*) FILTER (starts_with(qa_answer, reasoning_answer)), count(*)
            FROM answers
        """,
        "operator questions identifying a unique scenario": """
            SELECT count(*) FILTER(n=1), count(*) FROM
              (SELECT qa_question, count(DISTINCT scenario) AS n FROM answers GROUP BY 1)
        """,
        "test pairs with a prompt also seen in training": """
            SELECT count(*) FROM answers WHERE split='test'
                 AND qa_question IN (SELECT qa_question FROM answers WHERE split='train')
        """,
        "windows, matched windows, and windows with multiple labels by split": """
            SELECT split, count(*), count(*) FILTER (len(scenarios)>0),
                   count(*) FILTER (len(scenarios)>1)
            FROM windows GROUP BY ALL ORDER BY 1
        """,
        "five most common training scenarios": """
            SELECT scenario, count(*) AS n FROM answers WHERE split='train'
            GROUP BY 1 ORDER BY n DESC LIMIT 5
        """,
        "rare labels by split": """
            SELECT scenario, split, count(*), count(DISTINCT station_id)
            FROM answers WHERE scenario IN ('winding_spread_widens',
                                             'below_rated_reference',
                                             'near_flat_wind_while_generating')
            GROUP BY ALL ORDER BY 1,2
        """,
        "same-scenario pairs overlapping their nearest prior pair": """
            WITH preceding AS (
                SELECT split, cutoff_utc, lag(cutoff_utc) OVER(
                    PARTITION BY station_id, scenario, split, cohort ORDER BY cutoff_utc
                ) AS previous
                FROM answers
            )
            SELECT split, count(*), count(*) FILTER (cutoff_utc-previous < INTERVAL 6 HOURS)
            FROM preceding GROUP BY 1 ORDER BY 1
        """,
        "maintenance examples by scenario": """
            SELECT scenario, count(*) FROM answers WHERE family='maintenance_triage'
            GROUP BY ALL ORDER BY 2 DESC
        """,
    }
    for name, query in queries.items():
        print(name, connection.execute(query).fetchall(), flush=True)


def scan_values(root):
    """Recheck raw-value support separately from the rendered answers."""
    results = Counter()
    thermal_labels = ("windings_warm_with_load", "windings_cool_as_load_falls")
    channels = [
        "P",
        "W",
        "R",
        "C",
        "S",
        "ON",
        "READY",
        "ERR",
        "B",
        "A",
        "U",
        "V",
        "Z",
        "VR",
        "VS",
        "VT",
        "IR",
        "IS",
        "IT",
        "Q",
    ]
    for path in sorted(root.glob("20*_*")):
        windows = pq.read_table(
            path / "windows.parquet", columns=["station_id", "stop_step"]
        ).to_pandas()
        answers = pq.read_table(
            path / "answers.parquet", columns=["station_id", "stop_step", "scenario"]
        ).to_pandas()
        signals = pq.ParquetFile(path / "signals.parquet")
        for group in range(signals.num_row_groups):
            frame = signals.read_row_group(group).to_pandas()
            station = frame.station_id.iloc[0]
            cutoffs = windows.loc[
                windows.station_id.eq(station), "stop_step"
            ].to_numpy()
            missing = frame[channels].isna().all(axis=1).to_numpy()
            prefix = np.r_[0, missing.cumsum()]
            results["six-hour windows missing every channel"] += int(
                ((prefix[cutoffs] - prefix[cutoffs - 36]) == 36).sum()
            )
            for flag in ("ON", "READY", "ERR"):
                bad = frame[flag].lt(0) | frame[flag].gt(600)
                results[f"{flag} readings outside 0–600 seconds"] += int(bad.sum())
            station_answers = answers[answers.station_id.eq(station)]
            temperatures = frame[["U", "V", "Z"]]
            for label in thermal_labels:
                steps = (
                    station_answers.loc[station_answers.scenario.eq(label), "stop_step"]
                    - 1
                )
                hourly = temperatures.diff(6).mean(axis=1).iloc[steps].abs()
                jumps = (
                    temperatures.diff().mean(axis=1).abs().rolling(6).max().iloc[steps]
                )
                results[f"{label} mean change ≥30°C/hour"] += int(hourly.ge(30).sum())
                results[f"{label} mean step ≥20°C/10min"] += int(jumps.ge(20).sum())
    for name, count in results.items():
        print(name, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    with duckdb.connect() as connection:
        for table in ("answers", "windows"):
            files = sorted(str(path) for path in args.corpus.glob(f"*/{table}.parquet"))
            connection.read_parquet(files).create_view(table)
        report_tables(connection)
    scan_values(args.corpus)


if __name__ == "__main__":
    main()
