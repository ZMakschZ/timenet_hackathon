"""Build matching train, validation and test QA views with one evidence-and-answer target."""

import argparse
import json
from pathlib import Path

import duckdb

SPLITS = ("train", "validation", "test")


def build(corpus: Path, output: Path):
    """Use the existing chronological partitions and the same prompt format in each."""
    source = json.loads((corpus / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    with duckdb.connect() as con:
        con.read_parquet(
            [
                str(corpus / row["month"] / "answers.parquet")
                for row in source["months"]
            ],
            filename=True,
        ).create_view("answers")
        con.execute(
            "CREATE TABLE stations (station_id INTEGER, turbine_id VARCHAR, rated_capacity_kw INTEGER)"
        )
        con.executemany(
            "INSERT INTO stations VALUES (?, ?, ?)",
            [
                (row["Station ID"], row["Turbine Name"], row["Rated power (kW)"])
                for row in source["turbines"]
            ],
        )
        invalid = con.execute("""
            SELECT count(*) FROM answers LEFT JOIN stations USING (station_id)
            WHERE turbine_id IS NULL OR NOT starts_with(qa_answer, reasoning_answer || ' ')
              OR length(trim(substring(qa_answer, length(reasoning_answer) + 1))) = 0
        """).fetchone()[0]
        if invalid:
            raise ValueError(
                f"{invalid} rows missing station metadata or a distinct operator answer"
            )
        for split in SPLITS:
            view = con.sql(
                """
                SELECT regexp_extract(filename, '(20[0-9]{2}_[0-9]{2})/answers', 1) AS month,
                    station_id, turbine_id, rated_capacity_kw, split, cohort, family, scenario,
                    start_step, stop_step, cutoff_utc,
                    qa_question || ' Give a short evidence summary, then a final answer introduced by the label Answer:' AS question,
                    reasoning_answer AS evidence,
                    trim(substring(qa_answer, length(reasoning_answer) + 1)) AS final_answer,
                    reasoning_answer || chr(10) || 'Answer: ' ||
                        trim(substring(qa_answer, length(reasoning_answer) + 1)) AS target
                FROM answers JOIN stations USING (station_id)
                WHERE split = ?
                ORDER BY month, station_id, stop_step, scenario
            """,
                params=[split],
            )
            view.write_parquet(str(output / f"{split}.parquet"), compression="zstd")
        counts = dict(
            con.execute("SELECT split, count(*) FROM answers GROUP BY split").fetchall()
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": source["source"],
                "tasks": counts,
                "target_format": "<evidence>\nAnswer: <final answer>",
            },
            indent=2,
        )
        + "\n"
    )
    print(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.corpus, args.output)


if __name__ == "__main__":
    main()
