"""Build a 20,000-example SCADA analogue of OpenTSLM's curriculum."""

import argparse
import json
from pathlib import Path
from random import Random

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 2026
FIELDS = {
    "P": ("mean active power", "kW", 40),
    "W": ("mean wind speed", "m/s", 0.5),
    "R": ("mean generator speed", "rpm", 40),
    "U": ("mean winding 1U1 temperature", "°C", 1),
}
OPTIONS = ("increased", "decreased", "stayed about the same")


def make_warmup(root, output, stations, con):
    """Generate balanced trend MCQ and time-series captions from training windows."""
    con.read_parquet(str(root / "*" / "windows.parquet"), filename=True).create_view(
        "windows"
    )
    candidates = (
        con.sql("""
        WITH src AS (
            SELECT regexp_extract(filename, '(20[0-9]{2}_[0-9]{2})/windows', 1) AS month,
                * EXCLUDE(filename, scenarios)
            FROM windows WHERE split = 'train'
        )
        SELECT * FROM src
        QUALIFY row_number() OVER (
            PARTITION BY month, station_id
            ORDER BY hash(station_id, stop_step, month, 2026)
        ) <= 12
        ORDER BY month, station_id, stop_step
    """)
        .arrow()
        .read_all()
    )
    rng = Random(SEED)
    bins = {label: [] for label in OPTIONS}
    chosen = []
    for rows in candidates.to_batches():
        chosen.extend(rows.to_pylist())
    from itertools import groupby

    for month, records in groupby(chosen, key=lambda row: row["month"]):
        reader = pq.ParquetFile(root / month / "signals.parquet")
        for station, group in groupby(records, key=lambda row: row["station_id"]):
            index, name, rating = stations[station]
            signals = reader.read_row_group(index, columns=list(FIELDS))
            for row in group:
                start = row["start_step"]
                for channel, (label, unit, tolerance) in FIELDS.items():
                    values = signals[channel].slice(start, 36).to_numpy()
                    if not np.isfinite(values).all():
                        continue
                    first, last = float(values[:6].mean()), float(values[-6:].mean())
                    change = last - first
                    direction = (
                        OPTIONS[0]
                        if change >= tolerance
                        else OPTIONS[1]
                        if change <= -tolerance
                        else OPTIONS[2]
                    )
                    bins[direction].append(
                        (
                            {**row, "turbine_id": name, "rated_capacity_kw": rating},
                            label,
                            unit,
                            tolerance,
                            first,
                            last,
                        )
                    )
    for records in bins.values():
        rng.shuffle(records)
    used = set()
    for stage in ("mcq", "caption"):
        examples = []
        for direction, count in zip(OPTIONS, (1334, 1333, 1333), strict=True):
            for _ in range(count):
                while True:
                    row, label, unit, tolerance, first, last = bins[direction].pop()
                    key = row["month"], row["station_id"], row["stop_step"]
                    if key not in used:
                        used.add(key)
                        break
                if stage == "mcq":
                    options = list(OPTIONS)
                    rng.shuffle(options)
                    question = (
                        f"Compare {label} in the first six readings with the last six. "
                        f"Treat absolute changes below {tolerance:g} {unit} as steady. "
                        + " ".join(
                            f"{letter}. {option}."
                            for letter, option in zip("ABC", options, strict=True)
                        )
                    )
                    target = f"Answer: {'ABC'[options.index(direction)]}. {direction}."
                else:
                    question = (
                        f"Describe how {label} changed from the first six to the last six readings. "
                        "Report both means and the direction in one short observation."
                    )
                    target = f"The {label} {direction}, from {first:.1f} to {last:.1f} {unit}."
                examples.append({**row, "question": question, "target": target})
        examples.sort(
            key=lambda row: (row["month"], row["station_id"], row["stop_step"])
        )
        pq.write_table(
            pa.Table.from_pylist(examples),
            output / f"{stage}.parquet",
            compression="zstd",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("targeted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.corpus / "manifest.json").read_text())
    stations = {
        row["Station ID"]: (index, row["Turbine Name"], row["Rated power (kW)"])
        for index, row in enumerate(source["turbines"])
    }
    with duckdb.connect() as con:
        make_warmup(args.corpus, args.output, stations, con)
        con.read_parquet(str(args.targeted / "train.parquet")).create_view("qa")
        con.sql("""
            WITH independent AS (
                SELECT * FROM qa WHERE split = 'train'
                QUALIFY row_number() OVER (
                    PARTITION BY month, station_id, stop_step
                    ORDER BY hash(scenario, station_id, stop_step, 2026)
                ) = 1
            ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY substring(month, 1, 4), family
                    ORDER BY hash(month, station_id, stop_step, scenario, 2026)
                ) AS rank
                FROM independent
            ), selected AS (
                SELECT * EXCLUDE(rank) FROM ranked
                ORDER BY rank, hash(month, station_id, stop_step, 2026)
                LIMIT 12000
            )
            SELECT * FROM selected ORDER BY month, station_id, stop_step
        """).write_parquet(str(args.output / "qa.parquet"), compression="zstd")
    counts = {
        stage: pq.ParquetFile(args.output / f"{stage}.parquet").metadata.num_rows
        for stage in ("mcq", "caption", "qa")
    }
    if counts != {"mcq": 4000, "caption": 4000, "qa": 12000}:
        raise ValueError(f"Incorrect curriculum size: {counts}")
    print(counts)


if __name__ == "__main__":
    main()
