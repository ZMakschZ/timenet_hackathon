"""Fit fixed SCADA channel scaling using only the historical training signal records."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from build_corpus import segments
from build_examples import CHANNELS


def fit(corpus):
    """Share transforms within the winding, phase voltage and phase current groups."""
    names = tuple(CHANNELS)
    sums = np.zeros((3, len(names)))
    for folder in sorted(corpus.glob("20*_*")):
        spans = [
            (left, right)
            for left, right, split, _ in segments(folder.name)
            if split == "train"
        ]
        if not spans:
            continue
        file = pq.ParquetFile(folder / "signals.parquet")
        for station in range(file.num_row_groups):
            table = file.read_row_group(station, columns=list(names))
            for left, right in spans:
                for index, channel in enumerate(names):
                    values = table[channel].slice(left, right - left).to_numpy()
                    valid = np.isfinite(values)
                    if channel in ("ON", "READY", "ERR"):
                        valid &= (values >= 0) & (values <= 600)
                    readings = values[valid]
                    sums[:, index] += len(readings), readings.sum(), (readings**2).sum()
    for group in (("U", "V", "Z"), ("VR", "VS", "VT"), ("IR", "IS", "IT")):
        indices = [names.index(channel) for channel in group]
        combined = sums[:, indices].sum(axis=1)
        sums[:, indices] = combined[:, None]
    if np.any(sums[0] == 0):
        raise ValueError("Every channel requires at least one observed training value")
    center = sums[1] / sums[0]
    scale = np.sqrt(np.maximum(sums[2] / sums[0] - center**2, 0))
    return {
        "channels": names,
        "center": center.tolist(),
        "scale": np.where(scale > 0, scale, 1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(fit(args.corpus), indent=2) + "\n")


if __name__ == "__main__":
    main()
