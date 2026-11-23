"""Decide whether an evidence-first continuation earned promotion."""

import argparse
import json
from pathlib import Path

from evaluate_evidence_first import score_predictions


def read_scores(directory, checkpoint):
    source = directory / f"{checkpoint}_signals.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    per_family = {
        family: score_predictions(
            [(row, row["prediction"]) for row in rows if row["family"] == family], strict=True
        )
        for family in ("temperature", "maintenance")
    }
    return {
        "all": score_predictions([(row, row["prediction"]) for row in rows], strict=True),
        **per_family,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    prior = read_scores(args.evaluation, "previous")
    candidate = read_scores(args.evaluation, "evidence_first")
    pair_delta = {
        group: candidate[group]["paired_both_correct"] - prior[group]["paired_both_correct"]
        for group in prior
    }
    macro_f1_delta = candidate["all"]["macro_f1"] - prior["all"]["macro_f1"]
    format_rate = candidate["all"]["evidence_first_format"] / candidate["all"]["windows"]
    passed = (
        pair_delta["all"] >= 3
        and pair_delta["temperature"] >= 1
        and pair_delta["maintenance"] >= 1
        and macro_f1_delta >= -0.005
        and format_rate >= 0.95
    )
    result = {
        "passed": passed,
        "pair_delta": pair_delta,
        "macro_f1_delta": macro_f1_delta,
        "format_rate": format_rate,
        "baseline": prior,
        "candidate": candidate,
    }
    (args.evaluation / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print("promotion", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
