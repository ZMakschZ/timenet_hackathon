"""Derive operator handovers from supported model observations."""

import argparse
import json
from pathlib import Path

from evaluate_evidence_first import final_answer_match


def derive(family, output, actions):
    match = final_answer_match(output)
    outcome = match[1].strip().lower() if match else None
    entry = actions.get(family, {}).get(outcome)
    if entry is None:
        return {
            "outcome": None,
            "operator_answer": "Verify the live readings and controller state before choosing a follow-up.",
            "abstained": True,
        }
    return {
        "outcome": outcome,
        "operator_answer": f"{entry['observation']} {entry['check']}",
        "abstained": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    actions = json.loads(args.actions.read_text())
    total = matched = abstained = 0
    with args.predictions.open() as source, args.output.open("w") as stream:
        for line in source:
            row = json.loads(line)
            response = derive(row["family"], row["prediction"], actions)
            total += 1
            matched += response["outcome"] == row["expected_outcome"]
            abstained += response["abstained"]
            print(json.dumps({**row, **response}, ensure_ascii=False), file=stream)
    print("operator derivation", json.dumps({
        "windows": total, "matching_outcomes": matched, "abstained": abstained,
    }), flush=True)


if __name__ == "__main__":
    main()
