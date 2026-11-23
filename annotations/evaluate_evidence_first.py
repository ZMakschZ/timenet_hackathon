"""Score generated labels for evidence-first paired SCADA QA."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from evaluate_paired_qa import answers
from train_flamingo import load_model, restore
from tslm_inputs import iter_evaluation

ANSWER = re.compile(r"^\s*Answer:\s*([^.!?\n]+)", re.IGNORECASE | re.MULTILINE)
FINAL_LINE = re.compile(r"Answer:\s*([^.!?\n]+)\.?", re.IGNORECASE)


def final_answer_match(output):
    lines = output.strip().splitlines()
    return FINAL_LINE.fullmatch(lines[-1].strip()) if len(lines) > 1 else None


def score_predictions(predictions, *, strict=False):
    """Count invalid responses as misses and compute F1 across family/outcome classes."""
    expected, predicted, correct = Counter(), Counter(), Counter()
    pairs = defaultdict(list)
    valid_format = 0
    for row, output in predictions:
        output = output.strip()
        final = final_answer_match(output)
        match = final if strict else ANSWER.search(output)
        label = match[1].strip().lower() if match else None
        family = row["family"]
        truth = (family, row["expected_outcome"])
        guess = (family, label)
        expected[truth] += 1
        predicted[guess] += 1
        valid_format += bool(final)
        hit = guess == truth
        correct[truth] += hit
        pairs[row["pair_id"]].append((truth, row["question"], hit))
    if not pairs or any(
        len(pair) != 2
        or pair[0][0][0] != pair[1][0][0]
        or pair[0][0][1] == pair[1][0][1]
        or pair[0][1] != pair[1][1]
        for pair in pairs.values()
    ):
        raise ValueError("Expected complete pairs with a shared question and opposing outcomes")
    count = sum(expected.values())
    macro_f1 = sum(
        2 * correct[key] / (count_for_label + predicted[key])
        for key, count_for_label in expected.items()
    ) / len(expected)
    return {
        "windows": count,
        "correct": sum(correct.values()),
        "accuracy": sum(correct.values()) / count,
        "macro_f1": macro_f1,
        "pairs": len(pairs),
        "paired_both_correct": sum(all(member[2] for member in pair) for pair in pairs.values()),
        "evidence_first_format": valid_format,
    }


def generate(model, tokenizer, scaling, rows, *, without_signals=False):
    model.eval()
    if without_signals:
        # With constant media, windows sharing a prompt have identical model inputs.
        representatives = {example["question"]: (example, row) for example, row in rows}
        by_question = {
            example["question"]: output
            for (example, _), output in answers(
                model, tokenizer, scaling, list(representatives.values()), without_signals=True
            )
        }
        return [(row, by_question[example["question"]]) for example, row in rows]
    return [
        (row, output)
        for (_, row), output in answers(model, tokenizer, scaling, rows)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("evidence_first", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--signals-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    rows = list(iter_evaluation(args.corpus, args.data / "test.parquet"))
    if any(row["split"] != "test" for _, row in rows):
        raise ValueError("Only untouched test rows belong in the final comparison")
    scaling = json.loads((args.data / "scaling.json").read_text())
    checkpoint = torch.load(args.baseline, map_location="cpu", weights_only=True)
    model, tokenizer, revisions = load_model(
        "cuda", checkpoint["revisions"]["checkpoint"], checkpoint["revisions"]["base"]
    )
    params = {key: value for key, value in model.named_parameters() if value.requires_grad}
    summary = {}
    checkpoints = [("previous", args.baseline)]
    if not args.baseline_only:
        checkpoints.append(("evidence_first", args.evidence_first))
    for name, path in checkpoints:
        if name != "previous":
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["revisions"] != revisions:
            raise ValueError("Compared checkpoints must use identical base revisions")
        restore(params, checkpoint)
        del checkpoint
        modes = ("signals",) if args.signals_only else ("signals", "question_only")
        for mode in modes:
            predictions = generate(
                model, tokenizer, scaling, rows, without_signals=mode == "question_only"
            )
            key = f"{name}_{mode}"
            summary[key] = score_predictions(predictions)
            summary[key]["strict"] = score_predictions(predictions, strict=True)
            with (args.output / f"{key}.jsonl").open("w") as stream:
                for row, prediction in predictions:
                    stream.write(json.dumps({
                        "pair_id": row["pair_id"],
                        "family": row["family"],
                        "expected_outcome": row["expected_outcome"],
                        "question": row["question"],
                        "target": row["target"],
                        "prediction": prediction,
                    }, ensure_ascii=False) + "\n")
            print(key, json.dumps(summary[key]), flush=True)
    report = args.output / "test-results.json"
    temporary = report.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(report)


if __name__ == "__main__":
    main()
