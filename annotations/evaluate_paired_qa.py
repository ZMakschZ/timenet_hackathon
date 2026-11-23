"""Compare checkpoints on whether telemetry changes the answer to a fixed question."""

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import torch
from train_flamingo import batches, load_model, restore, tokens_and_media
from tslm_inputs import iter_evaluation

NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+)(?:\.\d+)?(?![\w.])")


@torch.inference_mode()
def compare_choices(model, tokenizer, scaling, rows, options):
    """Classify with candidate-answer likelihood without relying on output formatting."""
    inputs = list(
        chain.from_iterable(
            (
                (example, {"target": f"Answer: {option}."})
                for option in options[row["family"]]
            )
            for example, row in rows
        )
    )
    payload = tokens_and_media(inputs, tokenizer, scaling, "cuda", answer=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**payload).logits[:, :-1, :]
    targets = payload["labels"][:, 1:]
    mask = targets.ne(-100)
    # Exclude the common "Answer:" prefix and EOS.
    prefix_size = len(tokenizer.encode("Answer:", add_special_tokens=False))
    positions = mask.cumsum(dim=1)
    mask &= positions.gt(prefix_size) & positions.lt(positions[:, -1:])
    lengths = mask.sum(dim=1)
    losses = torch.nn.functional.cross_entropy(
        logits[mask].float(), targets[mask], reduction="none"
    )
    means = [float(part.mean()) for part in losses.split(lengths.tolist())]
    return [
        options[row["family"]][int(means[2 * i + 1] < means[2 * i])]
        for i, (_, row) in enumerate(rows)
    ]


@torch.inference_mode()
def answers(model, tokenizer, scaling, rows, *, without_signals=False):
    """Generate the evidence and follow-up on a small sample of complete pairs."""
    for batch in batches(iter(rows), 4):
        payload = tokens_and_media(batch, tokenizer, scaling, "cuda", answer=False)
        if without_signals:
            payload["vision_x"].zero_()
        ids = model.generate(
            **payload,
            max_new_tokens=96,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        texts = tokenizer.batch_decode(
            ids[:, payload["lang_x"].shape[1] :], skip_special_tokens=True
        )
        yield from zip(batch, texts, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("revised", type=Path)
    args = parser.parse_args()
    rows = list(iter_evaluation(args.corpus, args.data / "validation.parquet"))
    score_numbers = any(NUMBER.search(row["target"]) for _, row in rows)
    opts = defaultdict(set)
    paired = defaultdict(list)
    for example, row in rows:
        opts[row["family"]].add(row["expected_outcome"])
        paired[row["pair_id"]].append((example, row))
    options = {family: sorted(labels) for family, labels in opts.items()}
    assert all(len(values) == 2 for values in options.values())
    assert all(
        len(pair) == 2
        and pair[0][0]["question"] == pair[1][0]["question"]
        and pair[0][1]["expected_outcome"] != pair[1][1]["expected_outcome"]
        for pair in paired.values()
    )
    counts = Counter()
    sample = []
    for pair_id, pair in sorted(paired.items()):
        family = pair[0][1]["family"]
        if counts[family] < 2:
            sample.extend(pair)
            counts[family] += 1

    prior = torch.load(args.previous, map_location="cpu", weights_only=True)
    scaling = json.loads((args.data / "scaling.json").read_text())
    model, tokenizer, revisions = load_model(
        "cuda", prior["revisions"]["checkpoint"], prior["revisions"]["base"]
    )
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    positions = {
        (target["pair_id"], target["expected_outcome"]): i
        for i, (_, target) in enumerate(rows)
    }
    summaries = {}
    for name in ("previous", "revised"):
        checkpoint = (
            prior
            if name == "previous"
            else torch.load(args.revised, map_location="cpu", weights_only=True)
        )
        if checkpoint["revisions"] != revisions:
            raise ValueError(
                "Both checkpoints must use the same base and model revisions"
            )
        restore(params, checkpoint)
        model.eval()
        predictions = list(
            chain.from_iterable(
                compare_choices(model, tokenizer, scaling, batch, options)
                for batch in batches(iter(rows), 4)
            )
        )
        correct = [
            guess == row["expected_outcome"]
            for guess, (_, row) in zip(predictions, rows, strict=True)
        ]
        pair_correct = sum(
            all(
                correct[positions[(row["pair_id"], row["expected_outcome"])]]
                for _, row in pair
            )
            for pair in paired.values()
        )
        totals = Counter()
        with (args.data / f"{name}-validation.jsonl").open("w") as stream:
            for (example, row), text in answers(model, tokenizer, scaling, sample):
                head = re.match(r"\s*Answer:\s*([^.!?\n]+)", text, re.IGNORECASE)
                answer = head[1].strip().lower() if head else None
                totals["generated_outcome_correct"] += answer == row["expected_outcome"]
                if score_numbers:
                    reference_numbers = Counter(NUMBER.findall(row["target"]))
                    seen_numbers = Counter(NUMBER.findall(text))
                    totals.update(
                        matching_numeric_literals=sum(
                            (reference_numbers & seen_numbers).values()
                        ),
                        reference_numeric_literals=sum(reference_numbers.values()),
                    )
                stream.write(
                    json.dumps(
                        {
                            "pair_id": row["pair_id"],
                            "family": row["family"],
                            "turbine": row["turbine_id"],
                            "cutoff_utc": example["context"]["cutoff_utc"],
                            "question": row["question"],
                            "target": row["target"],
                            "prediction": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        summaries[name] = {
            "correct_choices": sum(correct),
            "windows": len(rows),
            "pairs_both_correct": pair_correct,
            "pairs": len(paired),
            "generated_windows": len(sample),
            **totals,
        }
        print(name, summaries[name], flush=True)
    (args.data / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
