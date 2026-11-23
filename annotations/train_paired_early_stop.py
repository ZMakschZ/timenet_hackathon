"""Continue paired SCADA QA with generation-based early stopping."""

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from random import Random

import pyarrow.parquet as pq
import torch
from evaluate_paired_qa import NUMBER, answers, compare_choices
from evaluate_evidence_first import final_answer_match, generate, score_predictions
from train_flamingo import (
    batches,
    load_model,
    model_loss,
    restore,
    shuffled_blocks,
    tokens_and_media,
)
from tslm_inputs import iter_evaluation


def validation_panel(corpus, data):
    """Prepare the unchanged 60 held-out pairs and a fixed 24-window generation panel."""
    rows = list(iter_evaluation(corpus, data / "validation.parquet"))
    labels, pairs = defaultdict(set), defaultdict(list)
    for example, row in rows:
        labels[row["family"]].add(row["expected_outcome"])
        pairs[row["pair_id"]].append((example, row))
    if len(pairs) != 60 or any(
        len(pair) != 2
        or pair[0][0]["question"] != pair[1][0]["question"]
        or pair[0][1]["expected_outcome"] == pair[1][1]["expected_outcome"]
        for pair in pairs.values()
    ):
        raise ValueError("Expected 60 complete held-out pairs with opposing outcomes")
    selected = Counter()
    sample = []
    for _, pair in sorted(pairs.items()):
        family = pair[0][1]["family"]
        if selected[family] < 2:
            sample.extend(pair)
            selected[family] += 1
    return rows, {family: sorted(values) for family, values in labels.items()}, sample


@torch.inference_mode()
def validate(model, tokenizer, scaling, panel):
    """Score pair discrimination and exact numeric evidence with extra numbers penalized."""
    rows, options, sample = panel
    model.eval()
    guesses = chain.from_iterable(
        compare_choices(model, tokenizer, scaling, batch, options)
        for batch in batches(iter(rows), 4)
    )
    right = Counter()
    for guess, (_, row) in zip(guesses, rows, strict=True):
        right[row["pair_id"]] += guess == row["expected_outcome"]
    correct_pairs = sum(count == 2 for count in right.values())
    matches, references, generated, generated_outcomes = 0, 0, 0, 0
    for (_, row), output in answers(model, tokenizer, scaling, sample):
        head = re.match(r"\s*Answer:\s*([^.!?\n]+)", output, re.IGNORECASE)
        generated_outcomes += (
            head is not None and head[1].strip().lower() == row["expected_outcome"]
        )
        reference = Counter(NUMBER.findall(row["target"]))
        seen = Counter(NUMBER.findall(output))
        matches += sum((reference & seen).values())
        references += sum(reference.values())
        generated += sum(seen.values())
    numeric_f1 = 2 * matches / (references + generated)
    return {
        "paired_both_correct": correct_pairs,
        "pairs": len(right),
        "correct_choices": sum(right.values()),
        "generated_outcomes_correct": generated_outcomes,
        "numeric_matches": matches,
        "reference_numbers": references,
        "generated_numbers": generated,
        "numeric_f1": numeric_f1,
        "score": 0.5 * (correct_pairs / len(right) + numeric_f1),
    }


def validate_outcomes(model, tokenizer, scaling, panel, *, diagnostics=None):
    """Select checkpoints using generated outcomes in evidence-first format."""
    predictions = generate(model, tokenizer, scaling, panel[0])
    scores = score_predictions(predictions, strict=True)
    if diagnostics is not None:
        families = defaultdict(list)
        invalid = Counter()
        with diagnostics.open("x") as stream:
            for row, output in predictions:
                family = row["family"]
                final = final_answer_match(output)
                label = final[1].strip().lower() if final else None
                families[family].append((row, output))
                invalid[family] += label not in panel[1][family]
                print(json.dumps({
                    "pair_id": row["pair_id"],
                    "family": family,
                    "turbine_id": row["turbine_id"],
                    "cutoff_utc": row["cutoff_utc"].isoformat(),
                    "question": row["question"],
                    "target": row["target"],
                    "expected_outcome": row["expected_outcome"],
                    "generated_outcome": label,
                    "prediction": output,
                }, ensure_ascii=False), file=stream)
        scores["invalid_labels"] = sum(invalid.values())
        scores["by_family"] = {
            family: {**score_predictions(rows, strict=True), "invalid_labels": invalid[family]}
            for family, rows in sorted(families.items())
        }
    return {
        **scores,
        "score": 0.5 * (
            scores["paired_both_correct"] / scores["pairs"] + scores["macro_f1"]
        ),
    }


def preserves_family_scores(baseline, candidate):
    """Require each focus family to improve where possible and preserve the others."""
    before = baseline["by_family"]
    after = candidate["by_family"]
    focus = {"temperature", "maintenance"}
    return all(
        after[family]["paired_both_correct"]
        >= min(score["pairs"], score["paired_both_correct"] + (family in focus))
        for family, score in before.items()
    )


def save_best(path, model, revisions, metrics, step, seen):
    """Write one atomic compact checkpoint, leaving the previous run intact."""
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "weights": {
                key: param.detach().to(device="cpu", dtype=torch.bfloat16)
                for key, param in model.named_parameters()
                if param.requires_grad
            },
            "revisions": revisions,
            "metrics": metrics,
            "step": step,
            "seen": seen,
        },
        temporary,
    )
    temporary.replace(path)


def main():
    launched = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--data", type=Path, default=Path("paired-qa-100k"))
    parser.add_argument(
        "--initial", type=Path, default=Path("runs/hill-paired-qa/best.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/hill-paired-qa-100k"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument(
        "--validate-every", type=int, default=5000, help="Examples between evaluations."
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--label-weight", type=float, default=1.0)
    parser.add_argument("--max-minutes", type=float, help="Time budget before final validation")
    parser.add_argument("--checkpoint-dir", type=Path, help="Override where candidate weights are written")
    parser.add_argument(
        "--retain-baseline",
        action="store_true",
        help="Require improvement over the initial checkpoint for evidence-first continuation",
    )
    parser.add_argument(
        "--evidence-first",
        action="store_true",
        help="Select on generated outcome scores for evidence-first QA targets",
    )
    parser.add_argument(
        "--balanced-diagnostic",
        action="store_true",
        help="Use the dataset's balanced training order and save per-window validation predictions",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This continuation requires CUDA")
    if args.balanced_diagnostic and not (args.evidence_first and args.retain_baseline):
        parser.error("--balanced-diagnostic requires --evidence-first and --retain-baseline")
    if args.label_weight != 1.0 and not args.evidence_first:
        parser.error("--label-weight requires evidence-first targets")
    if (
        min(args.batch, args.accumulate, args.validate_every, args.patience) <= 0
        or args.min_gain < 0 or args.lr <= 0 or args.label_weight <= 0
        or (args.max_minutes is not None and args.max_minutes <= 0)
    ):
        parser.error("Invalid training or early-stopping settings")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or args.output
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = args.output / "metrics.jsonl"
    if log.exists():
        parser.error("Choose a new output directory for a new experiment")
    saved = torch.load(args.initial, map_location="cpu", weights_only=True)
    torch.manual_seed(2026)
    model, tokenizer, revisions = load_model(
        "cuda", saved["revisions"]["checkpoint"], saved["revisions"]["base"]
    )
    params = {
        key: value for key, value in model.named_parameters() if value.requires_grad
    }
    restore(params, saved)
    del saved
    scaling = json.loads((args.data / "scaling.json").read_text())
    panel = validation_panel(args.corpus, args.data)
    def validate_checkpoint(tag):
        if not args.evidence_first:
            return validate(model, tokenizer, scaling, panel)
        diagnostics = args.output / f"validation-{tag}.jsonl" if args.balanced_diagnostic else None
        return validate_outcomes(model, tokenizer, scaling, panel, diagnostics=diagnostics)

    baseline = validate_checkpoint("000000")
    best = None if args.evidence_first and not args.retain_baseline else baseline
    selected, stale = args.initial, 0
    print("baseline", json.dumps(baseline), flush=True)
    path = args.data / "qa.parquet"
    file = pq.ParquetFile(path)
    count = file.metadata.num_rows
    if not args.evidence_first and count != 100000:
        raise ValueError(f"Expected 100,000 paired QA examples, found {count}")
    group_order = None
    if not args.balanced_diagnostic and (not args.evidence_first or args.retain_baseline):
        group_order = list(range(file.num_row_groups))
        Random(20261120).shuffle(group_order)
    source = iter_evaluation(args.corpus, path, row_groups=group_order)
    if args.balanced_diagnostic:
        # Read windows in Parquet order to reuse each station's signal row group.
        source = sorted(source, key=lambda item: item[1]["training_order"])
        if [row["training_order"] for _, row in source] != list(range(count)):
            raise ValueError("Expected one distinct training order per example")
    else:
        source = shuffled_blocks(source)
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(count / (args.batch * args.accumulate))
    warmup = max(1, math.ceil(steps * 0.1))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup
            if step < warmup
            else (steps - step) / max(1, steps - warmup)
        ),
    )
    started, seen, due = time.monotonic(), 0, args.validate_every
    deadline = launched + 60 * args.max_minutes if args.max_minutes is not None else math.inf
    time_limited = False
    model.train()
    with log.open("x") as metrics:
        print(
            json.dumps(
                {"step": 0, "examples": 0, "accepted": best is not None, **baseline}
            ),
            file=metrics,
            flush=True,
        )
        for step, group in enumerate(
            batches(batches(source, args.batch), args.accumulate),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                payload = tokens_and_media(
                    batch, tokenizer, scaling, "cuda", answer=True, label_weight=args.label_weight
                )
                loss = model_loss(model, payload, "cuda")
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                (loss / len(group)).backward()
                seen += len(batch)
            torch.nn.utils.clip_grad_norm_(
                params.values(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            schedule.step()
            if step == 1 or step % 100 == 0:
                print(
                    f"step={step} examples={seen}/{count} loss={loss.item():.4f} examples_s={seen / (time.monotonic() - started):.2f}",
                    flush=True,
                )
            time_limited = time.monotonic() >= deadline
            if seen < due and seen < count and not time_limited:
                continue
            due += args.validate_every
            observed = validate_checkpoint(f"{seen:06d}")
            accepted = best is None or observed["score"] >= best["score"] + args.min_gain
            if not args.evidence_first:
                accepted = (
                    accepted
                    and observed["paired_both_correct"] >= baseline["paired_both_correct"] - 2
                    and observed["generated_outcomes_correct"] >= baseline["generated_outcomes_correct"] - 2
                )
            elif args.retain_baseline:
                accepted &= observed["paired_both_correct"] >= baseline["paired_both_correct"]
            if args.balanced_diagnostic:
                accepted &= preserves_family_scores(baseline, observed)
            if accepted:
                best, stale = observed, 0
                selected = checkpoint_dir / "best.pt"
                save_best(selected, model, revisions, best, step, seen)
            else:
                stale += 1
            status = {
                "step": step,
                "examples": seen,
                "accepted": accepted,
                "stale_checks": stale,
                **observed,
            }
            print(json.dumps(status), file=metrics, flush=True)
            print("validation", json.dumps(status), flush=True)
            if time_limited or time.monotonic() >= deadline:
                time_limited = True
                print(f"time budget reached at {seen} examples", flush=True)
                break
            if stale >= args.patience:
                print(
                    f"early stopping at {seen} examples after {stale} checks without improvement",
                    flush=True,
                )
                break
            model.train()
    print(f"selected checkpoint: {selected}", flush=True)
    report = {
        "examples_seen": seen,
        "early_stopped": stale >= args.patience,
        "time_limited": time_limited,
        "label_weight": args.label_weight,
        "elapsed_seconds": round(time.monotonic() - launched, 1),
        "selected_checkpoint": str(selected),
        "baseline": baseline,
        "best_in_memory": best,
    }
    if selected != args.initial:
        checkpoint = torch.load(selected, map_location="cpu", weights_only=True)
        restore(params, checkpoint)
        report["saved_checkpoint_validation"] = validate_checkpoint("selected")
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
