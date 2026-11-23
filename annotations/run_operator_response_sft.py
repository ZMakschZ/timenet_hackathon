"""Fine-tune and evaluate generated operator responses on Nebius."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import torch
from aws_scada_inputs import CONFIG_SHA256, load_aws
from opentslm_microbatch import answer_losses, microbatches
from operator_response_targets import OPERATOR_CATEGORIES, operator_target
from run_nebius_continuation import load_delta, save_delta
from run_nebius_pilot import examples, training_groups


def select(rows, per_label, replay, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    replay_rows = []
    for row in rows:
        item = row["example"]
        if row.get("target_format") == "operator_response":
            label = row["label"]
            if label not in item.options or not item.answer.endswith(f"\nAnswer: {label}"):
                raise ValueError(f"Malformed expanded operator target for {item.category}")
            groups[item.category, label].append(row)
        elif item.category in OPERATOR_CATEGORIES:
            groups[item.category, item.answer].append(row | {"example": operator_target(item), "label": item.answer})
        else:
            replay_rows.append(row | {"label": item.answer})
    selected = [row for group in groups.values() for row in rng.sample(group, min(per_label, len(group)))]
    selected.extend(rng.sample(replay_rows, min(replay, len(replay_rows))))
    return selected


def _loss(aws, model, rows, rng=None):
    items = [row["example"] for row in rows]
    questions = [item.evaluated_question() if rng is None else item.question(rng) for item in items]
    with aws._autocast(model):
        return answer_losses(model, items, questions)


def benchmark(aws, model, rows, minimum, ceiling):
    """Check single-example loss parity, then benchmark larger microbatches."""
    rng = random.Random(9)
    candidates = microbatches(rows, ceiling, rng)
    short, long = (
        max(
            (batch for batch in candidates if len(batch) == ceiling and batch[0]["steps"] == steps),
            key=lambda batch: len(batch[0]["example"].series),
        )
        for steps in (36, 1008)
    )
    model.eval()
    parity_deltas = []
    with torch.no_grad():
        for batch in (short[:2], long[:2]):
            batched, token_counts = _loss(aws, model, batch)
            for reference, count, row in zip(batched, token_counts, batch, strict=True):
                item = row["example"]
                with aws._autocast(model):
                    native = model.loss(item.series, item.evaluated_question(), item.answer)
                # Compare per-token losses. Summing over long answers magnifies BF16 roundoff.
                torch.testing.assert_close(reference / count, native, rtol=0.02, atol=0.005)
                parity_deltas.append(float((reference / count - native).abs()))
    print(json.dumps({"largest_parity_delta_per_token": max(parity_deltas)}), flush=True)
    model.train()
    runs = []
    optimizer_gib = (
        2
        * sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters() if parameter.requires_grad
        )
        / 2**30
    )
    capacity_gib = torch.cuda.get_device_properties(model.device).total_memory / 2**30
    for size in (1, 2, 4, 6, 8):
        if size > ceiling:
            continue
        if minimum > 4 and size < 4:
            continue
        torch.cuda.empty_cache()
        try:
            warmup, _ = _loss(aws, model, short[:size])
            warmup.sum().backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            for samples in (short, long):
                for offset in range(0, len(samples), size):
                    losses, _ = _loss(aws, model, samples[offset : offset + size])
                    losses.sum().backward()
                    model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            runs.append({"microbatch": size, "out_of_memory": True})
        else:
            runs.append(
                {
                    "microbatch": size,
                    "examples_per_second": (len(short) + len(long)) / (time.monotonic() - start),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            )
        print(json.dumps(runs[-1]), flush=True)
    viable = [
        result
        for result in runs
        if result["microbatch"] >= minimum
        and "examples_per_second" in result
        and result["peak_gib"] + optimizer_gib + 4 <= capacity_gib
    ]
    if not viable:
        raise ValueError(f"No microbatch of at least {minimum} fit within the GPU memory estimate")
    return max(viable, key=lambda result: result["examples_per_second"])["microbatch"], runs


@torch.no_grad()
def validation_loss(aws, model, rows, size):
    model.eval()
    groups = defaultdict(list)
    for batch in microbatches(rows, size, random.Random(23)):
        sums, tokens = _loss(aws, model, batch)
        for value, count, row in zip(sums.tolist(), tokens, batch, strict=True):
            groups[row["example"].category].append(value / count)
    means = [sum(losses) / len(losses) for losses in groups.values()]
    score = sum(means) / len(means)
    if not math.isfinite(score):
        raise ValueError("Non-finite validation loss")
    return score


def train(aws, model, training, validation, groups, args, size):
    rng = random.Random(args.train_seed if args.train_seed is not None else args.seed)
    batches = []
    for _ in range(args.epochs):
        batches.extend(microbatches(training, size, rng))
    accumulation = max(1, args.effective_batch // size)
    steps = math.ceil(len(batches) / accumulation)
    for group in groups:
        group["lr"] *= args.lr_scale
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, fused=True)
    scheduler = aws.get_wsd_schedule(
        optimizer,
        num_warmup_steps=max(1, int(0.03 * steps)),
        num_decay_steps=max(1, int(0.2 * steps)),
        num_training_steps=max(2, steps),
        decay_type="linear",
    )
    initial = validation_loss(aws, model, validation, size)
    history = [{"update": 0, "validation_loss": initial}]
    best, best_step, stale, seen = initial, 0, 0, 0
    print(
        json.dumps(history[0] | {"planned_updates": steps, "microbatch": size, "accumulation": accumulation}),
        flush=True,
    )
    model.train()
    training_start = time.monotonic()
    time_limit = False
    for step, offset in enumerate(range(0, len(batches), accumulation), 1):
        count, observed = 0, 0
        losses = []
        for batch in batches[offset : offset + accumulation]:
            sums, tokens = _loss(aws, model, batch, rng)
            loss = sums.sum()
            loss.backward()
            losses.append(loss.detach())
            count += sum(tokens)
            observed += len(batch)
        total = torch.stack(losses).sum().item()
        if not math.isfinite(total):
            raise ValueError("Non-finite training loss")
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(count)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        seen += observed
        if step == 1 or step % 25 == 0:
            print(json.dumps({"update": step, "examples": seen, "train_loss": total / count}), flush=True)
        time_limit = args.max_train_seconds is not None and time.monotonic() - training_start >= args.max_train_seconds
        if step % args.validate_every != 0 and step != steps and not time_limit:
            continue
        score = validation_loss(aws, model, validation, size)
        if score < best - 0.002:
            best, best_step, stale = score, step, 0
            save_delta(model, args.output / "trained.safetensors")
        else:
            stale += 1
        entry = {
            "update": step,
            "examples": seen,
            "validation_loss": score,
            "best_update": best_step,
            "stale_checks": stale,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        (args.output / "validation-history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= args.patience or time_limit:
            break
        model.train()
    if best_step:
        load_delta(model, args.output / "trained.safetensors")
    else:
        load_delta(model, args.initial)
    reason = "completed"
    if stale >= args.patience:
        reason = "early_stopping"
    elif time_limit:
        reason = "time_budget"
    return {
        "examples": seen,
        "updates": step,
        "best_update": best_step,
        "initial_loss": initial,
        "best_loss": best,
        "stop_reason": reason,
    }


def evaluate(aws, model, rows):
    results = []
    model.eval()
    for row in rows:
        example = row["example"]
        with aws._autocast(model):
            output = model.generate(example.series, example.evaluated_question(), max_new_tokens=160)
        results.append(
            {
                "category": example.category,
                "turbine": row["turbine"],
                "cutoff": row["cutoff"],
                "prompt": example.prompt,
                "target": example.answer,
                "label": row["label"],
                "prediction": output,
                "label_correct": aws.chosen_option(output, example.options) == row["label"],
                "complete": all(section in output for section in ("Evidence:", "Assessment:", "Next step:", "Answer:")),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("initial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--utils", required=True, type=Path)
    parser.add_argument("--per-label", type=int, default=160)
    parser.add_argument("--replay", type=int, default=256)
    parser.add_argument("--additional-train", type=Path)
    parser.add_argument("--additional-validation", type=Path)
    parser.add_argument("--additional-test", type=Path)
    parser.add_argument("--test-per-label", type=int, default=4)
    parser.add_argument("--compare-initial", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-train-seconds", type=int, help="Training time budget excluding startup and test generation"
    )
    parser.add_argument("--microbatch-min", type=int, choices=(2, 4, 6, 8), default=2)
    parser.add_argument("--microbatch-ceiling", type=int, choices=(2, 4, 6, 8), default=4)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr-scale", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, help="Override the seed for training selection and batching only")
    args = parser.parse_args()
    if (
        min(args.per_label, args.test_per_label, args.epochs, args.validate_every, args.patience, args.lr_scale) <= 0
        or args.replay < 0
    ):
        parser.error("training counts and learning rate must be positive")
    if not args.microbatch_min <= args.microbatch_ceiling <= args.effective_batch:
        parser.error("microbatch bounds must fit the effective batch size")
    if args.max_train_seconds is not None and args.max_train_seconds <= 0:
        parser.error("training time budget must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("a CUDA GPU with bfloat16 support is required")
    if hashlib.sha256((args.base / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
        parser.error("expected the pinned tslm-nov26 checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    start = time.monotonic()
    aws = load_aws(args.utils)
    training = select(
        split_examples(aws, args.dataset / "train.jsonl.gz", args.additional_train),
        args.per_label,
        args.replay,
        args.train_seed if args.train_seed is not None else args.seed,
    )
    validation = select(
        split_examples(aws, args.dataset / "validation.jsonl.gz", args.additional_validation),
        3,
        24,
        args.seed,
    )
    print(
        json.dumps(
            {
                "training_rows": len(training),
                "validation_rows": len(validation),
                "training_by_category": dict(Counter(row["example"].category for row in training)),
            }
        ),
        flush=True,
    )
    model = aws.OpenTSLMFlamingo.from_file(args.base, device="cuda", dtype=torch.bfloat16)
    groups = training_groups(aws, model)
    load_delta(model, args.initial)
    size, profiling = benchmark(aws, model, training, args.microbatch_min, args.microbatch_ceiling)
    result = train(aws, model, training, validation, groups, args, size)
    del training, validation
    print(json.dumps({"training_finished": result}), flush=True)
    test = select(
        split_examples(aws, args.dataset / "test.jsonl.gz", args.additional_test), args.test_per_label, 0, args.seed
    )
    rows = evaluate(aws, model, test)
    report = {
        "microbatch": size,
        "benchmark": profiling,
        "training": result,
        "test_rows": len(rows),
        "correct_labels": sum(row["label_correct"] for row in rows),
        "complete_responses": sum(row["complete"] for row in rows),
        "seconds": round(time.monotonic() - start, 1),
    }
    (args.output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    if args.compare_initial:
        initial_rows = rows
        if result["best_update"]:
            load_delta(model, args.initial)
            initial_rows = evaluate(aws, model, test)
        (args.output / "initial-predictions.json").write_text(json.dumps(initial_rows, indent=2) + "\n")
        report["initial_correct_labels"] = sum(row["label_correct"] for row in initial_rows)
        report["initial_complete_responses"] = sum(row["complete"] for row in initial_rows)
        report["seconds"] = round(time.monotonic() - start, 1)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def split_examples(aws, original, additional):
    """Read the existing split followed by its optional supplement."""
    paths = (original, additional) if additional is not None else (original,)
    return chain.from_iterable(examples(aws, path) for path in paths)


if __name__ == "__main__":
    main()
