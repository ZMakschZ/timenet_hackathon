"""Fit and evaluate the pinned OpenTSLM model on a time-budgeted Nebius pilot."""

import argparse
import gzip
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from aws_scada_inputs import CONFIG_SHA256, load_aws
from safetensors.torch import save_file


def examples(aws, path):
    with gzip.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            fields = row["example"]
            series = [
                aws.SeriesInput(
                    **(item | {"values": [math.nan if value is None else value for value in item["values"]]})
                )
                for item in fields["series"]
            ]
            row["example"] = aws.Example(**(fields | {"series": series}))
            yield row


@torch.no_grad()
def evaluate(aws, model, rows, deadline):
    """Score canonical answers and retain generated captions and per-window outcomes."""
    model.eval()
    results = []
    for row in rows:
        if time.monotonic() >= deadline:
            break
        example = row["example"]
        question = example.evaluated_question()
        with aws._autocast(model):
            output = model.generate(example.series, question, max_new_tokens=40)
            loss = None if example.options else float(model.loss(example.series, question, example.answer))
        results.append(
            {
                "stage": row["stage"],
                "steps": row["steps"],
                "category": example.category,
                "turbine": row["turbine"],
                "cutoff": row["cutoff"],
                "question": question,
                "answer": example.answer,
                "prediction": output,
                "caption_loss": loss,
                "correct": (aws.chosen_option(output, example.options) == example.answer) if example.options else None,
            }
        )
    return results


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["category"]].append(row)
    summary = {}
    for category, rows in sorted(groups.items()):
        scores = defaultdict(list)
        for row in rows:
            if row["correct"] is not None:
                scores[row["answer"]].append(row["correct"])
        metrics = {"examples": len(rows)}
        if scores:
            metrics["accuracy"] = sum(sum(values) for values in scores.values()) / len(rows)
            metrics["balanced_accuracy"] = sum(sum(values) / len(values) for values in scores.values()) / len(scores)
        else:
            metrics["caption_loss"] = sum(row["caption_loss"] for row in rows) / len(rows)
        summary[category] = metrics
    return summary


def training_groups(aws, model):
    model.requires_grad_(False)
    encoder, perceiver, attention, embeddings = model.trained_modules()
    return [
        {
            "params": [
                parameter for module in modules for parameter in module.float().requires_grad_(True).parameters()
            ],
            "lr": aws.LEARNING_RATES[name],
        }
        for name, modules in (("encoder", (encoder,)), ("adapter", (perceiver, attention)), ("base", (embeddings,)))
    ]


def accumulate(aws, model, batch, rng):
    """Backpropagate summed answer-token loss for the pinned training recipe."""
    total, tokens = 0.0, 0
    for row in batch:
        example = row["example"]
        with aws._autocast(model):
            losses = model.loss(example.series, example.question(rng), example.answer, reduction="none")
        if not torch.isfinite(losses).all():
            raise ValueError("Non-finite training loss")
        losses.sum().backward()
        total += float(losses.detach().mean())
        tokens += len(losses)
    return total, tokens


def fit(aws, model, rows, size, seed, deadline, progress=None):
    rng = random.Random(seed)
    rng.shuffle(rows)
    groups = training_groups(aws, model)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, fused=model.device.type == "cuda")
    model.train()
    start = time.monotonic()
    accumulate(aws, model, rows[:size], rng)
    optimizer.zero_grad(set_to_none=True)
    step_time = max(0.01, time.monotonic() - start) * 1.2
    steps = min(math.ceil(len(rows) / size), int(max(0, deadline - time.monotonic()) / step_time))
    if steps < 1:
        raise TimeoutError("The pilot budget expired during setup and calibration")
    scheduler = aws.get_wsd_schedule(
        optimizer,
        num_warmup_steps=max(1, int(steps * aws.WARMUP_FRACTION)),
        num_decay_steps=max(1, int(steps * aws.DECAY_FRACTION)),
        num_training_steps=max(steps, 2),
        decay_type="linear",
    )
    seen, total, completed = 0, 0.0, 0
    for index in range(steps):
        if time.monotonic() + step_time >= deadline:
            break
        start = time.monotonic()
        batch = rows[index * size : (index + 1) * size]
        loss, tokens = accumulate(aws, model, batch, rng)
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad /= tokens
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_time = 0.85 * step_time + 0.15 * (time.monotonic() - start)
        seen += len(batch)
        total += loss
        completed += 1
        if index == 0 or completed % 25 == 0:
            status = {"updates": completed, "examples": seen, "loss": total / seen, "planned_updates": steps}
            print(json.dumps(status), flush=True)
            if progress:
                progress.write_text(json.dumps(status, indent=2) + "\n")
    if completed == 0:
        raise TimeoutError("The pilot budget expired before the first optimizer update")
    return {"updates": completed, "examples": seen, "loss": total / seen, "planned_updates": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--utils", type=Path, required=True)
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.minutes, args.accumulation) < 1:
        parser.error("duration and accumulation must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("the Nebius pilot requires a CUDA GPU with bfloat16 support")
    if hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
        parser.error("the checkpoint must be the pinned tslm-nov26 snapshot")

    started = time.monotonic()
    deadline = started + args.minutes * 60
    aws = load_aws(args.utils)
    train = list(examples(aws, args.dataset / "train.jsonl.gz"))
    validation = list(examples(aws, args.dataset / "validation.jsonl.gz"))
    if not train or not validation:
        parser.error("the pilot needs training and validation examples")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model = aws.OpenTSLMFlamingo.from_file(args.checkpoint, device="cuda", dtype=torch.bfloat16)
    start = time.monotonic()
    baseline = evaluate(aws, model, validation, min(deadline, start + args.minutes * 6))
    if not baseline:
        raise TimeoutError("No held-out examples could be scored before training")
    reserve = max(60, 2 * (time.monotonic() - start))
    training = fit(aws, model, train, args.accumulation, args.seed, deadline - reserve, args.output / "progress.json")
    weights = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.named_parameters() if tensor.requires_grad
    }
    temporary = args.output / "trained.tmp"
    save_file(weights, temporary)
    temporary.replace(args.output / "trained.safetensors")
    del weights
    after = evaluate(aws, model, validation[: len(baseline)], deadline)
    report = {
        "training": training,
        "validation_pairs": len(after),
        "validation_available": len(validation),
        "baseline": summarize(baseline[: len(after)]),
        "trained": summarize(after),
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.output / "predictions.json").write_text(json.dumps({"baseline": baseline, "trained": after}, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
