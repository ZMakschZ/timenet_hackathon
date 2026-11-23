"""Continue the OpenTSLM pilot for one pass with validation-based early stopping."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from aws_scada_inputs import CONFIG_SHA256, load_aws
from run_nebius_pilot import accumulate, evaluate, examples, summarize, training_groups
from safetensors.torch import load_file, save_file


@torch.no_grad()
def validation_loss(aws, model, rows):
    """Give each question category equal weight in checkpoint selection."""
    model.eval()
    losses = defaultdict(list)
    for row in rows:
        item = row["example"]
        with aws._autocast(model):
            loss = float(model.loss(item.series, item.evaluated_question(), item.answer))
        if not math.isfinite(loss):
            raise ValueError(f"Non-finite validation loss for {item.category}")
        losses[item.category].append(loss)
    per_category = {key: sum(values) / len(values) for key, values in sorted(losses.items())}
    return sum(per_category.values()) / len(per_category), per_category


def load_delta(model, path):
    """Restore exactly the parameters belonging to the time-series fine-tuning recipe."""
    values = load_file(path)
    expected = {name for name, tensor in model.named_parameters() if tensor.requires_grad}
    if set(values) != expected:
        raise ValueError("Checkpoint parameter names do not match the fine-tuning recipe")
    model.load_state_dict(values, strict=False)


def save_delta(model, path):
    temporary = path.with_suffix(".tmp")
    weights = {
        name: value.detach().cpu().contiguous() for name, value in model.named_parameters() if value.requires_grad
    }
    save_file(weights, temporary)
    temporary.replace(path)


def fit(aws, model, train, validation, groups, args):
    rng = random.Random(args.seed)
    rng.shuffle(train)
    for group in groups:
        group["lr"] *= args.lr_scale
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, fused=model.device.type == "cuda")
    steps = math.ceil(len(train) / args.accumulation)
    scheduler = aws.get_wsd_schedule(
        optimizer,
        num_warmup_steps=max(1, int(steps * aws.WARMUP_FRACTION)),
        num_decay_steps=max(1, int(steps * aws.DECAY_FRACTION)),
        num_training_steps=max(steps, 2),
        decay_type="linear",
    )
    initial, categories = validation_loss(aws, model, validation)
    history = [{"update": 0, "examples": 0, "validation_loss": initial, "per_category": categories}]
    print(json.dumps({"update": 0, "validation_loss": initial, "planned_updates": steps}), flush=True)
    best_loss, meaningful_loss, best_step, stale = initial, initial, 0, 0
    best_path = args.initial_delta
    seen, running_loss, reason = 0, 0.0, "completed_pass"
    model.train()
    for update in range(1, steps + 1):
        start = (update - 1) * args.accumulation
        batch = train[start : start + args.accumulation]
        loss, tokens = accumulate(aws, model, batch, rng)
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad /= tokens
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        seen += len(batch)
        running_loss += loss
        if update == 1 or update % 100 == 0:
            print(json.dumps({"update": update, "examples": seen, "train_loss": running_loss / seen}), flush=True)
        if update % args.validate_every != 0 and update != steps:
            continue

        score, categories = validation_loss(aws, model, validation)
        if score < best_loss:
            best_loss, best_step = score, update
            best_path = args.output / "trained.safetensors"
            save_delta(model, best_path)
        if score < meaningful_loss - args.min_delta:
            meaningful_loss, stale = score, 0
        else:
            stale += 1
        record = {
            "update": update,
            "examples": seen,
            "validation_loss": score,
            "best_update": best_step,
            "stale_checks": stale,
        }
        history.append(record | {"per_category": categories})
        (args.output / "validation-history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            reason = "early_stopping"
            break
        model.train()
    load_delta(model, best_path)
    if best_step == 0:
        (args.output / "trained.safetensors").symlink_to(args.initial_delta.resolve())
    return {
        "reason": reason,
        "updates": update,
        "planned_updates": steps,
        "examples": seen,
        "train_loss": running_loss / seen,
        "initial_validation_loss": initial,
        "best_validation_loss": best_loss,
        "best_update": best_step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("initial_delta", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--utils", type=Path, required=True)
    parser.add_argument("--pilot-panel", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--lr-scale", type=float, default=0.25)
    args = parser.parse_args()
    if min(args.accumulation, args.patience, args.validate_every) < 1 or args.min_delta < 0 or args.lr_scale <= 0:
        parser.error("invalid training or early-stopping settings")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("the continuation requires a CUDA GPU with bfloat16 support")
    if hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest() != CONFIG_SHA256:
        parser.error("expected the pinned tslm-nov26 snapshot")

    started = time.monotonic()
    aws = load_aws(args.utils)
    train = list(examples(aws, args.dataset / "train.jsonl.gz"))
    validation = list(examples(aws, args.dataset / "validation.jsonl.gz"))
    if not train or not validation:
        parser.error("training and validation samples are required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model = aws.OpenTSLMFlamingo.from_file(args.checkpoint, device="cuda", dtype=torch.bfloat16)
    groups = training_groups(aws, model)
    load_delta(model, args.initial_delta)
    training = fit(aws, model, train, validation, groups, args)
    del train, validation
    print(json.dumps({"training_finished": training}), flush=True)
    pilot = evaluate(aws, model, examples(aws, args.pilot_panel), math.inf)
    test = evaluate(aws, model, examples(aws, args.dataset / "test.jsonl.gz"), math.inf)
    report = {
        "training": training,
        "pilot_panel": summarize(pilot),
        "test": summarize(test),
        "patience": args.patience,
        "min_delta": args.min_delta,
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.output / "predictions.json").write_text(json.dumps({"pilot_panel": pilot, "test": test}, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
