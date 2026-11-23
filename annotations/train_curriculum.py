"""Run the OpenTSLM-style warmup and SCADA QA curriculum on one GPU."""

import argparse
import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from train_flamingo import (
    batches,
    load_model,
    model_loss,
    restore,
    shuffled_blocks,
    tokens_and_media,
    validation_rows,
)
from tslm_inputs import iter_evaluation


@torch.no_grad()
def validation(model, tokenizer, scaling, corpus, data, batch_size):
    """Score the same time-spread targeted QA validation sample as the direct run."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    rows = validation_rows(corpus, data / "validation.parquet", 128)
    for batch in batches(rows, batch_size):
        payload = tokens_and_media(batch, tokenizer, scaling, "cuda", answer=True)
        n = int((payload["labels"] != -100).sum())
        total_loss += n * model_loss(model, payload, "cuda").item()
        total_tokens += n
    return total_loss / total_tokens


def save(path, parameters, revisions, stage, step, seen, val_loss):
    """Atomically save only trainable weights and selection metadata."""
    temp = path.with_suffix(".tmp")
    torch.save(
        {
            "weights": {key: value.detach().cpu() for key, value in parameters.items()},
            "revisions": revisions,
            "stage": stage,
            "step": step,
            "seen": seen,
            "validation_loss": val_loss,
        },
        temp,
    )
    temp.replace(path)


def train_stage(args, stage, model, tokenizer, scaling, parameters, revisions):
    """Run a stage with AdamW, linear decay, and 10 percent warmup."""
    path = args.curriculum / f"{stage}.parquet"
    count = pq.ParquetFile(path).metadata.num_rows
    total_steps = math.ceil(count / (args.batch * args.accumulate))
    warmup = max(1, math.ceil(0.1 * total_steps))
    optimizer = torch.optim.AdamW(parameters.values(), lr=2e-4, weight_decay=0.01)

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return (total_steps - step) / (total_steps - warmup)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    microbatches = batches(
        shuffled_blocks(iter_evaluation(args.corpus, path)), args.batch
    )
    best = math.inf
    started, seen = time.monotonic(), 0
    model.train()
    for step, group in enumerate(batches(microbatches, args.accumulate), start=1):
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            payload = tokens_and_media(batch, tokenizer, scaling, "cuda", answer=True)
            loss = model_loss(model, payload, "cuda")
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {stage}/{step}")
            (loss / len(group)).backward()
            seen += len(batch)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters.values(), 1, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == total_steps:
            throughput = seen / (time.monotonic() - started)
            print(
                f"stage={stage} step={step}/{total_steps} examples={seen}/{count} "
                f"loss={loss.item():.4f} grad_norm={float(norm):.3f} examples_s={throughput:.2f}",
                flush=True,
            )
        if stage == "qa" and (step % 500 == 0 or step == total_steps):
            val = validation(
                model, tokenizer, scaling, args.corpus, args.data, args.batch
            )
            if math.isfinite(val) and val < best:
                best = val
                save(
                    args.output / "best.pt",
                    parameters,
                    revisions,
                    stage,
                    step,
                    seen,
                    val,
                )
            print(
                f"stage={stage} step={step} validation_loss={val:.4f} best={best:.4f}",
                flush=True,
            )
            model.train()
    if seen != count:
        raise RuntimeError(f"{stage}: expected {count} examples, got {seen}")
    if stage == "caption":
        val = validation(model, tokenizer, scaling, args.corpus, args.data, args.batch)
        print(f"warmup QA validation_loss={val:.4f}", flush=True)
        save(args.output / "warmup.pt", parameters, revisions, stage, step, seen, val)
    print(f"stage={stage} complete examples={seen}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--data", type=Path, default=Path("targeted-qa"))
    parser.add_argument("--curriculum", type=Path, default=Path("curriculum-20k"))
    parser.add_argument("--output", type=Path, default=Path("runs/hill-curriculum-20k"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--resume-qa", action="store_true")
    parser.add_argument("--warmup-checkpoint", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or min(args.batch, args.accumulate) < 1:
        parser.error("A CUDA GPU and positive batch settings are required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "best.pt").exists() or (
        (args.output / "warmup.pt").exists() and not args.resume_qa
    ):
        parser.error(
            "Existing checkpoints require a fresh output path or --resume-qa after warmup"
        )
    scaling = json.loads((args.data / "scaling.json").read_text())
    torch.manual_seed(2026)
    model, tokenizer, revisions = load_model(
        "cuda", args.checkpoint_revision, args.base_revision
    )
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    trainable = sum(p.numel() for p in parameters.values())
    total = sum(p.numel() for p in model.parameters())
    if not trainable or not total > trainable:
        raise RuntimeError("Expected both frozen and trainable parameters")
    print(f"trainable={trainable} frozen={total - trainable}", flush=True)
    if args.resume_qa:
        checkpoint = args.warmup_checkpoint or args.output / "warmup.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["revisions"] != revisions:
            raise ValueError("Warmup was run with different model revisions")
        restore(parameters, saved)
    else:
        base = validation(model, tokenizer, scaling, args.corpus, args.data, args.batch)
        print(f"pretrained QA validation_loss={base:.4f}", flush=True)
        for stage in ("mcq", "caption"):
            train_stage(args, stage, model, tokenizer, scaling, parameters, revisions)
    train_stage(args, "qa", model, tokenizer, scaling, parameters, revisions)
    print("curriculum complete examples=20000", flush=True)


if __name__ == "__main__":
    main()
