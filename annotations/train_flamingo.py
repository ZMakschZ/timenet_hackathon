"""Train one OpenTSLM Flamingo stage on the targeted Hill of Towie QA benchmark."""

import argparse
import json
import math
import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from build_examples import CHANNELS
from tslm_inputs import iter_evaluation

PRETRAINED = "OpenTSLM/llama-3.2-1b-tsqa-flamingo"
BASE = "meta-llama/Llama-3.2-1B"
DESCRIPTIONS = ", ".join(f"{field} ({unit})" for _, field, unit in CHANNELS.values())
FLAGS = [tuple(CHANNELS).index(flag) for flag in ("ON", "READY", "ERR")]


def numeric_tensor(example, scaling):
    """Represent each signal and its validity as adjacent frames in one media item."""
    values = np.asarray(example["values"], dtype=np.float32)
    valid = np.asarray(example["valid"], dtype=bool) & np.isfinite(values)
    valid[FLAGS] &= (values[FLAGS] >= 0) & (values[FLAGS] <= 600)
    normalized = (
        values - np.asarray(scaling["center"], dtype=np.float32)[:, None]
    ) / np.asarray(scaling["scale"], dtype=np.float32)[:, None]
    normalized = np.where(valid, normalized, 0)
    return torch.from_numpy(
        np.stack((normalized, valid.astype(np.float32)), axis=1).reshape(40, 36)
    )


def model_prompt(example):
    return (
        "<image> Six hours of SCADA at ten-minute cadence. Each signal is followed by its "
        f"validity mask. Signal order: {DESCRIPTIONS}. <|endofchunk|>\n"
        f"Question: {example['question']}\n"
    )


def tokens_and_media(rows, tokenizer, scaling, device, *, answer, label_weight=1.0):
    """Build one marker per media item and compute exact answer-only labels."""
    token_rows, labels, media, weights = [], [], [], []
    marker = tokenizer.convert_tokens_to_ids("<image>")
    for example, target in rows:
        prefix = tokenizer.encode(model_prompt(example), add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            prefix.insert(0, tokenizer.bos_token_id)
        if prefix.count(marker) != 1:
            raise ValueError("The prompt must have exactly one Flamingo media marker")
        output = []
        if answer:
            if label_weight == 1.0:
                output = tokenizer.encode(target["target"], add_special_tokens=False)
            else:
                text = target["target"]
                label = target["expected_outcome"]
                if not text.endswith(f"\nAnswer: {label}."):
                    raise ValueError("Expected an evidence-first target ending with its outcome")
                start = len(text) - len(label) - 1
                end = start + len(label)
                encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                output = encoded["input_ids"]
                outcome = [left < end and right > start for left, right in encoded["offset_mapping"]]
                if not any(outcome):
                    raise ValueError("No tokens cover the expected outcome")
                weights.append(
                    [1.0] * len(prefix)
                    + [label_weight if matched else 1.0 for matched in outcome]
                    + [1.0]
                )
            output.append(tokenizer.eos_token_id)
        token_rows.append(prefix + output)
        labels.append([-100] * len(prefix) + output)
        media.append(numeric_tensor(example, scaling))
    width = max(map(len, token_rows))
    pad = tokenizer.pad_token_id
    attention, input_ids, label_rows = [], [], []
    for ids, gold in zip(token_rows, labels, strict=True):
        padding = width - len(ids)
        input_ids.append([pad] * padding + ids)
        attention.append([0] * padding + [1] * len(ids))
        label_rows.append([-100] * padding + gold)
    inputs = {
        "vision_x": torch.stack(media).unsqueeze(1).to(device),
        "lang_x": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }
    if answer:
        inputs["labels"] = torch.tensor(label_rows, dtype=torch.long, device=device)
    if weights:
        inputs["loss_weights"] = torch.tensor(
            [[1.0] * (width - len(row)) + row for row in weights],
            dtype=torch.float32,
            device=device,
        )
    return inputs


def load_model(device, checkpoint_revision, base_revision):
    """Load the pinned upstream components and add distinct frame identities."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

    api = HfApi()
    checkpoint_revision = checkpoint_revision or api.model_info(PRETRAINED).sha
    base_revision = base_revision or api.model_info(BASE).sha
    try:
        model_path = snapshot_download(
            BASE, revision=base_revision, local_files_only=True
        )
    except LocalEntryNotFoundError:
        model_path = snapshot_download(
            BASE,
            revision=base_revision,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )
    wrapper = OpenTSLMFlamingo(
        device=device,
        llm_id=model_path,
        freeze_lm_embeddings=True,
        cross_attn_every_n_layers=1,
    )
    model = wrapper.model.to(device)
    weights = hf_hub_download(
        PRETRAINED, filename="model_checkpoint.pt", revision=checkpoint_revision
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("llm", checkpoint.get("model_state"))
    if state is None:
        raise ValueError("OpenTSLM checkpoint is missing its model state")
    expected = model.state_dict()
    prepared = {}
    for name, param in state.items():
        while name not in expected and name.startswith(("model.", "llm.", "module.")):
            name = name.split(".", 1)[1]
        if name in expected:
            prepared[name] = param
    rest = model.load_state_dict(prepared, strict=False)
    critical = (
        "vision_encoder.",
        "perceiver.",
        "lang_encoder.gated_cross_attn_layers.",
    )
    if any(name.startswith(critical) for name in rest.missing_keys):
        raise ValueError(
            "Base OpenTSLM checkpoint did not restore all time-series weights"
        )
    frame_ids = torch.empty(40, model.perceiver.latents.shape[-1], device=device)
    torch.nn.init.normal_(frame_ids, std=0.02)
    model.perceiver.frame_embs = torch.nn.Parameter(frame_ids)
    return (
        model,
        wrapper.text_tokenizer,
        {"checkpoint": checkpoint_revision, "base": base_revision},
    )


def batches(rows, size):
    """Keep at most one batch in memory."""
    while batch := list(islice(rows, size)):
        yield batch


def validation_rows(corpus, split_file, limit):
    """Take a fixed, time-spread sample without materializing its tensors."""
    count = pq.ParquetFile(split_file).metadata.num_rows
    stride = max(1, math.ceil(count / limit))
    return iter_evaluation(corpus, split_file, stride=stride)


def model_loss(model, payload, device):
    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
    ):
        if "loss_weights" not in payload:
            return model(**payload).loss
        inputs = {key: value for key, value in payload.items() if key not in ("labels", "loss_weights")}
        logits = model(**inputs).logits[:, :-1]
        labels = payload["labels"][:, 1:]
        supervised = labels.ne(-100)
        weights = payload["loss_weights"][:, 1:][supervised]
        loss = torch.nn.functional.cross_entropy(
            logits[supervised].float(), labels[supervised], reduction="none"
        )
        return (weights * loss).sum() / weights.sum()


def train(args, model, tokenizer, scaling, revisions):
    """Fit on train, select by validation loss, and save resumable checkpoints."""
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate)
    step, seen, best = 0, 0, math.inf
    if args.resume:
        saved = torch.load(
            args.output / "last.pt", map_location="cpu", weights_only=True
        )
        if (
            saved["revisions"] != revisions
            or saved["batch"] != args.batch
            or saved["accumulate"] != args.accumulate
        ):
            raise ValueError(
                "Resume requires the checkpoint revisions and original batch settings"
            )
        restore(parameters, saved)
        optimizer.load_state_dict(saved["optimizer"])
        step, seen, best = saved["step"], saved["seen"], saved["best"]
    if step >= args.max_steps:
        print(f"Checkpoint already reached step {step}", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)

    def save(name, best_loss):
        path = args.output / name
        temp = path.with_suffix(".tmp")
        torch.save(
            {
                "weights": {name: p.detach().cpu() for name, p in parameters.items()},
                **({"optimizer": optimizer.state_dict()} if name == "last.pt" else {}),
                "revisions": revisions,
                "step": step,
                "seen": seen,
                "best": best_loss,
                "batch": args.batch,
                "accumulate": args.accumulate,
            },
            temp,
        )
        temp.replace(path)

    @torch.no_grad()
    def validation():
        model.eval()
        numerator, denominator = 0.0, 0
        rows = validation_rows(
            args.corpus, args.data / "validation.parquet", args.validation_examples
        )
        for batch in batches(rows, args.batch):
            payload = tokens_and_media(
                batch, tokenizer, scaling, args.device, answer=True
            )
            count = int((payload["labels"] != -100).sum())
            numerator += model_loss(model, payload, args.device).item() * count
            denominator += count
        return numerator / denominator

    if step == 0:
        best = validation()
        save("best.pt", best)
        print(f"pretrained validation loss={best:.4f}", flush=True)
    model.train()
    source = iter_evaluation(args.corpus, args.data / "train.parquet")
    # Spread adjacent, highly correlated windows across optimizer batches with bounded storage.
    shuffled = shuffled_blocks(source)
    train_batches = batches(islice(shuffled, seen, None), args.batch)
    started, start_seen = time.monotonic(), seen
    for microbatches in batches(train_batches, args.accumulate):
        optimizer.zero_grad(set_to_none=True)
        for batch in microbatches:
            payload = tokens_and_media(
                batch, tokenizer, scaling, args.device, answer=True
            )
            loss = model_loss(model, payload, args.device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss before step {step + 1}")
            (loss / len(microbatches)).backward()
            seen += len(batch)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters.values(), 1, error_if_nonfinite=True
        )
        optimizer.step()
        step += 1
        if step % 10 == 0 or step == 1:
            rate = (seen - start_seen) / (time.monotonic() - started)
            print(
                f"step={step} examples={seen} loss={loss.item():.4f} grad_norm={float(norm):.4f} avg_examples_s={rate:.2f}",
                flush=True,
            )
        if step == 50:
            save("last.pt", best)
            restore(
                parameters,
                torch.load(
                    args.output / "last.pt", map_location="cpu", weights_only=True
                ),
            )
        if step % args.validate_every == 0 or step == args.max_steps:
            best = select_checkpoint(model, validation, save, best, step)
        if step == args.max_steps:
            break
    else:
        best = select_checkpoint(model, validation, save, best, step)


def shuffled_blocks(rows):
    rng = random.Random(2026)
    for block in batches(rows, 512):
        rng.shuffle(block)
        yield from block


def restore(parameters, checkpoint):
    weights = checkpoint["weights"]
    if set(parameters) != set(weights):
        raise ValueError("Saved and current trainable parameter names differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name])


def select_checkpoint(model, validate, save, best, step):
    loss = validate()
    if math.isfinite(loss) and loss < best:
        best = loss
        save("best.pt", best)
    save("last.pt", best)
    print(f"step={step} validation_loss={loss:.4f} best={best:.4f}", flush=True)
    model.train()
    return best


@torch.no_grad()
def evaluate(args, model, tokenizer, scaling):
    """Generate on the targeted test split only and save predictions for scoring."""
    model.eval()
    rows = iter_evaluation(args.corpus, args.data / "test.parquet")
    if args.test_limit is not None:
        rows = islice(rows, args.test_limit)
    with (args.output / "test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for batch in batches(rows, args.batch):
            payload = tokens_and_media(
                batch, tokenizer, scaling, args.device, answer=False
            )
            output = model.generate(
                **payload,
                max_new_tokens=200,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            predictions = tokenizer.batch_decode(
                output[:, payload["lang_x"].size(1) :], skip_special_tokens=True
            )
            for prediction, (_, row) in zip(predictions, batch, strict=True):
                result = {
                    key: row[key]
                    for key in ("station_id", "cohort", "family", "scenario", "target")
                }
                result.update(
                    cutoff_utc=row["cutoff_utc"].isoformat(), prediction=prediction
                )
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--data", type=Path, default=Path("targeted-qa"))
    parser.add_argument("--output", type=Path, default=Path("runs/hill-flamingo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop at this optimizer step. Defaults to one pass over train.",
    )
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--base-revision")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test-limit", type=int)
    args = parser.parse_args()
    if args.batch < 1 or args.accumulate < 1:
        parser.error("Batch size and accumulation must be positive")
    if args.max_steps is None:
        rows = pq.ParquetFile(args.data / "train.parquet").metadata.num_rows
        args.max_steps = math.ceil(rows / (args.batch * args.accumulate))
    if (
        min(
            args.batch,
            args.accumulate,
            args.max_steps,
            args.validation_examples,
            args.validate_every,
        )
        < 1
    ):
        parser.error("Batch sizes, step counts and validation size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable on this host. Run the trainer on an NVIDIA GPU host."
        )
    if args.mode == "train" and not args.resume and (args.output / "best.pt").exists():
        parser.error(
            "An existing run uses this output directory. Resume it or choose a fresh path."
        )
    if args.mode == "test" or args.resume:
        filename = "best.pt" if args.mode == "test" else "last.pt"
        saved = torch.load(
            args.output / filename, map_location="cpu", weights_only=True
        )
        args.checkpoint_revision = saved["revisions"]["checkpoint"]
        args.base_revision = saved["revisions"]["base"]
    scaling = json.loads((args.data / "scaling.json").read_text())
    if scaling["channels"] != list(CHANNELS):
        raise ValueError("Training scaler does not match the fixed channel order")
    torch.manual_seed(2026)
    model, tokenizer, revisions = load_model(
        args.device, args.checkpoint_revision, args.base_revision
    )
    if args.mode == "train":
        train(args, model, tokenizer, scaling, revisions)
    else:
        restore(
            {name: p for name, p in model.named_parameters() if p.requires_grad}, saved
        )
        evaluate(args, model, tokenizer, scaling)


if __name__ == "__main__":
    main()
