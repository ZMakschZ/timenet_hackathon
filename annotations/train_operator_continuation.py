"""Time-boxed mixed operator and paired QA continuation on a single GPU."""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch

from build_operator_continuation import STATES
from evaluate_checkpoint_comparison import paired_input, score_row, summarize
from evaluate_paired_qa import answers
from train_flamingo import batches, load_model, model_loss, restore, tokens_and_media
from train_paired_early_stop import save_best
from tslm_inputs import iter_evaluation

STATE_MATCHERS = tuple(
    map(
        re.compile,
        (
            r"\b(?:generated|generating|producing power|running smoothly|shows? generation)\b",
            r"\b(?:idle|stopped)\b",
            r"\b(?:unresolved|mixed|transitional|inconsistent)\b",
            r"\b(?:missing|unavailable|[Nn]one of the|no telemetry|no data)\b",
            r"\b(?:uncertain|unknown|cannot determine|can.t determine)\b",
        ),
    )
)
VERIFY = re.compile(
    r"\b(?:check|verify|confirm|compare|restore|obtain|review|monitor)\b", re.IGNORECASE
)


def operator_hit(text, state):
    """Score the lead diagnosis and look for evidence of unavailable telemetry."""
    text = re.sub(r"^\s*Answer:\s*", "", text.strip(), flags=re.IGNORECASE)
    lead = re.split(r"(?<=[.!?])\s+|\n", text, maxsplit=1)[0].lower()
    if state == STATES[3]:
        return bool(
            STATE_MATCHERS[4].search(lead) and STATE_MATCHERS[3].search(text.lower())
        )
    if state == STATES[4]:
        return bool(
            STATE_MATCHERS[4].search(lead)
            and re.search(
                r"\b(?:power|speed|running time|channel)\b", text, re.IGNORECASE
            )
        )
    return bool(STATE_MATCHERS[STATES.index(state)].search(lead))


def operator_predictions(model, tokenizer, scaling, rows):
    model.eval()
    records = []
    for (example, row), output in answers(model, tokenizer, scaling, rows):
        state = row["expected_state"]
        records.append(
            {
                "expected_state": state,
                "correct_state": operator_hit(output, state),
                "verification_language": bool(VERIFY.search(output)),
                "question": example["question"],
                "target": row.get("target", row.get("reference_answer")),
                "prediction": output,
                **{key: row[key] for key in ("month", "station_id", "stop_step")},
                "turbine": row["turbine_id"],
                "cutoff_utc": row["cutoff_utc"].isoformat(),
            }
        )
    return records


def operator_summary(records):
    totals = Counter(row["expected_state"] for row in records)
    return {
        "rows": len(records),
        "correct_state": sum(row["correct_state"] for row in records),
        "state_with_verification_language": sum(
            row["correct_state"] and row["verification_language"] for row in records
        ),
        "unknown_or_missing_correct": sum(
            row["correct_state"]
            for row in records
            if row["expected_state"] in STATES[3:]
        ),
        "unknown_or_missing_rows": sum(totals[state] for state in STATES[3:]),
        "by_state": {
            state: {
                "rows": count,
                "correct": sum(
                    row["correct_state"]
                    for row in records
                    if row["expected_state"] == state
                ),
            }
            for state, count in totals.items()
        },
    }


def mixed_batches(corpus, directory, rng):
    """Shuffle time blocks, then mix three operator batches for every two paired batches."""
    paths = [
        directory / name for name in ("operator-train.parquet", "paired-train.parquet")
    ]
    while True:
        iterators = []
        for path, size in zip(paths, (12, 8), strict=True):
            groups = list(range(pq.ParquetFile(path).num_row_groups))
            rng.shuffle(groups)
            source = iter_evaluation(corpus, path, row_groups=groups)
            iterators.append(batches(chain.from_iterable(_shuffled(source, rng)), size))
        # The two pools are sized in the same ratio. A new pass reshuffles both.
        for broad, paired in zip(*iterators, strict=True):
            group = broad + paired
            rng.shuffle(group)
            yield from batches(iter(group), 4)


def _shuffled(source, rng):
    for block in batches(source, 512):
        rng.shuffle(block)
        yield block


def keep_pairs(corpus, path):
    rows = list(paired_input(SimpleNamespace(corpus=corpus, panel=path)))
    families = defaultdict(set)
    for _, row in rows:
        families[row["family"]].add(row["pair_id"])
    rng = Random(20261121)
    selected = {
        pair for ids in families.values() for pair in rng.sample(sorted(ids), 6)
    }
    return [(example, row) for example, row in rows if row["pair_id"] in selected]


def evaluate(args, model, tokenizer, scaling, broad, pairs, label):
    operator = operator_predictions(model, tokenizer, scaling, broad)
    paired = []
    for (example, row), text in answers(model, tokenizer, scaling, pairs):
        paired.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "pair_id",
                        "family",
                        "expected_outcome",
                        "question",
                        "target",
                    )
                },
                "prediction": text,
                **score_row(example, row, text),
            }
        )
    for name, records in (("operator", operator), ("paired", paired)):
        with (args.output / f"{name}-{label}.jsonl").open("w") as handle:
            for row in records:
                print(json.dumps(row, ensure_ascii=False), file=handle)
    return {"operator": operator_summary(operator), "paired": summarize(paired)}


def run(args):
    launched = time.monotonic()
    torch.manual_seed(20261120)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    checkpoint = torch.load(args.initial, map_location="cpu", weights_only=True)
    model, tokenizer, revisions = load_model(
        "cuda", checkpoint["revisions"]["checkpoint"], checkpoint["revisions"]["base"]
    )
    params = {
        key: value for key, value in model.named_parameters() if value.requires_grad
    }
    restore(params, checkpoint)
    del checkpoint
    scaling = json.loads(args.scaling.read_text())
    broad = list(
        iter_evaluation(args.corpus, args.data / "operator-validation.parquet")
    )
    pairs = keep_pairs(args.corpus, args.paired_validation)
    baseline = evaluate(args, model, tokenizer, scaling, broad, pairs, "baseline")
    best = baseline
    selected = args.initial
    selected_seen = seen = stale = 0
    args.weights.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=0.01)
    source = mixed_batches(args.corpus, args.data, Random(20261120))
    deadline = launched + 60 * args.minutes
    every = 60 * args.check_minutes
    due = time.monotonic() + every
    started = time.monotonic()
    model.train()
    with (args.output / "metrics.jsonl").open("x") as log:
        print(json.dumps({"step": 0, "examples": 0, **baseline}), file=log, flush=True)
        print("baseline", json.dumps(baseline), flush=True)
        for step, group in enumerate(batches(source, 2), start=1):
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                loss = model_loss(
                    model,
                    tokens_and_media(batch, tokenizer, scaling, "cuda", answer=True),
                    "cuda",
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                (loss / len(group)).backward()
                seen += len(batch)
            torch.nn.utils.clip_grad_norm_(
                params.values(), 1.0, error_if_nonfinite=True
            )
            ramp = min(1.0, step / 100)
            optimizer.param_groups[0]["lr"] = args.lr * ramp
            optimizer.step()
            if step == 1 or step % 100 == 0:
                print(
                    f"step={step} examples={seen} loss={loss.item():.4f} "
                    f"examples_s={seen / (time.monotonic() - started):.2f}",
                    flush=True,
                )
            now = time.monotonic()
            if now < min(due, deadline):
                continue
            observed = evaluate(
                args, model, tokenizer, scaling, broad, pairs, str(seen)
            )
            operator = observed["operator"]
            accepted = (
                operator["state_with_verification_language"]
                >= best["operator"]["state_with_verification_language"] + 2
                and operator["unknown_or_missing_correct"]
                >= best["operator"]["unknown_or_missing_correct"]
                and observed["paired"]["both_correct"]
                >= baseline["paired"]["both_correct"] - 3
            )
            if accepted:
                selected = args.weights / "best.pt"
                save_best(selected, model, revisions, observed, step, seen)
                best, selected_seen, stale = observed, seen, 0
            else:
                stale += 1
            status = {"step": step, "examples": seen, "accepted": accepted, **observed}
            print(json.dumps(status), file=log, flush=True)
            print("validation", json.dumps(status), flush=True)
            if time.monotonic() >= deadline or stale >= 5:
                break
            due += every
            model.train()
    if args.transfer is not None:
        if selected != args.initial:
            restore(params, torch.load(selected, map_location="cpu", weights_only=True))
        transfer = operator_predictions(
            model, tokenizer, scaling, iter_evaluation(args.corpus, args.transfer)
        )
        with (args.output / "transfer.jsonl").open("w") as stream:
            for row in transfer:
                print(json.dumps(row, ensure_ascii=False), file=stream)
    else:
        transfer = []
    report = {
        "examples_seen": seen,
        "selected_seen": selected_seen,
        "selected_checkpoint": str(selected),
        "elapsed_minutes": round((time.monotonic() - launched) / 60, 2),
        "baseline": baseline,
        "best": best,
        "transfer": operator_summary(transfer),
    }
    temporary = args.output / "result.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output / "result.json")
    print("result", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "corpus",
        "data",
        "initial",
        "scaling",
        "paired-validation",
        "output",
        "weights",
    ):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--transfer", type=Path)
    parser.add_argument("--minutes", type=float, default=52)
    parser.add_argument("--check-minutes", type=float, default=10)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--memory-fraction", type=float, default=0.36)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.jsonl").exists():
        parser.error("Choose a new output directory")
    run(args)


if __name__ == "__main__":
    main()
