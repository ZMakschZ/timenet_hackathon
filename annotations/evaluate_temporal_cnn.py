"""Train a CPU temporal CNN and score the exact held-out OpenTSLM operator panel."""

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_trend_baseline import (
    SIGNALS,
    SIGNAL_INDEX,
    bootstrap_accuracy,
    metrics,
    operator_windows,
    read_operator_panel,
)


def raw_series(series):
    """Keep the original sample spacing and retain absent channels as missing."""
    series = dict(series)
    length = len(next(iter(series.values())))
    values = np.full((len(SIGNALS), length), np.nan, dtype=np.float32)
    for channel, samples in series.items():
        values[SIGNAL_INDEX[channel]] = samples
    return values


def grouped_windows(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.features.shape[-1]].append(index)
    return list(groups.values())


def fit_scale(training):
    counts = np.zeros(len(SIGNALS), dtype=np.int64)
    total = np.zeros(len(SIGNALS))
    squares = np.zeros(len(SIGNALS))
    for row in training:
        valid = np.isfinite(row.features)
        values = np.where(valid, row.features, 0).astype(np.float64)
        counts += valid.sum(axis=1)
        total += values.sum(axis=1)
        squares += (values ** 2).sum(axis=1)
    denominator = counts.clip(min=1)
    mean = total / denominator
    std = np.sqrt((squares / denominator - mean ** 2).clip(min=0))
    std[(std < 1e-6) | (counts == 0)] = 1
    return mean.astype(np.float32), std.astype(np.float32)


class TemporalCNN(nn.Module):
    def __init__(self, categories, labels, width=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2 * len(SIGNALS), width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=4, dilation=2),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=8, dilation=4),
            nn.BatchNorm1d(width),
            nn.SiLU(),
        )
        self.task_embedding = nn.Embedding(categories, 8)
        self.head = nn.Sequential(
            nn.Linear(10 * width + 8, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, labels)
        )

    def forward(self, samples, tasks):
        signal = self.encoder(samples)
        # Pool the encoded sequence in ordered quarters, with an additional view
        # of the last hour and the exact cutoff. Both input durations are supported.
        pools = (
            nn.functional.adaptive_avg_pool1d(signal, 4).flatten(1),
            nn.functional.adaptive_max_pool1d(signal, 4).flatten(1),
            signal[..., -6:].mean(dim=-1),
            signal[..., -1],
            self.task_embedding(tasks),
        )
        return self.head(torch.cat(pools, dim=1))


class Panel:
    def __init__(self, rows, categories, classes, scale, require_labels=False):
        self.rows = rows
        self.groups = grouped_windows(rows)
        self.tasks = torch.tensor([categories[row.category] for row in rows])
        self.targets = torch.tensor([classes.get((row.category, row.label), -1) for row in rows])
        options = torch.zeros((len(rows), len(classes)), dtype=torch.bool)
        for index, row in enumerate(rows):
            allowed = [classes[row.category, label] for label in row.options if (row.category, label) in classes]
            if not allowed or (require_labels and self.targets[index].item() not in allowed):
                raise ValueError(f"No supported answer for {row.key()}")
            options[index, allowed] = True
        self.options = options
        self.mean, self.std = (vector[None, :, None] for vector in scale)

    def batches(self, size, rng=None):
        groups = [rng.permutation(group) if rng is not None else group for group in self.groups]
        batches = [group[i:i + size] for group in groups for i in range(0, len(group), size)]
        if rng is not None:
            rng.shuffle(batches)
        for batch in batches:
            values = np.stack([self.rows[index].features for index in batch])
            observed = np.isfinite(values)
            scaled = np.where(observed, (values - self.mean) / self.std, 0)
            samples = torch.from_numpy(np.concatenate((scaled, observed), axis=1))
            yield batch, samples, self.tasks[batch], self.targets[batch], self.options[batch]


def logits_for_options(model, samples, tasks, options):
    return model(samples, tasks).masked_fill(~options, -torch.inf)


@torch.inference_mode()
def predict(model, panel, classes, batch_size):
    model.eval()
    guesses = [None] * len(panel.rows)
    for batch, samples, tasks, _, options in panel.batches(batch_size):
        chosen = logits_for_options(model, samples, tasks, options).argmax(dim=1)
        for index, label in zip(batch, chosen.tolist(), strict=True):
            guesses[index] = classes[label][1]
    return guesses


def train_model(training, validation, categories, classes, scale, args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    lookup = {label: index for index, label in enumerate(classes)}
    train = Panel(training, categories, lookup, scale, require_labels=True)
    valid = Panel(validation, categories, lookup, scale)
    model = TemporalCNN(len(categories), len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    # Temper the contribution of frequent labels while retaining all training rows.
    weights = torch.bincount(train.targets, minlength=len(classes)).float().rsqrt()
    best, stale = None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _, samples, tasks, targets, options in train.batches(args.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            logits = logits_for_options(model, samples, tasks, options)
            loss = nn.functional.cross_entropy(logits, targets, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
        guesses = predict(model, valid, classes, args.batch_size)
        score, _ = metrics(validation, guesses)
        history.append({"epoch": epoch, "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
        print(json.dumps(history[-1]), flush=True)
        if best is None or score["macro_f1"] > best["macro_f1"]:
            best = history[-1] | {"weights": deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best.pop("weights"))
    return model, best, history


def score_summary(rows, guesses):
    overall, labels = metrics(rows, guesses)
    groups = defaultdict(list)
    for row, guess in zip(rows, guesses, strict=True):
        groups[row.category].append((row, guess))
    by_category = {}
    for category, group in sorted(groups.items()):
        members, predictions = zip(*group, strict=True)
        scores, _ = metrics(members, list(predictions))
        by_category[category] = scores | {"labels": labels[category]}
    hits = [guess == row.label for row, guess in zip(rows, guesses, strict=True)]
    return overall | {
        "accuracy_ci95": bootstrap_accuracy([row.turbine for row in rows], hits),
        "category_macro_accuracy": float(np.mean([v["accuracy"] for v in by_category.values()])),
        "by_category": by_category,
    }


def comparison(rows, cnn, reference, positions):
    diff = np.array([
        int(cnn[index] == rows[index].label) - int(reference[index] == rows[index].label)
        for index in positions
    ])
    return {
        "cnn_minus_reference_accuracy": float(np.mean(diff)),
        "ci95": bootstrap_accuracy([rows[i].turbine for i in positions], diff),
        "only_cnn_correct": int((diff == 1).sum()),
        "only_reference_correct": int((diff == -1).sum()),
    }


def compare_to_archived(test, cnn, tslm, archived):
    reference = archived["predictions"]
    for row, guess, saved in zip(test, tslm, reference, strict=True):
        if (row.key(), row.label, guess) != (
            tuple(saved[k] for k in ("category", "turbine", "cutoff", "prompt")),
            saved["label"], saved["tslm"],
        ):
            raise ValueError(f"Benchmark comparison row differs: {row.key()}")
    methods = {"temporal_cnn": cnn, "tslm": tslm, "trend_logistic": [row["trend_logistic"] for row in reference]}
    forecast_categories = archived["forecast_subset"]["by_category"]
    forecasts = [i for i, row in enumerate(test) if row.category in forecast_categories]
    return {
        "checkpoint": archived["checkpoint"],
        "methods": {name: score_summary(test, guesses) for name, guesses in methods.items()},
        "paired_differences": {
            name: comparison(test, cnn, methods[name], range(len(test)))
            for name in ("tslm", "trend_logistic")
        },
        "forecast_subset": {
            "windows": len(forecasts),
            "correct": {name: sum(guesses[i] == test[i].label for i in forecasts) for name, guesses in methods.items()},
            "paired_differences": {
                name: comparison(test, cnn, methods[name], forecasts)
                for name in ("tslm", "trend_logistic")
            },
        },
        "predictions": [
            saved | {"temporal_cnn": guess}
            for saved, guess in zip(reference, cnn, strict=True)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path, help="New directory for the report and trained weights")
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path, help="Archived OpenTSLM predictions")
    parser.add_argument("--benchmark", required=True, type=Path, help="Verified logistic regression comparison")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("Epochs, patience, batch size and thread count must be positive")
    if args.output.exists():
        parser.error("Output directory already exists")
    torch.set_num_threads(args.threads)
    iterator, test, tslm = read_operator_panel(args, raw_series)
    categories = {name: i for i, name in enumerate(sorted({row.category for row in test}))}
    training = list(iterator)
    validation = [
        row
        for directory in (args.dataset, args.supplement)
        for row in operator_windows(directory / "validation.jsonl.gz", raw_series)
        if row.category in categories
    ]
    if max(row.cutoff for row in training) >= min(row.cutoff for row in validation):
        raise ValueError("Training and validation cutoffs overlap")
    if max(row.cutoff for row in validation) >= min(row.cutoff for row in test):
        raise ValueError("Validation and test cutoffs overlap")
    scale = fit_scale(training)
    support = Counter((row.category, row.label) for row in training)
    classes = sorted(support)
    archived = json.loads(args.benchmark.read_text())
    if support != Counter({
        (category, label): count
        for category, labels in archived["training_label_counts"].items()
        for label, count in labels.items()
    }):
        raise ValueError("Training label supports differ from the archived logistic baseline")
    print(f"{len(training):,} train, {len(validation):,} validation, {len(test):,} test", flush=True)
    model, best, history = train_model(training, validation, categories, classes, scale, args)
    test_panel = Panel(test, categories, {label: i for i, label in enumerate(classes)}, scale)
    cnn = predict(model, test_panel, classes, args.batch_size)
    report = compare_to_archived(test, cnn, tslm, archived) | {
        "training_rows": len(training),
        "validation_rows": len(validation),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "selected_epoch": best,
        "validation_history": history,
        "bootstrap": archived["bootstrap"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "weights": model.state_dict(),
            "categories": categories,
            "classes": classes,
            "signals": SIGNALS,
            "mean": torch.from_numpy(scale[0]),
            "scale": torch.from_numpy(scale[1]),
        },
        args.output / "temporal-cnn.pt",
    )
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: scores["accuracy"] for name, scores in report["methods"].items()}), flush=True)


if __name__ == "__main__":
    main()
