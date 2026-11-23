"""Evaluate CPU label baselines on the held-out Hill of Towie TSLM panels."""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_examples import CHANNELS
from operator_response_targets import OPERATOR_CATEGORIES, RESPONSE_INSTRUCTION

SIGNALS = (*CHANNELS, "DE", "NDE", "HS", "MAIN", "OIL", "PRESS")
SIGNAL_INDEX = {key: i for i, key in enumerate(SIGNALS)}
METHODS = ("training_majority", "trend_logistic", "tslm")
FINAL_LINE = re.compile(r"Answer:\s*([^.!?\n]+)\.?", re.IGNORECASE)


@dataclass
class Window:
    category: str
    turbine: str
    cutoff: str
    prompt: str
    label: str
    options: tuple
    features: np.ndarray
    pair: str | None = None

    def key(self):
        return self.category, self.turbine, self.cutoff, self.prompt


def summarize_series(values):
    """Measure within-channel change and availability over three trailing horizons."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    features = [1, float(valid[-1])]
    for width in (6, len(values) // 2, len(values)):
        observed = values[-width:][valid[-width:]]
        features.append(len(observed) / width)
        if len(observed) < 2:
            features.extend((0, 0, 0))
            continue
        midpoint = len(observed) // 2
        centered = observed - observed[0]
        features.extend((
            centered[-1],
            centered[midpoint:].mean() - centered[:midpoint].mean(),
            np.ptp(centered),
        ))
    return features


def trend_features(series):
    features = np.zeros((len(SIGNALS), 14), dtype=np.float32)
    for name, values in series:
        features[SIGNAL_INDEX[name]] = summarize_series(values)
    return features.ravel()


def operator_windows(path, featurize=trend_features):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            example = row["example"]
            if not example["options"]:
                continue
            category = example["category"]
            prompt = example["prompt"]
            if row.get("target_format") != "operator_response" and category in OPERATOR_CATEGORIES:
                prompt += RESPONSE_INSTRUCTION
            yield Window(
                category, row["turbine"], row["cutoff"], prompt,
                row.get("label", example["answer"]), tuple(example["options"]),
                featurize(
                    (item["name"].split(":", 1)[0], item["values"])
                    for item in example["series"]
                ),
            )


def paired_windows(corpus, path, split):
    from build_paired_qa import PROMPTS
    from tslm_inputs import iter_evaluation

    for example, row in iter_evaluation(corpus, path):
        if row["split"] != split:
            raise ValueError(f"Unexpected split in {path}: {row['split']}")
        yield Window(
            row["family"], row["turbine_id"], example["context"]["cutoff_utc"],
            row["question"], row["expected_outcome"], PROMPTS[row["family"]][1],
            trend_features(zip(CHANNELS, example["values"], strict=True)), row["pair_id"],
        )


def read_operator_panel(args, featurize=trend_features):
    references = json.loads(args.predictions.read_text())
    held_out = {}
    for dataset in (args.dataset, args.supplement):
        for window in operator_windows(dataset / "test.jsonl.gz", featurize):
            previous = held_out.setdefault(window.key(), window)
            if (
                previous.label != window.label or previous.options != window.options
                or not np.array_equal(previous.features, window.features, equal_nan=True)
            ):
                raise ValueError(f"Conflicting test rows for {window.key()}")
    test, guesses = [], []
    seen = set()
    for row in references:
        key = tuple(row[name] for name in ("category", "turbine", "cutoff", "prompt"))
        window = held_out[key]
        if key in seen or row["label"] != window.label:
            raise ValueError(f"Unexpected model prediction for {key}")
        seen.add(key)
        # Match the pinned OpenTSLM option evaluator.
        answer = row["prediction"].strip().lower().rsplit("answer:", 1)[-1]
        normalized = " ".join(answer.strip(".\n\t ").split())
        guess = next((label for label in window.options if label.lower() == normalized), None)
        if (guess == window.label) != row["label_correct"]:
            raise ValueError(f"Model label scoring differs for {key}")
        test.append(window)
        guesses.append(guess)
    categories = {row.category for row in test}
    training = (
        window
        for dataset in (args.dataset, args.supplement)
        for window in operator_windows(dataset / "train.jsonl.gz", featurize)
        if window.category in categories
    )
    return training, test, guesses


def read_paired_panel(args):
    test = list(paired_windows(args.corpus, args.dataset / "test.parquet", "test"))
    pairs = defaultdict(list)
    for row in test:
        pairs[row.pair].append(row)
    for pair in pairs.values():
        if len(pair) != 2 or pair[0].prompt != pair[1].prompt or pair[0].label == pair[1].label:
            raise ValueError("Expected complete pairs with a fixed prompt and opposing labels")
    references = {}
    for line in args.predictions.read_text().splitlines():
        row = json.loads(line)
        key = row["pair_id"], row["family"], row["question"], row["expected_outcome"]
        if key in references:
            raise ValueError(f"Repeated model prediction for {key}")
        references[key] = row["prediction"]
    guesses = []
    for row in test:
        lines = references.pop((row.pair, row.category, row.prompt, row.label)).strip().splitlines()
        match = FINAL_LINE.fullmatch(lines[-1].strip()) if len(lines) > 1 else None
        guess = match[1].strip().lower() if match else None
        guesses.append(guess if guess in row.options else None)
    if references:
        raise ValueError("Model predictions include additional rows outside the test panel")
    return paired_windows(args.corpus, args.dataset / "qa.parquet", "train"), test, guesses


def baseline_predictions(training, test):
    if not test:
        raise ValueError("The test panel is empty")
    groups = defaultdict(list)
    first_test_cutoff = min(row.cutoff for row in test)
    for row in training:
        if row.cutoff >= first_test_cutoff or row.label not in row.options:
            raise ValueError(f"Invalid training row: {row.key()}")
        groups[row.category].append(row)

    test_positions = defaultdict(list)
    for index, row in enumerate(test):
        test_positions[row.category].append(index)
    majority = [None] * len(test)
    logistic = [None] * len(test)
    support = {}
    for category, indices in test_positions.items():
        rows = groups[category]
        labels = Counter(row.label for row in rows)
        if not labels:
            raise ValueError(f"No training examples for {category}")
        support[category] = dict(sorted(labels.items()))
        classes = sorted(labels)
        if len(classes) > 1:
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            classifier.fit(np.stack([row.features for row in rows]), [row.label for row in rows])
            probabilities = classifier.predict_proba(np.stack([test[i].features for i in indices]))
        else:
            probabilities = np.ones((len(indices), 1))
        for index, scores in zip(indices, probabilities, strict=True):
            allowed = [label for label in classes if label in test[index].options]
            if not allowed:
                continue
            majority[index] = max(allowed, key=labels.get)
            logistic[index] = max(allowed, key=dict(zip(classes, scores, strict=True)).get)
    return majority, logistic, support


def metrics(rows, guesses):
    confusion = defaultdict(Counter)
    pairs = defaultdict(list)
    hits = 0
    for row, guess in zip(rows, guesses, strict=True):
        confusion[row.category, row.label][guess or "<invalid>"] += 1
        hits += guess == row.label
        if row.pair is not None:
            pairs[row.pair].append(guess == row.label)
    stats = {}
    predicted = Counter()
    for (category, _), counts in confusion.items():
        predicted.update({(category, label): count for label, count in counts.items()})
    f1, recall = [], []
    for (category, label), counts in sorted(confusion.items()):
        support = sum(counts.values())
        tp = counts[label]
        predicted_label = predicted[category, label]
        label_stats = {
            "support": support,
            "precision": tp / predicted_label if predicted_label else 0.0,
            "recall": tp / support,
            "f1": 2 * tp / (support + predicted_label),
            "predicted_counts": dict(sorted(counts.items())),
        }
        stats.setdefault(category, {})[label] = label_stats
        f1.append(label_stats["f1"])
        recall.append(label_stats["recall"])
    result = {
        "windows": len(rows), "correct": hits, "accuracy": hits / len(rows),
        "macro_f1": float(np.mean(f1)), "balanced_accuracy": float(np.mean(recall)),
        "invalid_labels": guesses.count(None),
    }
    if pairs:
        result.update(pairs=len(pairs), paired_both_correct=sum(map(all, pairs.values())))
    return result, stats


def bootstrap_accuracy(turbines, outcomes):
    """Bootstrap paired row outcomes with the turbine as the resampling unit."""
    _, cluster = np.unique(turbines, return_inverse=True)
    counts = np.bincount(cluster)
    sums = np.bincount(cluster, weights=outcomes)
    samples = np.random.default_rng(42).integers(len(counts), size=(10_000, len(counts)))
    scores = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return np.quantile(scores, [0.025, 0.975]).tolist()


def build_report(test, predictions, support, args):
    turbines = [row.turbine for row in test]
    report = {
        "mode": args.mode,
        "tslm_predictions": args.predictions.parent.name + "/" + args.predictions.name,
        "test_cutoffs_utc": [min(row.cutoff for row in test), max(row.cutoff for row in test)],
        "bootstrap": {"unit": "turbine", "clusters": len(set(turbines)), "replicates": 10_000, "seed": 42},
        "training_label_counts": support,
        "methods": {},
    }
    groups = defaultdict(list)
    for index, row in enumerate(test):
        groups[row.category].append(index)
    outcomes = {}
    for method, guesses in zip(METHODS, predictions, strict=True):
        hits = np.array([guess == row.label for row, guess in zip(test, guesses, strict=True)], dtype=int)
        outcomes[method] = hits
        overall, labels = metrics(test, guesses)
        by_category = {}
        for category, indices in sorted(groups.items()):
            scores, _ = metrics([test[i] for i in indices], [guesses[i] for i in indices])
            by_category[category] = scores | {"labels": labels[category]}
        report["methods"][method] = overall | {
            "accuracy_ci95": bootstrap_accuracy(turbines, hits),
            "category_macro_accuracy": float(np.mean([row["accuracy"] for row in by_category.values()])),
            "by_category": by_category,
        }
    report["tslm_accuracy_difference"] = {}
    for method in METHODS[:-1]:
        difference = outcomes["tslm"] - outcomes[method]
        report["tslm_accuracy_difference"][method] = {
            "difference": float(difference.mean()),
            "ci95": bootstrap_accuracy(turbines, difference),
            "only_tslm_correct": int((difference == 1).sum()),
            "only_baseline_correct": int((difference == -1).sum()),
        }
    report["predictions"] = [
        {
            "category": row.category, "turbine": row.turbine, "cutoff": row.cutoff,
            "prompt": row.prompt, "pair_id": row.pair, "label": row.label,
            **dict(zip(METHODS, guesses, strict=True)),
        }
        for row, guesses in zip(test, zip(*predictions, strict=True), strict=True)
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    paired = modes.add_parser("paired")
    paired.add_argument("--corpus", required=True, type=Path)
    operator = modes.add_parser("operator")
    operator.add_argument("--supplement", required=True, type=Path)
    for mode in (paired, operator):
        mode.add_argument("dataset", type=Path)
        mode.add_argument("output", type=Path)
        mode.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()
    read_panel = read_operator_panel if args.mode == "operator" else read_paired_panel
    training, test, model_guesses = read_panel(args)
    majority, logistic, support = baseline_predictions(training, test)
    result = build_report(test, (majority, logistic, model_guesses), support, args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
