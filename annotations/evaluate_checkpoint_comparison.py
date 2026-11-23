"""Evaluate generated SCADA answers under the same prompt and scoring rules."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from build_paired_qa import PROMPTS
from evaluate_paired_qa import answers
from train_flamingo import load_model, restore
from tslm_inputs import iter_evaluation

NUMBER = r"[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
QUANTITY = re.compile(
    rf"(?<![\w.])({NUMBER})\s*(kW|MW|W|rpm|s|seconds?|°\s*C|A|amperes?|m/s|V|kvar)(?![\w/])",
    re.IGNORECASE,
)
TRIPLE = re.compile(
    rf"({NUMBER})\s*(?:°\s*C)?\s*[,/]\s*({NUMBER})\s*(?:°\s*C)?"
    rf"\s*(?:,\s*(?:and\s*)?|and\s+|/\s*)({NUMBER})\s*°\s*C",
    re.IGNORECASE,
)
INTERVALS = re.compile(
    rf"({NUMBER})\s*(?:of|out of|/)\s*({NUMBER})\s*intervals", re.IGNORECASE
)
ACTION_WORD = re.compile(
    r"\b(check|confirm|compare|watch|monitor|retrieve|review|restore|verify|inspect|validate|consult|investigate)\b",
    re.IGNORECASE,
)
ACTIONS = {
    ("power", "tracking"): r"\b(gap|tracking|trend|reference|output|dispatch)\b",
    (
        "power",
        "shortfall",
    ): r"\b(wind|dispatch|limit\w*|setpoint\w*|controller|reference|curtailment)\b",
    (
        "operation",
        "generating",
    ): r"\b(availab\w*|controller|dispatch|alarm\w*|limit\w*)\b",
    (
        "operation",
        "idle",
    ): r"\b(controller|dispatch|alarm\w*|fault\w*|start|restrictions?|status)\b",
    ("temperature", "warming"): r"\b(load|cooling|limit\w*|temperature\w*|trend)\b",
    ("temperature", "cooling"): r"\b(load|cooling|limit\w*|temperature\w*|trend)\b",
    (
        "electrical",
        "small spread",
    ): r"\b(protection|meter\w*|alarm\w*|trend|phase|current)\b",
    ("electrical", "wide spread"): r"\b(protection|meter\w*|alarm\w*|phase|current)\b",
    ("telemetry", "reporting"): r"\b(readings?|telemetry|operation|controller|trend)\b",
    (
        "telemetry",
        "missing",
    ): r"\b(telemetry|visibility|live|communications?|independent|logger|data)\b",
    (
        "maintenance",
        "error present",
    ): r"\b(first.out|sequence|alarm\w*|event\w*|log\w*|controller)\b",
    (
        "maintenance",
        "no error recorded",
    ): r"\b(live|status|alarm\w*|log\w*|controller)\b",
}


def number(value):
    return float(value.replace(",", "").replace("−", "-"))


def quantities(text, family):
    """Identify reported measurements by field and normalize basic unit variants."""
    claims = []
    for match in QUANTITY.finditer(text):
        value, unit = number(match[1]), match[2].lower().replace(" ", "")
        before = re.split(r"[;\n]|\.(?!\d)", text[: match.start()])[-1].lower()
        if unit in {"kw", "mw", "w"}:
            value *= {"kw": 1, "mw": 1000, "w": 0.001}[unit]
            fields = list(
                re.finditer(
                    r"reference|setpoint|\bpower\b|\boutput\b|\bagainst\b", before
                )
            )
            field = (
                "reference_kw"
                if fields and fields[-1][0] in {"reference", "setpoint", "against"}
                else "power_kw"
            )
            if re.match(
                r"\s*(interval.end\s+)?(reference|setpoint)",
                text[match.end() :],
                re.IGNORECASE,
            ):
                field = "reference_kw"
            if re.search(
                r"\b(gap|difference|shortfall)\s*(?:was|of|is|by|:)?\s*$", before
            ):
                field = "power_gap_kw"
        else:
            field = {
                "rpm": "generator_rpm",
                "s": "running_seconds",
                "second": "running_seconds",
                "seconds": "running_seconds",
                "°c": "unresolved_temperature",
                "a": "phase_spread_a",
                "ampere": "phase_spread_a",
                "amperes": "phase_spread_a",
                "m/s": "wind_mps",
                "v": "voltage_v",
                "kvar": "reactive_kvar",
            }[unit]
        claims.append((field, value))
    if family == "temperature":
        triples = list(TRIPLE.finditer(text))
        if triples:
            claims = [
                (field, value)
                for field, value in claims
                if field != "unresolved_temperature"
            ]
            for triple in triples:
                claims.extend(
                    zip(
                        ("winding_u_delta_c", "winding_v_delta_c", "winding_w_delta_c"),
                        map(number, triple.groups()),
                        strict=True,
                    )
                )
    if family == "maintenance":
        claims.extend(
            ("error_intervals", number(match[1]))
            for match in INTERVALS.finditer(text)
            if number(match[2]) == 36
        )
    return claims


def expected_measurements(row, example):
    channels = dict(
        zip(
            (
                "P",
                "W",
                "R",
                "C",
                "S",
                "ON",
                "READY",
                "ERR",
                "B",
                "A",
                "U",
                "V",
                "Z",
                "VR",
                "VS",
                "VT",
                "IR",
                "IS",
                "IT",
                "Q",
            ),
            example["values"],
            strict=True,
        )
    )
    family = row["family"]
    required = dict(quantities(row["target"], family))
    latest = {
        field: channels[channel][-1]
        for field, channel in (
            ("power_kw", "P"),
            ("reference_kw", "C"),
            ("generator_rpm", "R"),
            ("running_seconds", "ON"),
            ("wind_mps", "W"),
            ("reactive_kvar", "Q"),
        )
    }
    latest.update(required)
    if latest["power_kw"] is not None and latest["reference_kw"] is not None:
        latest["power_gap_kw"] = abs(channels["C"][-1] - channels["P"][-1])
    return required, latest


def score_row(example, row, output):
    family, outcome = row["family"], row["expected_outcome"]
    choices = PROMPTS[family][1]
    labels = re.findall(r"\bAnswer:\s*([^.!?\n]+)", output, re.IGNORECASE)
    recognized = [
        item.strip().lower() for item in labels if item.strip().lower() in choices
    ]
    predicted = recognized[0] if recognized and len(set(recognized)) == 1 else None
    expected, supported = expected_measurements(row, example)
    matched = set()
    unverified = []
    for field, value in quantities(output, family):
        reference = supported.get(field)
        tolerance = 0.051 if field == "phase_spread_a" else 0.501
        if reference is not None and abs(value - reference) < tolerance:
            matched.add(field)
        else:
            unverified.append(
                {"field": field, "reported": value, "reference": reference}
            )
    # This screen checks topic and a verification verb. Engineering review is separate.
    guidance = any(
        ACTION_WORD.search(sentence)
        and re.search(ACTIONS[family, outcome], sentence, re.IGNORECASE)
        for sentence in re.split(r"(?<=[.!?])\s+|\n", output)
    )
    return {
        "label_correct": predicted == outcome,
        "label": predicted,
        "matching_measurements": sorted(matched & expected.keys()),
        "reference_measurements": expected,
        "unverified_measurements": unverified,
        "guidance_screen": guidance,
    }


def summarize(records):
    families = defaultdict(list)
    pairs = defaultdict(list)
    for record in records:
        families[record["family"]].append(record)
        pairs[record["pair_id"]].append(record)
    if any(
        len(pair) != 2
        or pair[0]["question"] != pair[1]["question"]
        or pair[0]["expected_outcome"] == pair[1]["expected_outcome"]
        for pair in pairs.values()
    ):
        raise ValueError(
            "Expected complete pairs with identical prompts and opposing outcomes"
        )
    return {
        "rows": len(records),
        "correct_labels": sum(row["label_correct"] for row in records),
        "pairs": len(pairs),
        "both_correct": sum(
            all(row["label_correct"] for row in pair) for pair in pairs.values()
        ),
        "both_correct_with_guidance": sum(
            all(row["label_correct"] and row["guidance_screen"] for row in pair)
            for pair in pairs.values()
        ),
        "reference_measurements": sum(
            len(row["reference_measurements"]) for row in records
        ),
        "matched_measurements": sum(
            len(row["matching_measurements"]) for row in records
        ),
        "answers_with_unverified_measurements": sum(
            bool(row["unverified_measurements"]) for row in records
        ),
        "guidance_screen_passes": sum(row["guidance_screen"] for row in records),
        "by_family": {
            family: {
                "rows": len(rows),
                "correct": sum(row["label_correct"] for row in rows),
            }
            for family, rows in sorted(families.items())
        },
    }


def paired_input(args):
    for example, row in iter_evaluation(args.corpus, args.panel):
        family = row["family"]
        prompt = PROMPTS[family][0].replace("below 3 A or above 8 A", "small or wide")
        left, right = PROMPTS[family][1]
        question = (
            f"{prompt} Include the label Answer: {left} or Answer: {right}. "
            "Briefly describe the evidence and the next check."
        )
        yield {**example, "question": question}, {**row, "question": question}


def generate(args):
    checkpoint = torch.load(
        args.checkpoints[0][1], map_location="cpu", weights_only=True
    )
    model, tokenizer, revisions = load_model(
        "cuda", checkpoint["revisions"]["checkpoint"], checkpoint["revisions"]["base"]
    )
    params = {
        key: value for key, value in model.named_parameters() if value.requires_grad
    }
    scaling = json.loads(args.scaling.read_text())
    summary = {}
    for index, (name, path) in enumerate(args.checkpoints):
        if index:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if revisions != checkpoint["revisions"]:
            raise ValueError("Compared checkpoints must use identical base revisions")
        restore(params, checkpoint)
        del checkpoint
        model.eval()
        rows = (
            iter_evaluation(args.corpus, args.panel)
            if args.broad
            else paired_input(args)
        )
        output = args.output / f"{name}.jsonl"
        with output.with_suffix(".tmp").open("w") as stream:
            records = []
            for (example, row), prediction in answers(model, tokenizer, scaling, rows):
                record = {
                    "question": example["question"],
                    "prediction": prediction,
                    "turbine": row["turbine_id"],
                    "cutoff_utc": row["cutoff_utc"].isoformat(),
                    "month": row["month"],
                    "station_id": row["station_id"],
                    "stop_step": row["stop_step"],
                }
                if args.broad:
                    record.update(
                        expected_state=row["expected_state"],
                        reference_answer=row["reference_answer"],
                        scenario_references=row["scenario_references"],
                    )
                else:
                    record.update(
                        pair_id=row["pair_id"],
                        family=row["family"],
                        expected_outcome=row["expected_outcome"],
                        target=row["target"],
                        **score_row(example, row, prediction),
                    )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            summary[name] = {"rows": len(records)} if args.broad else summarize(records)
        output.with_suffix(".tmp").replace(output)
        print(name, json.dumps(summary[name]), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("panel", type=Path)
    parser.add_argument("scaling", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--checkpoint", dest="checkpoints", action="append", nargs=2, required=True
    )
    parser.add_argument("--broad", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generate(args)


if __name__ == "__main__":
    main()
