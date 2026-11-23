"""Create a qualitative sibling of an existing paired or targeted QA dataset."""

import argparse
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

PAIRED_EVIDENCE = {
    ("power", "tracking"): "Mean power was close to the interval-end reference.",
    ("power", "shortfall"): "Mean power was well below the interval-end reference.",
    ("operation", "generating"): "Power output, generator speed and recorded running time indicate generation.",
    ("operation", "idle"): "Low power and generator speed accompanied an absence of recorded running time.",
    ("temperature", "warming"): "All generator winding temperatures rose over the latest hour.",
    ("temperature", "cooling"): "All generator winding temperatures fell over the latest hour.",
    ("electrical", "small spread"): "The phase-current averages were close together at cutoff.",
    ("electrical", "wide spread"): "The phase-current averages were widely separated at cutoff.",
    ("telemetry", "reporting"): "Power, wind, generator speed and running time all reported at cutoff.",
    ("telemetry", "missing"): "None of the selected channels reported at cutoff.",
    ("maintenance", "error present"): "The error flag was active at cutoff and in earlier readings.",
    ("maintenance", "no error recorded"): "The error flag stayed inactive throughout the window.",
}

TARGETED_EVIDENCE = {
    "rated_pitch_response": "Wind speed and blade A pitch rose while power held near its rating.",
    "below_rated_reference": "Wind speed rose while the below-rated reference stayed steady and power tracked it.",
    "wind_associated_ramp": "Power and wind speed rose over the latest hour while the power reference stayed nearly steady.",
    "falling_power_with_wind": "Power and wind speed fell while the reference held steady and the turbine kept operating.",
    "reference_reduction_and_power_drop": "The power reference and measured output both fell while wind speed changed little.",
    "multiple_start_stop_transitions": "Power, speed and running time crossed the generation screen repeatedly. Power and speed were low at cutoff.",
    "generation_after_repeated_starts": "The readings crossed the generation screen repeatedly earlier in the window. The turbine kept running and producing throughout the latest hour.",
    "ready_without_generation": "The turbine reported ready throughout the latest hour, with low speed and power and no logged running time.",
    "sustained_generation_after_idle": "Speed and power rose after an idle reading, and the turbine sustained generation through the latest readings.",
    "stop_with_error_flag": "Power and generator speed fell. The latest interval recorded an active turbine error and no running time.",
    "winding_spread_widens": "The hottest winding reading rose and the spread between windings widened at comparable load, speed and ambient temperature.",
    "windings_warm_with_load": "All winding temperatures rose together as power increased, with little change in the spread between windings.",
    "windings_cool_as_load_falls": "All winding temperatures fell as power output dropped.",
    "stable_winding_temperature_offset": "The windings stayed separated by a similar amount at comparable load, speed and ambient temperature.",
    "phase_spreads_move_differently": "The phase-current spread widened while the phase-voltage spread changed little.",
    "low_phase_spread_at_load": "The phase-current and phase-voltage averages stayed close together during high output.",
    "all_telemetry_missing_at_cutoff": "All selected channels were missing at cutoff, following earlier intervals with reported readings.",
    "reporting_resumes_after_gap": "Power, wind, generator speed and running time returned at cutoff after a gap across all selected channels.",
    "earlier_temperature_gaps": "Ambient and winding temperature readings had gaps earlier in the window and were present again at cutoff.",
    "near_flat_wind_while_generating": "The reported wind stayed nearly flat while power varied during generation.",
    "fresh_zero_running_time": "Running time reported no operation throughout the latest hour. Wind was recorded and generator speed and power stayed low.",
    "recurring_error_flag_after_generation": "The error flag was active in separate stretches and at cutoff. Latest power was low after earlier generation.",
    "repeated_thermal_separation": "Winding readings separated repeatedly during the window and remained apart at cutoff under comparable recent operating conditions.",
    "current_spread_under_steady_load": "Phase-current spread persisted while active and reactive power stayed relatively steady.",
    "live_operation_check_telemetry": "Ambient and winding temperature readings had gaps, while the latest power and running-time readings show generation.",
}

GUIDANCE_PHRASES = {
    "All three windings": "All windings",
    "across the last seven intervals": "throughout the latest hour",
    "the ten-minute averages": "the interval averages",
    "one persistent condition or several events": "a persistent condition or distinct events",
    "All three readings rose. One rose faster, which warrants comparison": "The widening warrants comparison",
}


def paired_text(row, *, evidence_first=False):
    label = row["expected_outcome"]
    prefix = f"Answer: {label}. "
    evidence, separator, action = row["target"].removeprefix(prefix).rpartition(". ")
    if not row["target"].startswith(prefix) or not separator or not evidence or not action:
        raise ValueError(f"Unexpected paired target in {row['pair_id']}")
    question = row["question"].replace("below 3 A or above 8 A", "small or wide")
    explanation = f"{PAIRED_EVIDENCE[row['family'], label]} {action}"
    if evidence_first:
        prompt, separator, options = question.partition(" Begin the answer with Answer: ")
        options, suffix, _ = options.partition(", then give brief evidence and a check.")
        options = options.replace(" or Answer: ", " or ")
        if not separator or not suffix:
            raise ValueError(f"Unexpected paired prompt in {row['pair_id']}")
        question = (
            f"{prompt} Briefly describe the evidence and the next check. "
            f"End on a new line with Answer: followed by {options}."
        )
        target = f"{explanation}\nAnswer: {label}."
    else:
        target = f"{prefix}{explanation}"
    return {"question": question, "target": target}


def direction(text, pattern):
    endpoints = re.search(pattern, text)
    if endpoints is None:
        raise ValueError(f"Missing expected endpoints in {text!r}")
    first, last = (float(value.replace(",", "")) for value in endpoints.groups())
    return "rose" if last > first else "fell" if last < first else "held steady"


def targeted_evidence(scenario, evidence):
    number = r"(-?[\d,.]+)"
    endpoints = f"{number} to {number}"
    if scenario == "thermal_handover_after_output_loss":
        trend = direction(evidence, f"winding moved from {endpoints} °C")
        return f"Power dropped to a low level and the hottest reported winding temperature {trend}. The turbine error was active in the latest interval."
    if scenario == "current_spread_eases_during_power_change":
        trend = direction(evidence, f"Power changed from {endpoints} kW")
        return f"The current spread eased from an earlier peak while power {trend} over the latest hour."
    if scenario == "grid_phase_voltage_spread_increases":
        trend = direction(evidence, f"Current spread went from {endpoints} A")
        return f"The phase-voltage spread widened. The phase-current spread {trend} over the same period."
    if scenario == "one_winding_moves_more":
        match = re.match(r"1([UVW])1 moved the most", evidence)
        if match is None:
            raise ValueError(f"Missing winding identifier in {evidence!r}")
        return f"The phase {match[1]} winding temperature changed the most at comparable load, speed and ambient temperature."
    if scenario == "steady_phase_current_spread":
        match = re.search(r"Phase ([RST]) is farthest", evidence)
        if match is None:
            raise ValueError(f"Missing phase identifier in {evidence!r}")
        return f"The phase-current spread persisted at comparable load. Phase {match[1]} was farthest from the phase-current average at cutoff."
    return TARGETED_EVIDENCE[scenario]


def targeted_text(row):
    evidence = targeted_evidence(row["scenario"], row["evidence"])
    guidance = row["final_answer"]
    for original, qualitative in GUIDANCE_PHRASES.items():
        guidance = guidance.replace(original, qualitative)
    return {
        "evidence": evidence,
        "final_answer": guidance,
        "target": f"{evidence}\nAnswer: {guidance}",
    }


def convert(source, output, *, evidence_first=False, test=None):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    files = tuple(path.name for path in sorted(source.glob("*.parquet")))
    if files == ("qa.parquet", "validation.parquet"):
        def render(row):
            return paired_text(row, evidence_first=evidence_first)
    elif not evidence_first and files == ("test.parquet", "train.parquet", "validation.parquet"):
        render = targeted_text
    else:
        raise ValueError(f"Unexpected QA splits: {files}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        staged.mkdir()
        inputs = {name: source / name for name in files}
        if test is not None:
            inputs["test.parquet"] = test
        for name, path in inputs.items():
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            with pq.ParquetWriter(staged / name, schema, compression="zstd") as writer:
                for batch in parquet.iter_batches(batch_size=8192):
                    rows = batch.to_pylist()
                    text = [render(row) for row in rows]
                    for row, updated in zip(rows, text, strict=True):
                        question = updated.get("question", row["question"])
                        if re.search(r"\d", question + updated["target"]):
                            raise ValueError(f"Numeric literal in qualitative {name}: {updated}")
                    table = pa.Table.from_batches([batch])
                    for field in text[0]:
                        index = schema.get_field_index(field)
                        table = table.set_column(
                            index, schema.field(index), pa.array([row[field] for row in text])
                        )
                    writer.write_table(
                        table, row_group_size=512 if evidence_first and name == "qa.parquet" else None
                    )
            print(f"{name}: {parquet.metadata.num_rows:,} rows", flush=True)

        shutil.copyfile(source / "scaling.json", staged / "scaling.json")
        if (source / "manifest.json").is_file():
            shutil.copyfile(source / "manifest.json", staged / "manifest.json")
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence-first", action="store_true")
    parser.add_argument("--test", type=Path)
    args = parser.parse_args()
    if args.test is not None and not args.evidence_first:
        parser.error("--test requires --evidence-first")
    convert(args.source, args.output, evidence_first=args.evidence_first, test=args.test)


if __name__ == "__main__":
    main()
