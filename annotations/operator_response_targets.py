"""Build evidence, outlook and operator follow-up targets for the SCADA demo."""

import math
from dataclasses import replace
from statistics import mean

OPERATOR_CATEGORIES = {
    "condition_monitoring",
    "maintenance_error_return_within_hour",
    "maintenance_gearbox_oil",
    "maintenance_thermal_elevation_within_hour",
    "pitch_lubrication_alarm_within_hour",
    "stop_within_hour",
    "stopping_alarm_within_hour",
}
RESPONSE_INSTRUCTION = (
    " Give the operator the observed evidence, your assessment or forecast, and a conditional next action. "
    "Finish with Answer: followed by the exact answer choice."
)


def _signal(example, key):
    return next((series for series in example.series if series.name.startswith(f"{key}: ")), None)


def _thermal_channel(example):
    return next(
        (item for item in example.series if "temperature" in item.name and not item.name.startswith("A: ")), None
    )


def _thermal_check(key):
    if key in {"U", "V", "Z"}:
        return "Verify generator cooling and compare phase loads."
    if key in {"HS", "OIL"}:
        return "Review gearbox oil pressure and cooling indications."
    if key in {"DE", "NDE", "MAIN"}:
        return "Review bearing lubrication and cooling indications."
    return "Review cooling and any applicable lubrication indications for the affected components."


def _thermal_trend(example):
    """Read the selected physical temperature channel at the two ends of the input."""
    channel = _thermal_channel(example)
    if channel is None:
        return "The available window gives incomplete temperature evidence."
    name = channel.name.split(": ", 1)[1].split(" (", 1)[0]
    first, last = channel.values[:6], channel.values[-6:]
    if any(not math.isfinite(value) for value in (*first, *last)):
        return f"The {name} readings have gaps in the comparison periods."
    change = mean(last) - mean(first)
    direction = "rose" if change >= 1 else "fell" if change <= -1 else "held broadly steady"
    return f"The {name} {direction} from the opening hour to the latest hour."


def _generation_evidence(example):
    """Describe cutoff operation and recent observations available to the forecast."""
    evidence = ["Power, generator speed and running time indicate generation at cutoff."]
    for key, name, tolerance in (("P", "Active power", 40), ("W", "Wind speed", 0.5)):
        series = _signal(example, key)
        if series is None:
            continue
        earlier, recent = series.values[-12:-6], series.values[-6:]
        if len(earlier) != 6 or len(recent) != 6 or not all(map(math.isfinite, (*earlier, *recent))):
            continue
        change = mean(recent) - mean(earlier)
        direction = "rose" if change >= tolerance else "fell" if change <= -tolerance else "held broadly steady"
        evidence.append(f"{name} {direction} in the latest hour compared with the preceding hour.")
    error = _signal(example, "ERR")
    if error is not None:
        recent = error.values[-6:]
        if any(math.isfinite(value) and value > 0 for value in recent):
            evidence.append("The error flag was active during the latest hour.")
        elif len(recent) == 6 and all(value == 0 for value in recent):
            evidence.append("The error flag was clear throughout the latest hour.")
    return " ".join(evidence)


def operator_target(example):
    """Combine a curated qualitative target with actions conditioned on live verification."""
    category, label = example.category, example.answer
    if category not in OPERATOR_CATEGORIES:
        return example
    if category == "maintenance_thermal_elevation_within_hour":
        evidence = _thermal_trend(example)
        channel = _thermal_channel(example)
        key = channel.name.split(":", 1)[0] if channel is not None else None
        positive = label == "persistent elevation observed"
        outlook = (
            "I expect elevated temperature over the next hour if comparable generation continues."
            if positive
            else "I do not expect sustained elevation over the next hour if comparable generation continues."
        )
        action = (
            f"Verify the live temperature and active alarms. {_thermal_check(key)} "
            "If confirmed temperatures meet the "
            "site's derating criteria, consider reducing power under the approved procedure. If a lubrication "
            "fault is confirmed, ask maintenance to start the approved lubrication response for that component."
            if positive
            else "Continue monitoring and review the live alarm state if the temperature rises."
        )
    elif category == "condition_monitoring":
        evidence = (
            "The available records do not establish a matched comparison for the latest day."
            if label == "insufficient evidence"
            else "The latest day's temperatures can be compared with earlier readings at similar load, "
            "speed and ambient temperature."
        )
        assessments = {
            "persistent elevation": "The latest day shows persistent elevation relative to matched earlier operation.",
            "no persistent elevation": "The matched comparison shows no persistent elevation over the latest day.",
            "insufficient evidence": "The matched history is insufficient to judge persistent elevation.",
        }
        outlook = assessments[label]
        action = (
            "Verify the live reading and open a condition review for the affected component if the elevation persists. "
            "Consider reducing power if the verified readings meet the site's approved derating criteria."
            if label == "persistent elevation"
            else "Monitor the temperature and obtain comparable readings before escalating a component inspection."
        )
    elif category == "maintenance_gearbox_oil":
        descriptions = {
            "warmer oil and lower pressure": "The oil is warmer and pressure is lower than during comparable operation.",
            "warmer oil": "Gearbox oil temperature is higher than during comparable operation.",
            "lower oil pressure": "Gearbox oil pressure is lower than during comparable operation.",
            "no sustained warming or pressure drop": "The matched readings show no sustained warming or pressure drop.",
            "insufficient evidence": "Comparable gearbox oil readings are insufficient to assess a sustained change.",
        }
        evidence = descriptions[label]
        pressure = label in {"warmer oil and lower pressure", "lower oil pressure"}
        outlook = (
            "The pressure change warrants a lubrication-system check."
            if pressure
            else "Review the operating context before attributing the readings to a gearbox fault."
        )
        action = (
            "Cross-check live oil pressure and controller alarms. If a lubrication fault is confirmed, ask maintenance "
            "to start the approved lubrication response and consider reducing power if the site procedure calls for it."
            if pressure
            else "Verify oil temperature and cooling indications against comparable operation."
        )
    elif category == "maintenance_error_return_within_hour":
        evidence = "The turbine error flag was active earlier and has cleared by the cutoff."
        positive = label == "error flag returned"
        outlook = (
            "I expect the error flag to return within the next hour."
            if positive
            else "I expect the error flag to stay clear over the next hour."
        )
        action = "Check the live controller and capture the event sequence if the flag returns."
    else:
        evidence = _generation_evidence(example)
        positive = label in {"stop observed", "alarm observed"}
        event = {
            "stop_within_hour": "a sustained stop",
            "stopping_alarm_within_hour": "a stopping alarm",
            "pitch_lubrication_alarm_within_hour": "a pitch lubrication alarm",
        }[category]
        outlook = f"I expect {event} in the next hour." if positive else f"I do not expect {event} in the next hour."
        if positive and category == "pitch_lubrication_alarm_within_hour":
            action = (
                "Verify the live pitch lubrication alarm. If it is confirmed, ask maintenance to start the "
                "approved lubrication response for the indicated pitch system."
            )
        elif positive:
            action = (
                "Verify the live controller alarms and prepare for a possible stop. Consider reducing power "
                "if live indications meet the site's approved derating criteria."
            )
        else:
            action = "Continue monitoring and follow the site response procedure if live alarms appear."
    return replace(
        example,
        prompt=example.prompt + RESPONSE_INSTRUCTION,
        answer=f"Evidence: {evidence}\nAssessment: {outlook}\nNext step: {action}\nAnswer: {label}",
    )
