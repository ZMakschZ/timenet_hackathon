"""Additional SCADA-grounded operator answers for retrospective questions."""

import math
from dataclasses import replace

from operator_response_targets import RESPONSE_INSTRUCTION, _signal, _thermal_check

POWER_REFERENCE = "power_reference_tracking"
TELEMETRY = "telemetry_availability"
_UNKNOWN = "insufficient evidence"
_WINDINGS = ("U", "V", "Z")
_CONDITION_TEXT = {
    "anomaly_end": {
        "recovered": "The earlier elevation cleared before cutoff.",
        "still elevated": "The elevation persisted through the latest interval.",
        "no elevation": "The matched comparison showed no elevation.",
    },
    "anomaly_categorization": {
        "brief elevation": "The elevation was brief.",
        "persistent elevation": "Elevated readings were consecutive.",
        "intermittent elevation": "Elevated readings appeared and cleared within the window.",
        "no elevation": "The matched comparison showed no elevation.",
    },
}


def _operator_answer(example, evidence, assessment, action):
    return replace(
        example,
        prompt=example.prompt + RESPONSE_INSTRUCTION,
        answer=f"Evidence: {evidence}\nAssessment: {assessment}\nNext step: {action}\nAnswer: {example.answer}",
    )


def _channel_name(channel):
    return channel.name.split(": ", 1)[1].split(" (", 1)[0]


def _missing(example, keys=(), steps=None):
    if keys:
        channels = [(key, _signal(example, key)) for key in keys]
    else:
        channels = [(_channel_name(channel), channel) for channel in example.series]
    return [
        key
        for key, channel in channels
        if channel is None
        or any(not math.isfinite(x) for x in (channel.values if steps is None else channel.values[-steps:]))
    ]


def _telemetry(example):
    if example.answer == "unavailable":
        return (
            "The window has no usable SCADA readings.",
            "Turbine operating status is unresolved.",
            "Check the live controller, independent metering and the SCADA link before updating availability.",
        )
    if example.answer == "reporting":
        return (
            "The selected signals have readings throughout the window.",
            "SCADA coverage is complete for this comparison.",
            "Use the live controller to verify current conditions before making an operating change.",
        )
    missing = _missing(example)
    tags = f" Recorded gaps include {', '.join(missing[:3])}." if missing else ""
    return (
        "Some expected readings are missing from the window." + tags,
        "The available channels give a partial view of turbine operation.",
        "Check the live controller or independent unit metering for the missing information before updating availability.",
    )


def _operating_state(example):
    states = {
        "generating": (
            "Latest power, generator speed and running time are consistent with generation.",
            "The turbine generated in the latest interval.",
            "Check the live controller and dispatch availability before committing its output.",
        ),
        "idle or stopped": (
            "Latest power and generator speed are low, with no logged running time.",
            "The turbine appears idle or stopped.",
            "Check live status, readiness and active alarms before requesting a restart.",
        ),
        "transitioning": (
            "Latest power, generator speed and running time give a mixed operating picture.",
            "Treat the state as transitional until later readings agree.",
            "Watch subsequent power and speed and confirm the current state in the controller.",
        ),
        _UNKNOWN: (
            "The latest operating readings are incomplete or invalid.",
            "The operating state is unresolved.",
            "Verify power, generator speed and running status independently before updating dispatch.",
        ),
    }
    return states[example.answer]


def _error_history(example):
    labels = {
        "separated error-active periods": "The error flag was active in separate parts of the window.",
        "one error-active period": "The error flag shows one continuous observed period of activity.",
        "no observed error activity": "The error flag was clear throughout the observed window.",
        _UNKNOWN: "The error flag history contains missing or unusable readings.",
    }
    return (
        labels[example.answer],
        "The recorded flag describes error activity. The controller event sequence is needed to establish the cause.",
        "Retrieve the live alarms and event history. Include the observed error periods in the handover if present.",
    )


def _phase_spread(example):
    labels = {
        "persistent current and voltage spread": "Current and voltage both show sustained phase-to-phase spread under steady load.",
        "persistent current spread": "Phase-current readings remained separated under steady load.",
        "no persistent current spread": "Current spread did not persist across the steady-load comparison.",
        _UNKNOWN: "The window lacks a complete steady-load comparison across all phases.",
    }
    return (
        labels[example.answer],
        "The phase comparison is a screening indication. Independent metering can help separate an electrical deviation from a measurement issue.",
        "Compare time-aligned controller and independent phase readings, and verify the protection indications with electrical maintenance.",
    )


def _winding_telemetry(example):
    gaps = _missing(example, _WINDINGS, steps=6)
    if example.answer == "winding telemetry complete":
        return (
            "All selected winding temperatures reported in the latest hour.",
            "Winding telemetry is available for recent monitoring.",
            "Continue temperature monitoring and verify any protection indication on the live controller.",
        )
    evidence = (
        f"The latest hour has gaps in winding readings {', '.join(gaps)}."
        if gaps
        else "The winding or operating evidence is incomplete."
    )
    assessment = (
        "Some winding temperatures are missing during confirmed generation."
        if example.answer == "winding telemetry missing during generation"
        else "The available readings do not establish whether winding telemetry was missing during generation."
    )
    return evidence, assessment, "Ask SCADA maintenance to compare the affected tags with live controller readings."


def _thermal_followup(example):
    if example.answer == _UNKNOWN:
        return (
            "The temperature history or comparable operating readings are incomplete.",
            "There is insufficient evidence to classify the thermal pattern.",
            "Verify the affected temperatures and obtain comparable load, speed and ambient readings.",
        )
    evidence = "Temperatures were compared with earlier readings at similar load, speed and ambient temperature."
    if example.category == "anomaly_identification":
        channel = _signal(example, example.answer)
        if channel is not None:
            assessment = f"The {_channel_name(channel)} shows elevation relative to matched earlier operation."
        elif example.answer == "multiple elevated channels":
            assessment = "More than one temperature channel shows elevation relative to matched earlier operation."
        else:
            assessment = "The matched comparison shows no thermal elevation."
        if example.answer == "no elevation":
            action = "Continue monitoring the thermal channels under similar load."
        else:
            action = f"Verify the elevated readings on the controller. {_thermal_check(example.answer)}"
        return evidence, assessment, action
    assessment = _CONDITION_TEXT[example.category][example.answer]
    if example.answer in {"recovered", "no elevation"}:
        action = "Watch the same temperature under similar load and verify the live alarms if the elevation returns."
    elif example.answer == "brief elevation":
        action = (
            "Recheck the live temperature under similar load and review alarms if the elevation persists or recurs."
        )
    else:
        key = example.series[0].name.split(":", 1)[0]
        action = f"Verify the live temperature. {_thermal_check(key)}"
        if example.answer == "still elevated":
            action += " If the site derating criteria are met, consider reducing power under the approved procedure."
    return evidence, assessment, action


_FOLLOWUPS = {
    TELEMETRY: _telemetry,
    "operating_state": _operating_state,
    "maintenance_error_recurrence": _error_history,
    "maintenance_phase_current": _phase_spread,
    "maintenance_thermal_telemetry": _winding_telemetry,
    "anomaly_end": _thermal_followup,
    "anomaly_categorization": _thermal_followup,
    "anomaly_identification": _thermal_followup,
}


def expanded_target(example):
    """Return a full response for a supported existing category."""
    builder = _FOLLOWUPS.get(example.category)
    return _operator_answer(example, *builder(example)) if builder is not None else None


def power_reference_target(example):
    """Reuse a full-channel telemetry window to assess tracking at a steady reference."""
    if example.category != TELEMETRY:
        return None
    keys = ("P", "C", "R", "ON", "W")
    by_key = {key: channel for key in keys if (channel := _signal(example, key)) is not None}
    if not {"P", "C"} & by_key.keys():
        return None
    latest = {key: channel.values[-3:] for key, channel in by_key.items()}
    required = ("P", "C", "R", "ON")
    complete = all(key in latest and len(latest[key]) == 3 and all(map(math.isfinite, latest[key])) for key in required)
    label = _UNKNOWN
    evidence = "Recent power, reference or operating readings are incomplete."
    assessment = "The comparable readings are insufficient to judge output tracking."
    action = "Verify live active power, the current reference and the operating state before assessing a shortfall."
    if complete:
        power, reference, speed, running = (latest[key] for key in required)
        comparable = (
            min(power) >= 50
            and min(speed) > 300
            and min(running) > 0
            and max(running) <= 600
            and min(reference) >= 400
            and max(reference) - min(reference) <= 40
        )
        if comparable:
            gaps = [target - measured for target, measured in zip(reference, power, strict=True)]
            if all(abs(gap) <= 120 for gap in gaps):
                label = "tracking reference"
                evidence = (
                    "Recent active power stayed close to a steady power reference while the turbine was generating."
                )
                assessment = "Output is tracking the supplied reference."
                action = "Continue monitoring and check dispatch and active limits if the gap opens."
            elif all(gap >= 300 for gap in gaps):
                label = "persistent shortfall"
                evidence = "Recent active power stayed below a steady power reference while the turbine was generating."
                assessment = "The sustained gap warrants a check of available power and controller limits."
                action = "Check wind availability, dispatch constraints and the live controller before attributing the gap to a fault."
            else:
                evidence = (
                    "Recent output did not consistently meet either the close-tracking or sustained-shortfall screen."
                )
        else:
            evidence = "The latest readings do not show steady-reference generation for the comparison."
    original = replace(
        example,
        series=list(by_key.values()),
        category=POWER_REFERENCE,
        answer=label,
        prompt="The SCADA readings are spaced ten minutes apart. Is active power tracking its recent reference?",
        options=["tracking reference", "persistent shortfall", _UNKNOWN],
    )
    return _operator_answer(original, evidence, assessment, action)
