"""Render short evidence traces and operator answers from observed windows."""

WINDINGS = ("U", "V", "Z")
PHASES = ("R", "S", "T")


def n(value):
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def at(timestamp):
    return timestamp.strftime("%H:%M UTC")


def change(first, last, field, unit):
    return f"{n(first[field])} to {n(last[field])} {unit}"


def spread(row, fields):
    return row[list(fields)].max() - row[list(fields)].min()


def pair(reason_question, evidence, qa_question, guidance):
    return (reason_question, evidence), (qa_question, f"{evidence} {guidance}")


def power(case, frame):
    first, last = frame.iloc[-7], frame.iloc[-1]
    if case == "rated_pitch_response":
        recent = frame.tail(7)
        return pair(
            "How did pitch and power respond to rising wind over the last hour?",
            f"Wind rose from {change(first, last, 'W', 'm/s')} and blade A pitch changed from "
            f"{change(first, last, 'B', 'degrees')}. Power stayed between {n(recent.P.min())} and "
            f"{n(recent.P.max())} kW around the 2,300 kW rating.",
            "Wind is picking up. Why is output barely moving, and should I adjust anything?",
            "The readings are consistent with regulating near rated power. Check for a sustained "
            "departure from the reference or an active controller alarm before revisiting the setting.",
        )
    if case == "below_rated_reference":
        return pair(
            "How did rising wind affect tracking of the below-rated reference?",
            f"Wind rose from {change(first, last, 'W', 'm/s')} in an hour. Latest power was "
            f"{n(last.P)} kW against a {n(last.C)} kW reference, a "
            f"{n(abs(last.C - last.P))} kW difference. The reference changed by less than 10 kW across the hour.",
            "Why isn't this turbine picking up more power as the wind increases?",
            "The controller reference is below rated power and the output is close to it. Check the "
            "dispatch request and active limits before treating this as underperformance or raising the request.",
        )
    if case == "wind_associated_ramp":
        return pair(
            "What was the average power ramp rate over the last hour and what happened to wind?",
            f"Power rose from {change(first, last, 'P', 'kW')}, averaging "
            f"{n((last.P - first.P) / 60)} kW/min between the two reported means. Wind rose from "
            f"{change(first, last, 'W', 'm/s')} with less than 10 kW variation in the reference.",
            "What drove the increase in output, and do I need to intervene?",
            "The rise is consistent with increasing wind under a near-constant reference. If the wind "
            "keeps rising, watch how power and pitch behave as the turbine approaches its rated output.",
        )
    if case == "falling_power_with_wind":
        return pair(
            "Did the reference fall with power, and how did wind change?",
            f"Over the last hour power fell from {change(first, last, 'P', 'kW')} and wind "
            f"fell from {change(first, last, 'W', 'm/s')}. The reference varied by less than "
            "10 kW and the turbine recorded 600 seconds in operation in every interval.",
            "Should I treat the falling output as a turbine problem or adjust the dispatch forecast?",
            "The decline is consistent with weakening wind during continued operation. Check whether "
            "the drop follows the site wind trend. If output remains depressed after wind recovers, "
            "review the live controller state and active limits.",
        )
    return pair(
        "How much did the reference and measured power fall, and how much did wind vary?",
        f"Over one hour the reference fell from {change(first, last, 'C', 'kW')} and measured power "
        f"fell from {change(first, last, 'P', 'kW')}. Wind varied by {n(frame.W.tail(7).max() - frame.W.tail(7).min())} m/s.",
        "Output fell while the wind changed little. What should I check first?",
        "The observed endpoints are consistent with a lower power request. Check the dispatch and "
        "limit history to learn why the reference changed. The interval means and interval-end references "
        "leave the exact sequence within each interval unresolved.",
    )


def operation(case, frame):
    first, last = frame.iloc[-7], frame.iloc[-1]
    producing = frame.R.gt(300) & frame.P.gt(50) & frame.ON.gt(0)
    starts = (producing & ~producing.shift(fill_value=False)).iloc[1:]
    stops = (~producing & producing.shift(fill_value=False)).iloc[1:]
    if case == "multiple_start_stop_transitions":
        return pair(
            "How many start-like and stop-like transitions can you identify in this window?",
            f"The six-hour view contains {int(starts.sum())} start-like and {int(stops.sum())} stop-like "
            "transitions across a screening condition of over 300 rpm, over 50 kW and positive time in operation. "
            f"The latest power is {n(last.P)} kW and speed is {n(last.R)} rpm.",
            "Is this turbine ready to carry load, or are we still cycling?",
            "Output is low at cutoff after multiple crossings of the generation screen. Check the controller start permits "
            "and event sequence to distinguish physical restarts from operation near the screening threshold.",
        )
    if case == "generation_after_repeated_starts":
        return pair(
            "What changed between the earlier crossings of the generation screen and the latest hour?",
            f"The generation screen was entered {int(starts.sum())} times and exited {int(stops.sum())} times "
            "in the six-hour view. The last seven intervals each show 600 seconds in operation "
            f"and power of at least {n(frame.P.tail(7).min())} kW.",
            "The unit looks to be generating steadily now. Can I count on it?",
            "Generation is sustained in the latest readings. Confirm the earlier transitions with the event log and check live alarm clearance and site "
            "restart acceptance criteria before declaring normal availability. Watch for another RPM or power drop.",
        )
    if case == "ready_without_generation":
        return pair(
            "Does the ready flag coincide with actual generation?",
            "Each of the last seven intervals shows 600 seconds ready and zero seconds in operation. "
            f"The latest generator speed is {n(last.R)} rpm and active power is {n(last.P)} kW.",
            "The unit says ready. Should I schedule its output?",
            "There is no active generation in the latest readings. Check the start request, "
            "start permissions and live availability before scheduling the output.",
        )
    if case == "sustained_generation_after_idle":
        return pair(
            "What evidence distinguishes the latest generation from the idle reading an hour ago?",
            f"Generator speed went from {change(first, last, 'R', 'rpm')} and power went from "
            f"{change(first, last, 'P', 'kW')}. Each of the last three intervals recorded 600 seconds "
            "in operation and more than 100 kW.",
            "What is the unit doing after the idle period, and what should the next shift watch?",
            "The unit is generating again. Watch for a return to idle and confirm it meets the site's "
            "normal availability criteria before relying on the recovery.",
        )
    return pair(
        "What happened to power, generator speed and the turbine-error flag in the latest hour?",
        f"Power fell from {change(first, last, 'P', 'kW')} and speed fell from {change(first, last, 'R', 'rpm')}. "
        f"The latest interval had {n(last.ERR)} seconds with turbine error active and zero seconds in operation.",
        "The unit has stopped producing and the error flag is up. What should I do before restart?",
        "Check the live controller's first-out alarm and start permissions. The error flag establishes "
        "that an error was active during the reporting interval. The component and restart eligibility remain unresolved.",
    )


def temperature(case, frame):
    first, last = frame.iloc[-7], frame.iloc[-1]
    differences = [last[key] - first[key] for key in WINDINGS]
    values = "/".join(n(value) for value in differences)
    gap = f"{n(spread(first, WINDINGS))} to {n(spread(last, WINDINGS))} °C"
    load = f"Power went from {change(first, last, 'P', 'kW')}."
    range_text = (
        f"Over those readings power spanned {n(frame.P.tail(7).min())}–{n(frame.P.tail(7).max())} kW, "
        f"speed varied by {n(frame.R.tail(7).max() - frame.R.tail(7).min())} rpm "
        f"and ambient varied by {n(frame.A.tail(7).max() - frame.A.tail(7).min())} °C."
    )
    if case == "winding_spread_widens":
        return pair(
            "How much did winding spread widen under comparable operating conditions?",
            f"Over the last hour the winding spread went from {gap}. Winding 1U1/1V1/1W1 "
            f"changed by {values} °C. {range_text}",
            "Is one winding pulling away from the others, and should I reduce output now?",
            "Flag the widening spread and verify the live winding readings, cooling and protection state. "
            "All three readings rose. One rose faster, which warrants comparison with the unit's normal behaviour at similar load. "
            "The applicable limit and dwell time are needed to decide on a derate.",
        )
    if case == "windings_warm_with_load":
        return pair(
            "Did the windings warm together as load increased?",
            f"In an hour, 1U1/1V1/1W1 warmed by {values} °C, and the spread went from {gap}. {load}",
            "The generator is getting warmer as power rises. What do I watch?",
            "The common rise is consistent with warming under increased load. Check the live maximum "
            "and watch for continued heating at steady output or a widening spread. Confirm any action "
            "against the controller's temperature limits.",
        )
    if case == "windings_cool_as_load_falls":
        return pair(
            "How much did each winding cool over the hour of falling output?",
            f"1U1/1V1/1W1 changed by {values} °C over the hour. {load}",
            "Output is lower and temperatures are falling. Can I turn it back up?",
            "All three windings read cooler at the end of the hour, consistent with lower load. Confirm why "
            "output fell and check live temperatures and the restoration criteria before increasing load.",
        )
    if case == "stable_winding_temperature_offset":
        return pair(
            "Did the hottest-to-coolest spread hold steady, and how high was it?",
            f"The winding spread went from {gap}. Across the latest seven readings it ranged from "
            f"{n(min(spread(row, WINDINGS) for _, row in frame.tail(7).iterrows()))} to "
            f"{n(max(spread(row, WINDINGS) for _, row in frame.tail(7).iterrows()))} °C. {range_text}",
            "There's a visible temperature difference between the windings. How do we decide whether to raise a maintenance ticket?",
            "Compare the offset with this turbine's earlier readings at similar load, RPM and ambient "
            "conditions, and verify the sensor values. Widening under these conditions or a confirmed alarm "
            "would raise the priority.",
        )
    key = max(WINDINGS, key=lambda name: abs(last[name] - first[name]))
    name = {"U": "1U1", "V": "1V1", "Z": "1W1"}[key]
    return pair(
        "Which winding changed most over an hour, and how much did the other two change?",
        f"{name} moved the most by magnitude. Winding 1U1/1V1/1W1 changed by {values} °C, "
        f"and the winding spread went from {gap}. {range_text}",
        "One temperature is behaving differently. What should the operator check?",
        "Compare the small difference with this unit’s historical readings at similar load and check live protection "
        "status. If one reading repeatedly moves independently, have maintenance compare the "
        "winding sensor and its measurement chain before diagnosing a thermal fault.",
    )


def electrical(case, frame):
    first, last = frame.iloc[-7], frame.iloc[-1]
    volts = tuple("V" + phase for phase in PHASES)
    amps = tuple("I" + phase for phase in PHASES)
    amp_spread = n(spread(last, amps))
    volt_spread = n(spread(last, volts))
    if case == "steady_phase_current_spread":
        mean = last[list(amps)].mean()
        phase = max(PHASES, key=lambda key: abs(last["I" + key] - mean))
        return pair(
            "Which phase current differs most from the three-phase mean at the cutoff?",
            f"The phase current mean is {n(mean)} A. Phase {phase} is farthest from it at "
            f"{n(last['I' + phase] - mean)} A relative to the mean. Latest current and voltage spreads "
            f"are {amp_spread} A and {volt_spread} V.",
            "Do the phase readings warrant an electrical check on this turbine?",
            "The current spread persisted across the last seven intervals at comparable load. Check local "
            "metering and compare with the station meter or synchronized peers. Use protection data to determine urgency.",
        )
    if case == "phase_spreads_move_differently":
        return pair(
            "Did current spread and voltage spread increase together over the last hour?",
            f"Current spread rose from {n(spread(first, amps))} to {amp_spread} A. Voltage spread "
            f"went from {n(spread(first, volts))} to {volt_spread} V.",
            "The phase currents have separated. Does that point to a grid problem?",
            "Start by comparing the turbine measurements against independent metering or synchronized peer "
            "turbines. The local voltage spread changed little while current spread increased, leaving the location unresolved.",
        )
    if case == "current_spread_eases_during_power_change":
        recent = frame.tail(7)
        peaks = recent[list(amps)].max(axis=1) - recent[list(amps)].min(axis=1)
        return pair(
            "When did current spread peak and what does it read now?",
            f"Current spread reached {n(peaks.max())} A at {at(peaks.idxmax())} and is now "
            f"{amp_spread} A. Power changed from {change(first, last, 'P', 'kW')} over the hour.",
            "The current spread narrowed over an hour of changing power. What follow-up makes sense?",
            "The spread has narrowed. Review whether it recurs at steady output and check live electrical "
            "alarms. The meter averages alone leave any protection assessment open.",
        )
    if case == "grid_phase_voltage_spread_increases":
        return pair(
            "How did the voltage spread change, and what happened to the current spread?",
            f"The measured phase-voltage spread increased from {n(spread(first, volts))} to {volt_spread} V. "
            f"Current spread went from {n(spread(first, amps))} to {amp_spread} A.",
            "What should I ask the grid desk about the separating phase voltages?",
            "Ask whether the station meter and other turbines show the same phase pattern at the same time. "
            "Use time-aligned waveforms or protection records for an assessment against a grid criterion.",
        )
    return pair(
        "What phase-voltage and current spreads are present under high output?",
        f"At {n(last.P)} kW the latest voltage spread is {volt_spread} V and the current spread is "
        f"{amp_spread} A. The previous hour also had measured spreads below 5 V and 4 A.",
        "Is a sustained separation visible in the phase averages under load?",
        "Only small absolute spreads are visible in the ten-minute averages throughout this hour. Check independent "
        "metering if the controller flags an event or a short disturbance is suspected.",
    )


def telemetry(case, frame):
    last = frame.iloc[-1]
    missing = frame.isna().all(axis=1)
    temp_missing = frame[["A", *WINDINGS]].isna().all(axis=1)
    if case == "all_telemetry_missing_at_cutoff":
        valid = frame.index[~missing]
        return pair(
            "How old is the most recent reported interval for this turbine?",
            f"All 20 selected channels are missing at the cutoff. The most recent interval with "
            f"any selected reading ended at {at(valid[-1])}, "
            f"{int((frame.index[-1] - valid[-1]).total_seconds() / 60)} minutes before cutoff.",
            "Has the turbine stopped or have we lost visibility?",
            "The archive has lost visibility in the selected channels. Confirm the live turbine and "
            "communications state before classifying the unit as stopped or estimating lost production.",
        )
    if case == "reporting_resumes_after_gap":
        return pair(
            "Which operating readings returned at the cutoff and what was missing just before it?",
            f"All 20 selected readings were absent for the preceding interval. At cutoff, power "
            f"reports {n(last.P)} kW, wind {n(last.W)} m/s, generator speed {n(last.R)} rpm "
            f"and running time {n(last.ON)} seconds.",
            "Telemetry has returned. Can I clear the issue?",
            "Core operating readings are back in the latest interval. Confirm the stream stays current "
            "and reconcile the gap before closing the telemetry issue. Check plant logs if you need to "
            "reconstruct what the unit did while readings were missing.",
        )
    if case == "earlier_temperature_gaps":
        return pair(
            "Can you trace all the winding temperatures continuously over the last six hours?",
            f"Ambient and all three selected winding temperatures are absent in "
            f"{int(temp_missing.sum())} of 36 intervals. All four report again at cutoff.",
            "Can I use this window to establish when a thermal trend started?",
            "Use the latest reported temperatures for a spot comparison. Recover the missing "
            "intervals from the controller historian before estimating when heating started or how long it lasted.",
        )
    if case == "near_flat_wind_while_generating":
        recent = frame.tail(7)
        return pair(
            "How much did the wind channel vary while power was moving?",
            f"Over seven readings wind ranged from {n(recent.W.min())} to {n(recent.W.max())} m/s. "
            f"Power ranged from {n(recent.P.min())} to {n(recent.P.max())} kW.",
            "The wind channel looks nearly flat while power moves. What do I check?",
            "Compare the wind channel with an independent reading and check the power reference "
            "and controller state. A narrow reported wind range alone leaves both the feed freshness "
            "and the reason for the power change unresolved.",
        )
    return pair(
        "Are the zero running-time values measured or missing?",
        f"The last seven running-time readings are valid measured zeros. Generator speed stayed below "
        f"100 rpm, latest active power is {n(last.P)} kW and latest wind speed is {n(last.W)} m/s.",
        "Is this a telemetry outage or is the turbine idle?",
        "The selected feeds are reporting, and the observed speed, power and running time are "
        "consistent with idle operation. Check the latest controller state for the reason it is idle.",
    )


def maintenance(case, frame):
    first, last = frame.iloc[-7], frame.iloc[-1]
    if case == "recurring_error_flag_after_generation":
        flagged = int(frame.ERR.gt(0).sum())
        mask = frame.ERR.gt(0)
        episodes = int((mask & ~mask.shift(fill_value=False)).sum())
        return pair(
            "How many error-active intervals and separated flag runs are visible before handover?",
            f"The error flag was active for some time in {flagged} ten-minute intervals across "
            f"{episodes} separate runs of consecutive flagged intervals. Latest speed is "
            f"{n(last.R)} rpm and power is {n(last.P)} kW.",
            "What should go into a maintenance ticket for this turbine?",
            "Log the error-active intervals and the loss of generation. Retrieve the controller event "
            "sequence and fault codes to establish whether the flags reflect one persistent condition or "
            "several events. Verify the start permits before any restart decision.",
        )
    if case == "repeated_thermal_separation":
        differences = frame[list(WINDINGS)].max(axis=1) - frame[list(WINDINGS)].min(
            axis=1
        )
        return pair(
            "How often did the winding spread exceed the 1.5 °C annotation screening value?",
            f"The spread exceeded 1.5 °C in {int(differences.gt(1.5).sum())} of 36 intervals and "
            f"reads {n(differences.iloc[-1])} °C at cutoff. The 1.5 °C value is a screening comparison.",
            "What maintenance follow-up is justified for this winding pattern?",
            "Trend the spread against the same unit at similar recent loading, speed and ambient temperature. "
            "Ask maintenance to check the winding sensors and cooling indications if the spread persists "
            "or widens. Use actual controller limits for decisions about output.",
        )
    if case == "current_spread_under_steady_load":
        amps = ("IR", "IS", "IT")
        recent = frame.tail(7)
        minimum = min(spread(row, amps) for _, row in recent.iterrows())
        return pair(
            "Did phase-current spread persist while active and reactive output were relatively stable?",
            f"Phase-current spread was at least {n(minimum)} A over seven readings. "
            f"Active power ranged from {n(recent.P.min())} to {n(recent.P.max())} kW and "
            f"reactive power ranged from {n(recent.Q.min())} to {n(recent.Q.max())} kvar.",
            "What should I ask the electrical maintenance team to verify?",
            "Request a time-aligned controller and independent meter comparison, with phase identifiers, "
            "active and reactive load, and alarm history. This can distinguish an instrumentation issue "
            "from a persistent electrical deviation that merits further investigation.",
        )
    if case == "live_operation_check_telemetry":
        count = int(frame[["A", *WINDINGS]].isna().all(axis=1).sum())
        return pair(
            "How much thermal coverage is missing while the latest operating channels are present?",
            f"All four selected temperature values are absent in {count} of 36 intervals. "
            f"The latest power is {n(last.P)} kW with {n(last.ON)} seconds in operation.",
            "Who should investigate first, and what evidence should go in the ticket?",
            "Ask controls or SCADA maintenance to reconcile the missing thermal intervals with the "
            "live controller. Include the affected tags and the power and running-time readings. "
            "Route to the generator team if live temperatures or protection indications corroborate a thermal issue.",
        )
    hottest = max(last[key] for key in WINDINGS)
    return pair(
        "How do output and the hottest winding compare between an hour ago and the interval showing turbine error?",
        f"One hour apart, power went from {change(first, last, 'P', 'kW')} and the hottest reported "
        f"winding moved from {n(max(first[key] for key in WINDINGS))} to {n(hottest)} °C. "
        f"The latest interval recorded {n(last.ERR)} seconds with turbine error active.",
        "The turbine is out of production. What thermal evidence should I hand over to maintenance?",
        "Include the winding trends, the output drop and the error-active interval. Ask for the "
        "first-out alarm, live temperatures and cooling status before linking the stoppage to temperature "
        "or authorizing a restart.",
    )


def scenarios(family, renderer, labels):
    splits = ("train", "train", "train", "validation", "stress_test")
    return {
        label: (family, split, lambda frame, case=label: renderer(case, frame))
        for label, split in zip(labels, splits, strict=True)
    }


ANSWERS = {
    **scenarios(
        "power_ramps",
        power,
        (
            "wind_associated_ramp",
            "falling_power_with_wind",
            "reference_reduction_and_power_drop",
            "below_rated_reference",
            "rated_pitch_response",
        ),
    ),
    **scenarios(
        "repeated_starts",
        operation,
        (
            "multiple_start_stop_transitions",
            "ready_without_generation",
            "stop_with_error_flag",
            "generation_after_repeated_starts",
            "sustained_generation_after_idle",
        ),
    ),
    **scenarios(
        "winding_temperature",
        temperature,
        (
            "winding_spread_widens",
            "stable_winding_temperature_offset",
            "windings_cool_as_load_falls",
            "one_winding_moves_more",
            "windings_warm_with_load",
        ),
    ),
    **scenarios(
        "electrical_phase_imbalance",
        electrical,
        (
            "steady_phase_current_spread",
            "phase_spreads_move_differently",
            "current_spread_eases_during_power_change",
            "grid_phase_voltage_spread_increases",
            "low_phase_spread_at_load",
        ),
    ),
    **scenarios(
        "stale_telemetry",
        telemetry,
        (
            "all_telemetry_missing_at_cutoff",
            "reporting_resumes_after_gap",
            "earlier_temperature_gaps",
            "near_flat_wind_while_generating",
            "fresh_zero_running_time",
        ),
    ),
    **scenarios(
        "maintenance_triage",
        maintenance,
        (
            "recurring_error_flag_after_generation",
            "current_spread_under_steady_load",
            "thermal_handover_after_output_loss",
            "repeated_thermal_separation",
            "live_operation_check_telemetry",
        ),
    ),
}
