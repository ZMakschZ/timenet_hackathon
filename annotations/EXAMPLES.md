# Annotated Hill of Towie SCADA windows

Each case uses a real six-hour window. Cutoffs are UTC. The [dataset notes](README.md) explain the shared inputs, sampling rules and limits.

## Power ramps

### Wind associated ramp

T11 · 2024-08-12 01:00 UTC · train

**Reasoning question.** What was the average power ramp rate over the last hour and what happened to wind?

Power rose from 147 to 2,014.51 kW, averaging 31.13 kW/min between the two reported means. Wind rose from 4.58 to 12.26 m/s with less than 10 kW variation in the reference.

**Operator question.** What drove the increase in output, and do I need to intervene?

Power rose from 147 to 2,014.51 kW, averaging 31.13 kW/min between the two reported means. Wind rose from 4.58 to 12.26 m/s with less than 10 kW variation in the reference. The rise is consistent with increasing wind under a near-constant reference. If the wind keeps rising, watch how power and pitch behave as the turbine approaches its rated output.

### Falling power with wind

T20 · 2024-08-12 14:00 UTC · train

**Reasoning question.** Did the reference fall with power, and how did wind change?

Over the last hour power fell from 2,227.09 to 231.4 kW and wind fell from 12.81 to 5.19 m/s. The reference varied by less than 10 kW and the turbine recorded 600 seconds in operation in every interval.

**Operator question.** Should I treat the falling output as a turbine problem or adjust the dispatch forecast?

Over the last hour power fell from 2,227.09 to 231.4 kW and wind fell from 12.81 to 5.19 m/s. The reference varied by less than 10 kW and the turbine recorded 600 seconds in operation in every interval. The decline is consistent with weakening wind during continued operation. Check whether the drop follows the site wind trend. If output remains depressed after wind recovers, review the live controller state and active limits.

### Reference reduction and power drop

T21 · 2024-09-10 22:00 UTC · train

**Reasoning question.** How much did the reference and measured power fall, and how much did wind vary?

Over one hour the reference fell from 2,300 to 4 kW and measured power fell from 2,249.23 to 3.17 kW. Wind varied by 1.82 m/s.

**Operator question.** Output fell while the wind changed little. What should I check first?

Over one hour the reference fell from 2,300 to 4 kW and measured power fell from 2,249.23 to 3.17 kW. Wind varied by 1.82 m/s. The observed endpoints are consistent with a lower power request. Check the dispatch and limit history to learn why the reference changed. The interval means and interval-end references leave the exact sequence within each interval unresolved.

### Below rated reference

T18 · 2026-03-29 09:00 UTC · stress test

**Reasoning question.** How did rising wind affect tracking of the below-rated reference?

Wind rose from 9.33 to 13.07 m/s in an hour. Latest power was 430.6 kW against a 431 kW reference, a 0.4 kW difference. The reference changed by less than 10 kW across the hour.

**Operator question.** Why isn't this turbine picking up more power as the wind increases?

Wind rose from 9.33 to 13.07 m/s in an hour. Latest power was 430.6 kW against a 431 kW reference, a 0.4 kW difference. The reference changed by less than 10 kW across the hour. The controller reference is below rated power and the output is close to it. Check the dispatch request and active limits before treating this as underperformance or raising the request.

### Rated pitch response

T13 · 2026-03-23 22:00 UTC · stress test

**Reasoning question.** How did pitch and power respond to rising wind over the last hour?

Wind rose from 16.66 to 22.62 m/s and blade A pitch changed from 9.99 to 20.46 degrees. Power stayed between 2,277.28 and 2,301.08 kW around the 2,300 kW rating.

**Operator question.** Wind is picking up. Why is output barely moving, and should I adjust anything?

Wind rose from 16.66 to 22.62 m/s and blade A pitch changed from 9.99 to 20.46 degrees. Power stayed between 2,277.28 and 2,301.08 kW around the 2,300 kW rating. The readings are consistent with regulating near rated power. Check for a sustained departure from the reference or an active controller alarm before revisiting the setting.

## Repeated starts and availability

### Multiple start stop transitions

T05 · 2024-09-02 11:00 UTC · train

**Reasoning question.** How many start-like and stop-like transitions can you identify in this window?

The six-hour view contains 5 start-like and 6 stop-like transitions across a screening condition of over 300 rpm, over 50 kW and positive time in operation. The latest power is -10.62 kW and speed is 0.65 rpm.

**Operator question.** Is this turbine ready to carry load, or are we still cycling?

The six-hour view contains 5 start-like and 6 stop-like transitions across a screening condition of over 300 rpm, over 50 kW and positive time in operation. The latest power is -10.62 kW and speed is 0.65 rpm. Output is low at cutoff after multiple crossings of the generation screen. Check the controller start permits and event sequence to distinguish physical restarts from operation near the screening threshold.

### Ready without generation

T03 · 2024-07-08 08:00 UTC · train

**Reasoning question.** Does the ready flag coincide with actual generation?

Each of the last seven intervals shows 600 seconds ready and zero seconds in operation. The latest generator speed is 27.55 rpm and active power is -10.25 kW.

**Operator question.** The unit says ready. Should I schedule its output?

Each of the last seven intervals shows 600 seconds ready and zero seconds in operation. The latest generator speed is 27.55 rpm and active power is -10.25 kW. There is no active generation in the latest readings. Check the start request, start permissions and live availability before scheduling the output.

### Stop with error flag

T13 · 2024-08-09 09:00 UTC · train

**Reasoning question.** What happened to power, generator speed and the turbine-error flag in the latest hour?

Power fell from 2,288.5 to -1.58 kW and speed fell from 1,552.03 to 98.83 rpm. The latest interval had 600 seconds with turbine error active and zero seconds in operation.

**Operator question.** The unit has stopped producing and the error flag is up. What should I do before restart?

Power fell from 2,288.5 to -1.58 kW and speed fell from 1,552.03 to 98.83 rpm. The latest interval had 600 seconds with turbine error active and zero seconds in operation. Check the live controller's first-out alarm and start permissions. The error flag establishes that an error was active during the reporting interval. The component and restart eligibility remain unresolved.

### Generation after repeated starts

T17 · 2024-10-20 04:00 UTC · validation

**Reasoning question.** What changed between the earlier crossings of the generation screen and the latest hour?

The generation screen was entered 7 times and exited 7 times in the six-hour view. The last seven intervals each show 600 seconds in operation and power of at least 303.81 kW.

**Operator question.** The unit looks to be generating steadily now. Can I count on it?

The generation screen was entered 7 times and exited 7 times in the six-hour view. The last seven intervals each show 600 seconds in operation and power of at least 303.81 kW. Generation is sustained in the latest readings. Confirm the earlier transitions with the event log and check live alarm clearance and site restart acceptance criteria before declaring normal availability. Watch for another RPM or power drop.

### Sustained generation after idle

T15 · 2026-03-25 04:00 UTC · stress test

**Reasoning question.** What evidence distinguishes the latest generation from the idle reading an hour ago?

Generator speed went from 56.76 to 1,214.32 rpm and power went from -7.65 to 685.32 kW. Each of the last three intervals recorded 600 seconds in operation and more than 100 kW.

**Operator question.** What is the unit doing after the idle period, and what should the next shift watch?

Generator speed went from 56.76 to 1,214.32 rpm and power went from -7.65 to 685.32 kW. Each of the last three intervals recorded 600 seconds in operation and more than 100 kW. The unit is generating again. Watch for a return to idle and confirm it meets the site's normal availability criteria before relying on the recovery.

## Generator winding temperature

### Winding spread widens

T12 · 2024-12-04 19:00 UTC · validation

**Reasoning question.** How much did winding spread widen under comparable operating conditions?

Over the last hour the winding spread went from 0.82 to 1.37 °C. Winding 1U1/1V1/1W1 changed by 2.63/2.08/2.33 °C. Over those readings power spanned 2,287.27–2,299.28 kW, speed varied by 1.13 rpm and ambient varied by 0 °C.

**Operator question.** Is one winding pulling away from the others, and should I reduce output now?

Over the last hour the winding spread went from 0.82 to 1.37 °C. Winding 1U1/1V1/1W1 changed by 2.63/2.08/2.33 °C. Over those readings power spanned 2,287.27–2,299.28 kW, speed varied by 1.13 rpm and ambient varied by 0 °C. Flag the widening spread and verify the live winding readings, cooling and protection state. All three readings rose. One rose faster, which warrants comparison with the unit's normal behaviour at similar load. The applicable limit and dwell time are needed to decide on a derate.

### Stable winding temperature offset

T03 · 2024-08-12 09:00 UTC · train

**Reasoning question.** Did the hottest-to-coolest spread hold steady, and how high was it?

The winding spread went from 1.58 to 1.73 °C. Across the latest seven readings it ranged from 1.49 to 1.73 °C. Over those readings power spanned 2,287.07–2,296.8 kW, speed varied by 1.3 rpm and ambient varied by 0.77 °C.

**Operator question.** There's a visible temperature difference between the windings. How do we decide whether to raise a maintenance ticket?

The winding spread went from 1.58 to 1.73 °C. Across the latest seven readings it ranged from 1.49 to 1.73 °C. Over those readings power spanned 2,287.07–2,296.8 kW, speed varied by 1.3 rpm and ambient varied by 0.77 °C. Compare the offset with this turbine's earlier readings at similar load, RPM and ambient conditions, and verify the sensor values. Widening under these conditions or a confirmed alarm would raise the priority.

### Windings cool as load falls

T14 · 2024-08-09 04:00 UTC · train

**Reasoning question.** How much did each winding cool over the hour of falling output?

1U1/1V1/1W1 changed by -42.87/-43.11/-42.61 °C over the hour. Power went from 2,299.23 to 2.75 kW.

**Operator question.** Output is lower and temperatures are falling. Can I turn it back up?

1U1/1V1/1W1 changed by -42.87/-43.11/-42.61 °C over the hour. Power went from 2,299.23 to 2.75 kW. All three windings read cooler at the end of the hour, consistent with lower load. Confirm why output fell and check live temperatures and the restoration criteria before increasing load.

### One winding moves more

T08 · 2024-12-24 13:00 UTC · validation

**Reasoning question.** Which winding changed most over an hour, and how much did the other two change?

1U1 moved the most by magnitude. Winding 1U1/1V1/1W1 changed by 0.87/0.62/0.57 °C, and the winding spread went from 1.32 to 1.62 °C. Over those readings power spanned 2,271.95–2,299.92 kW, speed varied by 1.76 rpm and ambient varied by 0.97 °C.

**Operator question.** One temperature is behaving differently. What should the operator check?

1U1 moved the most by magnitude. Winding 1U1/1V1/1W1 changed by 0.87/0.62/0.57 °C, and the winding spread went from 1.32 to 1.62 °C. Over those readings power spanned 2,271.95–2,299.92 kW, speed varied by 1.76 rpm and ambient varied by 0.97 °C. Compare the small difference with this unit’s historical readings at similar load and check live protection status. If one reading repeatedly moves independently, have maintenance compare the winding sensor and its measurement chain before diagnosing a thermal fault.

### Windings warm with load

T14 · 2026-03-11 20:00 UTC · stress test

**Reasoning question.** Did the windings warm together as load increased?

In an hour, 1U1/1V1/1W1 warmed by 43.83/43.9/43.56 °C, and the spread went from 0.51 to 0.19 °C. Power went from 11.9 to 2,300.41 kW.

**Operator question.** The generator is getting warmer as power rises. What do I watch?

In an hour, 1U1/1V1/1W1 warmed by 43.83/43.9/43.56 °C, and the spread went from 0.51 to 0.19 °C. Power went from 11.9 to 2,300.41 kW. The common rise is consistent with warming under increased load. Check the live maximum and watch for continued heating at steady output or a widening spread. Confirm any action against the controller's temperature limits.

## Electrical phase imbalance

### Steady phase current spread

T20 · 2024-08-13 19:00 UTC · train

**Reasoning question.** Which phase current differs most from the three-phase mean at the cutoff?

The phase current mean is 1,900.98 A. Phase R is farthest from it at -6.57 A relative to the mean. Latest current and voltage spreads are 13.07 A and 2.03 V.

**Operator question.** Do the phase readings warrant an electrical check on this turbine?

The phase current mean is 1,900.98 A. Phase R is farthest from it at -6.57 A relative to the mean. Latest current and voltage spreads are 13.07 A and 2.03 V. The current spread persisted across the last seven intervals at comparable load. Check local metering and compare with the station meter or synchronized peers. Use protection data to determine urgency.

### Phase spreads move differently

T20 · 2024-09-11 20:00 UTC · train

**Reasoning question.** Did current spread and voltage spread increase together over the last hour?

Current spread rose from 5.31 to 12.74 A. Voltage spread went from 2.83 to 2.92 V.

**Operator question.** The phase currents have separated. Does that point to a grid problem?

Current spread rose from 5.31 to 12.74 A. Voltage spread went from 2.83 to 2.92 V. Start by comparing the turbine measurements against independent metering or synchronized peer turbines. The local voltage spread changed little while current spread increased, leaving the location unresolved.

### Current spread eases during power change

T20 · 2024-09-11 10:00 UTC · train

**Reasoning question.** When did current spread peak and what does it read now?

Current spread reached 11.57 A at 09:40 UTC and is now 4.71 A. Power changed from 1,874.18 to 496.82 kW over the hour.

**Operator question.** The current spread narrowed over an hour of changing power. What follow-up makes sense?

Current spread reached 11.57 A at 09:40 UTC and is now 4.71 A. Power changed from 1,874.18 to 496.82 kW over the hour. The spread has narrowed. Review whether it recurs at steady output and check live electrical alarms. The meter averages alone leave any protection assessment open.

### Grid phase voltage spread increases

T09 · 2024-11-19 14:00 UTC · validation

**Reasoning question.** How did the voltage spread change, and what happened to the current spread?

The measured phase-voltage spread increased from 2.58 to 4 V. Current spread went from 3.6 to 6.38 A.

**Operator question.** What should I ask the grid desk about the separating phase voltages?

The measured phase-voltage spread increased from 2.58 to 4 V. Current spread went from 3.6 to 6.38 A. Ask whether the station meter and other turbines show the same phase pattern at the same time. Use time-aligned waveforms or protection records for an assessment against a grid criterion.

### Low phase spread at load

T14 · 2026-03-02 23:00 UTC · stress test

**Reasoning question.** What phase-voltage and current spreads are present under high output?

At 2,301.15 kW the latest voltage spread is 1.45 V and the current spread is 1.29 A. The previous hour also had measured spreads below 5 V and 4 A.

**Operator question.** Is a sustained separation visible in the phase averages under load?

At 2,301.15 kW the latest voltage spread is 1.45 V and the current spread is 1.29 A. The previous hour also had measured spreads below 5 V and 4 A. Only small absolute spreads are visible in the ten-minute averages throughout this hour. Check independent metering if the controller flags an event or a short disturbance is suspected.

## Stale telemetry and missingness

### All telemetry missing at cutoff

T15 · 2024-08-19 15:00 UTC · train

**Reasoning question.** How old is the most recent reported interval for this turbine?

All 20 selected channels are missing at the cutoff. The most recent interval with any selected reading ended at 10:00 UTC, 300 minutes before cutoff.

**Operator question.** Has the turbine stopped or have we lost visibility?

All 20 selected channels are missing at the cutoff. The most recent interval with any selected reading ended at 10:00 UTC, 300 minutes before cutoff. The archive has lost visibility in the selected channels. Confirm the live turbine and communications state before classifying the unit as stopped or estimating lost production.

### Reporting resumes after gap

T01 · 2024-08-06 11:00 UTC · train

**Reasoning question.** Which operating readings returned at the cutoff and what was missing just before it?

All 20 selected readings were absent for the preceding interval. At cutoff, power reports -4.48 kW, wind 1.78 m/s, generator speed 11.41 rpm and running time 0 seconds.

**Operator question.** Telemetry has returned. Can I clear the issue?

All 20 selected readings were absent for the preceding interval. At cutoff, power reports -4.48 kW, wind 1.78 m/s, generator speed 11.41 rpm and running time 0 seconds. Core operating readings are back in the latest interval. Confirm the stream stays current and reconcile the gap before closing the telemetry issue. Check plant logs if you need to reconstruct what the unit did while readings were missing.

### Earlier temperature gaps

T04 · 2024-08-13 13:00 UTC · train

**Reasoning question.** Can you trace all the winding temperatures continuously over the last six hours?

Ambient and all three selected winding temperatures are absent in 24 of 36 intervals. All four report again at cutoff.

**Operator question.** Can I use this window to establish when a thermal trend started?

Ambient and all three selected winding temperatures are absent in 24 of 36 intervals. All four report again at cutoff. Use the latest reported temperatures for a spot comparison. Recover the missing intervals from the controller historian before estimating when heating started or how long it lasted.

### Near flat wind while generating

T21 · 2024-11-19 08:00 UTC · validation

**Reasoning question.** How much did the wind channel vary while power was moving?

Over seven readings wind ranged from 8.1 to 8.33 m/s. Power ranged from 904.49 to 1,051.08 kW.

**Operator question.** The wind channel looks nearly flat while power moves. What do I check?

Over seven readings wind ranged from 8.1 to 8.33 m/s. Power ranged from 904.49 to 1,051.08 kW. Compare the wind channel with an independent reading and check the power reference and controller state. A narrow reported wind range alone leaves both the feed freshness and the reason for the power change unresolved.

### Fresh zero running time

T14 · 2026-03-02 04:00 UTC · stress test

**Reasoning question.** Are the zero running-time values measured or missing?

The last seven running-time readings are valid measured zeros. Generator speed stayed below 100 rpm, latest active power is -0.6 kW and latest wind speed is 11.08 m/s.

**Operator question.** Is this a telemetry outage or is the turbine idle?

The last seven running-time readings are valid measured zeros. Generator speed stayed below 100 rpm, latest active power is -0.6 kW and latest wind speed is 11.08 m/s. The selected feeds are reporting, and the observed speed, power and running time are consistent with idle operation. Check the latest controller state for the reason it is idle.

## Maintenance triage

### Recurring error flag after generation

T16 · 2024-08-11 14:00 UTC · train

**Reasoning question.** How many error-active intervals and separated flag runs are visible before handover?

The error flag was active for some time in 6 ten-minute intervals across 2 separate runs of consecutive flagged intervals. Latest speed is 14.02 rpm and power is -10.34 kW.

**Operator question.** What should go into a maintenance ticket for this turbine?

The error flag was active for some time in 6 ten-minute intervals across 2 separate runs of consecutive flagged intervals. Latest speed is 14.02 rpm and power is -10.34 kW. Log the error-active intervals and the loss of generation. Retrieve the controller event sequence and fault codes to establish whether the flags reflect one persistent condition or several events. Verify the start permits before any restart decision.

### Current spread under steady load

T16 · 2024-08-13 19:00 UTC · train

**Reasoning question.** Did phase-current spread persist while active and reactive output were relatively stable?

Phase-current spread was at least 9.33 A over seven readings. Active power ranged from 1,933.35 to 2,114.61 kW and reactive power ranged from 62.88 to 95.94 kvar.

**Operator question.** What should I ask the electrical maintenance team to verify?

Phase-current spread was at least 9.33 A over seven readings. Active power ranged from 1,933.35 to 2,114.61 kW and reactive power ranged from 62.88 to 95.94 kvar. Request a time-aligned controller and independent meter comparison, with phase identifiers, active and reactive load, and alarm history. This can distinguish an instrumentation issue from a persistent electrical deviation that merits further investigation.

### Thermal handover after output loss

T15 · 2024-08-13 20:00 UTC · train

**Reasoning question.** How do output and the hottest winding compare between an hour ago and the interval showing turbine error?

One hour apart, power went from 2,279.24 to -1.24 kW and the hottest reported winding moved from 97.34 to 81.96 °C. The latest interval recorded 600 seconds with turbine error active.

**Operator question.** The turbine is out of production. What thermal evidence should I hand over to maintenance?

One hour apart, power went from 2,279.24 to -1.24 kW and the hottest reported winding moved from 97.34 to 81.96 °C. The latest interval recorded 600 seconds with turbine error active. Include the winding trends, the output drop and the error-active interval. Ask for the first-out alarm, live temperatures and cooling status before linking the stoppage to temperature or authorizing a restart.

### Repeated thermal separation

T03 · 2024-10-24 03:00 UTC · validation

**Reasoning question.** How often did the winding spread exceed the 1.5 °C annotation screening value?

The spread exceeded 1.5 °C in 20 of 36 intervals and reads 1.79 °C at cutoff. The 1.5 °C value is a screening comparison.

**Operator question.** What maintenance follow-up is justified for this winding pattern?

The spread exceeded 1.5 °C in 20 of 36 intervals and reads 1.79 °C at cutoff. The 1.5 °C value is a screening comparison. Trend the spread against the same unit at similar recent loading, speed and ambient temperature. Ask maintenance to check the winding sensors and cooling indications if the spread persists or widens. Use actual controller limits for decisions about output.

### Live operation check telemetry

T11 · 2026-03-02 08:00 UTC · stress test

**Reasoning question.** How much thermal coverage is missing while the latest operating channels are present?

All four selected temperature values are absent in 35 of 36 intervals. The latest power is 697.26 kW with 600 seconds in operation.

**Operator question.** Who should investigate first, and what evidence should go in the ticket?

All four selected temperature values are absent in 35 of 36 intervals. The latest power is 697.26 kW with 600 seconds in operation. Ask controls or SCADA maintenance to reconcile the missing thermal intervals with the live controller. Include the affected tags and the power and running-time readings. Route to the generator team if live temperatures or protection indications corroborate a thermal issue.
