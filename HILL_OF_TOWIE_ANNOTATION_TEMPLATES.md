# Hill of Towie Operator Annotation Templates

The [observed-window pilot](annotations/EXAMPLES.md) applies these templates to 30 real SCADA windows. The [full archive and TimeNet export](annotations/README.md) apply all 30 scenario screens across 124 source months.

An operator can ask, **“What is this turbine undergoing right now, what could happen next, and what should I do?”** The answer should explain how the measurements fit together, give the action supported by that evidence, and say what would change the recommendation.

The catalogue has five **Chain-of-thought reasoning** templates and five **Chain-of-thought QA** templates per family. Reasoning answers show checkable measurements, calculations, and event order. QA answers use those findings to explain the operating state and guide a decision. Both can be represented by scoped TimeF `AnswerTask` examples. A brief supporting comparison can go in `rationale` if the training adapter reads it. [TimeF task model](https://docs.timenet.ai/data-model/tasks.html)

## Shared Flamingo input

Use one input schema for both annotation stages, validation, and deployment. The OpenTSLM Flamingo
implementation pads the time dimension and stacks examples in a batch, so examples in a batch need
the same number of series with the stock batching code. It does not require a globally fixed set
of SCADA fields. For this project, fixing the fields also gives each series position and description
a consistent meaning across questions. [OpenTSLM Flamingo batching code](https://github.com/OpenTSLM/OpenTSLM/blob/main/src/opentslm/model/llm/OpenTSLMFlamingo.py#L137-L175)

- **History.** Start with one hourly cutoff per example and a six-hour lookback, with 36 ten-minute
  intervals per series. Every question gets the same ordered series and stable descriptions with
  units. Use only readings available by the cutoff.
- **Candidate channel order.** Active power, active wind speed, generator RPM, power reference,
  main-shaft RPM, seconds running, seconds ready, seconds in turbine error, blade A pitch, ambient
  temperature, winding temperatures 1U1/1V1/1W1, grid voltages R/S/T, grid currents R/S/T, and
  reactive power. This is the proposed enrichment schema for all six families.
- **Missing readings.** Keep the same slots for every example. A missing sample must remain
  distinguishable from a measured zero, especially for stale telemetry. The stock Flamingo code
  uses zeros for length padding. Test a separate, fixed per-channel validity representation in the
  training adapter with both missing samples and valid zeros before using those templates.
- **Text context.** Use the same named fields for turbine ID, cutoff, rated capacity, alarm and
  controller status, applicable operating rules, matched baseline, and cross-turbine or independent
  evidence available to the operator. An unavailable field carries an explicit `unknown` value.
  Add a field to the schema once if it is available consistently at inference. The question and
  the observed values change across examples.

The current TimeF forecasting release carries active power, active wind speed, and generator RPM.
The additional channels and context above describe the proposed annotation data and training
adapter. The project should validate their historical coverage before fixing the final schema.

## How to populate an example

- The **Evidence required** column is a checklist for selecting or validating an example. It names
  the readings or context that must be available within the shared schema for the answer to be
  justified. It is a catalogue column for annotators. If the shared input lacks evidence for a
  particular conclusion, choose an uncertainty answer or leave that example out.
- Resolve numbers, timestamps, labels, and units from the input before generating the answer. Use positive durations, retain the sign of differences, and align compared channels to the same measurement intervals. Values from after the question cutoff can serve as targets in a future-outcome task.
- Generate a row only when its input conditions are met. If the data support several interpretations, use the uncertainty example and name the observation that would help decide. A blank, a missing timestamp, and a measured zero carry different meanings.
- Fit expected behaviour from training data and compare like operating regimes. Cite source fields in the evidence. A verified, applicable site rule or decoded protection instruction is required for an answer that recommends a specific operational intervention. Supply it with the example and preserve its measurement basis and action. The templates use placeholders and the two worked examples use invented readings. [Hill of Towie field definitions](https://zenodo.org/records/22662930/files/Hill_of_Towie_turbine_fields_description.csv?download=1) · [SCADA normal-behaviour modelling](https://wes.copernicus.org/articles/5/1375/2020/)

## Worked examples

The tables show only the readings discussed in each answer. Both samples use the shared input schema.

### Rapid power ramp following a reference change

**Input excerpt (illustrative)**

| Interval end UTC | Power mean | Power reference at interval end | Active wind speed mean | Seconds in operation |
| --- | ---: | ---: | ---: | ---: |
| 12:00 | 980 kW | 1,000 kW | 10.2 m/s | 600 |
| 12:10 | 995 kW | 1,500 kW | 10.3 m/s | 600 |
| 12:20 | 1,340 kW | 1,500 kW | 10.2 m/s | 600 |
| 12:30 | 1,490 kW | 1,500 kW | 10.3 m/s | 600 |

**Operator:** What is the turbine doing right now, and should I change anything?

**Answer:** Keep the 1,500 kW request. The reference first showed 1,500 kW at 12:10, then the interval mean rose from 995 to 1,490 kW over the next 20 minutes. Wind held between 10.2 and 10.3 m/s, and the turbine ran throughout. That pattern is consistent with the controller following the higher reference. At the latest interval the output is 10 kW below the request. Check that it continues to track while wind remains steady. If the gap opens again, review the active limits and operating state before raising the request. The ten-minute summaries establish the interval-level sequence.

### Winding temperature diverging under steady operation

**Input excerpt (illustrative)**

| Interval end UTC | Winding 1U1 | Winding 1V1 | Winding 1W1 | Ambient | Power | Generator speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14:00 | 68°C | 67°C | 67°C | 9°C | 1,840 kW | 1,520 rpm |
| 14:20 | 73°C | 68°C | 68°C | 9°C | 1,845 kW | 1,522 rpm |
| 14:40 | 79°C | 69°C | 68°C | 9°C | 1,842 kW | 1,521 rpm |

The example includes a load-adjusted expected range for 1U1 of 65–72°C and verified steady load over the preceding hour. The site action criterion and live alarm state are unavailable in this view. The table shows selected intervals.

**Operator:** What’s happening in the generator? Do I need to reduce output?

**Answer:** Flag the generator temperature trend to the control room now. Over 40 minutes, 1U1 rose 11°C, while 1V1 rose 2°C and 1W1 rose 1°C. Power, speed, and ambient temperature barely changed. The latest 1U1 reading is 7°C above its expected range, so the concern is localized heating or an anomalous temperature measurement. If the heating is real and continues, the margin to a protective limit may shrink. Confirm 1U1 on the live controller, check cooling and alarm status, and apply the site derate rule if its condition is met. This view supports an immediate check and escalation. The output decision depends on the verified limit and current protection state.

## Power ramps

Use active power mean `wtc_ActPower_mean`, active wind speed mean `wtc_AcWindSp_mean`, power reference at interval end `wtc_PowerRef_endvalue`, generator speed `wtc_GenRpm_mean`, and seconds in operation `wtc_ScInOper_timeon`. Use rated power from turbine metadata and measured blade pitch where needed. An interval mean and an end-of-interval signal support coarse ordering across intervals. [Hill of Towie data dictionary](https://zenodo.org/records/22662930/preview/Hill_of_Towie_data_dictionary.md?include_deleted=0) · [NREL active power controls](https://www.energy.gov/sites/default/files/2014/01/f6/Active%20Power%20Controls%20from%20Wind%20Power.pdf)

| Type | Question | Evidence required | Answer template |
| --- | --- | --- | --- |
| Chain-of-thought reasoning | How large and fast was the ramp? | Two power means and their interval-end timestamps. | Power moved from [A] to [B] kW over [N] minutes, an average change of [signed rate] kW/min between the reported means. |
| Chain-of-thought reasoning | Which change was visible first? | Aligned wind, reference, and power readings and a supplied rule for a material change. | The first material change in [signal] was visible in the interval ending [T1]. The power change was visible by [T2]. [State the interval order or report a tie.] |
| Chain-of-thought reasoning | Did the gap to the reference close? | Consecutive complete intervals of power and an unchanged reference after a step. | Power moved from [A] to [B] kW against a [R] kW reference. Its absolute gap changed from [X] to [Y] kW. |
| Chain-of-thought reasoning | How close to rated output was the turbine as wind increased? | Power, rated capacity, active wind, pitch and reference. | Power changed from [A]% to [B]% of rated capacity. Wind rose [X] m/s, blade pitch moved [Y]°, and the reference [observed change or fixed value]. |
| Chain-of-thought reasoning | Did running time change during the drop? | Complete power, RPM and running-time readings around a drop. | The reported mean power fell [X] kW and RPM fell [Y]. The running-time counter changed from [A] to [B] seconds per interval, with the first change recorded at [T]. |
| Chain-of-thought QA | What is the turbine doing now, and should I change anything? | Reference rises before the first visible output ramp. Wind is steady, running time stays at 600 s, and the gap to the new reference closes. | Keep the current request. The reference first reached [R] kW at [T], and output then rose from [A] to [B] kW with wind between [V1] and [V2] m/s and continuous running. That fits reference following. The latest gap is [X] kW. Watch for the gap to reopen while wind stays steady. |
| Chain-of-thought QA | Can I request more power as the wind picks up? | Power stays near rated, reference stays at rated, and measured pitch changes as wind rises. | Keep the current mode. Wind increased by [X] m/s, but output stayed around [P] kW against a [R] kW rated reference. Pitch moved by [Y]°, consistent with the turbine regulating near rated power. Use the reference and available capacity when assessing any higher request. |
| Chain-of-thought QA | Can I count on the extra output for dispatch? | Below-rated output rises with wind, reference is unchanged and above observed output, and running time stays at 600 s. | Treat the extra [X] kW as wind-dependent. Wind rose [V] m/s while the reference stayed at [R] kW and running continued. The ramp is consistent with more wind being available, and it could reverse if wind falls. Confirm sustained availability before counting on that output. |
| Chain-of-thought QA | What is holding the turbine back right now? | Output stays below a steady reference for several intervals, with wind in a comparable historical band and the turbine running. Current limit reason is unavailable. | Investigate the tracking shortfall before increasing demand. Output has stayed [X] kW below [R] kW for [N] minutes while running and while measured wind stayed in [band]. The readings establish a sustained gap. Check the active limit, availability and alarm state to determine its cause. |
| Chain-of-thought QA | Power has dropped. Do I need to intervene immediately? | Wind, reference and running state all change in one reporting interval. The current protection state or higher-resolution record is unavailable. | Confirm current operating state before changing the command. Power fell [X] kW in the interval when wind, reference and running time also changed. The interval summaries leave the initiating change unresolved. Check the controller event and verified availability before relying on the previous output for dispatch. |

## Repeated starts

Use generator speed `wtc_GenRpm_mean`, main shaft speed `wtc_MainSRpm_mean`, active power, active wind, running time `wtc_ScInOper_timeon`, ready time `wtc_ScReToOp_timeon`, turbine error time `wtc_ScTurSto_timeon`, and decoded alarm times where available. Call RPM cycles **possible start attempts** unless a corresponding start event or controller state confirms them. A partial running interval can hide several events. [Hill of Towie data dictionary](https://zenodo.org/records/22662930/preview/Hill_of_Towie_data_dictionary.md?include_deleted=0)

| Type | Question | Evidence required | Answer template |
| --- | --- | --- | --- |
| Chain-of-thought reasoning | How many possible attempts can we distinguish? | Contiguous RPM and power windows, with an explicit rule for identifying a rise and return toward idle. | [N] separate RPM rises met the rule and returned toward idle. [N2] were followed by sustained power under the supplied definition. |
| Chain-of-thought reasoning | How far did the most recent attempt progress? | Both RPM traces, power and running seconds through the latest cycle. | Shaft and generator speed rose by [A] and [B] rpm. Power [response] and running time reached [S] seconds. [Report whether sustained generation became visible.] |
| Chain-of-thought reasoning | When was readiness lost relative to the speed drop? | Aligned ready-time and RPM readings. | Ready time first fell in the interval ending [T1]. RPM first fell in the interval ending [T2]. [State the interval order or report a tie.] |
| Chain-of-thought reasoning | How did wind differ between attempts that held and those that did not? | Matched attempt windows, wind coverage and a supplied sustained-generation rule. | Wind ranged from [A] to [B] m/s before [N] sustained runs and from [C] to [D] before [M] interrupted runs. The ranges [describe observed overlap]. |
| Chain-of-thought reasoning | Which alarms overlapped the visible attempts? | Visible attempt intervals, decoded alarm intervals, and known alarm coverage. | [Decoded alarm] overlapped [N] of [M] visible attempts. Alarm coverage is [state] for the other attempts. |
| Chain-of-thought QA | What is happening with this turbine, and can dispatch count on it? | Several separate RPM rises and returns toward idle. The latest run has generated for less than the supplied acceptance criterion. | Treat it as provisionally available. After [N] interrupted accelerations, it has generated [P] kW for [M] minutes. That is short of the [criterion] for accepting a stable restart. Watch running time and whether RPM and power hold before updating dispatch. |
| Chain-of-thought QA | The turbine just stopped again. Should I request a restart? | Confirmed stop, decoded active inhibit or stopping alarm, and a matching restart instruction. | Keep the restart request on hold. RPM and power dropped at [T], and [alarm or inhibit] remains active. Under the supplied procedure [release condition] must be met first. Send the event time and alarm to the control room for clearance. |
| Chain-of-thought QA | Why does it keep spinning up and dropping back? | Visible acceleration cycles, wind near a supplied start condition then falling, ready flag maintained, and known alarm coverage. | The cycling is consistent with marginal wind. Wind briefly reached [X] m/s and fell to [Y] around the interrupted cycles while readiness remained [state]. Watch for sustained eligible wind and a stable run before counting the capacity. Recheck alarms if cycling continues in eligible wind. |
| Chain-of-thought QA | It shows ready. Can I count on it coming back soon? | Ready time stays at 600 s over multiple intervals, with low RPM and no sustained power. Start permission is unavailable. | Availability is still unconfirmed. It has shown ready for [N] minutes, but RPM stayed near [X] and power near [P]. It has yet to demonstrate sustained generation. Confirm the start request, permission and current alarms before scheduling its output. |
| Chain-of-thought QA | It has finally started. What should we watch now? | Sustained power, stable RPM, 600 seconds running per interval, and decoded status of the earlier alarm. | The unit is generating again at [P] kW and has held [RPM] for [N] minutes. The earlier [alarm] is [state]. Watch for the same speed drop or alarm recurrence, and use the supplied restart acceptance criteria before declaring normal availability. |

## Generator winding temperature imbalance

Use winding temperatures `wtc_Gen1U1Tm_mean`, `wtc_Gen1V1Tm_mean`, `wtc_Gen1W1Tm_mean`, ambient temperature `wtc_AmbieTmp_mean`, power, and generator speed. Where present, add cooling measurements and decoded temperature alarm context. Recent load and ambient conditions matter when comparing temperature trends. [SCADA field definitions](https://zenodo.org/records/22662930/files/Hill_of_Towie_turbine_fields_description.csv?download=1) · [SCADA normal-behaviour modelling](https://wes.copernicus.org/articles/5/1375/2020/)

| Type | Question | Evidence required | Answer template |
| --- | --- | --- | --- |
| Chain-of-thought reasoning | Which winding’s gap to its peers widened most? | Aligned start and end values for all three windings. | [Winding] went from [A] to [B]°C relative to the mean of the other two. Its signed gap changed by [C]°C over [N] minutes. |
| Chain-of-thought reasoning | Are the windings warming together or spreading apart? | All three winding temperatures at both ends of the same interval. | The three temperatures changed by [A], [B] and [C]°C. The mean changed by [M]°C and the hottest-to-coolest spread changed by [S]°C. |
| Chain-of-thought reasoning | How far outside its expected range is the winding? | Winding temperature and a supplied baseline matched for recent load, RPM and ambient temperature. | [Winding] reads [X]°C at [load and RPM] and [ambient]°C. The matched upper bound is [Y]°C, giving a signed difference of [X−Y]°C. |
| Chain-of-thought reasoning | Did the temperature recover after the load dropped? | Power and temperature through a load drop, plus a comparable expected cooldown curve if available. | Output fell from [A] to [B] kW at [T]. The winding changed by [X]°C over the next [N] minutes. [Compare with the supplied cooldown or report the measured change alone.] |
| Chain-of-thought reasoning | How much margin remains to the supplied action point? | A winding reading and an applicable rule that uses the same measurand and aggregation, plus any required dwell time. | The reading is [X]°C against [Y]°C, a signed remaining margin of [Y−X]°C. The reading has met the rule's temperature condition for [N] minutes against the required [M]. |
| Chain-of-thought QA | What’s happening to this unit and should I reduce output now? | Localized sustained heating above a load-adjusted baseline, plus a verified applicable derate rule whose trigger and duration are met. | Recommend the approved reduced-load mode now. [Winding] rose [X]°C in [N] minutes, is [Y]°C above its expected range at this load, and has met [verified trigger and dwell]. Continued heating could lead to a protective trip. After reducing load, watch the hottest winding and the spread. Escalate if temperatures keep rising under the approved procedure. |
| Chain-of-thought QA | Wind is rising and the generator is warming. What should I do? | Load rises with wind. All three windings warm together within their matched ranges. Verified action limits and current alarms are available and untriggered. | Continue at the current approved output. Power rose [A] kW and winding temperatures rose [X/Y/Z]°C together. The hottest winding is [H]°C within [expected range], with [M]°C to the verified action point. Reassess if one winding diverges, the margin erodes, or an alarm appears. |
| Chain-of-thought QA | What should the control room know about this temperature trend? | One winding diverges under steady recent load, RPM and ambient conditions. Site action limit and live alarm state are unavailable. | Flag the localized rise now. [Winding] warmed [X]°C while peers changed [Y/Z]°C at steady [P] kW, [RPM] and [A]°C ambient. The pattern could be localized heating or a faulty temperature reading. Confirm live temperature, cooling and alarm status, then use the applicable procedure to decide whether a load change is required. |
| Chain-of-thought QA | We lowered output. Can we restore it yet? | Actual load reduction, verified current reduced mode, a delayed cooldown relative to a matched baseline, and an applicable restoration rule whose criteria are unmet. | Hold the approved reduced output. Power fell from [A] to [B] kW, but [winding] remains [X]°C above the expected cooldown at [N] minutes and [recovery condition] is still unmet. Check cooling and confirm the sensor. Reassess when the specified recovery criteria are met. |
| Chain-of-thought QA | One temperature reading just jumped. What should I do now? | One winding jumps in a single interval, peers and operating conditions remain steady, and independent confirmation is unavailable. | Verify the reading promptly. [Winding] jumped [X]°C while peer temperatures, load and RPM stayed around [values]. This is an isolated indication that could reflect a local event or measurement issue. Check the live controller and protection status and follow any active protection instruction. |

## Electrical phase imbalance

Use grid-side phase voltages `wtc_VoltPhR_mean`, `wtc_VoltPhS_mean`, `wtc_VoltPhT_mean` and currents `wtc_AmpPhR_mean`, `wtc_AmpPhS_mean`, `wtc_AmpPhT_mean`, plus active and reactive power, grid frequency, and synchronized peer readings. These are turbine grid-side ten-minute averages. Use independent protection data for a protection-limit claim. [SCADA field definitions](https://zenodo.org/records/22662930/files/Hill_of_Towie_turbine_fields_description.csv?download=1)

| Type | Question | Evidence required | Answer template |
| --- | --- | --- | --- |
| Chain-of-thought reasoning | Which phase differs most from the phase mean? | Three simultaneous values for the same quantity and unit. | The three-phase mean is [M] [unit]. Phase [R/S/T] has the largest absolute deviation and is [signed difference] [unit] from that mean. |
| Chain-of-thought reasoning | Did voltage and current spread change together? | Complete three-phase voltage and current readings before and after the event. | Voltage spread moved from [A] to [B] V and current spread from [C] to [D] A. The first visible changes were in the intervals ending [T1/T2]. |
| Chain-of-thought reasoning | How long did the departure persist? | A supplied spread criterion and continuous complete samples through the event. | The spread first met the criterion at [T] and stayed there for [N] consecutive intervals. [State when it returned or whether the latest interval still meets it.] |
| Chain-of-thought reasoning | Where else was the same voltage change visible? | Synchronized phase voltages, aligned phase identities, and peer reporting coverage. | At [T], the corresponding phase voltage changed by [X] V on this turbine. [N] of [M] reporting peers showed the same direction of change. |
| Chain-of-thought reasoning | How unusual is the current spread at this load? | Three-phase current and a supplied baseline matched for active and reactive output. | At [P] kW and [Q] kvar, the spread is [X] A versus a comparable range of [A–B] A, based on [N] qualified historical intervals. |
| Chain-of-thought QA | Is this a local turbine issue or a plant-wide event? | Several synchronized turbines show a similar phase-voltage change. Independent grid-side protection evidence is unavailable. | Notify the grid desk of a likely shared electrical disturbance. Voltage on the corresponding phase moved [X] V at [T] on [N] of [M] reporting turbines. That pattern is consistent with a shared event. Ask the grid desk to confirm it with station metering and protection logs. |
| Chain-of-thought QA | What is happening electrically on this unit? | Persistent phase-current deviation at comparable active and reactive power. Local voltage and peer currents remain in their supplied ranges. | Flag this unit for electrical review. Current on phase [R/S/T] has differed by [X] A for [N] minutes at comparable output, while the local voltages and peer currents stayed within [ranges]. A sustained real imbalance can add heating. Check the unit’s metering and protection indications to locate the issue. |
| Chain-of-thought QA | The phases separated during the ramp. Do we need to act now? | Transient phase-current spread during an observed dispatch ramp, return to a matched normal range, and no active alarm in verified data. | Keep the present approved setting and monitor. Current spread reached [X] A during the ramp, then returned to [Y] A at steady output, within [range], with no active matching alarm. Recheck if it recurs under steady load or an alarm appears. |
| Chain-of-thought QA | The electrical alarm is active. Should output change now? | Decoded active protection alarm, relevant independent measurement and a matching verified rule that prescribes a load limit. | Apply the approved load limit through the control room now. [Protection measurement] is [X] and [decoded alarm] is active, meeting [site criterion]. Verify the controller acknowledges the limit and the protection indication responds, and escalate under the procedure if it persists. |
| Chain-of-thought QA | Can I confirm an electrical limit breach from these readings? | A sustained phase spread in ten-minute means. The applicable protection measurement or its configured limit is unavailable. | The SCADA shows a phase spread of [X] [unit] for [N] minutes. Whether a protection limit was breached remains unresolved. Check the relay measurement, time scale and event log before relying on the averages for an output decision. |

## Stale telemetry

Use `TimeStamp`, `StationId`, active power, active wind speed, generator speed, running time, and controller time `wtc_CurTime_endvalue` where available. Supply corroborating data at the level needed for the question. A farm export meter can corroborate aggregate export, while the individual turbine needs a unit-level reading. [Hill of Towie archive and timestamp conventions](https://zenodo.org/records/22662930)

| Type | Question | Evidence required | Answer template |
| --- | --- | --- | --- |
| Chain-of-thought reasoning | Which reporting intervals are missing? | An expected ten-minute grid, a window with the latest present sample, and the requested station and channel. | [N] expected intervals are missing between [T1] and [T2]. The latest present reading ended at [T0]. |
| Chain-of-thought reasoning | Which signals froze while other signals changed? | Several contiguous intervals of repeated values and at least one changing channel. | [Channels] repeated the same values for [N] intervals. [Other channel] changed by [X] [unit] during those intervals. |
| Chain-of-thought reasoning | How old is the latest reading at decision time? | Decision time, latest measurement timestamp and controller timestamps when available. | The latest measurement ended at [T], [N] minutes before the decision cutoff. The controller clock [reported behaviour or unavailable]. |
| Chain-of-thought reasoning | How broad is the reporting gap? | Expected coverage by turbine and synchronized rows from more than one table or peer. | [N] of [M] turbines lost rows in the same [time span]. [Second table or source] [observed coverage]. |
| Chain-of-thought reasoning | Do fresh measurements support a stationary state? | Fresh power, RPM and running-time readings plus recording timestamps. | Power was [P] kW, RPM [R], and running time [S] seconds across [N] fresh intervals. The combined readings are [consistent with stationary operation or do not establish it]. |
| Chain-of-thought QA | Can I use this unit’s output for dispatch right now? | Exact power repeats, changing corroborating unit signals, and no independent confirmation of unit power. | Treat its current power as unverified. The value stayed at [P] kW for [N] intervals while [unit signal] moved [X]. That makes the reported power suspect. Check the live unit controller or independent unit metering before counting that output. |
| Chain-of-thought QA | Did the turbines trip, or have we lost visibility? | Several stations lose rows simultaneously. No independent unit status is available. | We have lost SCADA visibility for [N] turbines since [T]. The gap leaves their generation state unknown. Check farm export for the plant total and the live controllers for individual status before making dispatch or restart decisions. |
| Chain-of-thought QA | Is the unit parked or are we seeing stale zeros? | Fresh, advancing timestamps, power and RPM measured at zero, and zero seconds running. | The unit appears parked. Power and RPM read zero across [N] fresh intervals and running time stayed at zero. Check the operating state, readiness and alarms to determine when it can return. |
| Chain-of-thought QA | Reporting is back. Can I trust it for dispatch yet? | Stream resumed. Consecutive fresh intervals are available, but the site’s recovery verification or an independent unit comparison is still missing. | Reporting resumed at [T] and timestamps have advanced for [N] intervals. Verify current unit output against the live controller or independent unit reading before restoring it as a dispatch input. Retain the earlier period as a telemetry gap. |
| Chain-of-thought QA | Wind appears frozen but power is changing. What can I still use? | Wind repeats across several fresh intervals, power changes and agrees with independent unit power. | Keep the corroborated power reading for status monitoring. Wind stayed at [V] m/s for [N] intervals while unit output moved [X] kW and agreed with [unit reference]. Flag the wind channel and use a verified alternate source for wind-dependent decisions until it is checked. |

## Maintenance triage

Maintenance questions use the **same six-hour input and text context** as both training stages.
SCADA and decoded alarms can support a maintenance handover, a proposed inspection priority, and
checks that would narrow down the cause. The public pack includes alarm intervals and a separate
shutdown-duration summary that ends in August 2024. The shutdown summary documents downtime.
Confirmed component failures, completed repairs, and remaining useful life need linked inspection
findings, work orders, or other verified outcome evidence. [Hill of Towie release](https://zenodo.org/records/22662930)
· [SCADA condition monitoring study](https://wes.copernicus.org/articles/5/1375/2020/)

Use only the existing temperature, power, wind, RPM, phase, and operating-state channels and
the shared baseline, alarm, procedure, and corroboration fields. A recurrence count refers to
the supplied window. If a stopping alarm or restart restriction is available, retain its exact
scope in the handover. When that information is missing, report the observed symptom and name
the record or check needed to establish priority.

| Type | Question | Evidence required | Answer template |
| --- | --- | --- | --- |
| Chain-of-thought reasoning | Is the temperature deviation sustained or does it recur? | Complete winding readings, a matched upper range, and a supplied rule for counting consecutive or separate excursions. | [Winding] exceeded [upper range] for [N] consecutive intervals, returned within range for [M], and exceeded it again at [T]. There were [K] distinct excursions in the supplied window. |
| Chain-of-thought reasoning | Which symptoms co-occur at comparable operating points? | Repeated observations of winding spread and grid-side phase-current spread, matched on active and reactive output and operating state. | At [P] kW and [Q] kvar, winding spread and grid-side current spread were both elevated in [N] of [M] comparable intervals. Their separate first observed onset times were [T1] and [T2]. |
| Chain-of-thought reasoning | Did the symptom precede the decoded stopping alarm? | Complete intervals before a stop, the first visible symptom, and the alarm interval for the same turbine. | [Symptom] first became visible in the interval ending [T1]. The decoded [alarm] began at [T2]. These times establish [the observed order or that the measurements leave the order unresolved]. |
| Chain-of-thought reasoning | Did the symptom return with the next comparable run? | Before- and after-restart readings within the window, at comparable load and RPM, with fresh timestamps. | Before the restart, [metric] was [X] at [load/RPM]. After the restart, it reached [Y] at comparable load and RPM. [State whether it reappeared in the observed window.] |
| Chain-of-thought reasoning | Which data gaps affect the proposed diagnosis? | Expected timestamps for the affected channels and an independently reported event time. | The [channel] is absent for [N] intervals around [T]. [Other channel] remains available. We can establish [supported observation], while [specific comparison] remains unresolved. |
| Chain-of-thought QA | What should I put in a maintenance ticket for this unit? | Persistent load-adjusted winding deviation, steady recent operating conditions, and turbine ID, timestamps, and alarm status. | Open a condition-monitoring ticket for [turbine]. Winding [name] remained [X]°C above its matched range for [N] minutes while load and ambient stayed near [values]. Attach the winding traces and [alarm status]. Request verification of the reading and a cooling review, then compare the trend under the next similar load. |
| Chain-of-thought QA | Should I ask maintenance to investigate the temperature sensor or the generator? | One abrupt winding jump with steady peer windings and load, and no independent confirmation of that channel. | Start by verifying the temperature reading through the live controller or an independent check. [Winding] jumped [X]°C in one interval while the other windings and load stayed near [values]. If the temperature is confirmed or the applicable alarm becomes active, escalate for generator and cooling assessment using the site procedure. |
| Chain-of-thought QA | How urgent is the maintenance handover after this stop? | Observed drop in RPM and power, a decoded active stopping alarm, and an applicable site restart restriction. | Hand this to the control room and maintenance now. [Alarm] is active after the unit lost [X] kW and slowed at [T]. The supplied procedure requires [verified clearance condition] before restart. Include the alarm code, interval timing, preceding readings, and current operating state. |
| Chain-of-thought QA | It ran normally again after the restart. Can I close the issue? | Fresh telemetry shows a restart, earlier deviation is documented in the window, and only limited comparable post-restart exposure is available. | Keep the issue open for follow-up. The restart restored [P] kW and [RPM], and the earlier deviation has been absent for [N] minutes at comparable load. That shows short-term recovery. Check for recurrence during the site’s review period or at the operating point where the symptom first appeared. |
| Chain-of-thought QA | Is this a controls or telemetry handover, or do we need a mechanical inspection? | Frozen measurements on one path, corroborated live unit power or RPM in the shared context, and no available evidence of a mechanical fault in the observed window. | Route this first to SCADA or controls support. [Channel] repeated [X] for [N] intervals while the independent unit reading changed [Y]. Report the affected path, last trusted timestamp, and corroborating trace. Reassess mechanical symptoms once measurement reliability is restored. |

## Review before training

Check that each generated answer can be derived from the supplied channels and context at the cutoff. Score the measured values, the interpretation supported by those values, the action and its urgency, and the condition that would change the recommendation. Include matched normal and ambiguous examples so the model learns when to maintain operation, when to verify an indication, and when a documented rule supports immediate action.
