# Pilot audit

I replayed all 30 original windows against the four raw SCADA tables. All 60 exported examples retain the same source windows. Each 20 × 36 matrix matched the source after the documented three-decimal rounding, and each mask matched the source nulls. I independently rebuilt the UTC step grid for each row, checked that the two stages share their measurements, and applied the 30 scenario screens to the recovered source windows.

| Family | Evidence checked | Editorial audit |
| --- | --- | --- |
| Power ramps | Power and wind endpoints, one-hour ramp rate, reference range or endpoint gap, pitch and rated capacity | Reference tracking and rated regulation are consistent interpretations. The data leave dispatch origin and within-interval order open. |
| Repeated starts | Counts of entries and exits from the generation screen, recent running seconds, speed, output and error-active seconds | Revised answers to distinguish threshold crossings from verified restarts. Ready and error flags indicate time in a reported state. |
| Winding temperature | All three winding changes, max–min spread, load, RPM and ambient range | The measured differences can be small. The widening-spread case now states that all three windings warmed and prioritizes comparison with historical behaviour. The cooling case describes the two observed endpoints. |
| Electrical phase imbalance | Simultaneous phase means, maximum deviations and spreads, peak timestamps and active/reactive load | The answers limit the observation to these phase means. An independent meter or protection record is needed to diagnose a grid event or assess an operating limit. |
| Stale telemetry | Latest valid slot, missing slots, restored temperatures, near-flat wind and observed zero running time | The examples distinguish measurements that were absent, resumed, or nearly flat. A narrow wind range alone leaves sensor freshness unresolved. |
| Maintenance triage | Error-active intervals and separated runs, winding spread, load ranges, missing coverage, and output loss | The guidance identifies what to include in a ticket or handover. The observed flags and temperature trends leave failure cause and restart eligibility open. |

The 60 examples are the original review pilot. Their March 2026 `stress_test` label survives in the pilot JSONL. In the full corpus those windows use `test` and the TimeNet west-and-east trial cohort.
