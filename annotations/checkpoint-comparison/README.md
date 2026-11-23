# Hill of Towie checkpoint comparison

The October–December 2024 validation panel has 20 opposing-outcome pairs in each of six families. The final January 2025–April 2026 test panel has the same family quotas and covers all four reporting and yaw-trial periods. Both panels cover all 21 turbines. The builder excluded six-hour windows that overlap earlier scored panels listed in the [panel audit](panels.json). Two validation windows appear under two different family questions. All 240 test windows are distinct.

Both checkpoints generated answers with greedy decoding under the same question, normalization and fixed 20-channel SCADA input. The prompt permits the answer label at either end of the response. Selection uses how often the model generates the correct outcomes for both windows in a pair. Both windows share the question, so guessing a fixed outcome from the wording cannot pass a pair.

The measurement check associates reported quantities with fields and units, then checks against reference rounding. It counts expected quantities recovered and flags parsed quantities it cannot verify. A qualitative answer can correctly describe the window while omitting exact numbers. The guidance screen checks for a verification action on a relevant topic. The screen measures topical relevance. Engineering review is needed to judge the full recommendation.

The open-ended transfer panel is a separate stratified check drawn from the 242,445-window operator test view. It uses one common question and keeps the operating state, any applicable diagnostic scenarios and the reference measurements outside the model input. It covers the four chronological cohorts and five reference operating states. The sample is deliberately enriched for rare states and has no window overlap with the paired test panels. Report counts from this panel as a diagnostic, without extrapolating them to the full test population.

## Validation and checkpoint selection

| Generated validation metric | Paired QA | Evidence-first |
| --- | ---: | ---: |
| Correct outcome labels | 220/240 | 203/240 |
| Both outcomes correct | 101/120 | 86/120 |
| Both outcomes correct and both guidance screens passed | 80/120 | 58/120 |
| Reference measurements recovered | 136/400 | 0/400 |
| Answers with at least one unverified parsed measurement | 158/240 | 0/240 |

The selected checkpoint is `runs/hill-paired-qa/best.pt`. It led on complete pairs and on pairs with a relevant next check. Its quantitative evidence remains unreliable. On T01 at 21:00 UTC on 3 October 2024 it correctly identified warming, but reported increases of 4 °C on all three windings. The reference increases round to 12 °C each. Evidence-first supplied qualitative descriptions and no reference measurements. It used “close spread” in eight electrical answers. Credit for all eight would still leave paired QA ahead on complete pairs. The common-prompt results measure performance under a shared question format and are separate from scores under the original training prompts. [Validation scores](validation.json)

## Untouched paired test

The selected checkpoint generated the correct outcome for 220 of 240 windows and both outcomes for 101 of 120 pairs. Both responses passed the guidance screen in 81 of 120 pairs. Individual outcome accuracy by cohort was 137/147 before commissioning, 31/37 during commissioning, 11/14 during the west yaw trial and 41/42 during the combined west and east trial. Cohort sample sizes differ.

The model recovered 131 of 400 reference measurements. The parser found at least one unverified measurement in 164 of 240 answers. For example, it correctly identified winding cooling for T13 on 24 January 2025 at 17:00 UTC, but reported a drop of 7 °C per winding where the readings show about 15 °C per winding. The test demonstrates useful discrimination under focused questions, with quantitative evidence still unsuitable for making an operating decision. [Test scores](test.json)

## Open-ended operator transfer

> What is this turbine doing now, what should I check next, and what would change the decision? Use the available telemetry and state any uncertainty.

The selected checkpoint failed the 63-window transfer check. It reported idle operation on 11 of the 12 windows with unavailable telemetry and omitted any acknowledgment of uncertainty on all 15 windows with unavailable telemetry or unknown state. For T01 on 15 September 2025 at 20:00 UTC, the answer supplied −9 kW, 1 rpm and a north direction even though the selected channels were missing at cutoff. Another response instructed a change to 1,551 kW without supporting evidence or operating limits. The paired benchmark scores support a narrow QA prototype. The open-ended output is unsuitable for operator guidance at this stage. The full 242,445-window open-ended test view remains available, with this run limited to the 63-window diagnostic. [Transfer review and examples](operator.json)

Run the evaluator from the annotation directory with CUDA and the existing Flamingo environment. The panel definitions and generated responses are stored under `runs/hill-checkpoint-comparison`.

```sh
python evaluate_checkpoint_comparison.py corpus \
  runs/hill-checkpoint-comparison/validation.parquet paired-qa/scaling.json \
  runs/hill-checkpoint-comparison/fresh-validation \
  --checkpoint paired runs/hill-paired-qa/best.pt \
  --checkpoint evidence_first runs/hill-evidence-first/best.pt
```
