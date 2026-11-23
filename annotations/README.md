# Hill of Towie SCADA annotations

[Full-archive corpus](#full-archive) · [TimeNet integration](#timenet-integration) · [Operator audit](#operator-audit-before-training) · [Primary QA benchmark](#primary-qa-benchmark) · [Broad operator evaluation](#full-automated-evaluation)

[Read the 60 pilot examples](EXAMPLES.md), [see the audit](PILOT_AUDIT.md), or use [the pilot JSONL](examples.jsonl). Each of the six annotation families has five observed scenarios, with a reasoning question and an operator-facing QA question for each window. The reasoning answer provides a short, checkable evidence summary. The QA answer explains the operating implications and gives a follow-up the data support.

## Reading a row

`input.question`, `input.context`, `input.values` and `input.valid` are the model inputs. Scenario labels, stages, split keys and source locators are dataset metadata. Both stages receive the same six-hour SCADA snapshot per `window_id`. Each snapshot ends at an hourly cutoff in UTC and contains 36 consecutive ten-minute slots. The first slot ends 350 minutes before the cutoff, and the last ends at the cutoff. A missing sample is JSON `null` with a false validity bit. Measured zeros have a true bit. Values retain up to three decimal places for the pilot. Source station, month, and cutoff identify the readings in the raw tables.

The context supplies the turbine identifier, cutoff and published rating. Live alarms, live controller state, site operating rules, matched history and independent meter readings have fixed keys with null values. The numeric ready, turbine error and in-operation flags remain available. They report seconds within the interval and cannot identify the active protection or first-out cause. Interpreting a power reference sampled at interval end alongside power averaged over the interval also limits timing precision.

Use the table order for the 20 series. Every series has the same mask shape. The JSONL is an annotation interchange format. A Flamingo adapter needs to encode its missingness masks and retain the fixed channel order and units when constructing inputs for either stage.

| Position | Input | Source field | Unit |
| ---: | --- | --- | --- |
| 1 | Active power | `wtc_ActPower_mean` | kW |
| 2 | Active wind speed | `wtc_AcWindSp_mean` | m/s |
| 3 | Generator speed | `wtc_GenRpm_mean` | rpm |
| 4 | Power reference at interval end | `wtc_PowerRef_endvalue` | kW |
| 5 | Main-shaft speed | `wtc_MainSRpm_mean` | rpm |
| 6 | In operation | `wtc_ScInOper_timeon` | s |
| 7 | Ready to operate | `wtc_ScReToOp_timeon` | s |
| 8 | Turbine error active | `wtc_ScTurSto_timeon` | s |
| 9 | Blade A pitch | `wtc_PitcPosA_mean` | degrees |
| 10 | Ambient temperature | `wtc_AmbieTmp_mean` | °C |
| 11 | Generator winding temperature 1U1 | `wtc_Gen1U1Tm_mean` | °C |
| 12 | Generator winding temperature 1V1 | `wtc_Gen1V1Tm_mean` | °C |
| 13 | Generator winding temperature 1W1 | `wtc_Gen1W1Tm_mean` | °C |
| 14 | Grid phase R voltage | `wtc_VoltPhR_mean` | V |
| 15 | Grid phase S voltage | `wtc_VoltPhS_mean` | V |
| 16 | Grid phase T voltage | `wtc_VoltPhT_mean` | V |
| 17 | Grid phase R current | `wtc_AmpPhR_mean` | A |
| 18 | Grid phase S current | `wtc_AmpPhS_mean` | A |
| 19 | Grid phase T current | `wtc_AmpPhT_mean` | A |
| 20 | Reactive power | `wtc_ReactPwr_mean` | kvar |

## Sampling and engineering limits

The miner screens hourly windows from July–September 2024 for training, October–December 2024 for validation, and March 2026 for the later-regime stress set. This follows the chronological TimeNet boundaries and leaves the main 2025 test unmined. It prefers three training windows, one validation window and one stress window per family. It uses another eligible split when the preferred split has no suitable nonoverlapping window. The assignments in the example index are authoritative. Both stages stay with their source window, and selected windows on the same turbine share no timestamps. Scenario-based selection makes these examples suitable for a pilot and biases any estimate of incident prevalence or model performance across routine operation.

The labels describe observable patterns. Start and stop counts are transitions across a documented speed, power and running-time screen. Electrical spreads are the maximum minus minimum of the three reported phase means. The screening criteria identify candidate windows. Operator interventions depend on site-approved limits and a verified live state. The answers use verification, monitoring or escalation where the data leave the decision open. In particular, the maintenance labels support a ticket or handover with measured evidence. Establishing a failed component, repair, or remaining useful life would require independent evidence. A load-adjusted healthy baseline would also be needed to call a winding temperature abnormal for its operating conditions. [Source data dictionary](https://zenodo.org/records/22662930/preview/Hill_of_Towie_data_dictionary.md?include_deleted=0) · [SCADA condition-monitoring study](https://wes.copernicus.org/articles/5/1375/2020/)

## Rebuild the pilot

Place the `2024.zip` and `2026.zip` archives from the pinned release in one directory. Pass the path to the turbine metadata CSV as the second argument. Run from the directory containing this README with Python 3.11 or later and NumPy and pandas installed.

```sh
python build_examples.py /path/to/archives /path/to/Hill_of_Towie_turbine_metadata.csv
```

Source measurements are from Clerc, A., and Lingkan, E. (2026), *Hill of Towie wind farm open dataset*, version 2.1.0, released by RES on behalf of TRIG under CC BY 4.0. This pilot selects and rounds samples and adds annotations. [Zenodo release](https://doi.org/10.5281/zenodo.22662930) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Full archive

The full release is under [corpus/](corpus/). It covers all 124 source months from January 2016 through April 2026 and all 21 turbines. The pilot above used selected examples for review. The full pipeline evaluates every eligible hourly cutoff and emits every matching scenario. Most screens require complete readings across all 20 channels. The gap-specific screens are designed to admit missing values when there is enough recorded evidence for their answer. A window may qualify for several scenarios. Unmatched windows stay in the index with an empty scenario list, which means the catalog provides no supported label for that window. These windows remain unclassified.

| Split | Matched scenario pairs | Text tasks |
| --- | ---: | ---: |
| Train | 786,373 | 1,572,746 |
| Validation | 27,199 | 54,398 |
| Test | 118,500 | 237,000 |
| **Total** | **932,072** | **1,864,144** |

Of 1,888,257 evaluated windows, 614,270 matched at least one scenario. All 30 labels occur somewhere in the release. Coverage is uneven. For example, the widening-winding-spread screen matched one training window and one validation window. Use the per-split counts when constructing training and evaluation batches.

Each monthly folder has three Parquet files.

| File | Contents |
| --- | --- |
| `signals.parquet` | 20 aligned, nullable SCADA channels. One row group per turbine, ordered by interval end. The first row ends at 00:10 UTC on the first of the month. |
| `windows.parquet` | Every eligible cutoff, the turbine, split and cohort, and all matching scenario labels. `[start_step, stop_step)` refers to the full monthly turbine row group. |
| `answers.parquet` | One row per window and matching scenario with both text questions and both answers. The monthly offsets join each row to its 20-channel source window. |

[The manifest](corpus/manifest.json) records channel order, turbine metadata and monthly coverage. [The answer counts](corpus/answer_counts.json) break down the materialized pairs by split, cohort, family and scenario. Keep the physical turbine and window together when filtering by stage or downsampling overlapping cases. These are deterministic, rule-generated annotations. Training examples can serve as weak supervision. Validation and test use later chronological partitions, but evaluating engineering accuracy requires independent labels or engineer adjudication. Template-matched scores measure agreement with the same rules used to build the targets.

Rebuild from the directory containing this README with all 11 pinned annual archives, turbine metadata and PyArrow installed.

```sh
python build_corpus.py /path/to/archives /path/to/Hill_of_Towie_turbine_metadata.csv corpus
python materialize_answers.py corpus
python verify_corpus.py corpus
```

## TimeNet integration

Use a separate `res/hill-of-towie-scada-qa` release alongside the three-channel forecasting release. This gives the 20-channel input and the QA targets their own version, card and evaluation history. The QA export stores one shared TimeF `Record` per turbine, month and cohort segment. Its Arrow signals preserve source nulls. Each scenario row becomes two `AnswerTask` objects with the same six-hour scope and chronological split. The prompt holds the question, and the text target holds either the evidence summary or the operator answer. The stage, scenario, family, cohort and paired window ID live in task metadata. Keep that metadata out of the prompt. The evidence summary is an explicit stage-one target. [TimeNet task model](https://docs.timenet.ai/data-model/tasks.html)

The [local TimeF release](timef-registry/res/hill-of-towie-scada-qa/1.0.0/manifest.json) contains 2,667 shared turbine records and 1,864,144 scoped answer tasks. The release checks verified file hashes, task counts by stage and split, and reloaded scoped values, masks and targets in all three splits. Rebuild with the TimeNet SDK installed. Without `--months`, the exporter writes all source months.

```sh
python export_timef.py corpus /path/to/local-registry
```

Use `TimeNet(sources=[registry]).open("res/hill-of-towie-scada-qa")` and `reader.iter_tasks(split=...)`. Filter on `task.metadata["stage"]` and feed the result through `tslm_inputs.task_example(task)`. That helper reads just the scoped 36 steps and returns the fixed values and validity arrays before tensor conversion. A Flamingo training reader still needs to encode the validity bits, for example as 20 additional fixed-order mask series. Any neutral fill used at tensor conversion must remain distinguishable from physical zero. The physical channel order, units and input fields must match across both stages, validation and use. [TimeF Arrow and missing values](https://docs.timenet.ai/data-model/time-series.html)

For a registered contribution, use the existing Hill of Towie archive pins and source parser, then register the reviewed QA dataset card and connector around this exporter. A representative task is the T18 below-rated-reference example at 09:00 UTC on 2026-03-29. The record has the same 20 nullable series described above and the task asks why output is not rising with wind. Its target explains that power was 430.6 kW against a 431 kW reference and asks the operator to check dispatch and active limits. The sample belongs to the west-and-east trial cohort in the test split.

## Operator audit before training

The full release is suitable for rule-generated supervision and a scoped pipeline smoke test. A
credible operator evaluation needs a revised question set and independently reviewed answers.
The audit below covers the 1,888,257 indexed windows, 932,072 paired targets and all 30 scenario
types. The release and its targets are versioned and remain available for reproducibility. Run
`python audit_operator.py corpus` to reproduce the counts.

| Finding | Evidence from v1.0.0 | Implication |
| --- | --- | --- |
| The wording reveals the scenario | All 30 QA question strings each map to exactly one label. All 118,500 test pairs use question strings seen in training. All 932,072 QA answers start with the paired reasoning answer verbatim. | A prompt lookup can score perfectly on test scenario identity. Keep the targeted questions for evidence drills. Use shared, neutral operator prompts and professionally reviewed targets to test whether the model reads the series. |
| Raw counts overstate variety | The five most common scenarios account for 487,707 of 786,373 training pairs (62.0%). In 445,173 training pairs (56.6%), the same turbine and scenario had another cutoff less than six hours earlier in the same cohort. This also applies to 65,279 test pairs (55.1%). | Sample or weight by event group. Keep overlapping windows in their chronological partition and cluster contemporaneous turbines for genuinely site-wide events. The two QA stages should retain the same group key. |
| Operator uncertainty is underrepresented | 1,273,987 windows have no catalog match. Of those, 21,915 are fully blank across all 20 channels for the six-hour input. The outage rule requires some prior data within the window. | Include full-outage cases with unknown last-good time, plus reviewed routine or ambiguous unmatched windows. An empty scenario list cannot establish healthy operation. |
| Thermal targets need operating context | For the common warming/cooling scenarios, 2,932 pairs showed at least a 30 °C absolute change in the mean of the three winding channels in an hour. Only one showed a 20 °C mean step in one ten-minute interval. On March 11, 2026 at 20:00 UTC, 20 turbines matched common warming at the same cutoff. The widening-spread class has one training and one validation example. | Stratify quality review by load transition and steady load. A rapid common-mode change can reflect a real load transition. Assess abnormality with matched load, RPM, ambient and prior thermal state. Keep pilot screens separate from protection thresholds. [Wind-farm normal behaviour modelling study](https://wes.copernicus.org/articles/8/893/2023/wes-8-893-2023.html) |
| Flag quality affects missingness | T10 reports −144 seconds for the in-operation flag in the interval ending 2018-02-02 22:00 UTC. Four annotated windows contain that reading across five scenario pairs. No other running, ready or error flag was outside 0–600 seconds. | Preserve the archived reading for traceability. Quarantine its affected examples or mark the sample invalid in the training input with a documented quality rule. |
| Maintenance coverage is narrow | 32,629 of 47,005 maintenance pairs (69.4%) are phase-current review. There are 505 recurrent-error handovers and 553 error-active output-loss handovers with thermal evidence. | Teach the handover and the next confirmation step. Fault confirmation needs first-out history, trusted independent readings or maintenance outcomes. |

### QA that requires reading the data

Use the same neutral question for different outcomes, such as **“What is this turbine doing now,
what should I check next, and what would change the decision?”** For each case, make the answer
lead with the current operating state and cite the few observations that drive the next action.

- **T13, March 23, 2026 at 22:00 UTC.** Output held near rated power at 2,277–2,301 kW while
  wind rose from 16.66 to 22.62 m/s and blade A pitch increased from 9.99° to 20.46°. The unit
  appears to be regulating near rated output. Check further if power departs from the reference
  or a protection indication appears.
- **T18, March 29, 2026 at 09:00 UTC.** The turbine is tracking a below-rated request. Wind
  rose from 9.33 to 13.07 m/s and power was 430.6 kW against a 431 kW reference. Check the
  dispatch and active limits before investigating an efficiency loss.
- **T15, August 19, 2024 at 15:00 UTC.** All selected channels are absent and the last interval
  with any reading ended at 10:00 UTC. Current operating state is unknown. Restore live visibility
  or confirm state through an independent source before assigning downtime.
- **T14, March 2, 2026 at 04:00 UTC.** Telemetry is reporting. The latest power was −0.6 kW
  and the last seven intervals each recorded zero running seconds. The turbine appears idle.
  Check the controller state and any current start restrictions for the reason.

- **T14, March 11, 2026 at 20:00 UTC.** Output rose from 11.9 to 2,300.4 kW in an hour and
  all three winding readings rose by 43.6–43.9 °C. The hottest was 73.9 °C at cutoff and the
  winding spread narrowed from 0.51 to 0.19 °C. The common rise coincided with a large load
  increase. Check the live cooling and alarm state, then watch whether temperatures keep
  rising after output stabilizes. A derate decision requires the applicable limit and dwell time.

When several patterns apply, write one handover per window with an operating state, the measured
evidence, a conditional risk, and the next check. Include a direct output or restart recommendation
only when the prompt contains a verified applicable site rule and the evidence satisfies it.
Later measurements or inspection results can serve as evaluation labels for separate tasks.

### Training and repository decision

Keep the dataset adapter, training loop, evaluation and demo in `timenet_hackathon/`. Use
OpenTSLM as a pinned dependency, starting from commit
`2968f4b891baab4307f7e9d0043e87677b593a30`. Keep the TimeNet fork for dataset work.
OpenTSLM exposes training loss, generation and checkpoint methods. Its curriculum script selects
its own task datasets, and the stock QA dataset class builds in-memory lists for all three splits.
Use a project-owned loader that reads bounded batches from TimeF or the monthly Parquet export.
A separate OpenTSLM fork becomes useful if shared model changes need their own upstream PR or a
maintained release history. The hackathon checkout currently has an unborn local Git branch, so
the first training code and config still need a reproducible commit. [OpenTSLM commit](https://github.com/OpenTSLM/OpenTSLM/commit/2968f4b891baab4307f7e9d0043e87677b593a30) · [Flamingo API](https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b891baab4307f7e9d0043e87677b593a30/src/opentslm/model/llm/OpenTSLMFlamingo.py) · [QA dataset loader](https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b891baab4307f7e9d0043e87677b593a30/src/opentslm/time_series_datasets/QADataset.py)

1. **Use the matched QA view.** Keep the 20 physical channels, 36 steps and explicit masks.
   Train and evaluate on scenario-specific questions drawn from the same template catalog. The
   response contains a short evidence summary followed by the final operator guidance. Keep
   audit metadata and scenario labels out of the model input. See the primary benchmark below.
2. **Verify the Flamingo input before a training run.** Start from the Llama 3.2 1B TSQA Flamingo
   checkpoint if base model access is available. Record both Hugging Face revisions and the
   dependency lock in a separate Linux CUDA environment. Fit scaling on observed training
   values per physical channel, preserving between-window levels and paired-phase comparisons.
   Apply a neutral model fill after masking invalid or absent readings. A zero reading with a true
   mask must remain distinct from a fill with a false mask. [Checkpoint](https://huggingface.co/OpenTSLM/llama-3.2-1b-tsqa-flamingo) · [OpenTSLM setup](https://github.com/OpenTSLM/OpenTSLM/tree/2968f4b891baab4307f7e9d0043e87677b593a30#llm-setup)
3. **Check channel identity and media routing.** The inspected implementation groups series
   into one media item, yet the prompt builder adds one media token per series description.
   The OpenFlamingo defaults in the inspected code also omit frame embeddings and use immediate
   media attention. With independently encoded, identically positioned frames, channel swaps
   can become invisible to the pooled representation. Start with a local wrapper that emits one
   media marker for the whole window and adds distinct positional or channel embeddings before
   pooling the 20 value frames and 20 validity frames. Confirm media shape and marker count,
   then verify that swapping same-unit channels or toggling validity changes the corresponding
   answer behavior. This is a source-code finding. Confirm it with the pinned runtime before
   assuming a particular failure mode. [OpenTSLM wrapper](https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b891baab4307f7e9d0043e87677b593a30/src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py) · [OpenFlamingo attention](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py)
4. **Train one SCADA QA stage first.** Start from the pretrained checkpoint and train evidence
   plus answer together. Begin with a small batch to verify EOS handling, answer-only loss,
   finite gradients and checkpoint reload. Initially freeze the base language model and train
   the time-series encoder, resampler, cross-attention and any added channel embeddings.
   A separate SCADA evidence-only stage is an optional ablation. [OpenTSLM curriculum](https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b891baab4307f7e9d0043e87677b593a30/curriculum_learning.py)
5. **Measure operator value.** Compare with prompt-only and rule-based baselines. Check whether
   outputs change appropriately when the telemetry changes under the same question. Score state,
   cited numbers and units, action appropriateness, evidence needed to resolve uncertainty, and
   unsupported intervention rates. Report by event and test cohort so routine operation and
   commissioning effects stay visible. Select on validation and report the full held-out test by cohort. Use overlap-aware
   aggregation alongside the all-window score.

## Full automated evaluation

This secondary evaluation uses **every held-out hourly window**, including routine, unclassified and completely
blank inputs. The original chronological split keeps training rows available for fitting. The
[validation set](evaluation/validation.parquet) contains 46,053 windows and the
[test set](evaluation/test.parquet) contains 242,445. The
[manifest](evaluation/manifest.json) records their state counts and the single shared operator question.

Each row has a rule-derived operating state, five cutoff measurements, the age of the last available
recording inside the input window, and a short reference handover. Rows with matching scenario
annotations also retain the applicable reference answers. A generating label requires power of at
least 50 kW, generator speed above 300 rpm and positive running seconds. Idle or stopped requires
power within ±30 kW, generator speed below 100 rpm and zero running seconds. These screening rules
support consistency scoring. The reference answers leave fault diagnoses and site intervention
decisions contingent on the applicable evidence. For a fully blank input, the last-observation age
is null, since the six-hour window supplies no last-good time.

`tslm_inputs.iter_evaluation(corpus, evaluation_file)` yields the same 20 × 36 observations, validity
masks and context keys as the TimeF training loader. The reference row is returned separately. Score
state and exact measurement fields first. Report matched-scenario evidence and cohort slices
separately. Use a fixed validation slice for frequent checkpoint comparisons, then run the full
validation and test sets once the candidate is selected. Reach and scores here refer to the rules and
source readings used to build the references.

```sh
python build_evaluation.py corpus evaluation
```

### Direct QA run on the full training split

Start with the direct QA model and the primary benchmark below. Use all **786,373 training QA targets** for one planned epoch if the measured throughput fits the budget. The broad operator evaluation measures transfer to a different question format. Treat the first **50 optimizer updates** as the startup
checkpoint of that run. An effective batch of eight sees **400 examples**, about **0.05% of an
epoch**. Check finite loss and gradients, mask and media routing, and that the checkpoint reloads.
Continue from the same checkpoint once those checks pass. This startup slice can validate the
training loop. An early change in loss on its own gives limited evidence of improvement on held-out
turbine cases. [OpenTSLM Flamingo implementation](https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b891baab4307f7e9d0043e87677b593a30/src/opentslm/model/llm/OpenTSLMFlamingo.py)

As a first progress comparison, evaluate the pretrained and fine-tuned models on the same validation
slice at roughly **2,000 optimizer updates**. That is 16,000 examples at effective batch eight,
about 2% of the QA epoch. The wall-clock estimates below exclude model downloads, checkpoint I/O
and evaluation. Measure the actual rate on the target GPU after warm-up.

| Measured training rate | First 50 updates | First 2,000 updates | One QA epoch |
| ---: | ---: | ---: | ---: |
| 2 examples/s | 3m 20s | 2h 13m | 109.2h |
| 10 examples/s | 40s | 26m 40s | 21.8h |
| 30 examples/s | 13s | 8m 53s | 7.3h |

32-bit or mixed-precision throughput, generation and training throughput can differ substantially.
For the full test, divide 242,445 by measured generation examples per second and budget that time
separately. At 10 generated answers per second, the full test alone is about 6.7 hours.
Try evidence-first training as a comparison after the direct run establishes a baseline.

## Primary QA benchmark

Use the [matched QA files](targeted-qa/manifest.json) for the main OpenTSLM-style experiment.
All three chronological splits use scenario-specific questions from the same template catalog.
Each training or evaluation example presents the same 20-channel SCADA input format and requests a
short evidence summary followed by a final answer. There is one text target per scenario window,
with the evidence followed by `Answer:` and the operator guidance. The original QA text starts
with the evidence paragraph, so this view separates the two parts without duplicating that paragraph.
[OpenTSLM curriculum and objective](https://arxiv.org/html/2510.02410v1#S4.SS1) · [ECG-QA prompt and target code](https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b891baab4307f7e9d0043e87677b593a30/src/opentslm/time_series_datasets/ecg_qa/ECGQACoTQADataset.py#L90-L149)

| Split | Examples | Use |
| --- | ---: | --- |
| Train | 786,373 | Fine-tune the model |
| Validation | 27,199 | Select the checkpoint and scoring settings |
| Test | 118,500 | Score the selected checkpoint |

The primary score should check the measurements, units, and appropriateness of the final action.
The question text predicts the scenario label in this corpus, so scenario-label accuracy alone
cannot establish signal grounding. Compare against a prompt-only baseline. The earlier broad
evaluation remains available as a separate transfer experiment across every held-out window.

Rebuild with `python build_targeted_qa.py corpus targeted-qa`. Use
`tslm_inputs.iter_evaluation(corpus, split_file)` to retrieve the masked input and a separate
reference row whose `target` contains the full response. Apply the model tokenizer's EOS token
when building the supervised batch.

### Run the first Flamingo training job

Run from this directory on a Linux NVIDIA GPU with Python 3.12 or newer and Hugging Face read
access to Meta's Llama 3.2 base model. Put the existing `corpus/` and `targeted-qa/` folders
alongside the scripts. The loader reads the Parquet directly. [Base model access](https://huggingface.co/meta-llama/Llama-3.2-1B) · [Starting Flamingo checkpoint](https://huggingface.co/OpenTSLM/llama-3.2-1b-tsqa-flamingo)

```sh
uv venv --python 3.12
uv pip install pyarrow "open-flamingo==2.0.1" "opentslm @ git+https://github.com/OpenTSLM/OpenTSLM.git@2968f4b891baab4307f7e9d0043e87677b593a30"

.venv/bin/python train_flamingo.py train --max-steps 50
.venv/bin/python train_flamingo.py train --resume --max-steps 2000
.venv/bin/python train_flamingo.py train --resume
.venv/bin/python train_flamingo.py test
```

The second command resumes the same run to the 2,000-update comparison point. The third finishes
one pass over the training file. The fourth uses the best targeted validation checkpoint to generate
predictions on the targeted test set. The open-ended evaluation files are outside that workflow.
The default validation sample has up to 512 time-spread rows. Keep it fixed when resuming so the
checkpoint selection losses are comparable. Model revisions are resolved at startup and saved with
checkpoints. The fitted scaler is `targeted-qa/scaling.json`, which can be rebuilt with
`python fit_scaling.py corpus targeted-qa/scaling.json`. It uses only training measurements and shares
transforms within physical phase and winding groups.

This runner supplies one Flamingo media item with 40 frames, alternating the 20 signals and their
validity masks. It adds learned frame identities to the upstream resampler and uses one media
marker in the prompt. The backbone is frozen and the time-series components are trainable.
The CPU integration check exercised real SCADA input, answer-only masking, optimizer updates,
checkpoint resume and targeted prediction export with a stand-in model. A pretrained CUDA forward
and backward pass still needs to be verified on the training host.

### OpenTSLM-style 20,000-example curriculum comparison

The OpenTSLM paper describes an encoder warmup on multiple-choice trend questions and
time-series captions, followed by task-specific reasoning QA. In Flamingo, the language
backbone stays frozen while the patch encoder, Perceiver and gated cross-attention are
trained. The released Flamingo wrapper has an input-embedding option. The SoftPrompt
curriculum applies LoRA after captioning. Full LLM fine-tuning is outside the published
Flamingo recipe. [Paper methods and curriculum](https://arxiv.org/html/2510.02410v1#S4.SS1) · [Released curriculum](https://github.com/OpenTSLM/OpenTSLM/blob/2968f4b891baab4307f7e9d0043e87677b593a30/curriculum_learning.py)

The Hill of Towie run uses 4,000 trend MCQs, 4,000 short captions, and 12,000 existing
operator QA examples. It adapts the upstream task sequence to SCADA and keeps the same
20 channels, masks, Llama 3.2 1B Flamingo initialization, and frozen language backbone
as the direct QA control. The builder draws from the historical training split. QA
selection spreads examples across year and family, and uses one scenario per sampled
window. The paper's AdamW learning rate of 0.0002, 10% warmup, linear decay and gradient
clip of 1 are applied separately to each substage. The run makes one pass over each
substage. [Paper training configuration](https://arxiv.org/html/2510.02410v1#A1.SS1)

```sh
python build_curriculum.py corpus targeted-qa curriculum-20k
python train_curriculum.py --checkpoint-revision 923332423e1b892d78d4dc8e5854d424b6fa9a92 \
  --base-revision 4e20de362430cd3b72f300e6b0f18e50e7166e08
```

Select by the same 128-window targeted validation sample as the direct run and compare
generated text on the same 12 prior validation cases. Inspect numeric agreement and
same-question signal swaps before using the full test or deciding on the larger run.
This compares a bundle of curriculum, sampling, and optimizer changes. It leaves the
contribution of each individual change open.

The 20,000-example curriculum finished on Nebius. Targeted validation loss was 24.3364 before
training, 10.7333 after warmup, and 0.3834 at the selected QA checkpoint. The direct run's
lowest logged loss was 0.5504 and its retained checkpoint scored 1.0016. On the same 12
generation cases, the curriculum produced 12 answer markers versus 9 for the direct run and
matched 10 of 58 reference numeric literals versus zero. The literal count includes fixed
numbers in templates. In three same-question pairs, swapping telemetry changed teacher-forced
loss by at most 0.012. Two pairs still got identical full answers with different measurements.
The lower validation loss and cleaner format did not establish reliable signal grounding.
[Generated validation comparison](curriculum-20k/validation-comparison.jsonl)

### Paired-question supervision comparison

This comparison reuses the 8,000-example caption-warmup checkpoint and replaces the 12,000
operator QA targets. Each of six questions is held fixed across windows with opposing
observed outcomes. The pairs cover power reference tracking, generation versus idle,
winding trends, phase-current spread, telemetry visibility and error-active handover.
Numbers come from the 36 readings available by the cutoff. A short answer gives the
outcome, the measured evidence and the next check. The numbers are supplied as training
targets. Model input remains the raw 20-channel window and its validity masks.

The training view has 6,000 paired examples of the question design, or 12,000 individual
question–answer rows over 105 months and 21 turbines. The validation view has 60 pairs
from October through December 2024. The older and revised QA checkpoints start from the
same Llama/Flamingo initialization and Hill of Towie warmup. The revised run uses the
same QA optimizer, schedule and precision as the earlier curriculum run.

Compare the two checkpoints on both members of each validation pair. Candidate-answer
likelihood lets us score outcome discrimination even when a model skips the requested
format. A 24-window generation sample separately checks emitted outcomes and numeric
literals. Numeric matches can include fixed quantities from the question and are a
diagnostic, so read example answers alongside the score. The original targeted test and
the broad operator test are reserved.

```sh
python build_paired_qa.py corpus paired-qa
cp targeted-qa/scaling.json paired-qa/scaling.json
python train_curriculum.py --resume-qa --warmup-checkpoint runs/hill-curriculum-20k/warmup.pt \
  --data paired-qa --curriculum paired-qa --output runs/hill-paired-qa \
  --checkpoint-revision 923332423e1b892d78d4dc8e5854d424b6fa9a92 \
  --base-revision 4e20de362430cd3b72f300e6b0f18e50e7166e08
python evaluate_paired_qa.py corpus paired-qa runs/hill-curriculum-20k/best.pt runs/hill-paired-qa/best.pt
```

The paired run completed and both checkpoints were scored on the same revised validation
questions. Forced-choice accuracy rose from 57/120 to 109/120. The number of pairs with both
outcomes correct rose from 0/60 to 50/60. In a separate 24-window generation sample, the revised
model emitted 23 correct outcome labels. It matched 19 of 46 reference numeric literals,
compared with three for the prior checkpoint. Some counts are fixed by the question or template.
The model still misreported specific values, such as giving 2.9 A for a measured 1.8 A phase-current
spread. [Scores](paired-qa/comparison.json) · [Revised predictions](paired-qa/revised-validation.jsonl)

### Qualitative QA variants

`paired-qa-qualitative` and `targeted-qa-qualitative` reuse the source windows, split
assignments, row order, and fitted scaler. Evidence describes directions, relative
relationships, and operating states. For example, when the windings warm together,
the evidence reports the common rise and the answer retains the checks on load and
cooling. The paired electrical prompt asks whether the phase-current spread is
small or wide. Its original thresholds still define which windows qualify.

```sh
python build_qualitative_qa.py paired-qa paired-qa-qualitative
python build_qualitative_qa.py targeted-qa targeted-qa-qualitative
```

Use `paired-qa-qualitative` for both `--data` and `--curriculum` when resuming the
paired QA stage. Use `targeted-qa-qualitative` as `--data` in the full QA runner.
The paired evaluator scores outcome accuracy and complete pairs as before. When the
references contain no numeric literals, it omits numeric-match metrics. Review the
saved generated answers for whether the evidence supports the outcome and whether
the follow-up reflects the engineering guidance.

### 100,000-example paired QA continuation

The continuation starts from the selected paired-QA checkpoint. The training view contains
50,000 opposing-outcome pairs from 105 months and 21 turbines. Source windows used in the
prior 12,000-example paired run are excluded. Each family has the same number of rows per
outcome. The number of pairs per family depends on how many eligible windows the archive
contains. The trainer mixes month-local chunks across the epoch to expose the model to
different years before the first checkpoint. The 60 validation pairs are reused byte-for-byte.

The continuation checks all 60 pairs and generates on the same fixed 24-window sample before
training and every 5,000 examples. Checkpoint selection uses the mean of both-correct pair
accuracy and exact numeric-literal F1, which penalizes extra generated numbers. A checkpoint
needs at least a 0.5 percentage-point gain over the best score and must preserve the baseline
outcome counts within two pairs and two generated answers. Three consecutive failed checks stop
training. The run saves improved weights atomically in BF16 to fit the VM disk and re-evaluates
the selected saved checkpoint at the end. If no checkpoint improves, the starting checkpoint
remains selected. The test set stays outside checkpoint selection.

```sh
python build_paired_qa.py corpus paired-qa-100k --train-pairs 50000 \
  --exclude-previous paired-qa/qa.parquet
cp paired-qa/validation.parquet paired-qa/scaling.json paired-qa-100k/
python train_paired_early_stop.py --validate-every 5000 --patience 3 --min-gain 0.005
```

The five-check retry stopped at 25,000 examples. It retained the starting paired-QA checkpoint.
At the last check, 51 of 60 pairs were correct versus 50 at the start, and 17 of 46 reference
numeric literals matched versus 19 at the start. The joint score decreased from 0.6232 to
0.6098. [Run report](paired-qa-100k/patience5/result.json) · [All five checks](paired-qa-100k/patience5/metrics.jsonl)

### Generated-answer checkpoint comparison

The paired-QA and evidence-first checkpoints were compared on fresh October–December
2024 validation pairs under identical prompts and decoding. The selected paired-QA
checkpoint was then evaluated on the untouched January 2025–April 2026 comparison
panel. The final open-ended transfer check uses the same operator question across a
separate sample. [Protocol, scores and operator examples](checkpoint-comparison/README.md)

The [Mixed operator QA continuation](operator-continuation/README.md) adds open-ended operator training and uses the paired benchmark to measure retention.

## AWS OpenTSLM curriculum data

The [AWS dataset views](aws-curriculum/manifest.json) reference the original
six-hour Hill of Towie windows. The curriculum uses multiple-choice trend
questions, qualitative captions, then multichannel SCADA trend and state
questions. Held-out warmup tasks are derived from the same chronological
validation and test windows as domain QA. Domain examples keep their pair IDs
and carry qualitative evidence separately from the option text the stock AWS
trainer and scorer use as the answer. Operator checks remain downstream.

`aws_scada_inputs.model_examples(aws, corpus, path)` yields native AWS
`Example` objects directly from a stage Parquet file. The caller can pass a
list of these objects to the AWS `train()` function. The underlying
`iter_examples()` also yields the task row for reporting, and returns a
`None` example for an entirely unobserved window. For the telemetry task,
route such windows to the availability response. Report this path separately
from the model score and include it when evaluating the full operator workflow.

The adapter gives each physical channel a name, unit, and its own media item.
The AWS loader receives raw readings and NaN for missing samples, and
computes normalization and validity masks itself. Channels with no observed
samples are omitted from the media and named in the question context. This
accommodates the AWS normalization routine, which requires an observation
in every media item. A recorded zero remains an observed value.

Rebuild and verify from the annotations directory. Use the public source and
checkpoint configuration at the pinned revisions in
[`aws_scada_inputs.py`](aws_scada_inputs.py).

```sh
python build_aws_curriculum.py aws-curriculum
python aws_scada_inputs.py aws-curriculum \
  --aws-utils /path/to/utils.py --checkpoint-config /path/to/config.json \
  --report aws-curriculum/audit.json
```

The [audit](aws-curriculum/audit.json) records model and availability-route
counts. It uses the pinned AWS input preparation and option scorer on
representative tasks from every view. The stock AWS caption score is exact
text agreement. Use it as such when reporting caption results.
[AWS sample and training instructions](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/b83a9901d401e9e872ee9048fe61e6c9df9a5b78/examples/opentslm-finetune-arfbench) · [Pinned checkpoint config](https://huggingface.co/opentime/tslm-nov26/blob/5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983/config.json)

## Full-history TimeNet curriculum connector

Build the registered connector from the TimeNet workspace:

```sh
uv run timenet-build build res/hill-of-towie-scada-curriculum@1.0.0 \
  --local-registry /path/to/registry --keep-cache
```

The connector uses the pinned Hill of Towie 2.1.0 annual archives and the
existing chronological split and campaign boundaries. It registers the raw
26-channel history once per turbine and period, with nulls preserved. It
streams scoped task rows, using hourly cutoffs that fit within the period.
The new bearing, oil-temperature, and oil-pressure channels augment the
original 20 signals. Every window with an eligible task contributes to the
release. Training can filter tasks by the `stage` metadata value.

| Stage | Principal targets |
| --- | --- |
| `trend_mcq` | Rising, falling, or steady for an observed channel |
| `caption` | Qualitative descriptions of the selected channels |
| `domain_qa` | Telemetry availability, operating state, anomaly presence, onset, recovery, magnitude, pattern, channel identification, co-movement and relative timing, plus the multi-day condition screen and maintenance triage |
| `forecast` | A sustained stop, stopping alarm onset, pitch-lubrication alarm onset, return of a cleared error flag, or persistent bearing or oil temperature elevation within the next hour |

The thermal questions use an earlier comparison at similar load, speed and
ambient temperature. The condition screen compares the most recent day with
matched generating intervals from the preceding six days in the same record.
A persistent elevation has an inspection rationale for possible wear or
lubrication issues. Physical wear and scheduled work orders require separate
evidence. The alarm forecast uses the stopping codes listed in the publisher's
alarm description and keeps
pitch lubrication as its own label. Stop labels require two consecutive
observed stopped intervals. Gaps in the required future operating telemetry
exclude that forecast task. Forecast targets stay outside the task's input
scope. These outcomes describe what actually followed in the historical data.
Estimating the effect of an operator intervention requires intervention records.

The maintenance forecasts use observations after the cutoff as targets.
Error recurrence requires an error-active interval in the input, a cleared
flag at cutoff and valid future flag readings. The thermal target compares
the next hour with matched generating intervals from the earlier half of
the input window. It uses load, generator speed and ambient temperature at
cutoff to define comparable operation. The thermal task is skipped when the
baseline is inadequate or the forecast hour lacks comparable generation or
required readings.
The labels support comparing controller events and reviewing affected components.
Actual maintenance scheduling and repair outcomes require work-order records.

Maintenance triage adds a rotating question at each hourly cutoff with available
telemetry. Select these questions by the `maintenance_` category prefix.

| Question | Qualitative target and operator follow-up |
| --- | --- |
| Error recurrence | Distinguish separated flag periods, a single observed flag period, and an inactive flag. Retrieve the controller event sequence for the handover. |
| Phase spread | Identify persistent phase-current spread under steady load and whether voltage spread also persists. Request a time-aligned independent meter comparison. |
| Gearbox oil | Compare oil temperature and pressure with earlier intervals at similar load, speed and ambient temperature. Verify the pressure indication and review oil cooling when the readings warrant it. |
| Thermal telemetry | Identify missing winding readings during verified generation. Ask SCADA maintenance to compare affected tags with the live controller. |

The gearbox comparison requires complete oil readings over the latest hour
and enough matched observations earlier in the window. The phase and oil
criteria are annotation screens. Their thresholds encode training labels and
carry no authority for turbine operation. The qualitative label is the AWS
target. The TimeF rationale supplies a follow-up for the operator handover,
subject to live verification.

For AWS training, open the release with TimeNet and call
`aws_scada_inputs.example_from_task(aws, model, task)` for tasks in the intended
stage and split. This reads only the channels and steps in the task scope.
Keep fully unobserved tasks in the availability path when reporting end-to-end
coverage. The AWS text target is the canonical label for classification and the
caption text for captioning. Classification rationales remain available as
separate evidence.

### One-hour Nebius pilot

`build_nebius_pilot.py` reads the curriculum's scoped tasks from selected
archive months and samples each category and answer within its original split.
The default pilot uses January and July 2018, March and July 2024 for training,
and October 2024 for validation. The exported JSONL files retain both window
lengths, the native OpenTSLM prompts, and the missing-value masks as nulls.
The test split stays outside this pilot.

Run `run_nebius_pilot.py` with the pilot directory, the locally cached pinned
`opentime/tslm-nov26` snapshot, and a new output directory on the GPU instance.
Pass `--utils` with the pinned reference implementation used by
`aws_scada_inputs.py` and `--minutes 60` for the setup, training and validation
budget after the model is cached. The runner measures baseline validation and
training throughput before selecting its step count. It saves the tuned
parameter tensors in `trained.safetensors` and the paired baseline and tuned
validation results in `report.json` and `predictions.json`. Keep the base
snapshot to reload the tuned parameters. Compute, data, and results stay on
Nebius or the local workspace.

For a larger continuation, build a subset with `--train-limit 40000` and a
separate test panel with `--test-per-label`. `run_nebius_continuation.py` takes
that directory, the cached base snapshot, the pilot's tuned parameter file,
and a new output directory. It also takes `--utils` and `--pilot-panel` to
score the same pilot questions after training. It checks category-averaged
validation loss every 500 optimizer updates, stops after three checks without
an improvement of at least 0.003, and reloads the lowest-loss checkpoint for
evaluation. The separate test panel is used after checkpoint selection.

### Generated operator responses

`run_operator_response_sft.py` continues the best checkpoint on full text
targets for condition monitoring, gearbox oil, and all five one-hour forecast
tasks. Each response has observed evidence, an assessment, a conditional
next action, and the original answer label. Selected short-label tasks remain
in the training mix. Action text is curated guidance. It depends on verified
live readings and the applicable site procedure.

The runner checks loss parity against the original single-example path,
benchmarks true microbatches of two and four on Nebius, and retains about
eight examples per optimizer update. It selects by validation loss and
measures generated response structure and answer labels on held-out windows.
Run it with the existing continuation data directory, the pinned base
snapshot, the best continuation parameter file, a new output directory,
and `--utils` pointing to the pinned reference implementation.

`build_expanded_operator_qa.py` exports a supplement using the continuation
dataset and its existing chronological splits. It upgrades telemetry
coverage, operating-state handover, error recurrence, phase spread, winding
telemetry, and thermal recovery, pattern, and channel identification to
full answers. It also derives power-reference tracking from scoped
power, reference, speed, running time, and wind signals. The tracking
screen uses consecutive generating intervals at a steady reference.
A sustained gap calls for checking wind availability and dispatch limits
before investigating a fault.

Pass the continuation data directory, a new output directory, and
`--utils`. To sample additional held-out normal-tracking examples from the
source archive, also pass `--archive` and `--months 2024-10 2025-01`. The
export includes a manifest with counts by category, answer choice, and split.
Pass the supplement files to `run_operator_response_sft.py` using
`--additional-train`, `--additional-validation`, and `--additional-test`.
Use the continuation dataset from which the supplement was built as the
runner's base dataset.

#### Operator target audit

The audited supplement is in `runs/operator-qa-targets-audited`. It covers
operating state, power-reference tracking, SCADA coverage, recurring errors,
phase spread, winding telemetry, and thermal detection, persistence and
recovery. The base dataset supplies matched seven-day thermal screening,
gearbox oil checks, and the five next-hour forecasts.

The audit matched all 6,799 forecast labels to their source SCADA or alarm
windows. It also checked that the input and outcome intervals stayed within
the assigned chronological split. Forecast evidence uses readings through the
cutoff. A sustained stop is a SCADA operating-state transition. Documented
stopping alarms and pitch lubrication alarms are code-based onsets. The error
return target applies when the error flag was active earlier and had cleared
at cutoff. Thermal outlooks apply under comparable generation.

The sampled test split has eight positives and eight negatives per forecast.
It measures performance on those cases. Estimating alert precision or
false alarms per turbine-day requires a chronological test at the observed
event rate. Error return has 52 positive training windows. The supplement
has no positive cases for winding telemetry loss during generation or
simultaneous persistent phase-current and voltage spread. Component wear
and maintenance due dates need inspection or work-order labels.

### CPU baselines for the trend-trained TSLM

`evaluate_trend_baseline.py` fits two label baselines. The majority baseline
chooses the most frequent eligible training label for each category and breaks
ties alphabetically. The logistic regression fits one model per category
with fixed regularization (`C=1`) and training-only standardization. It uses
within-channel endpoint change, half-to-half mean change, range and
observation coverage over the full window, its trailing half and its trailing
hour. Channel presence and cutoff availability retain missingness information.
Turbine ID and cutoff identify evaluation rows and group uncertainty estimates.
The classifiers use neither as a feature.

The operator comparison uses the 371 saved test windows from the completed
audited `tslm-nov26` run and fits on all 41,579 eligible labeled rows in the
base and audited supplement training exports. Its saved model results are
from update 1,500. The subsequent continuation has a local launcher but no
saved test predictions at the time of this comparison. The script joins the
prediction file to the fixed held-out source rows by category, turbine,
cutoff and prompt. Use the continuation's prediction file to compare its
results on the same panel.

| Operator test, 371 windows | Correct | Accuracy (95% CI) | Category macro accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Training majority | 128 | 34.5% (27.6–42.0%) | 37.9% | 0.166 |
| Trend logistic regression | 259 | 69.8% (65.1–74.5%) | 69.9% | 0.644 |
| Audited `tslm-nov26` | 242 | 65.2% (60.6–69.8%) | 66.7% | 0.554 |

TSLM accuracy minus logistic regression accuracy was −4.6 percentage points
(95% CI −11.3 to +2.7). TSLM alone was right on 35 windows. Logistic
regression alone was right on 52. These are label metrics. The sampled forecast
cases support a comparison on the panel. Field alert precision requires
sampling at the observed event rate. Evaluating the generated evidence and
actions requires separate review.

The earlier `large-trend-data` run supplies a paired test with identical
questions and opposing outcomes. Logistic regression fits on its 100,000
training windows. The run retained the preceding evidence-first checkpoint,
whose archived results appear below.

| Earlier trend test, 240 windows | Correct | Accuracy | Macro F1 | Both correct, 120 pairs |
| --- | ---: | ---: | ---: | ---: |
| Training majority | 120 | 50.0% | 0.333 | 0 |
| Trend logistic regression | 234 | 97.5% | 0.975 | 114 |
| Retained evidence-first checkpoint | 194 | 80.8% | 0.799 | 79 |

The JSON reports contain the individual predictions, per-category and
per-label counts, precision, recall and F1, and 95% percentile intervals
computed with 10,000 turbine-cluster bootstrap replicates. Accuracy
differences resample the same turbines for both methods. Balanced accuracy
and macro F1 average over the category/label combinations observed in the
test set. [Operator test statistics](trend-baseline/operator-test.json) ·
[Earlier paired test statistics](trend-baseline/large-trend-test.json)

Run from the annotations directory with NumPy, scikit-learn, pandas and
PyArrow installed. Set `dataset` to the local continuation export directory
containing `train.jsonl.gz` and `test.jsonl.gz`. Pass a fresh output path.

```bash
python evaluate_trend_baseline.py operator "$dataset" operator-baseline.json \
  --supplement runs/operator-qa-targets-audited \
  --predictions runs/nebius-operator-audited-20261122/predictions.json

python evaluate_trend_baseline.py paired large-trend-data paired-baseline.json \
  --corpus corpus --predictions runs/hill-large-trend/previous_signals.jsonl
```

#### Temporal CNN on the submission panel

`evaluate_temporal_cnn.py` trains a CPU classifier directly on the SCADA
sequences used by the audited OpenTSLM run. It aligns the scoped channels by
name and gives each channel an observed-value mask, preserving measured zeros,
missing readings, and the original ten-minute spacing. It fits channel
normalization on the training split. Three temporal convolutions learn features
from six-hour and seven-day windows. The model uses the task category as an
embedding and restricts predictions to the question's eligible answer choices.
It outputs a label for each window.

The CNN uses the same training exports and label counts as the trend logistic
regression. It selects the epoch by validation macro F1 and then scores the
371 archived submission windows in their original order. The report includes
the one-hour forecast subset and paired accuracy differences, with
turbine-cluster bootstrap intervals. The output directory contains the
report and a PyTorch checkpoint with the weights, scaling, channel order,
and label vocabulary.

The 43,477-parameter CNN selected epoch 5 on 453 validation rows. Results
were reproduced by reloading the saved weights and scoring the archived panel.

| Method | Correct, 371 windows | Accuracy | Macro F1 | One-hour forecasts |
| --- | ---: | ---: | ---: | ---: |
| Temporal CNN | 275 | 74.1% | 0.676 | 54/80 |
| Trend logistic regression | 259 | 69.8% | 0.644 | 50/80 |
| Submission OpenTSLM | 242 | 65.2% | 0.554 | 53/80 |

The paired CNN-minus-OpenTSLM accuracy difference was +8.9 percentage points
(95% CI +4.4 to +13.6). The CNN-minus-logistic difference was +4.3 points
(95% CI −0.3 to +9.1). The intervals resample turbines and reflect uncertainty
on the test panel for this fitted CNN. [Per-window CNN report](trend-baseline/temporal-cnn-test.json)

From the annotations directory, use the continuation export as `dataset`
and the same audited supplement used for the logistic regression run.

```bash
python evaluate_temporal_cnn.py "$dataset" runs/temporal-cnn-submission \
  --supplement runs/operator-qa-targets-audited \
  --predictions runs/nebius-operator-audited-20261122/predictions.json \
  --benchmark trend-baseline/demo-checkpoint-test.json
```
