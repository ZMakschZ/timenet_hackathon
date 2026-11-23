# Evidence-first paired QA experiment

Test whether a qualitative, evidence-first answer format improves generated SCADA
outcomes and whether the model needs the recorded signals to produce those outcomes.

- Reuse the original 12,000 training examples and 120 validation examples. Keep
  the same time series, validity masks, scopes, splits, scaler, and candidate
  outcomes. Rewrite the targets as a brief observation and engineering check,
  followed by a final `Answer:` line with the outcome label.
- Start from the same SCADA warmup checkpoint as the original paired QA run.
  Train for up to one pass and evaluate generated answers on the complete
  validation panel every 4,000 examples. Select by the mean of both-correct pair
  accuracy and macro-F1 across family/outcome classes. During selection, require
  a final answer line. Numeric literals receive no credit.
- Reserve 120 new opposing-outcome pairs from the 2025–2026 test split. Evaluate
  the original paired checkpoint and the new selected checkpoint on exactly
  these prompts with real SCADA and with the media tensor held constant at zero
  for every window. The latter keeps the prompt and model path fixed and
  removes all window-specific signal and missingness information.
- Report generated-label accuracy, macro-F1, the number of pairs where both
  outcomes are correct, and evidence-first format compliance. Score generated
  labels independently of placement for the checkpoint comparison, and report
  strict final-line results alongside them. Preserve the full generated answers
  for inspection. The results measure agreement with rule-derived outcomes.

## Build and run

From the annotations directory, build the test panel and convert the source
dataset. Training never reads the test file.

```sh
python build_paired_qa.py corpus paired-qa-fresh-test --test-pairs 120
python build_qualitative_qa.py paired-qa paired-qa-evidence-first \
  --evidence-first --test paired-qa-fresh-test/test.parquet
bash run_evidence_first.sh
```

On Nebius, run from the existing `hill-towie-qa` project with the existing
checkpoint, corpus, and environment. The launcher uses the installed Nebius
Python by default. Set `EXPERIMENT_PYTHON` to use another CUDA environment.
The run saves a selected checkpoint, validation history, a test score report,
and per-window predictions under `runs/hill-evidence-first`.

## Targeted continuation and scale gate

Continue from the selected evidence-first checkpoint with 12,000 new paired
examples. Temperature and maintenance contribute 4,000 rows each, and the
other families contribute 1,000 each. Both outcomes stay balanced within each
family. The quantitative QA pool supplies the original windows and checks,
which are rendered as qualitative evidence-first targets. The starting
checkpoint has seen none of these cutoff windows.

Use the original validation panel every 4,000 examples and retain the starting
weights unless a candidate improves the selection score and preserves the
number of complete pairs answered correctly. Use a lower learning rate of
0.00005 for the continuation. If there is a candidate, compare it with the
starting checkpoint on 120 fresh test pairs.

The scale gate requires at least three more complete pairs overall, at least
one more in each of temperature and maintenance, macro-F1 within half a
percentage point of the baseline or higher, and at least 95% compliance with the
evidence-first format. Only after passing that gate, continue for up to 76,000
additional examples, for a 100,000-example total including the original
evidence-first run. Check validation every 8,000 additional examples and stop
after three checks without an improvement. Assess the larger checkpoint on a
second fresh panel using the same promotion criteria. The two panels exclude
windows that overlap the original test panel or each other.

```sh
python prepare_evidence_continuation.py evidence-continuation-data
bash run_evidence_continuation.sh
```

## Balanced replay diagnostic

Run a short continuation from the original evidence-first winner after the
targeted continuation failed. Each optimizer batch contains both outcomes for
every family, with half of its rows drawn from original QA training and half
drawn from the new pool. The run uses 6,000 rows and a learning rate of 0.00002.
Score the existing validation panel near every 1,000 examples and save all its
generated answers, parsed outcomes, invalid-label counts, and family scores.

Advance a checkpoint only if it improves the validation selection score and
complete-pair scores for temperature and maintenance, except that a family
already at the ceiling must stay there. The other family pair scores must
hold or improve. Compare any qualifying checkpoint with the original on the
unused fresh panel before promoting it.

```sh
python prepare_balanced_diagnostic.py balanced-diagnostic-data
bash run_balanced_diagnostic.sh
```

## Time-capped trend training and operator answers

Start from the retained evidence-first checkpoint. Mix the 12,000 original
training rows with 88,000 rows from the previously prepared continuation pools.
Keep opposite outcomes together and interleave every question family and data
source throughout the pass. Only the original training rows are replay for this
starting checkpoint.

Train the model to report the observed trend or state and end with the
outcome label. Keep operator checks out of its prompts and training targets.
Give outcome tokens eight times the weight of the other answer tokens and
continue at a learning rate of 0.000008.

Validate every 6,000 examples and save the generated trends and per-family
scores. Keep the starting checkpoint unless the aggregate validation score
improves, temperature and maintenance improve where possible, and each other
family preserves its complete-pair score. Confirm any qualifying checkpoint on
the unused test panel before promoting it.

After trend prediction, look up the operator check using the question family
and generated outcome. Keep the observation and its check in a separate
operator response. An unsupported outcome leads to a request to verify live
readings and controller state. Run this step on every held-out prediction from
the retained checkpoint and record its answers and abstentions.

Stop optimizer updates around 48 minutes after launch and leave time for
validation and any test comparison. Cap the job at 59 minutes on the VM.
The job leaves the VM running.

```sh
python prepare_large_evidence_qa.py large-evidence-data
python prepare_trend_stage.py large-evidence-data large-trend-data
timeout --kill-after=15s 59m bash run_large_trend.sh
```
