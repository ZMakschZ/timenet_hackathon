# Mixed operator QA continuation

The paired checkpoint generalized to focused questions, but the earlier open-ended
test exposed false idle calls when telemetry was missing and inaccurate numeric
claims. This continuation starts from that checkpoint and mixes new open-ended
operator QA with focused QA from the larger historical pool. Both tasks use the
same 20 SCADA channels over 36 ten-minute steps and their existing validity masks.

The training pool has 60,000 operator QA rows and 40,000 focused QA rows from the
chronological training split. The operator examples cover all 21 turbines and
105 months. Targets describe reported generation, idle conditions, mixed evidence,
missing visibility and invalid readings. They give a verification step and describe
what evidence could change the assessment. The 45 distinct examples of unknown
state with some telemetry present each appear 12 times across three prompt variants.
The 4,507 unavailable-visibility samples come from observed gaps. See the
[data audit](data.json) for the full distribution.

Training mixes the two tasks in a 3:2 ratio, with an effective batch size of eight
and a peak learning rate of 0.00006. The runner starts at the selected paired QA
weights. It scores generated answers every 9.5 minutes and near its 50-minute limit.
To replace the initial checkpoint, a candidate must improve the number of broad
answers with a correct state and verification language by at least two rows,
preserve the number of correct unknown or missing cases, and stay within three
of the initial 36-pair panel score. The end check uses the original 63-window
operator transfer panel, which is excluded from training and selection.

The verification-language score is a screening metric. Inspect generated responses
against the raw SCADA and the reference answers for false measurements, unsupported
operating changes, and whether the suggested verification would resolve the uncertainty.

```sh
python build_operator_continuation.py corpus runs/hill-operator-continuation/data paired-qa-100k/qa.parquet
python train_operator_continuation.py \
  --corpus corpus --data runs/hill-operator-continuation/data \
  --initial runs/hill-paired-qa/best.pt --scaling paired-qa/scaling.json \
  --paired-validation runs/hill-checkpoint-comparison/validation.parquet \
  --transfer runs/hill-checkpoint-comparison/operator-transfer.parquet \
  --output runs/hill-operator-continuation --weights runs/hill-operator-continuation \
  --minutes 50 --check-minutes 9.5
```

## Results

The continuation ran for 51.05 minutes including validation and the open-ended
transfer check. It processed 50,112 examples and selected the weights at 30,056
examples. The selected weights passed the broad state-and-verification screen on
61/70 validation windows, up from 15/70. Focused validation rose from 30/36 to
31/36 complete pairs. See the [run result](result.json) and [validation checks](metrics.jsonl).

| Same open-ended test windows | Starting checkpoint | Mixed checkpoint |
| --- | ---: | ---: |
| Correct state with verification language | 13/63 | 56/63 |
| Recognized unavailable telemetry | 0/12 | 12/12 |
| Correctly characterized partial telemetry with unknown state | 0/3 | 0/3 |

For T01 at 20:00 UTC on 15 September 2025, the starting checkpoint called the
turbine idle and supplied readings. The mixed checkpoint said that the operating
state was unknown, identified the telemetry gap and asked for an independent
status. On all three partial-visibility test cases, it expressed uncertainty
but described a generic telemetry gap. These answers missed which signals still
reported and which channels prevented a reliable diagnosis. Four of 16 mixed or
transitional windows were still misclassified. The test is stratified and its
scores describe these 63 windows.

The selected weights were copied out of the VM's temporary storage. The local
checkpoint is `runs/hill-operator-continuation/best.pt`. The
[SHA-256 digest](checkpoint.sha256) matches the VM copy.

### Focused test and parallel run

On the same 120 focused test pairs as the original checkpoint, the mixed model
got both outcomes right on 97 pairs versus 101 for the original. Individual
outcomes were correct on 216/240 windows versus 220/240. It recovered 132/400
reference measurements, compared with 131/400, and 160/240 answers contained at
least one parsed measurement the scorer could not verify. The mixed checkpoint
works better for open-ended handovers on this panel. The original checkpoint
still leads on the focused benchmark, and quantitative evidence remains
unreliable for both. [Focused test scores](paired-test.json)

The parallel trend experiment reached its time budget after 49,932 examples and
retained its starting evidence-first checkpoint. It saved its result and final
test report before shutdown.
