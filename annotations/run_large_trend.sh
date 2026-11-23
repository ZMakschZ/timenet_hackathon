#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${EXPERIMENT_PYTHON:-$HOME/hot-paired-96/.venv/bin/python}"
data="$scripts/large-trend-data"
initial="$PWD/runs/hill-evidence-first/best.pt"
output="$PWD/runs/hill-large-trend"

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$scripts:$PWD:$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)"
if (( free_mib < 43008 )); then
    echo "Insufficient free GPU memory for the shared run."
    exit 1
fi
test -s "$initial"
scratch="$(mktemp -d /dev/shm/hill-large-trend.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
"$python" "$scripts/train_paired_early_stop.py" --evidence-first --retain-baseline \
    --balanced-diagnostic --data "$data" --initial "$initial" --output "$output" \
    --checkpoint-dir "$scratch" \
    --batch 4 --accumulate 3 --lr 0.000008 --label-weight 8 \
    --validate-every 6000 --patience 32 --max-minutes 48

selected="$initial"
prediction=previous_signals
if [[ -s "$scratch/best.pt" ]]; then
    "$python" "$scripts/evaluate_evidence_first.py" corpus "$data" \
        "$initial" "$scratch/best.pt" "$output" --signals-only
    "$python" "$scripts/evidence_continuation_gate.py" "$output"
    if "$python" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$output/gate.json"; then
        cp "$scratch/best.pt" "$output/best.pt.tmp"
        mv "$output/"{best.pt.tmp,best.pt}
        selected="$output/best.pt"
        prediction=evidence_first_signals
    fi
else
    "$python" "$scripts/evaluate_evidence_first.py" corpus "$data" \
        "$initial" "$initial" "$output" --signals-only --baseline-only
fi
"$python" "$scripts/render_operator_answers.py" "$data/operator-actions.json" \
    "$output/$prediction.jsonl" "$output/operator-answers.jsonl"
printf '%s\n' "$selected" > "$output/selected-checkpoint.txt"
