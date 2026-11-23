#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${EXPERIMENT_PYTHON:-$HOME/hot-paired-96/.venv/bin/python}"
data="$scripts/balanced-diagnostic-data"
initial="$PWD/runs/hill-evidence-first/best.pt"
output="$PWD/runs/hill-balanced-diagnostic"

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$scripts:$PWD:$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
    echo "GPU is occupied by another process."
    exit 1
fi
test -s "$initial"
"$python" "$scripts/train_paired_early_stop.py" --evidence-first --retain-baseline \
    --balanced-diagnostic --data "$data" --initial "$initial" --output "$output" \
    --batch 4 --accumulate 3 --lr 0.00002 --validate-every 1000 --patience 6

if [[ -s "$output/best.pt" ]]; then
    "$python" "$scripts/evaluate_evidence_first.py" corpus "$data" \
        "$initial" "$output/best.pt" "$output" --signals-only
    "$python" "$scripts/evidence_continuation_gate.py" "$output"
fi
