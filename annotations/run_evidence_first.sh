#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${EXPERIMENT_PYTHON:-$HOME/hot-paired-96/.venv/bin/python}"
data="$scripts/paired-qa-evidence-first"
output="runs/hill-evidence-first"
warmup="runs/hill-curriculum-20k/warmup.pt"
previous="runs/hill-paired-qa/best.pt"

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$scripts:$PWD:$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"

for path in "$warmup" "$previous" "$data/qa.parquet" "$data/validation.parquet" "$data/test.parquet"; do
    test -s "$path"
done

"$python" "$scripts/train_paired_early_stop.py" \
    --evidence-first --data "$data" --initial "$warmup" --output "$output" \
    --validate-every 4000 --patience 3
"$python" "$scripts/evaluate_evidence_first.py" \
    corpus "$data" "$previous" "$output/best.pt" "$output"
