#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${EXPERIMENT_PYTHON:-$HOME/hot-paired-96/.venv/bin/python}"
data="$scripts/evidence-continuation-data"
run="$PWD/runs/hill-evidence-continuation"
initial="$PWD/runs/hill-evidence-first/best.pt"

export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$scripts:$PWD:$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"

test -s "$initial"
mkdir "$run"
scratch="$(mktemp -d /dev/shm/hill-evidence-continuation.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT

train_stage() {
    local name="$1" checkpoint="$2" interval="$3"
    mkdir "$run/$name"
    "$python" "$scripts/train_paired_early_stop.py" --evidence-first --retain-baseline \
        --data "$data/$name" --initial "$checkpoint" --output "$scratch/$name" \
        --lr 0.00005 --validate-every "$interval" --patience 3
    cp "$scratch/$name/"{metrics.jsonl,result.json} "$run/$name/"
}

evaluate_stage() {
    local name="$1" checkpoint="$2"
    "$python" "$scripts/evaluate_evidence_first.py" corpus "$data/$name" \
        "$checkpoint" "$scratch/$name/best.pt" "$run/$name" --signals-only
    "$python" "$scripts/evidence_continuation_gate.py" "$run/$name"
}

passed() {
    "$python" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' \
        "$run/$1/gate.json"
}

echo "Waiting for any existing paired QA GPU run to finish."
while tmux has-session -t =hill-paired-qa-100k-patience5 2>/dev/null; do
    sleep 15
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
    echo "GPU is occupied by another process."
    exit 1
fi

train_stage focused "$initial" 4000
if [[ ! -s "$scratch/focused/best.pt" ]]; then
    echo "Focused continuation did not improve validation. Keeping the initial checkpoint."
    printf '%s\n' "$initial" > "$run/selected-checkpoint.txt"
    exit 0
fi

evaluate_stage focused "$initial"
if ! passed focused; then
    echo "Focused continuation did not pass the holdout gate. Keeping the initial checkpoint."
    printf '%s\n' "$initial" > "$run/selected-checkpoint.txt"
    exit 0
fi

cp "$scratch/focused/best.pt" "$run/focused/best.pt.tmp"
mv "$run/focused/"{best.pt.tmp,best.pt}
printf '%s\n' "$run/focused/best.pt" > "$run/selected-checkpoint.txt"
train_stage scale "$run/focused/best.pt" 8000
if [[ -s "$scratch/scale/best.pt" ]]; then
    evaluate_stage scale "$run/focused/best.pt"
    if passed scale; then
        cp "$scratch/scale/best.pt" "$run/scale/best.pt.tmp"
        mv "$run/scale/"{best.pt.tmp,best.pt}
        printf '%s\n' "$run/scale/best.pt" > "$run/selected-checkpoint.txt"
    fi
fi
