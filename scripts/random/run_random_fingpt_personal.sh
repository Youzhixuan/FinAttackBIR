#!/bin/bash
# Run random baseline on fintrust_fairness (personal-level) for FinGPT
# Usage: nohup bash run_random_fingpt_personal.sh > logs/random_fingpt_personal.log 2>&1 &

set -e

cd "$(dirname "$0")/../.."
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate pair_final

TARGET_MODEL="fingpt"
N_SAMPLES=300
TASK="fintrust_fairness"

echo "=== Random Baseline: ${TARGET_MODEL} / ${TASK} ==="
echo "Samples: $N_SAMPLES"
echo ""

POOL_FILE="data/attack_pools/${TARGET_MODEL}/${TASK}.jsonl"
if [ ! -f "$POOL_FILE" ]; then
    echo "[WARNING] Attack pool not found: $POOL_FILE, skipping..."
    exit 1
fi

POOL_SIZE=$(wc -l < "$POOL_FILE")
echo "[INFO] Attack pool size: $POOL_SIZE samples"

if [ "$POOL_SIZE" -eq 0 ]; then
    echo "[WARNING] Empty attack pool, exiting..."
    exit 1
fi

echo "[$(date)] Starting task: $TASK"

python random_baseline_classification.py \
    --task $TASK \
    --target-model $TARGET_MODEL \
    --n-samples $N_SAMPLES

echo "[$(date)] Completed: ${TARGET_MODEL} / ${TASK}"
