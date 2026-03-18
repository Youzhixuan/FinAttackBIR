#!/bin/bash
# Run BIR attack on fintrust_fairness (personal-level) for FinGPT
# Usage: nohup bash run_attack_fingpt_personal.sh > logs/attack_fingpt_personal.log 2>&1 &

set -e

cd "$(dirname "$0")/../.."
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate pair_final

TARGET_MODEL="fingpt"
N_SAMPLES=300
TASK="fintrust_fairness"

ATTACK_PARAMS="--logits-control soft --delta 0.5 --blockwise --block-size 10 --max-suffix-length 30 --block-iterations 4 --n-streams 20"

# Multi-GPU mode (H200 / dual-GPU): attacker on cuda:0, target on cuda:1
ATTACK_PARAMS="$ATTACK_PARAMS --no-offload --attacker-device cuda:0 --target-device cuda:1"
# Single-GPU mode: uncomment below and comment out the line above
# ATTACK_PARAMS="$ATTACK_PARAMS"

echo "=== BIR Attack: ${TARGET_MODEL} / ${TASK} ==="
echo "Samples: $N_SAMPLES"
echo "Params: $ATTACK_PARAMS"
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

python financial_attack_main_classification.py \
    --task $TASK \
    --target-model $TARGET_MODEL \
    --n-samples $N_SAMPLES \
    $ATTACK_PARAMS

echo "[$(date)] Completed: ${TARGET_MODEL} / ${TASK}"
