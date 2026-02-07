#!/bin/bash
# Run attack experiments for FinGPT
# Usage: nohup bash run_attack_fingpt.sh > logs/attack_fingpt.log 2>&1 &

set -e

cd "$(dirname "$0")/../.."

# Configuration
TARGET_MODEL="fingpt"
N_SAMPLES=300
TASKS="flare_ma flare_fiqasa flare_fpb flare_headlines flare_cra_polish fintrust_fairness"

# Common attack parameters
ATTACK_PARAMS="--logits-control soft --delta 0.5 --blockwise --block-size 10 --max-suffix-length 30 --block-iterations 4 --n-streams 20"

echo "=== Starting Attack Experiments for $TARGET_MODEL ==="
echo "Tasks: $TASKS"
echo "Samples per task: $N_SAMPLES"
echo ""

# Run each task sequentially
for TASK in $TASKS; do
    echo "=================================================="
    echo "[$(date)] Starting task: $TASK"
    echo "=================================================="
    
    # Check if attack pool exists
    POOL_FILE="data/attack_pools/${TARGET_MODEL}/${TASK}.jsonl"
    if [ ! -f "$POOL_FILE" ]; then
        echo "[WARNING] Attack pool not found: $POOL_FILE, skipping..."
        continue
    fi
    
    # Get pool size
    POOL_SIZE=$(wc -l < "$POOL_FILE")
    echo "[INFO] Attack pool size: $POOL_SIZE samples"
    
    if [ "$POOL_SIZE" -eq 0 ]; then
        echo "[WARNING] Empty attack pool, skipping..."
        continue
    fi
    
    # Run attack
    python financial_attack_main_classification.py \
        --task $TASK \
        --target-model $TARGET_MODEL \
        --n-samples $N_SAMPLES \
        $ATTACK_PARAMS
    
    echo "[$(date)] Completed task: $TASK"
    echo ""
done

echo "=== All Attack Experiments Complete ==="
