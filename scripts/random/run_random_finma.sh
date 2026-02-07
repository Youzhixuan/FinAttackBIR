#!/bin/bash
# Run random baseline experiments for FinMA-7B
# Usage: nohup bash run_random_finma.sh > logs/random_finma.log 2>&1 &

set -e

cd "$(dirname "$0")/../.."

# Configuration
TARGET_MODEL="finma"
N_SAMPLES=300
TASKS="flare_ma flare_fiqasa flare_fpb flare_headlines flare_cra_polish fintrust_fairness"

echo "=== Starting Random Baseline for $TARGET_MODEL ==="
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
    
    # Run random baseline
    python random_baseline_classification.py \
        --task $TASK \
        --target-model $TARGET_MODEL \
        --n-samples $N_SAMPLES
    
    echo "[$(date)] Completed task: $TASK"
    echo ""
done

echo "=== All Random Baseline Experiments Complete ==="
