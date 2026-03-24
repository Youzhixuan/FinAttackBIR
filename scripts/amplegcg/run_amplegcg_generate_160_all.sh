#!/bin/bash
# Generate shared AmpleGCG 160-suffix pools for all tasks
# Usage: nohup bash run_amplegcg_generate_160_all.sh > ../../logs/amplegcg_generate_160_all.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pair_final

NUM_SUFFIXES=160
GENERATE_BATCH_SIZE=160
N_SAMPLES=0
PROMPTER_MODEL="../models/AmpleGCG"
SUFFIX_DIR="./results/amplegcg_160"
LOG_DIR="./logs/amplegcg_160"
TASKS="flare_fpb flare_fiqasa flare_ma flare_cra_polish flare_headlines fintrust_fairness"
COMMON_PARAMS="--generate-only --merge-pools --num-suffixes $NUM_SUFFIXES --generate-batch-size $GENERATE_BATCH_SIZE --prompter-model $PROMPTER_MODEL --suffix-dir $SUFFIX_DIR"

mkdir -p "$SUFFIX_DIR" "$LOG_DIR"

echo "=== Starting AmpleGCG suffix generation (shared 160-suffix pools) ==="
echo "Tasks: $TASKS"
echo "Suffix dir: $SUFFIX_DIR"
echo "Log dir: $LOG_DIR"
echo "Prompter model: $PROMPTER_MODEL"
echo ""

for TASK in $TASKS; do
    echo "=================================================="
    echo "[$(date)] Starting suffix generation: $TASK"
    echo "=================================================="

    python -m amplegcg.attack \
        --task "$TASK" \
        --n-samples "$N_SAMPLES" \
        $COMMON_PARAMS \
        2>&1 | tee "$LOG_DIR/generate_${TASK}.log"

    echo "[$(date)] Completed suffix generation: $TASK"
    echo ""
done

echo "=== All AmpleGCG suffix generation complete ==="
echo "Suffixes saved under: $SUFFIX_DIR"
