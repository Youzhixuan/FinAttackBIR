#!/bin/bash
# Run AmpleGCG attack on Fin-R1 using shared 160-suffix pools
# Usage: nohup bash run_amplegcg_attack_finr1_160.sh > ../../logs/amplegcg_attack_finr1_160.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pair_final

TARGET_MODEL="finr1"
NUM_SUFFIXES=160
ATTACK_BATCH_SIZE=24
SUFFIX_DIR="./results/amplegcg_160"
OUTPUT_DIR="./results/amplegcg"
LOG_DIR="./logs/amplegcg_160"
TASKS=(
  "flare_fpb 300"
  "flare_fiqasa 96"
  "flare_ma 300"
  "flare_cra_polish 300"
  "flare_headlines 300"
  "fintrust_fairness 300"
)

mkdir -p "$OUTPUT_DIR" "$SUFFIX_DIR" "$LOG_DIR"

echo "=== Starting AmpleGCG attacks for $TARGET_MODEL ==="
echo "Suffix dir: $SUFFIX_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"
echo ""

for ITEM in "${TASKS[@]}"; do
    read -r TASK N_SAMPLES <<< "$ITEM"
    SUFFIX_FILE="$SUFFIX_DIR/suffixes_${TASK}_${NUM_SUFFIXES}.json"

    echo "=================================================="
    echo "[$(date)] Starting attack: $TARGET_MODEL / $TASK / n_samples=$N_SAMPLES"
    echo "=================================================="

    if [ ! -f "$SUFFIX_FILE" ]; then
        echo "[ERROR] Missing suffix file: $SUFFIX_FILE"
        echo "Please run scripts/amplegcg/run_amplegcg_generate_160_all.sh first."
        exit 1
    fi

    python -m amplegcg.attack \
        --target-model "$TARGET_MODEL" \
        --task "$TASK" \
        --n-samples "$N_SAMPLES" \
        --num-suffixes "$NUM_SUFFIXES" \
        --attack-batch-size "$ATTACK_BATCH_SIZE" \
        --suffix-dir "$SUFFIX_DIR" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/${TARGET_MODEL}_${TASK}.log"

    echo "[$(date)] Completed attack: $TARGET_MODEL / $TASK"
    echo ""
done

echo "=== All AmpleGCG attacks complete for $TARGET_MODEL ==="
echo "Results saved under: $OUTPUT_DIR"
