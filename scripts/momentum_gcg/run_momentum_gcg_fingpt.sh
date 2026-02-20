#!/bin/bash
# Momentum-GCG white-box attack on FinGPT (all tasks)
# modified by yzx
# Requires: 1 GPU (white-box attack, no separate attacker model)
# FinGPT vocab=128K, needs smaller batch than 32K-vocab models

set -e
cd "$(dirname "$0")/../.."

N_SAMPLES=300

# H200 80GB 推荐参数；如在较小显存 GPU 上运行，改为 --batch-size 8 --fwd-batch-size 1
COMMON_PARAMS="--n-samples $N_SAMPLES \
  --steps 100 \
  --batch-size 128 \
  --fwd-batch-size 64 \
  --topk 256 \
  --mu 0.4 \
  --eval-interval 10 \
  --output-dir ./results/momentum_gcg"

TASKS="flare_fpb flare_fiqasa flare_ma flare_cra_polish flare_headlines fintrust_fairness"

for TASK in $TASKS; do
  echo "========== Momentum-GCG: fingpt / ${TASK} =========="
  python -m momentum_gcg.attack \
    --target-model fingpt \
    --task "$TASK" \
    $COMMON_PARAMS 2>&1 | tee "logs/momentum_gcg_fingpt_${TASK}.log"
done

echo "All FinGPT Momentum-GCG experiments complete!"
