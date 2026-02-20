#!/bin/bash
# Momentum-GCG white-box attack on Fin-R1 (all tasks)
# modified by yzx
# Requires: 1 GPU (white-box attack, no separate attacker model)
# Fin-R1 vocab=151K (largest), needs smallest batch

set -e
cd "$(dirname "$0")/../.."

N_SAMPLES=300

# H200 80GB 推荐参数；如在较小显存 GPU 上运行，改为 --batch-size 4 --fwd-batch-size 1
COMMON_PARAMS="--n-samples $N_SAMPLES \
  --steps 100 \
  --batch-size 64 \
  --fwd-batch-size 32 \
  --topk 256 \
  --mu 0.4 \
  --eval-interval 10 \
  --output-dir ./results/momentum_gcg"

TASKS="flare_fpb flare_fiqasa flare_ma flare_cra_polish flare_headlines fintrust_fairness"

for TASK in $TASKS; do
  echo "========== Momentum-GCG: finr1 / ${TASK} =========="
  python -m momentum_gcg.attack \
    --target-model finr1 \
    --task "$TASK" \
    $COMMON_PARAMS 2>&1 | tee "logs/momentum_gcg_finr1_${TASK}.log"
done

echo "All Fin-R1 Momentum-GCG experiments complete!"
