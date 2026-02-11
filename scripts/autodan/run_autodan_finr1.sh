#!/bin/bash
# AutoDAN attack on Fin-R1 (all tasks, GA + HGA)
# Budget: 20 x 12 = 240 per sample
# Requires: 2 GPUs (attacker on cuda:0, target on cuda:1)
# Note: Fin-R1 uses larger max-new-tokens (512) for chain-of-thought output

set -e
cd "$(dirname "$0")/../.."

N_SAMPLES=300

COMMON_PARAMS="--attack-model ../models/Llama-3.1-8B \
  --n-samples $N_SAMPLES \
  --num-steps 12 \
  --batch-size 20 \
  --max-suffix-tokens 30 \
  --mutation-rate 0.15 \
  --ppl-lambda 0.1 \
  --ppl-threshold 50.0 \
  --max-new-tokens 512 \
  --attacker-device cuda:0 \
  --target-device cuda:1"

TASKS="flare_fpb flare_fiqasa flare_ma flare_cra_polish flare_headlines fintrust_fairness"

for TASK in $TASKS; do
  echo "========== AutoDAN-GA: finr1 / ${TASK} =========="
  python -m autodan.attack \
    --target-model finr1 \
    --task "$TASK" \
    --variant ga \
    $COMMON_PARAMS \
    --output-dir ./results/autodan 2>&1 | tee "logs/autodan_ga_finr1_${TASK}.log"

  echo "========== AutoDAN-HGA: finr1 / ${TASK} =========="
  python -m autodan.attack \
    --target-model finr1 \
    --task "$TASK" \
    --variant hga \
    $COMMON_PARAMS \
    --output-dir ./results/autodan 2>&1 | tee "logs/autodan_hga_finr1_${TASK}.log"
done

echo "All Fin-R1 AutoDAN experiments complete!"
