#!/bin/bash
# AutoDAN attack on FinGPT (all tasks, GA + HGA)
# Budget: 20 x 12 = 240 per sample
# Requires: 2 GPUs (attacker on cuda:0, target on cuda:1)

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
  --attacker-device cuda:0 \
  --target-device cuda:1"

TASKS="flare_fpb flare_fiqasa flare_ma flare_cra_polish flare_headlines fintrust_fairness"

for TASK in $TASKS; do
  echo "========== AutoDAN-GA: fingpt / ${TASK} =========="
  python -m autodan.attack \
    --target-model fingpt \
    --task "$TASK" \
    --variant ga \
    $COMMON_PARAMS \
    --output-dir ./results/autodan 2>&1 | tee "logs/autodan_ga_fingpt_${TASK}.log"

  echo "========== AutoDAN-HGA: fingpt / ${TASK} =========="
  python -m autodan.attack \
    --target-model fingpt \
    --task "$TASK" \
    --variant hga \
    $COMMON_PARAMS \
    --output-dir ./results/autodan 2>&1 | tee "logs/autodan_hga_fingpt_${TASK}.log"
done

echo "All FinGPT AutoDAN experiments complete!"
