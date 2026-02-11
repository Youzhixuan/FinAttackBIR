#!/bin/bash
# AutoDAN attack on XuanYuan-6B (all tasks, GA + HGA)
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
  echo "========== AutoDAN-GA: xuanyuan / ${TASK} =========="
  python -m autodan.attack \
    --target-model xuanyuan \
    --task "$TASK" \
    --variant ga \
    $COMMON_PARAMS \
    --output-dir ./results/autodan 2>&1 | tee "logs/autodan_ga_xuanyuan_${TASK}.log"

  echo "========== AutoDAN-HGA: xuanyuan / ${TASK} =========="
  python -m autodan.attack \
    --target-model xuanyuan \
    --task "$TASK" \
    --variant hga \
    $COMMON_PARAMS \
    --output-dir ./results/autodan 2>&1 | tee "logs/autodan_hga_xuanyuan_${TASK}.log"
done

echo "All XuanYuan AutoDAN experiments complete!"
