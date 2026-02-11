#!/bin/bash
# TextFooler attack on XuanYuan-6B (all tasks)
# Budget: 30 (importance) + 30*7 (replacement) = 240 per sample
# Requires: 2 GPUs (attacker on cuda:0, target on cuda:1)

set -e
cd "$(dirname "$0")/../.."

N_SAMPLES=300

COMMON_PARAMS="--attack-model ../models/Llama-3.1-8B \
  --n-samples $N_SAMPLES \
  --max-suffix-tokens 30 \
  --candidates-per-pos 7 \
  --attacker-device cuda:0 \
  --target-device cuda:1"

TASKS="flare_fpb flare_fiqasa flare_ma flare_cra_polish flare_headlines fintrust_fairness"

for TASK in $TASKS; do
  echo "========== TextFooler: xuanyuan / ${TASK} =========="
  python -m textfooler.attack \
    --target-model xuanyuan \
    --task "$TASK" \
    $COMMON_PARAMS \
    --output-dir ./results/textfooler 2>&1 | tee "logs/textfooler_xuanyuan_${TASK}.log"
done

echo "All XuanYuan TextFooler experiments complete!"
