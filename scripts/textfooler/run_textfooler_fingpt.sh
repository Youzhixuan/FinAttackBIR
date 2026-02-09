#!/bin/bash
# TextFooler attack on FinGPT (all tasks)
# Budget: 30 (importance) + 30*7 (replacement) = 240 per sample
# Requires: 2 GPUs (attacker on cuda:0, target on cuda:1)

set -e
cd "$(dirname "$0")/../.."

COMMON_PARAMS="--attack-model ../models/Llama-3.1-8B \
  --max-suffix-tokens 30 \
  --candidates-per-pos 7 \
  --attacker-device cuda:0 \
  --target-device cuda:1"

TASKS="flare_fpb flare_fiqasa flare_ma flare_cra_polish flare_headlines fintrust_fairness"

for TASK in $TASKS; do
  echo "========== TextFooler: fingpt / ${TASK} =========="
  python -m textfooler.attack \
    --target-model fingpt \
    --task "$TASK" \
    $COMMON_PARAMS \
    --output-dir ./results/textfooler 2>&1 | tee "logs/textfooler_fingpt_${TASK}.log"
done

echo "All FinGPT TextFooler experiments complete!"
