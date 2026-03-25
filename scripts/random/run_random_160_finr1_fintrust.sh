#!/bin/bash
# Random Baseline 160: Fin-R1 / fintrust_fairness
# Usage: nohup bash run_random_160_finr1_fintrust.sh > ../../logs/random_160_finr1_fintrust.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pair_final

mkdir -p result/random_baseline_160 logs/random_baseline_160

echo "[$(date)] Starting: finr1 / fintrust_fairness / n_samples=300"

python random_baseline_classification.py \
  --target-model finr1 \
  --task fintrust_fairness \
  --n-samples 300 \
  --output-dir result/random_baseline_160

echo "[$(date)] Finished: finr1 / fintrust_fairness"
