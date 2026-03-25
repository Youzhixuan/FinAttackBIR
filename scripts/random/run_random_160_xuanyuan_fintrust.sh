#!/bin/bash
# Random Baseline 160: XuanYuan / fintrust_fairness
# Usage: nohup bash run_random_160_xuanyuan_fintrust.sh > ../../logs/random_160_xuanyuan_fintrust.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pair_final

mkdir -p result/random_baseline_160 logs/random_baseline_160

echo "[$(date)] Starting: xuanyuan / fintrust_fairness / n_samples=129"

python random_baseline_classification.py \
  --target-model xuanyuan \
  --task fintrust_fairness \
  --n-samples 129 \
  --output-dir result/random_baseline_160

echo "[$(date)] Finished: xuanyuan / fintrust_fairness"
