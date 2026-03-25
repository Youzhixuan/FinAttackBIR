#!/bin/bash
# Random Baseline 160: XuanYuan / flare_headlines
# Usage: nohup bash run_random_160_xuanyuan_headlines.sh > ../../logs/random_160_xuanyuan_headlines.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pair_final

mkdir -p result/random_baseline_160 logs/random_baseline_160

echo "[$(date)] Starting: xuanyuan / flare_headlines / n_samples=300"

python random_baseline_classification.py \
  --target-model xuanyuan \
  --task flare_headlines \
  --n-samples 300 \
  --output-dir result/random_baseline_160

echo "[$(date)] Finished: xuanyuan / flare_headlines"
