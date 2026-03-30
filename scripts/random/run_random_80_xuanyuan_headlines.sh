#!/bin/bash
# Random Baseline 80: XuanYuan / flare_headlines
# Usage: nohup bash run_random_80_xuanyuan_headlines.sh > ../../logs/random_80_xuanyuan_headlines.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pair_final

mkdir -p result/random_baseline_80 logs

echo "[$(date)] Starting: xuanyuan / flare_headlines / budget=80 / n_samples=300"

python random_baseline_classification.py \
  --target-model xuanyuan \
  --task flare_headlines \
  --n-samples 300 \
  --budget 80 \
  --output-dir result/random_baseline_80

echo "[$(date)] Finished: xuanyuan / flare_headlines"
