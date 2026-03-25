#!/bin/bash
# Random Baseline 160: Fin-R1 / flare_fiqasa
# Usage: nohup bash run_random_160_finr1_fiqasa.sh > ../../logs/random_160_finr1_fiqasa.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pair_final

mkdir -p result/random_baseline_160 logs/random_baseline_160

echo "[$(date)] Starting: finr1 / flare_fiqasa / n_samples=96"

python random_baseline_classification.py \
  --target-model finr1 \
  --task flare_fiqasa \
  --n-samples 96 \
  --output-dir result/random_baseline_160

echo "[$(date)] Finished: finr1 / flare_fiqasa"
