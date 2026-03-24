#!/bin/bash
# Run AmpleGCG full 160-suffix pipeline sequentially
# Usage: bash scripts/amplegcg/run_amplegcg_all_160.sh

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs results/amplegcg results/amplegcg_160

echo "=============================="
echo "AmpleGCG 160 Full Experiment Suite"
echo "=============================="

echo "[1/5] Generating shared suffix pools"
bash scripts/amplegcg/run_amplegcg_generate_160_all.sh

echo "[2/5] Attacking FinMA"
bash scripts/amplegcg/run_amplegcg_attack_finma_160.sh

echo "[3/5] Attacking XuanYuan"
bash scripts/amplegcg/run_amplegcg_attack_xuanyuan_160.sh

echo "[4/5] Attacking FinGPT"
bash scripts/amplegcg/run_amplegcg_attack_fingpt_160.sh

echo "[5/5] Attacking Fin-R1"
bash scripts/amplegcg/run_amplegcg_attack_finr1_160.sh

echo ""
echo "=============================="
echo "All AmpleGCG 160 experiments done!"
echo "=============================="
echo "Suffixes in: results/amplegcg_160/"
echo "Results in: results/amplegcg/"
echo "Logs in: logs/amplegcg_160/"
