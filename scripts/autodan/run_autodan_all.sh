#!/bin/bash
# Run AutoDAN attacks on ALL models sequentially
# Usage: bash scripts/autodan/run_autodan_all.sh

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs results/autodan

echo "=============================="
echo "AutoDAN Full Experiment Suite"
echo "=============================="

bash scripts/autodan/run_autodan_finma.sh
bash scripts/autodan/run_autodan_xuanyuan.sh
bash scripts/autodan/run_autodan_fingpt.sh
bash scripts/autodan/run_autodan_finr1.sh

echo ""
echo "=============================="
echo "All AutoDAN experiments done!"
echo "=============================="
echo "Results in: results/autodan/"
echo "Logs in: logs/"
