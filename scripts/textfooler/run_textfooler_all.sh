#!/bin/bash
# Run TextFooler attack on ALL target models sequentially.
# Budget: 240 per sample (30 importance + 30*7 replacement)
# Requires: 2 GPUs (attacker on cuda:0, target on cuda:1)

set -e
SCRIPT_DIR="$(dirname "$0")"

echo "===== Starting TextFooler experiments for ALL models ====="
echo ""

echo ">>> [1/4] FinMA-7B"
bash "$SCRIPT_DIR/run_textfooler_finma.sh"
echo ""

echo ">>> [2/4] XuanYuan-6B"
bash "$SCRIPT_DIR/run_textfooler_xuanyuan.sh"
echo ""

echo ">>> [3/4] FinGPT"
bash "$SCRIPT_DIR/run_textfooler_fingpt.sh"
echo ""

echo ">>> [4/4] Fin-R1"
bash "$SCRIPT_DIR/run_textfooler_finr1.sh"
echo ""

echo "===== ALL TextFooler experiments complete! ====="
