#!/bin/bash
# Run Momentum-GCG white-box attack on ALL target models sequentially.
# modified by yzx
# Requires: 1 GPU (H200 80GB recommended)
# Estimated total: ~18 hours on H200

set -e
SCRIPT_DIR="$(dirname "$0")"

echo "===== Starting Momentum-GCG experiments for ALL models ====="
echo ""

echo ">>> [1/4] FinMA-7B (vocab=32K, ~2h)"
bash "$SCRIPT_DIR/run_momentum_gcg_finma.sh"
echo ""

echo ">>> [2/4] XuanYuan-6B (vocab=39K, ~6h)"
bash "$SCRIPT_DIR/run_momentum_gcg_xuanyuan.sh"
echo ""

echo ">>> [3/4] FinGPT (vocab=128K, ~6h)"
bash "$SCRIPT_DIR/run_momentum_gcg_fingpt.sh"
echo ""

echo ">>> [4/4] Fin-R1 (vocab=151K, ~4h)"
bash "$SCRIPT_DIR/run_momentum_gcg_finr1.sh"
echo ""

echo "===== ALL Momentum-GCG experiments complete! ====="
