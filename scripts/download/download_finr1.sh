#!/bin/bash
# Download Fin-R1 model
# Usage: bash download_finr1.sh

set -e

MODEL_DIR="../models"
MODEL_NAME="Fin-R1"
HF_MODEL="SUFE-AIFLM-Lab/Fin-R1"

mkdir -p $MODEL_DIR

echo "=== Downloading Fin-R1 ==="
echo "Target: $MODEL_DIR/$MODEL_NAME"

# Use hf-mirror for China users
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download $HF_MODEL --local-dir $MODEL_DIR/$MODEL_NAME --local-dir-use-symlinks False

echo "=== Download Complete ==="
echo "Model saved to: $MODEL_DIR/$MODEL_NAME"
