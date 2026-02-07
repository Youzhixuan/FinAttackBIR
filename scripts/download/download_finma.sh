#!/bin/bash
# Download FinMA-7B model
# Usage: bash download_finma.sh

set -e

MODEL_DIR="../models"
MODEL_NAME="finma-7b-full"
HF_MODEL="TheFinAI/finma-7b-full"

mkdir -p $MODEL_DIR

echo "=== Downloading FinMA-7B ==="
echo "Target: $MODEL_DIR/$MODEL_NAME"

# Use hf-mirror for China users
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download $HF_MODEL --local-dir $MODEL_DIR/$MODEL_NAME --local-dir-use-symlinks False

echo "=== Download Complete ==="
echo "Model saved to: $MODEL_DIR/$MODEL_NAME"
