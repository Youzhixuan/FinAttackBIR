#!/bin/bash
# Download XuanYuan-6B model
# Usage: bash download_xuanyuan.sh

set -e

MODEL_DIR="../models"
MODEL_NAME="XuanYuan-6B"
HF_MODEL="Duxiaoman-DI/XuanYuan-6B"

mkdir -p $MODEL_DIR

echo "=== Downloading XuanYuan-6B ==="
echo "Target: $MODEL_DIR/$MODEL_NAME"

# Use hf-mirror for China users
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download $HF_MODEL --local-dir $MODEL_DIR/$MODEL_NAME --local-dir-use-symlinks False

echo "=== Download Complete ==="
echo "Model saved to: $MODEL_DIR/$MODEL_NAME"
