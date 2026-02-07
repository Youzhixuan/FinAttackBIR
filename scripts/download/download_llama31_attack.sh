#!/bin/bash
# Download Llama-3.1-8B (Attack Model)
# Usage: bash download_llama31_attack.sh

set -e

MODEL_DIR="../models"
MODEL_NAME="Llama-3.1-8B"
HF_MODEL="meta-llama/Llama-3.1-8B"

mkdir -p $MODEL_DIR

echo "=== Downloading Llama-3.1-8B (Attack Model) ==="
echo "Target: $MODEL_DIR/$MODEL_NAME"

# Use hf-mirror for China users
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download $HF_MODEL --local-dir $MODEL_DIR/$MODEL_NAME --local-dir-use-symlinks False

echo "=== Download Complete ==="
echo "Model saved to: $MODEL_DIR/$MODEL_NAME"
