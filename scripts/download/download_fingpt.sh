#!/bin/bash
# Download FinGPT models (Base + LoRA)
# Usage: bash download_fingpt.sh

set -e

MODEL_DIR="../models"

mkdir -p $MODEL_DIR

# Use hf-mirror for China users
export HF_ENDPOINT=https://hf-mirror.com

echo "=== Downloading Meta-Llama-3-8B (Base Model) ==="
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir $MODEL_DIR/Meta-Llama-3-8B --local-dir-use-symlinks False

echo "=== Downloading FinGPT LoRA Adapter ==="
huggingface-cli download FinGPT/fingpt-mt_llama3-8b_lora --local-dir $MODEL_DIR/fingpt-mt_llama3-8b_lora --local-dir-use-symlinks False

echo "=== Download Complete ==="
echo "Base model: $MODEL_DIR/Meta-Llama-3-8B"
echo "LoRA adapter: $MODEL_DIR/fingpt-mt_llama3-8b_lora"
