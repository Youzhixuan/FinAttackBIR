# FinancialAdversarialAttack

Adversarial attacks on Financial Large Language Models (FinLLMs) for classification tasks.

## Overview

This project implements adversarial suffix attacks against financial LLMs, evaluating their robustness on sentiment analysis, financial classification, and decision-making tasks.

**Target Models:**
- FinMA-7B (TheFinAI/finma-7b-full)
- FinGPT (Meta-Llama-3-8B + fingpt-mt_llama3-8b_lora)
- XuanYuan-6B (Duxiaoman-DI/XuanYuan-6B)
- Fin-R1 (SUFE-AIFLM-Lab/Fin-R1)

**Attack Model:**
- Llama-3.1-8B (meta-llama/Llama-3.1-8B)

**Tasks:**
- FinBen: flare_fpb, flare_fiqasa, flare_headlines, flare_ma, flare_cra_polish
- FinTrust: fintrust_fairness (personal-level)

## Directory Structure

```
FinancialAdversarialAttack/
├── data/
│   └── attack_pools/           # Pre-built attack sample pools
│       ├── finma/              # FinMA-7B attack pools
│       ├── fingpt/             # FinGPT attack pools
│       ├── xuanyuan/           # XuanYuan-6B attack pools
│       └── finr1/              # Fin-R1 attack pools
├── scripts/
│   ├── download/               # Model download scripts
│   ├── attack/                 # Attack experiment scripts
│   └── random/                 # Random baseline scripts
├── logs/                       # Experiment logs (gitignored)
├── results/                    # Experiment results (gitignored)
├── financial_attack_main_classification.py   # Main attack script
├── random_baseline_classification.py         # Random baseline
├── build_attack_pool.py        # Build attack pools for new models
├── task_prompts.py             # Task definitions and prompt handling
├── conversers.py               # Target model implementations
├── judges.py                   # Classification judge
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n finattack python=3.10 -y
conda activate finattack
pip install -r requirements.txt
pip install datasets peft
```

Key dependencies: PyTorch 2.1+, Transformers 4.43+, PEFT 0.5+ (for FinGPT LoRA). Two GPUs with 24 GB+ VRAM each are recommended (attacker and target models reside on separate GPUs). Single-GPU mode is also supported via CPU offloading (remove `--no-offload` from scripts).

### 2. Download Models

```bash
cd scripts/download

# Download target model (choose one)
bash download_finma.sh # finma跑完了不用下
bash download_fingpt.sh # fingpt跑完了不用下
bash download_xuanyuan.sh  # 下这个
bash download_finr1.sh

# Download attack model
bash download_llama31_attack.sh # 下这个
```

Models will be saved to `../models/`.

### 3. Run Experiments

Attack pools are pre-built and included. Choose experiment type:

**Option A: Our Attack (BIR)**
```bash
# 跑xuanyuan ours实验 -- 0207
cd scripts/attack
nohup bash run_attack_xuanyuan.sh > ../../logs/attack_xuanyuan.log 2>&1 &
```

**Option B: Random Baseline**
```bash
# 跑xuanyuan random实验 -- 0207
cd scripts/random
nohup bash run_random_xuanyuan.sh > ../../logs/random_xuanyuan.log 2>&1 &
```

### 4. Build Attack Pools (Optional) -- 因为可攻击样本池已经在代码里了，所以不用执行这步

If you need to build attack pools for a new model:

```bash
# Build all tasks for a model
python build_attack_pool.py --model finma --task all

# Build specific tasks
python build_attack_pool.py --model fingpt --task flare_fpb flare_fiqasa
```

## Attack Configuration --这个我在sh里面已经设好了

Default parameters (in shell scripts):
- `--block-size 10` - Tokens per block
- `--max-suffix-length 30` - Total suffix length
- `--block-iterations 4` - Iterations per block
- `--n-streams 20` - Parallel candidates
- `--n-samples 300` - Samples to attack

## Results

Results are saved to `results/` directory in JSON format, containing:
- Attack Success Rate (ASR)
- Per-sample success/failure details
- Generated adversarial suffixes

## Notes

- Attack pools for all target models are pre-built
- Use `hf-mirror.com` for faster downloads in China (set in download scripts)
- Default: multi-GPU mode (`--no-offload`), attacker on `cuda:0`, target on `cuda:1`
- For single-GPU setups, remove `--no-offload` from shell scripts to enable CPU offloading

## Citation

```bibtex

```
