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
├── autodan/                    # AutoDAN baseline (GA + HGA)
│   ├── attack.py               # Main AutoDAN attack script
│   ├── ga_utils.py             # Fitness function + GA/HGA operators
│   ├── suffix_manager.py       # Prompt construction + token slicing
│   └── initial_pool.py         # LLM-based initial population
├── textfooler/                 # TextFooler baseline (greedy importance-based)
│   ├── attack.py               # Main TextFooler attack script
│   └── greedy.py               # Importance scoring + greedy replacement
├── momentum_gcg/               # Momentum-GCG baseline (white-box gradient-based)
│   ├── attack.py               # Main Momentum-GCG attack script
│   ├── opt_utils.py            # GCG optimization utilities
│   ├── llm_attacks_utils.py    # Local vendor for llm-attacks functions
│   └── config.py               # Original GCG config (TEST_PREFIXES)
├── scripts/
│   ├── download/               # Model download scripts
│   ├── attack/                 # Attack experiment scripts
│   ├── random/                 # Random baseline scripts
│   ├── autodan/                # AutoDAN baseline scripts
│   ├── textfooler/             # TextFooler baseline scripts
│   └── momentum_gcg/           # Momentum-GCG baseline scripts
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
bash download_finr1.sh # 下这个

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

# 跑finr1 ours实验 -- 0208
nohup bash run_attack_finr1.sh > ../../logs/attack_finr1.log 2>&1 &
```

**Option B: Random Baseline**
```bash
# 跑xuanyuan random实验 -- 0207
cd scripts/random
nohup bash run_random_xuanyuan.sh > ../../logs/random_xuanyuan.log 2>&1 &

# 跑finr1 random实验 -- 0208
nohup bash run_random_finr1.sh > ../../logs/random_finr1.log 2>&1 &
```

**Option C: AutoDAN Baseline (GA + HGA)**

额外依赖（在 finattack 环境中安装）：
```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

跑全部模型（每个模型 6 tasks × 2 variants = 12 组，4 模型共 48 组）：
```bash
cd scripts/autodan
mkdir -p ../../logs ../../results/autodan

# 逐模型跑（每个模型大约需要几小时，可并行在不同机器上）
nohup bash run_autodan_finma.sh    > ../../logs/autodan_finma.log 2>&1 &
nohup bash run_autodan_xuanyuan.sh > ../../logs/autodan_xuanyuan.log 2>&1 &
nohup bash run_autodan_fingpt.sh   > ../../logs/autodan_fingpt.log 2>&1 &
nohup bash run_autodan_finr1.sh    > ../../logs/autodan_finr1.log 2>&1 &

# 或一键全跑（按顺序）
nohup bash run_autodan_all.sh > ../../logs/autodan_all.log 2>&1 &
```

AutoDAN 同样需要双卡（attacker on cuda:0, target on cuda:1），参数已在脚本中配好：Pop=20, Gen=12, suffix≤30 tokens。结果输出到 `results/autodan/`。

**Option D: TextFooler Baseline (Greedy Importance-Based)**

TextFooler baseline 使用 leave-one-out 重要性排序 + 贪心逐 token 替换，从 symbol_vocab 中采样候选。Budget=240（30 importance + 30×7 replacement），suffix≤30 tokens。无额外依赖。

```bash
cd scripts/textfooler
mkdir -p ../../logs ../../results/textfooler

# 逐模型跑
nohup bash run_textfooler_finma.sh    > ../../logs/textfooler_finma.log 2>&1 &
nohup bash run_textfooler_xuanyuan.sh > ../../logs/textfooler_xuanyuan.log 2>&1 &
nohup bash run_textfooler_fingpt.sh   > ../../logs/textfooler_fingpt.log 2>&1 &
nohup bash run_textfooler_finr1.sh    > ../../logs/textfooler_finr1.log 2>&1 &

# 或一键全跑（按顺序）
nohup bash run_textfooler_all.sh > ../../logs/textfooler_all.log 2>&1 &
```

TextFooler 同样需要双卡（attacker on cuda:0 用于生成初始 suffix, target on cuda:1 用于重要性评分和贪心搜索）。结果输出到 `results/textfooler/`。

**Option E: Momentum-GCG Baseline (White-box Gradient-Based)**

白盒梯度攻击 baseline，基于 [Boosting Jailbreak Attack with Momentum](https://openreview.net/pdf?id=WCar0kfHCF) (ICLR 2024 Workshop / ICASSP 2025)，改造为分类标签翻转攻击。只需**单卡**（无 attacker model，直接在 target model 上算梯度）。

额外依赖（在 pair_final 环境中安装）：
```bash
pip install ml_collections
```

> **注意**：原始代码依赖 `llm-attacks` 包（pin transformers==4.28.1），与我们环境不兼容。已本地实现所需工具函数于 `momentum_gcg/llm_attacks_utils.py`，无需安装 `llm-attacks`。

```bash
cd scripts/momentum_gcg
mkdir -p ../../logs ../../results/momentum_gcg

# 逐模型跑（H200 80GB 推荐，各模型 batch 参数已在脚本中配好）
nohup bash run_momentum_gcg_finma.sh    > ../../logs/momentum_gcg_finma.log 2>&1 &
nohup bash run_momentum_gcg_xuanyuan.sh > ../../logs/momentum_gcg_xuanyuan.log 2>&1 &
nohup bash run_momentum_gcg_fingpt.sh   > ../../logs/momentum_gcg_fingpt.log 2>&1 &
nohup bash run_momentum_gcg_finr1.sh    > ../../logs/momentum_gcg_finr1.log 2>&1 &

# 或一键全跑（按顺序，约 18 小时）
nohup bash run_momentum_gcg_all.sh > ../../logs/momentum_gcg_all.log 2>&1 &
```

Momentum-GCG 只需**单卡**即可运行。脚本中 batch 参数已针对 H200 80GB 优化（详见各脚本注释）。如在较小显存 GPU 上运行，按脚本注释缩减 `--batch-size` 和 `--fwd-batch-size`。结果输出到 `results/momentum_gcg/`。

**Option F: FinTrust Personal-Level Fairness 补充实验**

针对 `fintrust_fairness`（FinTrust personal-level fairness）任务的补充实验，共 3 个实验：

| # | 方法 | Target Model | 任务 | 脚本 |
|---|------|-------------|------|------|
| 1 | Random Baseline | FinMA-7B | fintrust_fairness | `scripts/random/run_random_finma_personal.sh` |
| 2 | Random Baseline | FinGPT | fintrust_fairness | `scripts/random/run_random_fingpt_personal.sh` |
| 3 | Ours (BIR) | FinGPT | fintrust_fairness | `scripts/attack/run_attack_fingpt_personal.sh` |

**运行方式（H200）：**

```bash
cd scripts/random

# 实验 1: Random Baseline + FinMA（单卡），结果输出到 result/random_baseline/
nohup bash run_random_finma_personal.sh > ../../logs/random_finma_personal.log 2>&1 &

# 实验 2: Random Baseline + FinGPT（单卡），结果输出到 result/random_baseline/
nohup bash run_random_fingpt_personal.sh > ../../logs/random_fingpt_personal.log 2>&1 &

# 实验 3: Ours BIR Attack + FinGPT（双卡: attacker cuda:0, target cuda:1），结果输出到 result/
cd ../attack
nohup bash run_attack_fingpt_personal.sh > ../../logs/attack_fingpt_personal.log 2>&1 &
```

> **注意**：
> - Random Baseline 只需**单卡**（仅加载 target model，无 attacker model），两个实验需顺序执行（不同 target model）。
> - BIR Attack 需要**双卡**（默认 `--no-offload --attacker-device cuda:0 --target-device cuda:1`）。

**H200 显存优化（可选，用显存换速度）：**

H200 80GB 显存充裕，以下两处参数默认值偏保守，可直接改大以提速，不影响实验结果：

1. **Random Baseline 推理 batch size**：`random_baseline_classification.py` 第 34 行
   ```python
   BATCH_SIZE = 20  # 改为 120（总预算 240 不变，forward pass 从 12 次降到 2 次，约 3x 加速）
   ```
2. **BIR Attack 中 FinGPT 的 OOM 保护阈值**：`conversers.py` 第 1078-1085 行
   ```python
   # 原始值（小显存卡设计，20 条 prompt 会被拆成 5 个 sub-batch）：
   if max_len > 1500:
       effective_batch_size = 4    # 最大可改为 40
   elif max_len > 1000:
       effective_batch_size = 8    # 最大可上改为 40
   # 改大后 20 条 prompt 一次推完，省去 4 次额外 forward pass
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
