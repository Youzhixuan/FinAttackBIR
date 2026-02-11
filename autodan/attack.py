"""
AutoDAN Financial Classification Attack.

Adapted from AutoDAN (Zhu et al., 2023) for financial label-flipping attacks.
Supports both GA and HGA variants with Llama-3.1-8B as attacker for mutation
and initial population generation.

Usage:
    python -m autodan.attack --target-model finma --task flare_fpb --variant ga ...
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# --- Add parent dir to path so we can import project modules ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodan.ga_utils import (
    autodan_sample_control,
    autodan_sample_control_hga,
    generate_response,
    get_fitness_score,
    truncate_suffix,
)
from autodan.initial_pool import generate_initial_population
from autodan.suffix_manager import FinancialSuffixManager

# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(description="AutoDAN Financial Attack")

    # --- Model ---
    parser.add_argument("--target-model", type=str, required=True,
                        choices=["finma", "xuanyuan", "fingpt", "finr1"],
                        help="Target model name")
    parser.add_argument("--attack-model", type=str,
                        default="../models/Llama-3.1-8B",
                        help="Path to attacker model (Llama-3.1-8B)")

    # --- Task & data ---
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., flare_fpb)")
    parser.add_argument("--pool-path", type=str, default=None,
                        help="Path to attack pool JSONL. Default: auto-detect from data/attack_pools/")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Number of samples to attack (0 = all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in the pool")

    # --- AutoDAN variant ---
    parser.add_argument("--variant", type=str, default="ga",
                        choices=["ga", "hga"],
                        help="AutoDAN variant: ga or hga")
    parser.add_argument("--hga-iter", type=int, default=3,
                        help="HGA: full GA every N steps (rest are word-level)")

    # --- GA hyperparameters ---
    parser.add_argument("--num-steps", type=int, default=12,
                        help="Number of evolution generations (default: 12)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Population size (default: 20)")
    parser.add_argument("--num-elites", type=float, default=0.05,
                        help="Elite fraction (default: 0.05)")
    parser.add_argument("--crossover", type=float, default=0.5,
                        help="Crossover probability")
    parser.add_argument("--num-points", type=int, default=5,
                        help="Number of crossover points")
    parser.add_argument("--mutation-rate", type=float, default=0.15,
                        help="Mutation rate (default: 0.15)")

    # --- Suffix constraints ---
    parser.add_argument("--max-suffix-tokens", type=int, default=30,
                        help="Max suffix length in tokens")

    # --- PPL penalty ---
    parser.add_argument("--ppl-lambda", type=float, default=0.1,
                        help="PPL penalty weight (0 to disable)")
    parser.add_argument("--ppl-threshold", type=float, default=50.0,
                        help="PPL threshold (penalise if below)")

    # --- GPU ---
    parser.add_argument("--attacker-device", type=str, default="cuda:0",
                        help="Device for attacker model")
    parser.add_argument("--target-device", type=str, default="cuda:1",
                        help="Device for target model")

    # --- Output ---
    parser.add_argument("--output-dir", type=str, default="./results/autodan",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)

    # --- Generation ---
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Max new tokens for response generation")

    return parser.parse_args()


# ======================================================================
# Model loading
# ======================================================================

def load_attacker(model_path: str, device: str):
    """Load Llama-3.1-8B as attacker for initial population + mutation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/3] Loading attacker model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use float32 on CPU (fp16 matmul not supported on CPU), fp16 on GPU
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Attacker loaded on {device} (dtype={dtype})")
    return model, tokenizer


def load_target(model_name: str, device: str):
    """
    Load target model and return (raw_model, tokenizer, targetLM).

    raw_model/tokenizer are used for fitness computation (forward pass).
    targetLM is used for response generation (via get_response).
    """
    # Determine model paths based on model name
    MODEL_PATHS = {
        "finma": {"model_path": "../models/finma-7b-full"},
        "xuanyuan": {"model_path": "../models/XuanYuan-6B"},
        "fingpt": {
            "base_model_path": "../models/Meta-Llama-3-8B",
            "lora_path": "../models/fingpt-mt_llama3-8b_lora",
        },
        "finr1": {"model_path": "../models/Fin-R1"},
    }

    paths = MODEL_PATHS[model_name]

    if model_name == "finma":
        from conversers import FinancialTargetLM
        target = FinancialTargetLM(model_path=paths["model_path"])
    elif model_name == "xuanyuan":
        from conversers import XuanYuanTargetLM
        target = XuanYuanTargetLM(model_path=paths["model_path"])
    elif model_name == "fingpt":
        from conversers import FinGPTTargetLM
        target = FinGPTTargetLM(
            base_model_path=paths["base_model_path"],
            lora_path=paths["lora_path"],
        )
    elif model_name == "finr1":
        from conversers import FinR1TargetLM
        target = FinR1TargetLM(model_path=paths["model_path"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Pin target to GPU (no offloading)
    target._skip_offload = True
    target.gpu_device = device
    print(f"[2/3] Moving target model to {device}...")
    target.model = target.model.to(device)
    target.model.eval()
    print(f"  Target ({target.model_name}) loaded on {device}")

    return target.model, target.tokenizer, target


# ======================================================================
# Data loading
# ======================================================================

def load_attack_pool(args) -> List[Dict]:
    """Load attack pool samples."""
    if args.pool_path:
        pool_path = args.pool_path
    else:
        pool_path = os.path.join(
            "data", "attack_pools", args.target_model, f"{args.task}.jsonl"
        )

    if not os.path.exists(pool_path):
        raise FileNotFoundError(
            f"Attack pool not found: {pool_path}\n"
            f"Run build_attack_pool.py first to generate the pool."
        )

    samples = []
    with open(pool_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        print(f"  [WARN] Attack pool is empty: {pool_path} (0 valid samples). Skipping this task.")
        return []

    samples = samples[args.start:]

    # Random sampling with fixed seed (aligned with BIR main attack for fair comparison)
    if args.n_samples > 0 and args.n_samples < len(samples):
        import random as _rand
        total_before = len(samples)
        _rand.seed(args.seed)
        samples = _rand.sample(samples, args.n_samples)
        print(f"  Loaded {len(samples)} samples from {pool_path} "
              f"(randomly sampled from {total_before} with seed={args.seed})")
    else:
        if args.n_samples > 0:
            samples = samples[:args.n_samples]
        print(f"  Loaded {len(samples)} samples from {pool_path}")

    return samples


def get_target_label(gold_label: str, choices: List[str]) -> str:
    """Select the attack target label (a wrong label different from gold)."""
    wrong = [c for c in choices if c.lower() != gold_label.lower()]
    if not wrong:
        return choices[0]
    return wrong[0]  # Pick first wrong label (deterministic)


# ======================================================================
# Main attack loop
# ======================================================================

def attack_single_sample(
    sample: Dict,
    args,
    target_model,
    target_tokenizer,
    target_lm,
    attacker_model,
    attacker_tokenizer,
    judge,
    crit: nn.Module,
) -> Dict:
    """
    Run AutoDAN GA/HGA attack on a single sample.

    Returns:
        Result dict with attack outcome, suffix, response, loss log, etc.
    """
    source_input = sample["source_input"]
    gold_label = sample["gold_label"]
    choices = sample.get("choices", [])
    answer_map = sample.get("answer_map", None)

    # Fallback: get choices and answer_map from task config if missing in pool
    from task_prompts import get_task_config
    task_config = get_task_config(args.task)
    if not choices:
        choices = task_config.get("choices", [])
    if not answer_map:
        answer_map = task_config.get("answer_map", None)

    target_label = get_target_label(gold_label, choices)

    # --- Initialize judge ---
    judge.set_gold_label(gold_label, choices, answer_map)

    # --- Initialize population ---
    population = generate_initial_population(
        attacker_model=attacker_model,
        attacker_tokenizer=attacker_tokenizer,
        source_input=source_input,
        gold_label=gold_label,
        target_label=target_label,
        choices=choices,
        batch_size=args.batch_size,
        max_tokens=args.max_suffix_tokens,
        device=args.attacker_device,
    )

    # Reference pool for mutation fallback
    reference = list(population)

    # GA parameters
    num_elites = max(1, int(args.batch_size * args.num_elites))
    word_dict = {}  # HGA word-level momentum

    # Result tracking
    result = {
        "source_input": source_input[:200],
        "gold_label": gold_label,
        "target_label": target_label,
        "is_success": False,
        "final_suffix": "",
        "final_response": "",
        "final_loss": float("inf"),
        "total_time": 0,
        "generations": 0,
        "log": {"loss": [], "suffix": [], "response": [], "success": [], "time": []},
    }

    start_time = time.time()

    for gen in range(args.num_steps):
        gen_start = time.time()

        with torch.no_grad():
            # 1. Compute fitness for all candidates
            losses = get_fitness_score(
                tokenizer=target_tokenizer,
                model=target_model,
                model_name=args.target_model,
                source_input=source_input,
                target_label=target_label,
                device=args.target_device,
                test_controls=population,
                crit=crit,
                ppl_lambda=args.ppl_lambda,
                ppl_threshold=args.ppl_threshold,
                forward_batch_size=args.batch_size,
            )
            score_list = losses.cpu().numpy().tolist()

            # 2. Find best candidate
            best_idx = losses.argmin().item()
            best_suffix = population[best_idx]
            best_loss = losses[best_idx].item()

            # 3. Generate response with best suffix for judge evaluation
            sm = FinancialSuffixManager(
                tokenizer=target_tokenizer,
                model_name=args.target_model,
                source_input=source_input,
                adv_suffix=best_suffix,
                target_label=target_label,
            )

            # For Fin-R1, use larger max_new_tokens
            max_gen_tokens = args.max_new_tokens
            if args.target_model == "finr1":
                max_gen_tokens = 512  # Fin-R1 needs more tokens for think+answer

            gen_str = generate_response(
                target_model, target_tokenizer, sm, args.target_device,
                max_new_tokens=max_gen_tokens,
            )

            # For Fin-R1, strip thinking part
            if args.target_model == "finr1" and hasattr(target_lm, "_strip_thinking"):
                gen_str = target_lm._strip_thinking(gen_str)

            # 4. Judge evaluation
            scores = judge.score([""], [gen_str])
            is_success = scores[0] == 10

            gen_time = round(time.time() - gen_start, 2)

            # Log
            result["log"]["loss"].append(best_loss)
            result["log"]["suffix"].append(best_suffix)
            result["log"]["response"].append(gen_str[:200])
            result["log"]["success"].append(is_success)
            result["log"]["time"].append(gen_time)

            print(
                f"  Gen {gen + 1}/{args.num_steps} | "
                f"Loss: {best_loss:.4f} | "
                f"Success: {is_success} | "
                f"Time: {gen_time}s | "
                f"Response: {gen_str[:80]}..."
            )

            if is_success:
                result["is_success"] = True
                result["final_suffix"] = best_suffix
                result["final_response"] = gen_str
                result["final_loss"] = best_loss
                result["generations"] = gen + 1
                break

            # 5. Evolve population
            if args.variant == "hga" and gen % args.hga_iter != 0:
                # HGA: word-level momentum replacement
                population, word_dict = autodan_sample_control_hga(
                    word_dict=word_dict,
                    control_suffixs=population,
                    score_list=score_list,
                    num_elites=num_elites,
                    batch_size=args.batch_size,
                    crossover=args.crossover,
                    mutation=args.mutation_rate,
                    attacker_model=attacker_model,
                    attacker_tokenizer=attacker_tokenizer,
                    attacker_device=args.attacker_device,
                    reference=reference,
                )
            else:
                # GA: standard evolution
                population = autodan_sample_control(
                    control_suffixs=population,
                    score_list=score_list,
                    num_elites=num_elites,
                    batch_size=args.batch_size,
                    crossover=args.crossover,
                    num_points=args.num_points,
                    mutation=args.mutation_rate,
                    attacker_model=attacker_model,
                    attacker_tokenizer=attacker_tokenizer,
                    attacker_device=args.attacker_device,
                    reference=reference,
                )

            # 6. Truncate suffixes to max tokens
            population = [
                truncate_suffix(s, target_tokenizer, args.max_suffix_tokens)
                for s in population
            ]

        gc.collect()
        torch.cuda.empty_cache()

    total_time = round(time.time() - start_time, 2)
    result["total_time"] = total_time

    if not result["is_success"]:
        result["final_suffix"] = best_suffix
        result["final_response"] = gen_str
        result["final_loss"] = best_loss
        result["generations"] = args.num_steps

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print(f"AutoDAN Financial Attack ({args.variant.upper()})")
    print(f"  Target: {args.target_model} | Task: {args.task}")
    print(f"  Budget: {args.batch_size} x {args.num_steps} = {args.batch_size * args.num_steps}")
    print(f"  Max suffix tokens: {args.max_suffix_tokens}")
    print(f"  PPL penalty: lambda={args.ppl_lambda}, threshold={args.ppl_threshold}")
    print(f"  Devices: attacker={args.attacker_device}, target={args.target_device}")
    print("=" * 70)

    # --- Load models ---
    attacker_model, attacker_tokenizer = load_attacker(args.attack_model, args.attacker_device)
    target_model, target_tokenizer, target_lm = load_target(args.target_model, args.target_device)

    # --- Load judge ---
    print("[3/3] Loading Judge (ClassificationJudge)...")
    from judges import ClassificationJudge

    # Create a minimal args-like object for ClassificationJudge
    class JudgeArgs:
        judge_model = None
        judge_max_n_tokens = 256
        judge_temperature = 0
        goal = ""
        target_str = ""
    judge = ClassificationJudge(JudgeArgs())
    print("  Judge loaded")

    # --- Load data ---
    print("\nLoading attack pool...")
    samples = load_attack_pool(args)

    # Loss criterion
    crit = nn.CrossEntropyLoss(reduction="mean")

    # --- Output setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"autodan_{args.variant}_{args.target_model}_{args.task}.json",
    )

    # --- Attack loop ---
    results = {}
    n_success = 0
    total_time = 0

    if len(samples) == 0:
        print(f"\n[WARN] No samples to attack for {args.target_model}/{args.task}. Skipping.")
        print("=" * 70)
        return

    print(f"\nStarting attack on {len(samples)} samples...")
    print("-" * 70)

    for i, sample in enumerate(samples):
        print(f"\n[Sample {i + 1}/{len(samples)}] "
              f"Gold: {sample['gold_label']} | "
              f"Input: {sample['source_input'][:60]}...")

        result = attack_single_sample(
            sample=sample,
            args=args,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            target_lm=target_lm,
            attacker_model=attacker_model,
            attacker_tokenizer=attacker_tokenizer,
            judge=judge,
            crit=crit,
        )

        results[i + args.start] = result
        if result["is_success"]:
            n_success += 1
        total_time += result["total_time"]

        asr = n_success / (i + 1) * 100
        print(f"  Result: {'SUCCESS' if result['is_success'] else 'FAIL'} "
              f"| Time: {result['total_time']}s "
              f"| Running ASR: {asr:.1f}% ({n_success}/{i + 1})")

        # Save intermediate results
        summary = {
            "config": {
                "target_model": args.target_model,
                "task": args.task,
                "variant": args.variant,
                "batch_size": args.batch_size,
                "num_steps": args.num_steps,
                "max_suffix_tokens": args.max_suffix_tokens,
                "mutation_rate": args.mutation_rate,
                "ppl_lambda": args.ppl_lambda,
                "ppl_threshold": args.ppl_threshold,
                "seed": args.seed,
            },
            "results": results,
            "summary": {
                "total_samples": i + 1,
                "n_success": n_success,
                "asr": asr,
                "total_time": total_time,
                "avg_time_per_sample": total_time / (i + 1),
            },
        }
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: AutoDAN-{args.variant.upper()} on {args.target_model}/{args.task}")
    print(f"  Samples: {len(samples)}")
    asr_pct = n_success / len(samples) * 100 if len(samples) > 0 else 0
    print(f"  ASR: {n_success}/{len(samples)} = {asr_pct:.1f}%")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/sample: {total_time / max(len(samples), 1):.1f}s")
    print(f"  Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
