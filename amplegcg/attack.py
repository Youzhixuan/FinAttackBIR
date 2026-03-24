"""
Usage:
    # Generate shared 160-suffix pool
    python -m amplegcg.attack --generate-only --merge-pools --task [TASK NAME]

    # Attack target model with saved suffixes
    python -m amplegcg.attack --target-model [MODEL NAME] --task [TASK NAME]
"""


import argparse
import gc
import json
import os
import sys
import time

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from pathlib import Path
from typing import Dict, List

import torch

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges import ClassificationJudge
from conversers import (
    FinancialTargetLM,
    FinGPTTargetLM,
    XuanYuanTargetLM,
    FinR1TargetLM,
)
from task_prompts import get_task_config
from amplegcg.generator import (
    set_seed,
    save_suffixes,
    load_suffixes,
    load_prompter_lm,
    unload_prompter_lm,
    generate_all_suffixes
)


def load_attack_pool(
    target_model: str, task: str, n_samples: int = 0
) -> List[Dict]:
    """
    Load financial data from attack_pools
    """
    pool_path = os.path.join(
        "data", "attack_pools", target_model, f"{task}.jsonl"
    )

    if not os.path.exists(pool_path):
        raise FileNotFoundError(
            f"Attack pool not found: {pool_path}\n"
            f"Please run build_attack_pool.py to generate the data first"
        )

    samples = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    samples.sort(key=lambda x: x.get("index", 0))

    if n_samples > 0:
        samples = samples[:n_samples]

    print(f"[DATA] Loaded {len(samples)} samples from {pool_path}")
    return samples


def load_merged_attack_pool(
    task: str, n_samples: int = 0
) -> List[Dict]:
    """
    Merge attack pools from four models and deduplicate by index
    """
    models = ["finma", "fingpt", "xuanyuan", "finr1"]
    all_samples = {}
    
    for model in models:
        pool_path = os.path.join(
            "data", "attack_pools", model, f"{task}.jsonl"
        )
        
        if os.path.exists(pool_path):
            with open(pool_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        sample_index = sample.get("index")
                        if sample_index not in all_samples:
                            all_samples[sample_index] = sample
            print(f"[DATA] Loaded data from {pool_path}")
        else:
            print(f"[DATA] Skipping non-existent file: {pool_path}")
    
    merged_samples = list(all_samples.values())
    merged_samples.sort(key=lambda x: x.get("index", 0))
    
    if n_samples > 0:
        merged_samples = merged_samples[:n_samples]
    
    print(f"[DATA] Merged into {len(merged_samples)} unique samples")
    return merged_samples


def load_target_lm(model_name: str, device: str):
    print(f"[MODEL] Loading target model: {model_name}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Target model requires GPU CUDA.")

    if model_name == "finma":
        target = FinancialTargetLM()
    elif model_name == "xuanyuan":
        target = XuanYuanTargetLM()
    elif model_name == "fingpt":
        target = FinGPTTargetLM()
    elif model_name == "finr1":
        target = FinR1TargetLM()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    target._skip_offload = True
    target.gpu_device = device
    target.model = target.model.to(target.gpu_device)
    target.model.eval()

    print(f"[MODEL] Target model loaded on device: {target.gpu_device}")
    return target


def test_suffixes(
    suffixes: List[str],
    target_lm,
    sample: Dict,
    judge,
    task: str,
    attack_batch_size: int,
) -> Dict:
    """
    Test suffixes and determine if attack is successful
    """
    source_input = sample["source_input"]
    gold_label = str(sample["gold_label"])
    choices = sample.get("choices", [])

    task_cfg = get_task_config(task) if task else {}
    answer_map = task_cfg.get("answer_map")
    judge.set_gold_label(gold_label, choices, answer_map=answer_map)

    success_found = False
    best_suffix = ""
    best_response = ""
    all_results = []

    batch_size = attack_batch_size
    for i in range(0, len(suffixes), batch_size):
        batch_suffixes = suffixes[i : i + batch_size]
        batch_prompts = [f"{source_input} {s}" for s in batch_suffixes]

        responses = target_lm.get_response(batch_prompts)
        scores = judge.score(batch_prompts, responses)

        for j, (suffix, response, score) in enumerate(
            zip(batch_suffixes, responses, scores)
        ):
            is_success = score == 10
            all_results.append(
                {
                    "suffix": suffix,
                    "response": response,
                    "score": score,
                    "success": is_success,
                }
            )

            if is_success and not success_found:
                success_found = True
                best_suffix = suffix
                best_response = response
                print(f"  Found successful suffix (suffix #{i+j+1})")

        if success_found:
            break

    return {
        "success": success_found,
        "best_suffix": best_suffix,
        "best_response": best_response,
        "num_suffixes_tested": len(all_results),
    }


def get_args():
    parser = argparse.ArgumentParser(
        description="AmpleGCG Financial Attack Evaluation"
    )
    
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        choices=["finma", "fingpt", "xuanyuan", "finr1"],
        help="Target model: finma, fingpt, xuanyuan, finr1 (optional for --generate-only)",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name, e.g., flare_fpb, flare_fiqasa, etc.",
    )

    parser.add_argument(
        "--prompter-model",
        type=str,
        default="../models/AmpleGCG",
        help="AmpleGCG Prompter LM local path",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="Number of test samples (0 = all)",
    )

    parser.add_argument(
        "--num-suffixes",
        type=int,
        default=160,
        help="Number of suffixes to generate/test per sample",
    )

    parser.add_argument(
        "--suffix-dir",
        type=str,
        default="./results/amplegcg_160",
        help="Directory for saved suffix pools",
    )

    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=60,
        help="Number of suffixes to generate per batch",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/amplegcg",
        help="Output directory for results",
    )

    parser.add_argument(
        "--attack-batch-size",
        type=int,
        default=8,
        help="Number of suffixes tested per target-model forward pass",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate suffixes and save, without attack",
    )

    parser.add_argument(
        "--logits-control",
        type=str,
        default="none",
        choices=["none", "soft", "hard"],
        help="Logits control mode: none (baseline), soft (boost symbols), hard (force symbols)",
    )

    parser.add_argument(
        "--logits-delta",
        type=float,
        default=10.0,
        help="Delta value for soft logits control (default: 10.0)",
    )

    parser.add_argument(
        "--merge-pools",
        action="store_true",
        help="Merge attack pools from all models even when --target-model is specified",
    )

    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    print("=" * 70)
    print("AmpleGCG Financial Attack Evaluation")
    print(f"  Target model: {args.target_model}")
    print(f"  Task: {args.task}")
    print(f"  Suffixes per sample: {args.num_suffixes}")
    print(f"  Logits control: {args.logits_control}" + (f" (delta={args.logits_delta})" if args.logits_control == 'soft' else ""))
    if args.generate_only:
        print("  Mode: Generate suffixes only")
    else:
        print("  Mode: Attack only")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU CUDA.")
    device = "cuda:0"
    print(f"[DEVICE] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.suffix_dir, exist_ok=True)

    # ========================================================================
    # Mode 1: Only generate suffixes (--generate-only)
    # ========================================================================
    if args.generate_only:
        if args.merge_pools or args.target_model is None:
            print(f"\n[MERGE] Merging attack pools from four models...")
            samples = load_merged_attack_pool(args.task, args.n_samples)
        else:
            samples = load_attack_pool(args.target_model, args.task, args.n_samples)

        suffixes_file = os.path.join(
            args.suffix_dir,
            f"suffixes_{args.task}_{args.num_suffixes}.json",
        )

        # Load full sample list (to preserve order)
        if args.merge_pools or args.target_model is None:
            all_samples = load_merged_attack_pool(args.task, args.n_samples)
        else:
            all_samples = load_attack_pool(args.target_model, args.task, args.n_samples)

        # Check if we can resume from a previous run
        existing_suffixes = {}
        existing_suffixes_list = []
        if os.path.exists(suffixes_file):
            print(f"\n[RESUME] Found existing suffixes file: {suffixes_file}")
            try:
                existing_suffixes = load_suffixes(suffixes_file)
                print(f"  Loaded {len(existing_suffixes)} already-generated samples")
                
                # Build existing suffixes list in order
                existing_suffixes_list = []
                samples_to_process = []
                
                for sample in all_samples:
                    sample_idx = sample.get("index", len(existing_suffixes_list))
                    if sample_idx in existing_suffixes:
                        existing_suffixes_list.append(existing_suffixes[sample_idx])
                    else:
                        samples_to_process.append(sample)
                
                if not samples_to_process:
                    print("\n[COMPLETE] All samples already generated!")
                    return
                
                print(f"  Resuming from sample {len(existing_suffixes_list) + 1}/{len(all_samples)}")
                samples = samples_to_process
                
            except Exception as e:
                print(f"  [WARNING] Could not load existing file: {e}")
                print("  Starting from scratch...")
                existing_suffixes = {}
                existing_suffixes_list = []
                samples = all_samples
        else:
            samples = all_samples

        print(f"\n[PHASE 1] Loading Prompter and generating suffixes...")
        print("-" * 70)
        prompter_model, prompter_tokenizer = load_prompter_lm(
            args.prompter_model, device
        )

        # Define incremental save callback
        def incremental_save(new_suffixes, processed_samples, current_idx_in_batch):
            combined_suffixes = existing_suffixes_list + new_suffixes
            combined_samples = all_samples[:len(existing_suffixes_list) + len(new_suffixes)]
            save_suffixes(combined_suffixes, combined_samples, suffixes_file)
            print(f"  [SAVE] Progress saved to {suffixes_file}")

        all_suffixes = generate_all_suffixes(
            samples=samples,
            prompter_model=prompter_model,
            prompter_tokenizer=prompter_tokenizer,
            num_suffixes=args.num_suffixes,
            device=device,
            batch_size=args.generate_batch_size,
            save_callback=incremental_save,
            logits_control=args.logits_control,
            delta=args.logits_delta
        )

        print("\n[PHASE 1 COMPLETE] All suffixes generated")
        unload_prompter_lm(prompter_model, prompter_tokenizer)

        # Final save (combine existing and new)
        complete_suffixes = existing_suffixes_list + all_suffixes
        save_suffixes(complete_suffixes, all_samples, suffixes_file)

        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print(f"  Total samples: {len(all_samples)}")
        print(f"  Suffixes saved to: {suffixes_file}")
        print("=" * 70)
        return

    # ========================================================================
    # Mode 2: Attack only (no --generate-only, auto-load suffixes)
    # ========================================================================
    else:
        if args.target_model is None:
            raise ValueError("Must specify --target-model for attack mode!")

        samples = load_attack_pool(args.target_model, args.task, args.n_samples)

        output_file = os.path.join(
            args.output_dir,
            f"amplegcg_{args.target_model}_{args.task}.json",
        )

        # Auto-detect suffixes file
        suffixes_file = os.path.join(
            args.suffix_dir,
            f"suffixes_{args.task}_{args.num_suffixes}.json",
        )

        if not os.path.exists(suffixes_file):
            raise FileNotFoundError(
                f"Suffixes file not found: {suffixes_file}\n"
                f"Please generate suffixes first using: python -m amplegcg.attack --generate-only --merge-pools --task {args.task} --suffix-dir {args.suffix_dir}"
            )

        print(f"\n[LOAD] Loading saved suffixes...")
        suffixes_dict = load_suffixes(suffixes_file)
        all_suffixes = []
        missing_indices = []
        
        for sample in samples:
            sample_index = sample.get("index")
            if sample_index in suffixes_dict:
                all_suffixes.append(suffixes_dict[sample_index])
            else:
                missing_indices.append(sample_index)
        
        if missing_indices:
            raise ValueError(
                f"Suffixes not found for samples: {missing_indices}! Please generate suffixes for these samples first."
            )
        print(f"[SUFFIXES] Successfully matched suffixes for {len(all_suffixes)} samples")

        # ========================================================================
        # Attack phase
        # ========================================================================
        print(f"\n[PHASE 2] Loading Target model and testing suffixes...")
        print("-" * 70)
        target_lm = load_target_lm(args.target_model, device)

        print("[MODEL] Loading Judge (ClassificationJudge)...")

        class JudgeArgs:
            judge_model = None
            judge_max_n_tokens = 256
            judge_temperature = 0
            goal = ""
            target_str = ""
        judge = ClassificationJudge(JudgeArgs())

        results = []
        success_count = 0
        start_time = time.time()

        for idx, sample in enumerate(tqdm(samples, desc="Testing suffixes")):
            print(f"\n[Sample {idx+1}/{len(samples)}] Testing suffixes")
            print(f"  Gold Label: {sample['gold_label']}")

            suffixes = all_suffixes[idx]

            test_result = test_suffixes(
                suffixes=suffixes,
                target_lm=target_lm,
                sample=sample,
                judge=judge,
                task=args.task,
                attack_batch_size=args.attack_batch_size,
            )

            results.append(
                {
                    "sample_id": idx,
                    "source_input": sample["source_input"][:200],
                    "gold_label": sample["gold_label"],
                    "success": test_result["success"],
                    "best_suffix": test_result["best_suffix"],
                    "best_response": test_result["best_response"],
                    "num_suffixes_tested": test_result["num_suffixes_tested"],
                }
            )

            if test_result["success"]:
                success_count += 1

            asr = success_count / (idx + 1) * 100
            print(
                f"  Result: {'Success' if test_result['success'] else 'Fail'} | "
                f"Current ASR: {asr:.1f}% ({success_count}/{idx+1})"
            )

            summary = {
                "config": {
                    "target_model": args.target_model,
                    "task": args.task,
                    "num_suffixes": args.num_suffixes,
                    "attack_batch_size": args.attack_batch_size,
                    "seed": args.seed,
                },
                "summary": {
                    "total_samples": idx + 1,
                    "success_count": success_count,
                    "asr": asr,
                    "total_time": time.time() - start_time,
                },
                "results": results,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            gc.collect()
            torch.cuda.empty_cache()

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print(f"  Total samples: {len(samples)}")
        print(f"  Success count: {success_count}")
        print(f"  ASR: {success_count/len(samples)*100:.1f}%")
        print(f"  Results saved to: {output_file}")
        print("=" * 70)


if __name__ == "__main__":
    main()
