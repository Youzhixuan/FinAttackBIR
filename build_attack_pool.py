#!/usr/bin/env python3
"""
Unified attack pool builder for all target models and tasks.

Usage:
    python build_attack_pool.py --model finma --task all
    python build_attack_pool.py --model fingpt --task flare_fpb flare_fiqasa
    python build_attack_pool.py --model finr1 --task flare_fpb

Outputs attack pools to data/attack_pools/{model}/{task_name}.jsonl

Created: 2026-02-06
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm

from task_prompts import (
    SUPPORTED_TASKS, 
    load_task_data, 
    doc_to_text, 
    judge_task,
    get_task_config
)

# Model configurations
MODEL_CONFIGS = {
    "finma": {
        "class": "FinancialTargetLM",
        "default_path": "../models/finma-7b-full"
    },
    "fingpt": {
        "class": "FinGPTTargetLM",
        "default_path": "../models/Meta-Llama-3-8B"  # Base model, LoRA loaded automatically
    },
    "xuanyuan": {
        "class": "XuanYuanTargetLM",
        "default_path": "../models/XuanYuan-6B"
    },
    "finr1": {
        "class": "FinR1TargetLM",
        "default_path": "../models/Fin-R1"
    }
}

OUTPUT_DIR = "data/attack_pools"
BATCH_SIZE = 4


def load_target_model(model_name: str, model_path: str = None, experiment_logger=None):
    """
    Load target model by name.
    
    Args:
        model_name: Model name (finma, fingpt, xuanyuan, finr1)
        model_path: Override default model path
        experiment_logger: Optional logger
    
    Returns:
        Model instance
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    path = model_path or config["default_path"]
    
    # Import and instantiate
    from conversers import (
        FinancialTargetLM, 
        FinGPTTargetLM, 
        XuanYuanTargetLM,
        FinR1TargetLM
    )
    
    model_class = {
        "FinancialTargetLM": FinancialTargetLM,
        "FinGPTTargetLM": FinGPTTargetLM,
        "XuanYuanTargetLM": XuanYuanTargetLM,
        "FinR1TargetLM": FinR1TargetLM
    }[config["class"]]
    
    print(f"\n[INFO] Loading {model_name} from {path}")
    
    # FinGPT uses different constructor params (base_model_path + lora_path)
    if model_name == "fingpt":
        model = model_class(base_model_path=path, experiment_logger=experiment_logger)
    else:
        model = model_class(model_path=path, experiment_logger=experiment_logger)
    return model


def build_pool_for_task(model, model_name: str, task_name: str, batch_size: int = 4,
                        debug: bool = False, max_samples: int = None) -> int:
    """
    Build attack pool for a single task.
    
    Args:
        model: Target model instance
        model_name: Model name for output path
        task_name: Task name
        batch_size: Batch size for inference
        debug: Enable detailed per-sample logging
        max_samples: Limit number of samples to evaluate (for testing)
    
    Returns:
        Number of samples in attack pool
    """
    print(f"\n{'='*60}")
    print(f"Building attack pool: {model_name} / {task_name}")
    print(f"{'='*60}")
    
    # Load task data
    try:
        samples = load_task_data(task_name)
    except FileNotFoundError as e:
        print(f"[WARNING] Skipping {task_name}: {e}")
        return 0
    
    if not samples:
        print(f"[WARNING] No samples loaded for {task_name}")
        return 0
    
    # Limit samples for testing
    if max_samples and max_samples < len(samples):
        print(f"[INFO] Limiting to first {max_samples} samples (out of {len(samples)})")
        samples = samples[:max_samples]
    
    config = get_task_config(task_name)
    print(f"[INFO] Task config: choices={config['choices']}, judge_type={config['judge_type']}, lower_case={config.get('lower_case', True)}")
    print(f"[INFO] Total samples to evaluate: {len(samples)}")
    
    # Get prompts
    prompts = [doc_to_text(s) for s in samples]
    
    if debug:
        print(f"\n[DEBUG] === Sample prompt preview (first sample) ===")
        print(f"[DEBUG] Prompt (first 300 chars): {prompts[0][:300]}...")
        print(f"[DEBUG] Gold label: {samples[0]['gold']}")
        print(f"[DEBUG] Choices: {samples[0]['choices']}")
        print(f"[DEBUG] ================================================\n")
    
    # Run inference in batches
    correct_samples = []
    incorrect_count = 0
    ambiguous_count = 0
    
    for i in tqdm(range(0, len(samples), batch_size), desc=f"Evaluating {task_name}"):
        batch_samples = samples[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        
        # Generate responses
        responses = model.get_response(batch_prompts)
        
        # Judge each response
        for j, (sample, response) in enumerate(zip(batch_samples, responses)):
            global_idx = i + j
            is_correct, cleaned, gold_label = judge_task(sample, response, task_name)
            
            # Parse prediction for debug info
            from task_prompts import parse_prediction, clean_output
            pred = parse_prediction(response, sample['choices'], config.get('lower_case', True),
                                    answer_map=config.get('answer_map'))
            
            if debug:
                status = "CORRECT" if is_correct else ("WRONG" if pred is not None else "AMBIGUOUS")
                print(f"\n[DEBUG] --- Sample {global_idx} ---")
                print(f"[DEBUG] Input (last 150 chars): ...{sample['query'][-150:]}")
                print(f"[DEBUG] Raw output: \"{cleaned[:200]}\"")
                print(f"[DEBUG] Extracted prediction: {pred}")
                print(f"[DEBUG] Gold label: {gold_label}")
                print(f"[DEBUG] Judgment: {status}")
                if is_correct:
                    print(f"[DEBUG] -> SAVED to attack pool")
            
            if is_correct:
                # Create attack pool entry
                entry = {
                    "index": sample["index"],
                    "source_input": sample["query"],
                    "gold_label": gold_label,
                    "choices": sample["choices"],
                    "model_output": cleaned  # Keep for reference
                }
                correct_samples.append(entry)
            elif pred is None:
                ambiguous_count += 1
            else:
                incorrect_count += 1
    
    # Save attack pool
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in correct_samples:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    total = len(samples)
    accuracy = len(correct_samples) / total * 100
    print(f"\n[RESULT] {task_name}:")
    print(f"  Total samples: {total}")
    print(f"  Correct (attack pool): {len(correct_samples)} ({accuracy:.2f}%)")
    print(f"  Incorrect: {incorrect_count} ({incorrect_count/total*100:.2f}%)")
    print(f"  Ambiguous: {ambiguous_count} ({ambiguous_count/total*100:.2f}%)")
    print(f"  Saved to: {output_path}")
    
    if debug and correct_samples:
        print(f"\n[DEBUG] === Attack pool sample check (first entry) ===")
        first = correct_samples[0]
        print(f"[DEBUG] index: {first['index']}")
        print(f"[DEBUG] gold_label: {first['gold_label']}")
        print(f"[DEBUG] choices: {first['choices']}")
        print(f"[DEBUG] model_output: \"{first['model_output'][:200]}\"")
        print(f"[DEBUG] source_input (first 200 chars): {first['source_input'][:200]}...")
        print(f"[DEBUG] ================================================")
    
    return len(correct_samples)


def main():
    parser = argparse.ArgumentParser(description="Build attack pools for financial adversarial attacks")
    parser.add_argument("--model", type=str, required=True, 
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Target model name")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override default model path")
    parser.add_argument("--task", type=str, nargs='+', default=["all"],
                        help="Task name(s) or 'all' for all tasks")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference (default: 4)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable detailed per-sample logging")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples for testing (default: all)")
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    
    # Determine tasks
    if "all" in args.task:
        tasks = SUPPORTED_TASKS
    else:
        tasks = args.task
        for t in tasks:
            if t not in SUPPORTED_TASKS:
                print(f"[ERROR] Unknown task: {t}")
                print(f"Supported tasks: {SUPPORTED_TASKS}")
                sys.exit(1)
    
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Tasks: {tasks}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Debug: {args.debug}")
    if args.max_samples:
        print(f"[INFO] Max samples: {args.max_samples}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    
    # Load model once, pin to GPU (no offloading needed for eval-only)
    model = load_target_model(args.model, args.model_path)
    
    gpu_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Pinning model to {gpu_device} (eval-only mode, no offloading)...")
    model.model = model.model.to(gpu_device)
    model.gpu_device = gpu_device
    model._skip_offload = True
    print(f"[INFO] Model pinned to GPU successfully")
    
    # Build pools for all tasks
    results = {}
    for task_name in tasks:
        pool_size = build_pool_for_task(model, args.model, task_name, batch_size,
                                         debug=args.debug, max_samples=args.max_samples)
        results[task_name] = pool_size
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_name, pool_size in results.items():
        status = "SKIPPED" if pool_size == 0 else f"{pool_size} samples"
        print(f"  {task_name}: {status}")
    
    print(f"\n[DONE] Attack pools saved to {OUTPUT_DIR}/{args.model}/")


if __name__ == "__main__":
    main()
