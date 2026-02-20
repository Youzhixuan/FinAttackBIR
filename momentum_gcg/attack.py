#!/usr/bin/env python3
"""
Momentum-enhanced GCG attack for financial text classification (label flipping).
modified by yzx: adapted from the original jailbreaking GCG to classification tasks.

Reference: "Boosting Jailbreak Attack with Momentum" (ICLR 2024 Workshop / ICASSP 2025)

Usage (run from project root JailbreakingLLMs-main/):
    python -m momentum_gcg.attack \
        --target-model finma \
        --task flare_fpb \
        --n-samples 3 \
        --steps 100 \
        --batch-size 32 \
        --mu 0.4
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn

# modified by yzx: import from same package
from .llm_attacks_utils import get_nonascii_toks
from .opt_utils import (
    load_model_and_tokenizer,
    token_gradients,
    sample_control,
    get_filtered_cands,
    get_logits,
    target_loss,
)

# modified by yzx: import from parent project (JailbreakingLLMs-main/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from task_prompts import get_task_config, SUPPORTED_TASKS
from judges import ClassificationJudge

logger = logging.getLogger(__name__)

# ============================================================================
# Model configs: path, prompt format, vocab notes
# modified by yzx
# ============================================================================
MODEL_CONFIGS = {
    "finma": {
        "path": "../models/finma-7b-full",
        "prompt_type": "raw",
        "max_new_tokens": 80,
        "prefer_last": False,
    },
    "xuanyuan": {
        "path": "../models/XuanYuan-6B",
        "prompt_type": "xuanyuan",
        "max_new_tokens": 80,
        "prefer_last": False,
    },
    "fingpt": {
        "path": "../models/Meta-Llama-3-8B",
        "lora_path": "../models/fingpt-mt_llama3-8b_lora",
        "prompt_type": "raw",
        "max_new_tokens": 80,
        "prefer_last": False,
    },
    "finr1": {
        "path": "../models/Fin-R1",
        "prompt_type": "finr1",
        "max_new_tokens": 1024,
        "prefer_last": True,
    },
}


# ============================================================================
# ClassificationSuffixManager
# modified by yzx: replaces the original SuffixManager for classification tasks
# ============================================================================
class ClassificationSuffixManager:
    """Manages adversarial suffix placement in classification prompts.

    Supports three prompt formats:
      - 'raw':      {source_input} {suffix}{target_label}
      - 'xuanyuan': Human: {source_input} {suffix}\\nAssistant:{target_label}
      - 'finr1':    <chat_template>{source_input} {suffix}</chat_template>{target_label}
    """

    def __init__(self, *, tokenizer, source_input, target_label, adv_string,
                 prompt_type="raw"):
        self.tokenizer = tokenizer
        self.source_input = source_input
        self.target_label = target_label
        self.adv_string = adv_string
        self.prompt_type = prompt_type

    def _build_parts(self, adv_string=None):
        """Return (prefix_before_suffix, suffix_str, bridge, target_str)."""
        if adv_string is not None:
            self.adv_string = adv_string

        if self.prompt_type == "raw":
            prefix = self.source_input + " "
            bridge = ""
        elif self.prompt_type == "xuanyuan":
            prefix = " Human: " + self.source_input + " "
            bridge = "\nAssistant:"
        elif self.prompt_type == "finr1":
            messages = [{"role": "user", "content": self.source_input + " " + "{{SUFFIX}}"}]
            template_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            parts = template_str.split("{{SUFFIX}}")
            prefix = parts[0]
            bridge = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Unknown prompt_type: {self.prompt_type}")

        return prefix, self.adv_string, bridge, self.target_label

    def get_prompt(self, adv_string=None):
        prefix, suffix, bridge, target = self._build_parts(adv_string)
        return prefix + suffix + bridge + target

    def get_input_ids(self, adv_string=None):
        """Build input_ids by tokenizing each segment separately to avoid boundary merging."""
        prefix, suffix, bridge, target = self._build_parts(adv_string)

        # modified by yzx: tokenize segments individually to guarantee stable slices
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False).input_ids
        bridge_ids = self.tokenizer(bridge, add_special_tokens=False).input_ids if bridge else []
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids

        n_pre = len(prefix_ids)
        n_suf = len(suffix_ids)
        n_bri = len(bridge_ids)
        n_tgt = len(target_ids)

        self._goal_slice = slice(0, n_pre)
        self._control_slice = slice(n_pre, n_pre + n_suf)
        self._assistant_role_slice = slice(n_pre + n_suf, n_pre + n_suf + n_bri)
        self._target_slice = slice(n_pre + n_suf + n_bri, n_pre + n_suf + n_bri + n_tgt)
        self._loss_slice = slice(self._target_slice.start - 1, self._target_slice.stop - 1)

        all_ids = prefix_ids + suffix_ids + bridge_ids + target_ids
        return torch.tensor(all_ids)


# ============================================================================
# Auto target label selection
# modified by yzx
# ============================================================================
def select_best_target_label(model, tokenizer, source_input, gold_label, choices,
                             prompt_type="raw"):
    """Pick the wrong label with the lowest initial loss (easiest to flip to)."""
    wrong_labels = [c for c in choices if c.lower() != gold_label.lower()]

    if len(wrong_labels) == 1:
        return wrong_labels[0]

    best_label = None
    best_loss = float('inf')
    dummy_suffix = "! ! ! ! !"

    for label in wrong_labels:
        sm = ClassificationSuffixManager(
            tokenizer=tokenizer, source_input=source_input,
            target_label=label, adv_string=dummy_suffix, prompt_type=prompt_type,
        )
        input_ids = sm.get_input_ids().to(model.device)

        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0)).logits
            crit = nn.CrossEntropyLoss()
            loss_val = crit(
                logits[0, sm._loss_slice, :],
                input_ids[sm._target_slice],
            ).item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_label = label

    return best_label


# ============================================================================
# Single-sample GCG attack
# modified by yzx
# ============================================================================
def attack_single_sample(model, tokenizer, sample, task_name, judge, args,
                         not_allowed_tokens, prompt_type):
    """Run momentum-GCG attack on a single classification sample."""
    source_input = sample["source_input"]
    gold_label = sample["gold_label"]
    choices = sample["choices"]
    sample_id = f"{task_name}_{sample['index']}"

    target_label = select_best_target_label(
        model, tokenizer, source_input, gold_label, choices, prompt_type
    )
    logger.info(f"  [{sample_id}] gold={gold_label}, target={target_label}")

    task_config = get_task_config(task_name)
    judge.set_gold_label(gold_label, choices, answer_map=task_config.get("answer_map"))

    adv_suffix = '! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !'
    sm = ClassificationSuffixManager(
        tokenizer=tokenizer, source_input=source_input,
        target_label=target_label, adv_string=adv_suffix, prompt_type=prompt_type,
    )

    momentum_grad = None
    loss_history = []
    result = {
        "sample_id": sample_id,
        "task_name": task_name,
        "gold_label": gold_label,
        "target_label": target_label,
        "choices": choices,
        "attack_success": False,
        "iterations_to_success": -1,
        "final_adversarial_suffix": adv_suffix,
        "final_prediction": "",
        "loss_history": [],
    }

    for step in range(args.steps):
        input_ids = sm.get_input_ids(adv_string=adv_suffix).to(model.device)

        coordinate_grad = token_gradients(
            model, input_ids,
            sm._control_slice, sm._target_slice, sm._loss_slice,
        )

        with torch.no_grad():
            mu = args.mu
            # modified by yzx: reset momentum when suffix token count changes
            if momentum_grad is None or momentum_grad.shape != coordinate_grad.shape:
                momentum_grad = coordinate_grad
            else:
                momentum_grad = momentum_grad * mu + coordinate_grad * (1 - mu)
                coordinate_grad = momentum_grad.clone()

            adv_suffix_tokens = input_ids[sm._control_slice].to(model.device)
            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens, coordinate_grad, args.batch_size,
                topk=args.topk, temp=1, not_allowed_tokens=not_allowed_tokens,
            )
            new_adv_suffix = get_filtered_cands(
                tokenizer, new_adv_suffix_toks,
                filter_cand=True, curr_control=adv_suffix,
            )

            logits, ids = get_logits(
                model=model, tokenizer=tokenizer, input_ids=input_ids,
                control_slice=sm._control_slice, test_controls=new_adv_suffix,
                return_ids=True, batch_size=args.fwd_batch_size,
            )
            losses = target_loss(logits, ids, sm._target_slice)
            best_idx = losses.argmin()
            best_suffix = new_adv_suffix[best_idx]
            current_loss = losses[best_idx].item()
            adv_suffix = best_suffix

        loss_history.append(current_loss)

        if (step + 1) % args.eval_interval == 0:
            gen_str = generate_response(
                model, tokenizer, sm, adv_suffix, args.max_new_tokens
            )
            scores = judge.score(
                [source_input + " " + adv_suffix], [gen_str]
            )
            is_success = scores[0] == 10

            logger.info(
                f"  [{sample_id}] Step {step+1}/{args.steps} | "
                f"Loss: {current_loss:.4f} | Success: {is_success} | "
                f"Pred: {gen_str[:60]}..."
            )

            if is_success:
                result["attack_success"] = True
                result["iterations_to_success"] = step + 1
                result["final_adversarial_suffix"] = adv_suffix
                result["final_prediction"] = gen_str
                result["loss_history"] = loss_history
                return result

        del coordinate_grad, adv_suffix_tokens
        gc.collect()
        torch.cuda.empty_cache()

    gen_str = generate_response(model, tokenizer, sm, adv_suffix, args.max_new_tokens)
    scores = judge.score([source_input + " " + adv_suffix], [gen_str])
    result["attack_success"] = scores[0] == 10
    result["final_adversarial_suffix"] = adv_suffix
    result["final_prediction"] = gen_str
    result["loss_history"] = loss_history
    if result["attack_success"]:
        result["iterations_to_success"] = args.steps

    return result


def generate_response(model, tokenizer, sm, adv_suffix, max_new_tokens):
    """Generate model response given the adversarial prompt."""
    input_ids = sm.get_input_ids(adv_string=adv_suffix).to(model.device)
    gen_ids = input_ids[:sm._assistant_role_slice.stop].unsqueeze(0)

    if sm._assistant_role_slice.stop == sm._assistant_role_slice.start:
        gen_ids = input_ids[:sm._control_slice.stop].unsqueeze(0)

    with torch.no_grad():
        output_ids = model.generate(
            gen_ids,
            attention_mask=torch.ones_like(gen_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

    gen_str = tokenizer.decode(output_ids[gen_ids.shape[1]:], skip_special_tokens=True).strip()
    return gen_str


# ============================================================================
# Data loading (from BIR project attack pools)
# modified by yzx
# ============================================================================
def load_attack_samples(task_name, target_model, n_samples, seed=42):
    """Load attack pool samples from project data directory."""
    import jsonlines

    # modified by yzx: data/ is in the parent project root
    project_root = os.path.join(os.path.dirname(__file__), '..')
    pool_path = os.path.join(project_root, 'data', 'attack_pools', target_model, f"{task_name}.jsonl")

    if not os.path.exists(pool_path):
        logger.warning(f"  Attack pool not found: {pool_path}")
        return []

    all_samples = []
    with jsonlines.open(pool_path) as reader:
        for obj in reader:
            all_samples.append(obj)

    # modified by yzx: n_samples=0 means use all samples
    if n_samples and n_samples > 0 and n_samples < len(all_samples):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(all_samples), size=n_samples, replace=False)
        all_samples = [all_samples[i] for i in sorted(indices)]

    return all_samples


# ============================================================================
# Main
# modified by yzx
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Momentum-GCG Classification Attack")
    parser.add_argument("--target-model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--task", required=True)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fwd-batch-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--mu", type=float, default=0.4)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/momentum_gcg")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.target_model]
    args.max_new_tokens = model_cfg["max_new_tokens"]

    # Setup logging
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/momentum_gcg_{args.target_model}_{args.task}_{timestamp}.log"
    logging.basicConfig(
        format='[%(asctime)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info(f"Args: {vars(args)}")

    # 1. Load model
    logger.info(f"Loading model: {args.target_model} from {model_cfg['path']}...")
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["path"], low_cpu_mem_usage=True, use_cache=False
    )

    # modified by yzx: handle LoRA for FinGPT
    if args.target_model == "fingpt" and "lora_path" in model_cfg:
        from peft import PeftModel
        logger.info(f"  Loading LoRA adapter: {model_cfg['lora_path']}")
        model = PeftModel.from_pretrained(model, model_cfg["lora_path"])
        model = model.merge_and_unload()
        model.eval()

    logger.info(f"  Vocab size: {tokenizer.vocab_size}, Device: {model.device}")
    not_allowed_tokens = get_nonascii_toks(tokenizer)

    # 2. Load judge
    class JudgeArgs:
        judge_model = "classification"
        judge_max_n_tokens = 500
        judge_temperature = 0.0
        goal = "misclassify financial text"
        target_str = ""

    judge = ClassificationJudge(JudgeArgs())
    if model_cfg.get("prefer_last"):
        judge.set_prefer_last(True)
        logger.info("  Judge: prefer_last=True")

    # 3. Load data
    samples = load_attack_samples(args.task, args.target_model, args.n_samples, args.seed)
    if not samples:
        logger.error(f"No samples found for {args.task}/{args.target_model}")
        return
    logger.info(f"Loaded {len(samples)} samples for {args.task}")

    # 4. Run attacks
    prompt_type = model_cfg["prompt_type"]
    results = []
    total_start = time.time()

    for i, sample in enumerate(samples):
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {i+1}/{len(samples)}: index={sample['index']}")
        logger.info(f"{'='*60}")

        sample_start = time.time()
        result = attack_single_sample(
            model, tokenizer, sample, args.task, judge, args,
            not_allowed_tokens, prompt_type,
        )
        elapsed = time.time() - sample_start
        result["time_seconds"] = round(elapsed, 1)

        results.append(result)
        logger.info(
            f"  Result: success={result['attack_success']}, "
            f"iters={result['iterations_to_success']}, time={elapsed:.1f}s"
        )

    # 5. Save results
    total_elapsed = time.time() - total_start
    successful = sum(1 for r in results if r["attack_success"])
    asr = successful / len(results) if results else 0

    output = {
        "experiment_config": {
            "method": "momentum_gcg",
            "target_model": args.target_model,
            "task": args.task,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "topk": args.topk,
            "mu": args.mu,
            "eval_interval": args.eval_interval,
            "n_samples": len(results),
        },
        "statistics": {
            "total_samples": len(results),
            "successful_attacks": successful,
            "attack_success_rate": round(asr, 4),
            "total_time_seconds": round(total_elapsed, 1),
        },
        "results": results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir,
        f"gcg_{args.target_model}_{args.task}_n{len(results)}_{timestamp}.json"
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE: ASR={successful}/{len(results)} = {asr*100:.1f}%")
    logger.info(f"Total time: {total_elapsed/60:.1f}min")
    logger.info(f"Results saved to: {out_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
