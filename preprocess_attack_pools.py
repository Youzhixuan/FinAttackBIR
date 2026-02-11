#!/usr/bin/env python3
"""
Preprocess attack pools: randomly sample up to N_SAMPLES from each full pool
and write them to data/attack_pools/ (the working pools).

The original full pools are preserved in data/attack_pools_full/.

Sampling logic is identical to BIR main attack (financial_attack_main_classification.py):
    random.seed(SEED)
    sampled = random.sample(all_samples, min(N_SAMPLES, len(all_samples)))

This ensures all three methods (BIR, AutoDAN, TextFooler) attack the *exact same*
subset of samples when using the same seed.

Usage:
    python preprocess_attack_pools.py [--seed 42] [--n-samples 300]
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Preprocess attack pools: sample subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-samples", type=int, default=300, help="Max samples per pool (default: 300)")
    parser.add_argument("--src-dir", type=str, default="data/attack_pools_full",
                        help="Source directory with full pools")
    parser.add_argument("--dst-dir", type=str, default="data/attack_pools",
                        help="Destination directory for sampled pools")
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    n_samples = args.n_samples
    seed = args.seed

    if not os.path.isdir(src_dir):
        print(f"[ERROR] Source directory not found: {src_dir}")
        return

    print(f"Preprocess attack pools")
    print(f"  Source:      {src_dir}")
    print(f"  Destination: {dst_dir}")
    print(f"  Max samples: {n_samples}")
    print(f"  Seed:        {seed}")
    print()

    # Iterate over model dirs
    models = sorted([d for d in os.listdir(src_dir)
                     if os.path.isdir(os.path.join(src_dir, d))])

    total_files = 0
    total_sampled = 0
    total_original = 0

    for model in models:
        model_src = os.path.join(src_dir, model)
        model_dst = os.path.join(dst_dir, model)
        os.makedirs(model_dst, exist_ok=True)

        pool_files = sorted([f for f in os.listdir(model_src) if f.endswith(".jsonl")])
        print(f"=== {model} ({len(pool_files)} tasks) ===")

        for fname in pool_files:
            src_path = os.path.join(model_src, fname)
            dst_path = os.path.join(model_dst, fname)

            # Load all samples
            all_samples = []
            with open(src_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            original_count = len(all_samples)

            if original_count == 0:
                # Empty pool — just copy the empty file
                with open(dst_path, "w", encoding="utf-8") as f:
                    pass
                print(f"  {fname}: 0 -> 0 (empty, copied as-is)")
                total_files += 1
                continue

            # Sample — use exactly the same logic as BIR
            random.seed(seed)
            actual_n = min(n_samples, original_count)
            sampled = random.sample(all_samples, actual_n)

            # Write sampled pool
            with open(dst_path, "w", encoding="utf-8") as f:
                for sample in sampled:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            status = "full (no sampling needed)" if actual_n == original_count else "sampled"
            print(f"  {fname}: {original_count} -> {actual_n} ({status})")

            total_files += 1
            total_sampled += actual_n
            total_original += original_count

    print()
    print(f"Done! Processed {total_files} pool files.")
    print(f"  Total original samples: {total_original}")
    print(f"  Total sampled samples:  {total_sampled}")


if __name__ == "__main__":
    main()
