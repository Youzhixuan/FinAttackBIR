"""
Initial population generator for AutoDAN financial attack.

Uses Llama-3.1-8B (attacker model) to generate seed adversarial suffixes.
Each candidate is generated independently (like BIR's n_streams), with
max_new_tokens controlling length naturally — no truncation needed.
Failed candidates are retried (up to MAX_RETRIES rounds). If still
empty after all retries, an empty string is kept as-is.
"""

from typing import List

import torch

MAX_RETRIES = 3          # max retry rounds for failed candidates
MIN_SUFFIX_LEN = 4       # minimum character length to accept a suffix


def _build_seed_prompt(
    source_input: str,
    gold_label: str,
    target_label: str,
    choices: List[str],
) -> str:
    """Build the prompt for a single suffix generation call."""
    source_preview = source_input[:300] + "..." if len(source_input) > 300 else source_input
    return (
        f"You are helping with an adversarial robustness experiment on financial NLP models.\n\n"
        f"Task: Generate a short text suffix (1-2 sentences) that will be appended after "
        f"a financial analysis question to change the model's prediction.\n\n"
        f"Financial query: \"{source_preview}\"\n"
        f"Current prediction: {gold_label}\n"
        f"Desired prediction: {target_label}\n"
        f"Valid labels: {', '.join(choices)}\n\n"
        f"Output ONLY the suffix text, nothing else.\n\n"
        f"Suffix:"
    )


def _batch_generate(
    attacker_model,
    attacker_tokenizer,
    prompt_text: str,
    n: int,
    max_tokens: int,
    device: str,
    temperature: float = 0.9,
) -> List[str]:
    """
    Generate n suffixes in a single batched forward pass.

    Returns a list of length n, where each element is either a valid
    suffix string or None (if generation failed for that candidate).
    """
    prompts = [prompt_text] * n
    inputs = attacker_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = attacker_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=attacker_tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results: List[str] = []
    for i in range(n):
        gen_tokens = outputs[i][input_len:]
        suffix = attacker_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        # Take first line only (remove any extra output)
        suffix = suffix.split("\n")[0].strip().strip('"').strip("'")
        if suffix and len(suffix) >= MIN_SUFFIX_LEN:
            results.append(suffix)
        else:
            results.append(None)

    return results


def generate_initial_population(
    attacker_model,
    attacker_tokenizer,
    source_input: str,
    gold_label: str,
    target_label: str,
    choices: List[str],
    batch_size: int = 20,
    max_tokens: int = 30,
    device: str = "cuda:0",
) -> List[str]:
    """
    Generate initial adversarial suffix population using Llama-3.1-8B.

    Each candidate is generated independently via a separate stream in a
    batched call (like BIR's n_streams). Length is controlled by
    max_new_tokens — no truncation. Failed candidates are retried up to
    MAX_RETRIES rounds to ensure the entire population comes from the LLM.

    Args:
        attacker_model:     Llama-3.1-8B model (on device)
        attacker_tokenizer: Llama-3.1-8B tokenizer
        source_input:       The financial query
        gold_label:         Correct label (e.g., "positive")
        target_label:       Wrong label to induce (e.g., "negative")
        choices:            All valid labels
        batch_size:         Number of candidates to generate
        max_tokens:         Max new tokens per suffix (controls length naturally)
        device:             CUDA device for attacker

    Returns:
        List of suffix strings, length == batch_size
    """
    prompt_text = _build_seed_prompt(source_input, gold_label, target_label, choices)

    # population[i] = suffix string or None (pending)
    population: List[str] = [None] * batch_size

    for attempt in range(1, MAX_RETRIES + 1):
        # Identify which slots still need generation
        pending_indices = [i for i, s in enumerate(population) if s is None]
        if not pending_indices:
            break  # all slots filled

        n_pending = len(pending_indices)
        print(f"[INFO] Initial population: attempt {attempt}/{MAX_RETRIES}, "
              f"generating {n_pending} candidate(s)...")

        try:
            results = _batch_generate(
                attacker_model, attacker_tokenizer,
                prompt_text, n_pending, max_tokens, device,
                temperature=0.9 + 0.05 * (attempt - 1),  # slightly raise temp on retries
            )
            for idx, suffix in zip(pending_indices, results):
                population[idx] = suffix  # could still be None if failed
        except Exception as e:
            print(f"[WARN] Batch generation failed on attempt {attempt}: {e}")

    # Replace remaining None with empty string
    n_from_llm = sum(1 for s in population if s is not None)
    n_empty = batch_size - n_from_llm
    for i in range(batch_size):
        if population[i] is None:
            population[i] = ""

    if n_empty > 0:
        print(f"[WARN] {n_empty} candidate(s) still empty after {MAX_RETRIES} retries")

    print(f"[INFO] Initial population ready: {batch_size} candidates "
          f"({n_from_llm} from LLM, {n_empty} empty)")

    return population
