"""
GA / HGA utilities for AutoDAN financial classification attack.

Adapted from AutoDAN (https://github.com/SheltonLiu-N/AutoDAN)
with the following modifications:
  1. Fitness = label-flipping loss + PPL penalty (instead of harmful-response loss)
  2. Mutation uses Llama-3.1-8B (instead of GPT-4)
  3. Suffix length truncation after every evolution step
  4. Population size = 20, generations = 12 (budget = 240)
"""

import gc
import math
import random
import re
from collections import OrderedDict, defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import nltk
from nltk.corpus import stopwords, wordnet

from autodan.suffix_manager import FinancialSuffixManager


# ======================================================================
# 1. Forward pass (batched, memory-efficient)
# ======================================================================

def forward(*, model, input_ids, attention_mask, batch_size=512):
    """Batched forward pass through the model. Returns logits."""
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = (
            attention_mask[i:i + batch_size] if attention_mask is not None else None
        )
        logits.append(
            model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        )
        gc.collect()
    return torch.cat(logits, dim=0)


# ======================================================================
# 2. Fitness function: label-flipping loss + PPL penalty
# ======================================================================

def get_fitness_score(
    tokenizer,
    model,
    model_name: str,
    source_input: str,
    target_label: str,
    device: str,
    test_controls: List[str],
    crit: nn.Module,
    ppl_lambda: float = 0.1,
    ppl_threshold: float = 50.0,
    forward_batch_size: int = 20,
) -> torch.Tensor:
    """
    Compute fitness = label_loss + ppl_penalty for each candidate suffix.

    label_loss  = CrossEntropyLoss(logits[target_slice], target_token_ids)
    ppl_penalty = ppl_lambda * max(0, ppl_threshold - suffix_PPL)

    Both are computed from the SAME forward pass (zero extra overhead).

    Args:
        tokenizer:    Target model tokenizer
        model:        Target model (on GPU)
        model_name:   One of "finma", "xuanyuan", "fingpt", "finr1"
        source_input: The financial query (fixed)
        target_label: The wrong label we want to induce (e.g., "negative")
        device:       CUDA device string
        test_controls: List of candidate adversarial suffixes
        crit:         nn.CrossEntropyLoss(reduction='mean')
        ppl_lambda:   Weight for PPL penalty term
        ppl_threshold: PPL values below this are penalised
        forward_batch_size: Batch size for forward pass

    Returns:
        Tensor of shape (len(test_controls),) with total fitness scores (lower is better)
    """
    # 1. Build input_ids and identify slices for each candidate
    input_ids_list = []
    target_slices = []
    control_slices = []

    for suffix in test_controls:
        sm = FinancialSuffixManager(
            tokenizer=tokenizer,
            model_name=model_name,
            source_input=source_input,
            adv_suffix=suffix,
            target_label=target_label,
        )
        ids = sm.get_input_ids().to(device)
        input_ids_list.append(ids)
        target_slices.append(sm.target_slice)
        control_slices.append(sm.control_slice)

    # 2. Pad to uniform length
    pad_tok = 0
    for ids in input_ids_list:
        while pad_tok in ids:
            pad_tok += 1

    max_len = max(ids.size(0) for ids in input_ids_list)
    padded = []
    for ids in input_ids_list:
        pad_length = max_len - ids.size(0)
        padded.append(
            torch.cat([ids, torch.full((pad_length,), pad_tok, device=device)], dim=0)
        )

    input_ids_tensor = torch.stack(padded, dim=0)
    attn_mask = (input_ids_tensor != pad_tok).to(input_ids_tensor.dtype)

    # 3. Forward pass
    logits = forward(
        model=model,
        input_ids=input_ids_tensor,
        attention_mask=attn_mask,
        batch_size=forward_batch_size,
    )

    # 4. Compute label_loss + ppl_penalty per candidate
    losses = []
    for idx in range(len(test_controls)):
        ts = target_slices[idx]
        cs = control_slices[idx]

        # --- Label loss ---
        loss_slice = slice(ts.start - 1, ts.stop - 1)
        logits_slice = logits[idx, loss_slice, :].unsqueeze(0).transpose(1, 2)
        targets = input_ids_tensor[idx, ts].unsqueeze(0)
        label_loss = crit(logits_slice, targets)

        # --- PPL penalty (from the same logits) ---
        ppl_penalty = torch.tensor(0.0, device=device)
        if ppl_lambda > 0 and cs.stop > cs.start:
            ctrl_loss_slice = slice(cs.start - 1, cs.stop - 1)
            ctrl_logits = logits[idx, ctrl_loss_slice, :]
            ctrl_targets = input_ids_tensor[idx, cs]
            # Per-token CE for suffix
            ce_per_tok = nn.functional.cross_entropy(
                ctrl_logits, ctrl_targets, reduction="mean"
            )
            suffix_ppl = torch.exp(ce_per_tok)
            # Penalise if PPL is too low (suffix looks like natural language)
            ppl_penalty = ppl_lambda * torch.clamp(ppl_threshold - suffix_ppl, min=0.0)

        losses.append(label_loss + ppl_penalty)

    del input_ids_list, target_slices, control_slices, input_ids_tensor, attn_mask
    gc.collect()
    return torch.stack(losses)


# ======================================================================
# 3. GA evolution operators
# ======================================================================

def roulette_wheel_selection(
    data_list: List[str],
    score_list: List[float],
    num_selected: int,
    if_softmax: bool = True,
) -> List[str]:
    """Select parents using roulette wheel (higher score = higher probability)."""
    scores = np.array(score_list, dtype=np.float64)
    if if_softmax:
        scores = np.exp(scores - np.max(scores))
        probs = scores / scores.sum()
    else:
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones_like(scores) / len(scores)

    selected_indices = np.random.choice(len(data_list), size=num_selected, p=probs, replace=True)
    return [data_list[i] for i in selected_indices]


def crossover(str1: str, str2: str, num_points: int = 5) -> Tuple[str, str]:
    """Multi-point crossover at sentence boundaries."""
    sentences1 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str1) if s.strip()]
    sentences2 = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str2) if s.strip()]

    if not sentences1:
        sentences1 = [str1]
    if not sentences2:
        sentences2 = [str2]

    # Align lengths
    max_len = max(len(sentences1), len(sentences2))
    while len(sentences1) < max_len:
        sentences1.append(random.choice(sentences1))
    while len(sentences2) < max_len:
        sentences2.append(random.choice(sentences2))

    # Multi-point crossover
    num_points = min(num_points, max_len - 1) if max_len > 1 else 0
    if num_points > 0:
        points = sorted(random.sample(range(1, max_len), num_points))
    else:
        points = []

    child1, child2 = [], []
    swap = False
    prev = 0
    for pt in points + [max_len]:
        if swap:
            child1.extend(sentences2[prev:pt])
            child2.extend(sentences1[prev:pt])
        else:
            child1.extend(sentences1[prev:pt])
            child2.extend(sentences2[prev:pt])
        swap = not swap
        prev = pt

    return " ".join(child1), " ".join(child2)


def llm_mutate(suffix: str, attacker_model, attacker_tokenizer, device: str = "cuda:0") -> str:
    """
    Mutate a suffix using Llama-3.1-8B (local attacker model).
    Replaces the original GPT-4 mutation in AutoDAN.
    """
    prompt = (
        "Rephrase the following short text while preserving its approximate meaning. "
        "Output ONLY the rephrased text, nothing else.\n\n"
        f"Original: {suffix}\n\n"
        "Rephrased:"
    )
    try:
        inputs = attacker_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = attacker_model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=attacker_tokenizer.eos_token_id,
            )
        generated = attacker_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # Take first line only, clean up
        result = generated.split("\n")[0].strip().strip('"').strip("'")
        if len(result) < 3:
            return suffix  # Fallback if generation failed
        return result
    except Exception as e:
        print(f"[WARN] LLM mutation failed: {e}")
        return suffix


def synonym_mutate(sentence: str, num: int = 5) -> str:
    """Mutate by replacing words with WordNet synonyms (fallback / HGA supplement)."""
    T = {"llama", "meta", "model", "assistant", "human"}
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    words = nltk.word_tokenize(sentence)
    uncommon = [w for w in words if w.lower() not in stop_words and w.lower() not in T]
    selected = random.sample(uncommon, min(num, len(uncommon)))

    for word in selected:
        syns = wordnet.synsets(word)
        if syns and syns[0].lemmas():
            syn = syns[0].lemmas()[0].name()
            sentence = sentence.replace(word, syn, 1)

    return sentence


def apply_mutation(
    offspring: List[str],
    mutation_rate: float,
    attacker_model,
    attacker_tokenizer,
    attacker_device: str,
    reference: Optional[List[str]] = None,
) -> List[str]:
    """
    Apply LLM-based mutation to offspring.
    Uses Llama-3.1-8B for mutation with fallback to synonym replacement.
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            if attacker_model is not None:
                offspring[i] = llm_mutate(
                    offspring[i], attacker_model, attacker_tokenizer, attacker_device
                )
            elif reference is not None:
                # Fallback: pick a random reference
                offspring[i] = random.choice(reference)
            else:
                offspring[i] = synonym_mutate(offspring[i])
    return offspring


def apply_crossover_and_mutation(
    selected_data: List[str],
    crossover_probability: float = 0.5,
    num_points: int = 5,
    mutation_rate: float = 0.15,
    attacker_model=None,
    attacker_tokenizer=None,
    attacker_device: str = "cuda:0",
    reference: Optional[List[str]] = None,
) -> List[str]:
    """Crossover + mutation pipeline."""
    offspring = []
    for i in range(0, len(selected_data), 2):
        parent1 = selected_data[i]
        parent2 = selected_data[i + 1] if (i + 1) < len(selected_data) else selected_data[0]

        if random.random() < crossover_probability:
            child1, child2 = crossover(parent1, parent2, num_points)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    offspring = apply_mutation(
        offspring, mutation_rate, attacker_model, attacker_tokenizer, attacker_device, reference
    )
    return offspring


def autodan_sample_control(
    control_suffixs: List[str],
    score_list: List[float],
    num_elites: int,
    batch_size: int,
    crossover: float = 0.5,
    num_points: int = 5,
    mutation: float = 0.15,
    attacker_model=None,
    attacker_tokenizer=None,
    attacker_device: str = "cuda:0",
    reference: Optional[List[str]] = None,
) -> List[str]:
    """
    GA evolution: elitism + roulette wheel + crossover + LLM mutation.
    """
    # Negate losses (lower loss = higher score)
    neg_scores = [-x for x in score_list]

    # Elites
    sorted_indices = sorted(range(len(neg_scores)), key=lambda k: neg_scores[k], reverse=True)
    elites = [control_suffixs[i] for i in sorted_indices[:num_elites]]

    # Roulette wheel selection for parents
    parents = roulette_wheel_selection(control_suffixs, neg_scores, batch_size - num_elites)

    # Crossover + mutation
    offspring = apply_crossover_and_mutation(
        parents,
        crossover_probability=crossover,
        num_points=num_points,
        mutation_rate=mutation,
        attacker_model=attacker_model,
        attacker_tokenizer=attacker_tokenizer,
        attacker_device=attacker_device,
        reference=reference,
    )

    next_gen = elites + offspring[: batch_size - num_elites]
    # Ensure correct size
    while len(next_gen) < batch_size:
        next_gen.append(random.choice(elites) if elites else random.choice(control_suffixs))
    return next_gen[:batch_size]


# ======================================================================
# 4. HGA-specific operators (word-level momentum)
# ======================================================================

def get_synonyms(word: str) -> List[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def construct_momentum_word_dict(
    word_dict: dict,
    control_suffixs: List[str],
    score_list: List[float],
) -> dict:
    """Build / update a word -> score dictionary with momentum."""
    T = {"llama", "meta", "model", "assistant", "human", "prompt"}
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    word_scores = defaultdict(list)
    for suffix, score in zip(control_suffixs, score_list):
        words = set(
            w for w in nltk.word_tokenize(suffix)
            if w.lower() not in stop_words and w.lower() not in T
        )
        for w in words:
            word_scores[w].append(score)

    for w, scores in word_scores.items():
        avg = sum(scores) / len(scores)
        if w in word_dict:
            word_dict[w] = (word_dict[w] + avg) / 2
        else:
            word_dict[w] = avg

    return dict(OrderedDict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True)))


def word_roulette_selection(word: str, word_scores: dict) -> str:
    if not word_scores:
        return word
    min_score = min(word_scores.values())
    adjusted = {k: v - min_score for k, v in word_scores.items()}
    total = sum(adjusted.values())
    if total == 0:
        return word
    pick = random.uniform(0, total)
    current = 0
    for synonym, score in adjusted.items():
        current += score
        if current > pick:
            return synonym.title() if word.istitle() else synonym
    return word


def apply_word_replacement(
    word_dict: dict,
    parents_list: List[str],
    crossover_prob: float = 0.5,
) -> List[str]:
    """HGA word-level replacement using momentum dictionary."""
    T = {"llama", "meta", "model", "assistant", "human", "prompt"}
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    result = []
    min_val = min(word_dict.values()) if word_dict else 0

    for sentence in parents_list:
        words = nltk.word_tokenize(sentence)
        count = 0
        for i, word in enumerate(words):
            if random.random() < crossover_prob:
                if word.lower() not in stop_words and word.lower() not in T:
                    syns = get_synonyms(word.lower())
                    word_scores = {s: word_dict.get(s, min_val) for s in syns}
                    best = word_roulette_selection(word, word_scores)
                    if best:
                        words[i] = best
                        count += 1
                        if count >= 5:
                            break
        result.append(" ".join(words))
    return result


def autodan_sample_control_hga(
    word_dict: dict,
    control_suffixs: List[str],
    score_list: List[float],
    num_elites: int,
    batch_size: int,
    crossover: float = 0.5,
    mutation: float = 0.15,
    attacker_model=None,
    attacker_tokenizer=None,
    attacker_device: str = "cuda:0",
    reference: Optional[List[str]] = None,
) -> Tuple[List[str], dict]:
    """
    HGA evolution: word-level momentum replacement + LLM mutation.
    """
    neg_scores = [-x for x in score_list]
    sorted_indices = sorted(range(len(neg_scores)), key=lambda k: neg_scores[k], reverse=True)
    elites = [control_suffixs[i] for i in sorted_indices[:num_elites]]
    parents = [control_suffixs[i] for i in sorted_indices[num_elites:]]

    # Update word dictionary
    word_dict = construct_momentum_word_dict(word_dict, control_suffixs, neg_scores)

    # Ensure enough parents
    if len(parents) < batch_size - num_elites:
        if reference:
            parents += random.choices(reference, k=batch_size - num_elites - len(parents))
        else:
            parents += random.choices(elites, k=batch_size - num_elites - len(parents))

    # Word-level replacement
    offspring = apply_word_replacement(word_dict, parents, crossover)

    # LLM mutation
    offspring = apply_mutation(
        offspring, mutation, attacker_model, attacker_tokenizer, attacker_device, reference
    )

    next_gen = elites + offspring[: batch_size - num_elites]
    while len(next_gen) < batch_size:
        next_gen.append(random.choice(elites) if elites else random.choice(control_suffixs))
    return next_gen[:batch_size], word_dict


# ======================================================================
# 5. Suffix length control
# ======================================================================

def truncate_suffix(suffix: str, tokenizer, max_tokens: int = 30) -> str:
    """Truncate suffix to at most max_tokens tokens."""
    tokens = tokenizer.encode(suffix, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return suffix
    tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


# ======================================================================
# 6. Response generation (for judge evaluation)
# ======================================================================

def generate_response(
    model,
    tokenizer,
    suffix_manager: FinancialSuffixManager,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Generate model response for the best candidate suffix (for judge evaluation).
    Truncates input at assistant_role_slice (doesn't include target_label).
    """
    input_ids = suffix_manager.get_input_ids()
    # Only feed up to assistant role (not the target label)
    gen_input = input_ids[: suffix_manager.assistant_role_slice.stop].to(device).unsqueeze(0)
    attn_mask = torch.ones_like(gen_input)

    with torch.no_grad():
        output_ids = model.generate(
            gen_input,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for deterministic evaluation
            pad_token_id=tokenizer.eos_token_id,
        )[0]

    gen_tokens = output_ids[suffix_manager.assistant_role_slice.stop:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
