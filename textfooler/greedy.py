"""
TextFooler-style greedy importance-based token replacement for financial
adversarial suffix attack.

Core algorithm:
  1. Leave-one-out importance scoring (30 forward passes)
  2. Greedy token replacement in importance order (30 x 7 forward passes)
  Total budget: 240 target model forward passes per sample.

All operations work at the token-ID level to avoid tokenization drift.
Loss is computed ONLY at target_slice (classification decision tokens).
"""

import gc
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodan.suffix_manager import FinancialSuffixManager


# ======================================================================
# Batched forward pass (memory-efficient)
# ======================================================================

def _forward(model, input_ids, attention_mask, batch_size=64):
    """Batched forward pass. Returns logits tensor."""
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        b_ids = input_ids[i:i + batch_size]
        b_mask = attention_mask[i:i + batch_size] if attention_mask is not None else None
        logits.append(model(input_ids=b_ids, attention_mask=b_mask).logits)
        gc.collect()
    return torch.cat(logits, dim=0)


# ======================================================================
# Loss computation at target_slice
# ======================================================================

def _compute_label_loss(logits, input_ids, target_slice, crit):
    """
    Compute CE loss at target_slice for a single sequence.
    logits: (seq_len, vocab_size)
    input_ids: (seq_len,)
    """
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    logits_at_target = logits[loss_slice, :]           # (T, V)
    target_ids = input_ids[target_slice]               # (T,)
    return crit(logits_at_target, target_ids)


def _batch_label_losses(model, all_input_ids, attention_mask,
                        target_slice, crit, device, fwd_batch=64):
    """
    Compute label_loss for a batch of input_ids that share the same
    target_slice (all sequences have same structure, only suffix differs).

    Returns: tensor of shape (N,) with loss per sequence.
    """
    logits = _forward(model, all_input_ids, attention_mask, batch_size=fwd_batch)
    losses = []
    for idx in range(all_input_ids.shape[0]):
        loss = _compute_label_loss(
            logits[idx], all_input_ids[idx], target_slice, crit
        )
        losses.append(loss)
    del logits
    gc.collect()
    return torch.stack(losses)


# ======================================================================
# Prepare suffix in target model's token space
# ======================================================================

def filter_symbol_ids_for_model(
    symbol_token_ids: List[int],
    tokenizer,
) -> List[int]:
    """
    Filter symbol_vocab token IDs to only include IDs within the target
    model's vocabulary. symbol_vocab.json is built from Llama-3.1-8B (128K),
    but target models may have smaller vocabs (e.g., FinMA uses 32K).
    """
    vocab_size = tokenizer.vocab_size
    valid = [tid for tid in symbol_token_ids if tid < vocab_size]
    print(f"  [INFO] Filtered symbol vocab: {len(valid)}/{len(symbol_token_ids)} "
          f"(target vocab_size={vocab_size})")
    return valid


def prepare_suffix_tokens(
    suffix_str: str,
    target_tokenizer,
    max_suffix_tokens: int,
    symbol_token_ids: List[int],
) -> List[int]:
    """
    Tokenize suffix with target model's tokenizer and ensure it is exactly
    max_suffix_tokens long. Pad with random symbols or trim if needed.
    symbol_token_ids should already be filtered for the target model.
    """
    toks = target_tokenizer.encode(suffix_str, add_special_tokens=False)
    if len(toks) > max_suffix_tokens:
        toks = toks[:max_suffix_tokens]
    while len(toks) < max_suffix_tokens:
        toks.append(random.choice(symbol_token_ids))
    return toks


# ======================================================================
# Build full input_ids with specific suffix token IDs
# ======================================================================

def build_input_ids_from_suffix_tokens(
    tokenizer, model_name: str, source_input: str,
    suffix_token_ids: List[int], target_label: str,
    device: str,
) -> Tuple[torch.Tensor, slice, slice, slice]:
    """
    Build full input_ids by:
      1. Using SuffixManager to get the prompt structure
      2. Replacing the control_slice region with the exact suffix_token_ids

    This ensures the suffix tokens are exactly what we specify,
    avoiding any tokenization drift from string decode/re-encode.

    Returns: (input_ids, control_slice, target_slice, assistant_role_slice)
    """
    # First, decode the suffix tokens to a string for SuffixManager
    suffix_str = tokenizer.decode(suffix_token_ids, skip_special_tokens=True)
    sm = FinancialSuffixManager(
        tokenizer=tokenizer,
        model_name=model_name,
        source_input=source_input,
        adv_suffix=suffix_str,
        target_label=target_label,
    )
    base_ids = sm.get_input_ids()
    ctrl = sm.control_slice
    tgt = sm.target_slice
    asst = sm.assistant_role_slice

    # The control_slice length might differ from len(suffix_token_ids)
    # due to tokenization boundary effects. We need to handle this.
    ctrl_len = ctrl.stop - ctrl.start
    suffix_len = len(suffix_token_ids)

    if ctrl_len == suffix_len:
        # Perfect match — directly replace
        base_ids[ctrl] = torch.tensor(suffix_token_ids, dtype=torch.long)
    elif ctrl_len < suffix_len:
        # SuffixManager produced fewer tokens; rebuild with exact IDs
        # by constructing: prefix + suffix_token_ids + postfix
        prefix_ids = base_ids[:ctrl.start]
        postfix_ids = base_ids[tgt.start:]  # target_label tokens + beyond
        suffix_t = torch.tensor(suffix_token_ids, dtype=torch.long)
        base_ids = torch.cat([prefix_ids, suffix_t, postfix_ids])
        # Recompute slices
        new_ctrl_end = ctrl.start + suffix_len
        ctrl = slice(ctrl.start, new_ctrl_end)
        tgt = slice(new_ctrl_end, new_ctrl_end + (tgt.stop - tgt.start))
        asst = slice(0, new_ctrl_end)
    else:
        # ctrl_len > suffix_len: SuffixManager expanded tokens
        prefix_ids = base_ids[:ctrl.start]
        postfix_ids = base_ids[tgt.start:]
        suffix_t = torch.tensor(suffix_token_ids, dtype=torch.long)
        base_ids = torch.cat([prefix_ids, suffix_t, postfix_ids])
        new_ctrl_end = ctrl.start + suffix_len
        ctrl = slice(ctrl.start, new_ctrl_end)
        tgt = slice(new_ctrl_end, new_ctrl_end + (tgt.stop - tgt.start))
        asst = slice(0, new_ctrl_end)

    return base_ids.to(device), ctrl, tgt, asst


# ======================================================================
# 1. Leave-one-out importance scoring
# ======================================================================

def compute_importance(
    model,
    tokenizer,
    model_name: str,
    source_input: str,
    suffix_token_ids: List[int],
    target_label: str,
    device: str,
    crit: nn.Module,
) -> Tuple[List[float], float]:
    """
    Leave-one-out importance scoring.

    For each suffix position i, replace token i with UNK and measure
    how much the label loss changes. Higher importance = this token
    contributes more to the current attack effectiveness.

    Budget: 30 forward passes (batched as 31: original + 30 variants).

    Args:
        model:            Target model on GPU
        tokenizer:        Target tokenizer
        model_name:       One of "finma", "xuanyuan", "fingpt", "finr1"
        source_input:     Financial query (frozen)
        suffix_token_ids: Current suffix as list of token IDs (length N)
        target_label:     Wrong label to induce
        device:           CUDA device
        crit:             nn.CrossEntropyLoss

    Returns:
        (importance_scores, base_loss)
        importance_scores: list of floats, length N
        base_loss: float, loss with the original suffix
    """
    N = len(suffix_token_ids)

    # Choose mask token: unk_token_id if available and valid, else a period token
    mask_id = getattr(tokenizer, "unk_token_id", None)
    if mask_id is None or mask_id < 0 or mask_id >= tokenizer.vocab_size:
        mask_id = tokenizer.encode(".", add_special_tokens=False)[0]

    # Build base input_ids (original suffix)
    base_ids, ctrl, tgt, _ = build_input_ids_from_suffix_tokens(
        tokenizer, model_name, source_input,
        suffix_token_ids, target_label, device,
    )

    # Build N variants: each with one suffix position replaced by mask
    all_ids = [base_ids.clone()]  # index 0 = original
    for i in range(N):
        variant = base_ids.clone()
        variant[ctrl.start + i] = mask_id
        all_ids.append(variant)

    # Stack and pad to uniform length (should already be same length)
    max_len = max(ids.size(0) for ids in all_ids)
    # Use a safe pad ID (some models have pad_token_id=-1 which is invalid)
    _raw_pad = tokenizer.pad_token_id
    pad_id = _raw_pad if (_raw_pad is not None and _raw_pad >= 0) else 0
    padded = []
    for ids in all_ids:
        if ids.size(0) < max_len:
            pad = torch.full((max_len - ids.size(0),), pad_id,
                             dtype=torch.long, device=device)
            padded.append(torch.cat([ids, pad]))
        else:
            padded.append(ids)

    input_ids_batch = torch.stack(padded)
    attn_mask = (input_ids_batch != pad_id).long()

    # Forward pass (31 sequences)
    with torch.no_grad():
        losses = _batch_label_losses(
            model, input_ids_batch, attn_mask, tgt, crit, device, fwd_batch=64
        )

    base_loss = losses[0].item()
    importance = []
    for i in range(N):
        # importance = how much loss increases when we mask this token
        # If masking hurts (loss goes up), this token was helping the attack
        importance.append(losses[i + 1].item() - base_loss)

    del all_ids, input_ids_batch, attn_mask, losses
    gc.collect()
    torch.cuda.empty_cache()

    return importance, base_loss


# ======================================================================
# 2. Greedy token-by-token replacement
# ======================================================================

def greedy_replace(
    model,
    tokenizer,
    model_name: str,
    source_input: str,
    suffix_token_ids: List[int],
    target_label: str,
    device: str,
    crit: nn.Module,
    importance_order: List[int],
    symbol_token_ids: List[int],
    candidates_per_pos: int = 7,
) -> Tuple[List[int], float]:
    """
    Greedy token-by-token replacement in importance order.

    For each position (most important first):
      1. Sample K random candidates from symbol_vocab
      2. Evaluate all K (single batched forward pass)
      3. Pick the one with lowest loss
      4. Lock and move to next position

    Budget: N * K forward passes (30 * 7 = 210).

    Args:
        model:              Target model
        tokenizer:          Target tokenizer
        model_name:         Model identifier
        source_input:       Financial query (frozen)
        suffix_token_ids:   Current suffix token IDs (will be modified)
        target_label:       Wrong label to induce
        device:             CUDA device
        crit:               nn.CrossEntropyLoss
        importance_order:   Positions sorted by importance (descending)
        symbol_token_ids:   Full symbol vocabulary token IDs
        candidates_per_pos: K, number of candidates per position

    Returns:
        (optimized_suffix_token_ids, final_loss)
    """
    current_suffix = list(suffix_token_ids)
    current_loss = float("inf")

    for pos in importance_order:
        # Sample K random candidates from symbol vocab
        candidates = random.sample(symbol_token_ids, min(candidates_per_pos, len(symbol_token_ids)))

        # Build K variants
        all_ids = []
        ctrl_ref = None
        tgt_ref = None
        for cand_id in candidates:
            trial_suffix = list(current_suffix)
            trial_suffix[pos] = cand_id
            ids, ctrl, tgt, _ = build_input_ids_from_suffix_tokens(
                tokenizer, model_name, source_input,
                trial_suffix, target_label, device,
            )
            all_ids.append(ids)
            if ctrl_ref is None:
                ctrl_ref = ctrl
                tgt_ref = tgt

        # Pad and batch
        max_len = max(ids.size(0) for ids in all_ids)
        _raw_pad2 = tokenizer.pad_token_id
        pad_id = _raw_pad2 if (_raw_pad2 is not None and _raw_pad2 >= 0) else 0
        padded = []
        for ids in all_ids:
            if ids.size(0) < max_len:
                pad = torch.full((max_len - ids.size(0),), pad_id,
                                 dtype=torch.long, device=device)
                padded.append(torch.cat([ids, pad]))
            else:
                padded.append(ids)

        input_ids_batch = torch.stack(padded)
        attn_mask = (input_ids_batch != pad_id).long()

        # Forward pass (K sequences)
        with torch.no_grad():
            losses = _batch_label_losses(
                model, input_ids_batch, attn_mask, tgt_ref, crit, device,
                fwd_batch=64,
            )

        # Pick best candidate
        best_idx = losses.argmin().item()
        best_loss = losses[best_idx].item()

        # Always accept the best candidate (greedy)
        current_suffix[pos] = candidates[best_idx]
        current_loss = best_loss

        del all_ids, input_ids_batch, attn_mask, losses
        gc.collect()

    torch.cuda.empty_cache()
    return current_suffix, current_loss


# ======================================================================
# 3. Suffix PPL computation
# ======================================================================

def compute_suffix_ppl(
    model,
    tokenizer,
    model_name: str,
    source_input: str,
    suffix_token_ids: List[int],
    target_label: str,
    device: str,
) -> float:
    """
    Compute perplexity of the suffix region using target model logits.
    PPL = exp(mean CE over suffix tokens).
    """
    ids, ctrl, tgt, _ = build_input_ids_from_suffix_tokens(
        tokenizer, model_name, source_input,
        suffix_token_ids, target_label, device,
    )

    if ctrl.stop <= ctrl.start:
        return 0.0

    input_ids = ids.unsqueeze(0)
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[0]

    # CE over suffix region
    ctrl_loss_slice = slice(ctrl.start - 1, ctrl.stop - 1)
    ctrl_logits = logits[ctrl_loss_slice, :]
    ctrl_targets = ids[ctrl]
    ce = nn.functional.cross_entropy(ctrl_logits, ctrl_targets, reduction="mean")
    ppl = torch.exp(ce).item()

    del logits
    gc.collect()
    return ppl


# ======================================================================
# 4. Response generation (for judge evaluation)
# ======================================================================

def generate_response(
    model,
    tokenizer,
    model_name: str,
    source_input: str,
    suffix_token_ids: List[int],
    target_label: str,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Generate target model response for judge evaluation.
    Input is prompt up to (but not including) target_label.
    """
    ids, ctrl, tgt, asst = build_input_ids_from_suffix_tokens(
        tokenizer, model_name, source_input,
        suffix_token_ids, target_label, device,
    )

    # Feed up to assistant_role_slice (before target_label)
    gen_input = ids[:asst.stop].unsqueeze(0)
    attn_mask = torch.ones_like(gen_input)

    with torch.no_grad():
        output_ids = model.generate(
            gen_input,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

    gen_tokens = output_ids[asst.stop:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
