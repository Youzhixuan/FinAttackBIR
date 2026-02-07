#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Symbol Logits Processors for Attacker LLM Generation Control
Created: 2025-12-11

Purpose: Control Attacker LLM to generate "gibberish" style suffixes by
manipulating logits during generation.

Two modes:
- Soft Control: Boost symbol token logits by delta (preserves model's language intuition)
- Hard Control: Force model to only select from symbol vocabulary (pure gibberish)
"""

import json
import torch
from typing import List, Optional
from transformers import LogitsProcessor, LogitsProcessorList


class SoftSymbolLogitsProcessor(LogitsProcessor):
    """
    Soft control: Boost symbol token logits by a constant delta.
    
    This preserves the model's "language intuition" - if a non-symbol token
    has extremely high probability (e.g., grammatically necessary), it can
    still be selected.
    
    Args:
        symbol_ids: List of token IDs in the symbol vocabulary
        delta: Constant to add to symbol token logits (default: 10.0)
    """
    
    def __init__(self, symbol_ids: List[int], delta: float = 10.0):
        self.symbol_ids = symbol_ids
        self.delta = delta
        # Pre-create tensor for efficiency (will be moved to correct device on first call)
        self._symbol_ids_tensor: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply soft control by boosting symbol token logits.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            scores: Logits for next token (batch_size, vocab_size)
        
        Returns:
            Modified scores with symbol tokens boosted
        """
        # Modified: 2025-12-11 - Device synchronization to avoid CUDA errors
        if self._symbol_ids_tensor is None or self._device != scores.device:
            self._symbol_ids_tensor = torch.tensor(
                self.symbol_ids, 
                dtype=torch.long, 
                device=scores.device
            )
            self._device = scores.device
        
        # Boost symbol token logits by delta
        scores[:, self._symbol_ids_tensor] += self.delta
        
        return scores


class HardSymbolLogitsProcessor(LogitsProcessor):
    """
    Hard control: Force model to only select from symbol vocabulary.
    
    Sets all non-symbol token logits to -inf, making it impossible for
    the model to generate anything outside the symbol vocabulary.
    
    WARNING: Ensure EOS token is in symbol_ids, otherwise generation
    will never stop!
    
    Args:
        symbol_ids: List of token IDs in the symbol vocabulary (must include EOS)
    """
    
    def __init__(self, symbol_ids: List[int]):
        self.symbol_ids = symbol_ids
        # Pre-create mask (will be moved to correct device on first call)
        self._mask: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._vocab_size: Optional[int] = None
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply hard control by setting non-symbol logits to -inf.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            scores: Logits for next token (batch_size, vocab_size)
        
        Returns:
            Modified scores with non-symbol tokens set to -inf
        """
        vocab_size = scores.shape[-1]
        
        # Modified: 2025-12-11 - Device synchronization and mask caching
        if (self._mask is None or 
            self._device != scores.device or 
            self._vocab_size != vocab_size):
            
            # Create mask: True for tokens to KEEP (symbol tokens)
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            
            # Filter symbol_ids to valid range
            valid_ids = [sid for sid in self.symbol_ids if sid < vocab_size]
            mask[valid_ids] = True
            
            self._mask = mask
            self._device = scores.device
            self._vocab_size = vocab_size
        
        # Set non-symbol tokens to -inf
        scores[:, ~self._mask] = float('-inf')
        
        return scores


def load_symbol_vocab(vocab_path: str) -> dict:
    """
    Load symbol vocabulary from JSON file.
    
    Args:
        vocab_path: Path to symbol_vocab.json
    
    Returns:
        Dictionary with keys: vocab_size, symbol_token_count, eos_token_id, symbol_token_ids
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_logits_processor(
    mode: str,
    symbol_ids: List[int],
    delta: float = 10.0
) -> Optional[LogitsProcessorList]:
    """
    Factory function to create LogitsProcessor based on mode.
    
    Args:
        mode: 'none', 'soft', or 'hard'
        symbol_ids: List of symbol token IDs
        delta: Delta value for soft control (ignored in hard mode)
    
    Returns:
        LogitsProcessorList or None if mode is 'none'
    """
    if mode == 'none':
        return None
    elif mode == 'soft':
        processor = SoftSymbolLogitsProcessor(symbol_ids, delta=delta)
        return LogitsProcessorList([processor])
    elif mode == 'hard':
        processor = HardSymbolLogitsProcessor(symbol_ids)
        return LogitsProcessorList([processor])
    else:
        raise ValueError(f"Unknown logits control mode: {mode}")


# =============================================================================
# Unit Tests
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LogitsProcessor Unit Tests")
    print("=" * 60)
    
    # Test parameters
    vocab_size = 1000
    batch_size = 2
    symbol_ids = [10, 20, 30, 40, 50]  # Mock symbol IDs
    delta = 10.0
    
    # Create mock logits (random values)
    torch.manual_seed(42)
    mock_input_ids = torch.randint(0, vocab_size, (batch_size, 5))
    mock_scores = torch.randn(batch_size, vocab_size)
    
    print("\n--- Test 1: SoftSymbolLogitsProcessor ---")
    soft_processor = SoftSymbolLogitsProcessor(symbol_ids, delta=delta)
    
    # Copy scores for comparison
    scores_before = mock_scores.clone()
    scores_after = soft_processor(mock_input_ids, mock_scores.clone())
    
    # Verify symbol IDs got boosted
    for sid in symbol_ids:
        diff = scores_after[0, sid] - scores_before[0, sid]
        status = "PASS" if abs(diff - delta) < 1e-6 else "FAIL"
        print(f"  Token {sid}: diff = {diff:.2f} (expected {delta}) [{status}]")
    
    # Verify non-symbol IDs unchanged
    non_symbol_id = 100  # Not in symbol_ids
    diff = scores_after[0, non_symbol_id] - scores_before[0, non_symbol_id]
    status = "PASS" if abs(diff) < 1e-6 else "FAIL"
    print(f"  Non-symbol {non_symbol_id}: diff = {diff:.2f} (expected 0) [{status}]")
    
    print("\n--- Test 2: HardSymbolLogitsProcessor ---")
    hard_processor = HardSymbolLogitsProcessor(symbol_ids)
    
    scores_before = torch.randn(batch_size, vocab_size)
    scores_after = hard_processor(mock_input_ids, scores_before.clone())
    
    # Verify symbol IDs are NOT -inf
    for sid in symbol_ids:
        is_valid = not torch.isinf(scores_after[0, sid])
        status = "PASS" if is_valid else "FAIL"
        print(f"  Token {sid}: value = {scores_after[0, sid]:.2f} (not -inf) [{status}]")
    
    # Verify non-symbol IDs are -inf
    non_symbol_id = 100
    is_inf = torch.isinf(scores_after[0, non_symbol_id]) and scores_after[0, non_symbol_id] < 0
    status = "PASS" if is_inf else "FAIL"
    print(f"  Non-symbol {non_symbol_id}: value = {scores_after[0, non_symbol_id]} (expected -inf) [{status}]")
    
    # Count how many tokens are NOT -inf
    valid_count = (~torch.isinf(scores_after[0])).sum().item()
    status = "PASS" if valid_count == len(symbol_ids) else "FAIL"
    print(f"  Valid tokens count: {valid_count} (expected {len(symbol_ids)}) [{status}]")
    
    print("\n--- Test 3: Device Synchronization (CPU) ---")
    # Test on CPU
    cpu_scores = torch.randn(batch_size, vocab_size)
    soft_processor_cpu = SoftSymbolLogitsProcessor(symbol_ids, delta=delta)
    try:
        result = soft_processor_cpu(mock_input_ids, cpu_scores)
        print(f"  CPU test: PASS (output device: {result.device})")
    except Exception as e:
        print(f"  CPU test: FAIL ({e})")
    
    print("\n--- Test 4: Factory Function ---")
    try:
        proc_none = create_logits_processor('none', symbol_ids)
        print(f"  mode='none': {proc_none} [PASS]")
        
        proc_soft = create_logits_processor('soft', symbol_ids, delta=5.0)
        print(f"  mode='soft': {type(proc_soft[0]).__name__} [PASS]")
        
        proc_hard = create_logits_processor('hard', symbol_ids)
        print(f"  mode='hard': {type(proc_hard[0]).__name__} [PASS]")
    except Exception as e:
        print(f"  Factory test: FAIL ({e})")
    
    print("\n" + "=" * 60)
    print("  All Unit Tests Completed")
    print("=" * 60)
