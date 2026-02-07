#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Static Symbol Vocabulary for Attacker LLM Logits Control
Created: 2025-12-11

Purpose: Extract token IDs from Llama-3.1-8B tokenizer that only contain
symbols, digits, and spaces. This vocabulary is used by LogitsProcessor
to control suffix generation.

Usage:
    python build_symbol_vocab.py [--model-path PATH] [--output PATH]
"""

import argparse
import json
import os
from typing import List, Set


def get_whitelist_chars() -> Set[str]:
    """
    Define the whitelist character set for symbol vocabulary.
    
    Includes:
    - Basic symbols: !@#$%^&*()_+-=[]{};':",./<>?\|`~
    - Digits: 0-9
    - Space character (important for Llama-3 tokenizer)
    
    Returns:
        Set of allowed characters
    """
    # Basic symbols
    symbols = set('!@#$%^&*()_+-=[]{};\':",./<>?\\|`~')
    
    # Digits 0-9
    digits = set('0123456789')
    
    # Space character (critical for Llama-3 BPE tokens with space prefix)
    space = set(' ')
    
    return symbols | digits | space


def build_symbol_vocab(tokenizer, whitelist: Set[str]) -> List[int]:
    """
    Build symbol vocabulary by filtering tokenizer vocab.
    
    Handles Llama-3 tokenizer's special space encoding:
    - Many tokens have space prefix encoded as special characters (e.g., Ġ)
    - We decode each token and check if stripped content is in whitelist
    
    Args:
        tokenizer: HuggingFace tokenizer
        whitelist: Set of allowed characters
    
    Returns:
        List of token IDs that only contain whitelist characters
    """
    symbol_ids = []
    vocab_size = tokenizer.vocab_size
    
    print(f"[INFO] Scanning vocabulary (size: {vocab_size})...")
    
    for token_id in range(vocab_size):
        try:
            # Decode token to string
            token_str = tokenizer.decode([token_id])
            
            # Strip leading/trailing whitespace for checking
            # This handles Llama-3's space prefix tokens (e.g., " 123" -> "123")
            stripped = token_str.strip()
            
            # Skip empty tokens
            if not stripped:
                # But keep pure space tokens
                if token_str == ' ':
                    symbol_ids.append(token_id)
                continue
            
            # Check if ALL characters in stripped string are in whitelist
            if all(c in whitelist for c in stripped):
                symbol_ids.append(token_id)
                
        except Exception as e:
            # Skip problematic tokens (e.g., special control tokens)
            continue
    
    return symbol_ids


def main():
    parser = argparse.ArgumentParser(
        description='Build static symbol vocabulary for Llama-3.1-8B'
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        default='../Llama-3.1-8B',
        help='Path to Llama-3.1-8B model/tokenizer'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='symbol_vocab.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Build Static Symbol Vocabulary")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Output file: {args.output}")
    
    # Load tokenizer
    print("\n[INFO] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"[INFO] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # Get EOS token ID (critical for preventing infinite generation)
    eos_token_id = tokenizer.eos_token_id
    print(f"[INFO] EOS token ID: {eos_token_id}")
    
    # Build whitelist
    whitelist = get_whitelist_chars()
    print(f"[INFO] Whitelist characters: {len(whitelist)} chars")
    print(f"       Symbols: {''.join(sorted(c for c in whitelist if not c.isdigit() and c != ' '))}")
    print(f"       Digits: 0-9")
    print(f"       Space: included")
    
    # Build symbol vocabulary
    print("\n[INFO] Building symbol vocabulary...")
    symbol_ids = build_symbol_vocab(tokenizer, whitelist)
    
    # Ensure EOS token is included (critical for HardSymbolLogitsProcessor)
    if eos_token_id not in symbol_ids:
        print(f"[INFO] Adding EOS token ID {eos_token_id} to symbol vocab")
        symbol_ids.append(eos_token_id)
    
    # Sort for consistency
    symbol_ids = sorted(set(symbol_ids))
    
    print(f"\n[INFO] Symbol vocabulary built:")
    print(f"       Total tokens: {len(symbol_ids)}")
    print(f"       Percentage of vocab: {len(symbol_ids) / tokenizer.vocab_size * 100:.2f}%")
    
    # Show some examples
    print("\n[INFO] Sample tokens in symbol vocab:")
    sample_count = min(20, len(symbol_ids))
    for i in range(sample_count):
        tid = symbol_ids[i]
        token_str = tokenizer.decode([tid])
        print(f"       ID {tid:6d}: '{token_str}'")
    
    # Prepare output data
    output_data = {
        "vocab_size": tokenizer.vocab_size,
        "symbol_token_count": len(symbol_ids),
        "eos_token_id": eos_token_id,
        "whitelist_chars": ''.join(sorted(whitelist)),
        "symbol_token_ids": symbol_ids
    }
    
    # Save to JSON
    print(f"\n[INFO] Saving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Done! Symbol vocabulary saved to: {args.output}")
    
    # Verification
    print("\n[INFO] Verification:")
    with open(args.output, 'r') as f:
        loaded = json.load(f)
    print(f"       Loaded {loaded['symbol_token_count']} symbol token IDs")
    print(f"       EOS token ID: {loaded['eos_token_id']}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
