"""
Suffix Generator module for AmpleGCG
Handles loading Prompter LM and generating adversarial suffixes
"""

import gc
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Import logits processors for symbol control
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logits_processors import create_logits_processor, load_symbol_vocab, LogitsProcessor, LogitsProcessorList


# Pre-load symbol vocabulary to avoid reloading for each sample
_SYMBOL_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "symbol_vocab.json"
)
_SYMBOL_VOCAB = None
_SYMBOL_IDS = None


def _get_symbol_ids():
    """
    Lazy-load symbol vocabulary and return symbol IDs
    """
    global _SYMBOL_VOCAB, _SYMBOL_IDS
    if _SYMBOL_VOCAB is None:
        _SYMBOL_VOCAB = load_symbol_vocab(_SYMBOL_VOCAB_PATH)
        _SYMBOL_IDS = _SYMBOL_VOCAB["symbol_token_ids"]
    return _SYMBOL_IDS


class SafeSoftSymbolLogitsProcessor(LogitsProcessor):
    """
    Safe version of SoftSymbolLogitsProcessor that handles out-of-bounds symbol IDs.
    
    Args:
        symbol_ids: List of token IDs in the symbol vocabulary
        delta: Constant to add to symbol token logits (default: 10.0)
    """
    
    def __init__(self, symbol_ids, delta=10.0):
        self.symbol_ids = symbol_ids
        self.delta = delta
        self._symbol_ids_tensor = None
        self._device = None
        self._vocab_size = None
    
    def __call__(self, input_ids, scores):
        vocab_size = scores.shape[-1]
        
        if self._symbol_ids_tensor is None or self._device != scores.device or self._vocab_size != vocab_size:
            # Filter symbol_ids to valid range
            valid_ids = [sid for sid in self.symbol_ids if sid < vocab_size]
            if valid_ids:
                self._symbol_ids_tensor = torch.tensor(
                    valid_ids, 
                    dtype=torch.long, 
                    device=scores.device
                )
            else:
                self._symbol_ids_tensor = torch.tensor([], dtype=torch.long, device=scores.device)
            self._device = scores.device
            self._vocab_size = vocab_size
        
        # Boost symbol token logits by delta
        if len(self._symbol_ids_tensor) > 0:
            scores[:, self._symbol_ids_tensor] += self.delta
        
        return scores


def create_safe_logits_processor(mode, symbol_ids, delta=10.0):
    """
    Safe version of create_logits_processor that uses SafeSoftSymbolLogitsProcessor.
    
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
        processor = SafeSoftSymbolLogitsProcessor(symbol_ids, delta=delta)
        return LogitsProcessorList([processor])
    elif mode == 'hard':
        # Use original hard processor which already has safety checks
        processor = create_logits_processor('hard', symbol_ids, delta)
        return processor
    else:
        raise ValueError(f"Unknown logits control mode: {mode}")


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_suffixes(suffixes: List[List[str]], samples: List[Dict], output_path: str):
    """
    Save generated suffixes to file (includes original index for cross-model matching)
    """
    data = []
    for idx, (sample, sample_suffixes) in enumerate(zip(samples, suffixes)):
        data.append({
            "index": sample.get("index", idx),
            "sample_id": idx,
            "source_input": sample["source_input"],
            "gold_label": sample["gold_label"],
            "choices": sample.get("choices", []),
            "suffixes": sample_suffixes
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[SUFFIXES] Suffixes saved to: {output_path}")


def load_suffixes(input_path: str) -> dict:
    """
    Load suffixes from file, returns dict {index: suffixes}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    suffixes_dict = {item["index"]: item["suffixes"] for item in data}
    print(f"[SUFFIXES] Loaded suffixes for {len(suffixes_dict)} samples from {input_path}")
    return suffixes_dict


def load_prompter_lm(model_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[MODEL] Loading Prompter LM: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Prompter model not found: {model_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Prompter requires GPU CUDA.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )

    model.eval()
    print(f"[MODEL] Prompter LM loaded on device: {device}")
    return model, tokenizer


def unload_prompter_lm(prompter_model, prompter_tokenizer):
    """Unload Prompter LM from GPU and clear memory"""
    print("[MEMORY] Unloading Prompter model and clearing GPU memory...")
    del prompter_model
    del prompter_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("[MEMORY] Prompter unloaded, GPU memory cleared")


def generate_suffixes(
    prompter_model,
    prompter_tokenizer,
    source_input: str,
    num_suffixes: int,
    device: str,
    batch_size: int = 30,
    logits_control: str = "none",
    delta: float = 10.0,
) -> List[str]:
    """
    Generate adversarial suffixes using AmpleGCG Prompter LM
    
    Args:
        logits_control: Logits control mode: 'none', 'soft', or 'hard'
        delta: Delta value for soft logits control
    """
    from transformers import GenerationConfig

    prompt = f"### Query:{source_input} ### Prompt:"

    prompter_tokenizer.padding_side = "left"
    if not prompter_tokenizer.pad_token:
        prompter_tokenizer.pad_token = prompter_tokenizer.eos_token

    inputs = prompter_tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Get pre-loaded symbol IDs
    symbol_ids = _get_symbol_ids()

    # Create logits processor if needed
    logits_processor = None
    if logits_control != "none":
        logits_processor = create_safe_logits_processor(
            mode=logits_control,
            symbol_ids=symbol_ids,
            delta=delta
        )

    all_adv_suffixes = []
    remaining = num_suffixes
    batch_idx = 0

    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        
        base_seed = 42
        batch_seed = base_seed + batch_idx
        
        random.seed(batch_seed)
        np.random.seed(batch_seed)
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        
        temperature = 0.9
        
        gen_config = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "max_new_tokens": 30,
            "min_new_tokens": 4,
            "num_return_sequences": current_batch_size,
            "pad_token_id": prompter_tokenizer.pad_token_id,
            "eos_token_id": prompter_tokenizer.eos_token_id,
        }

        generation_config = GenerationConfig(**gen_config)

        with torch.no_grad():
            outputs = prompter_model.generate(
                **inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
            )

        outputs = outputs[:, inputs["input_ids"].shape[-1]:]
        batch_suffixes = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_adv_suffixes.extend(batch_suffixes)
        
        remaining -= current_batch_size
        batch_idx += 1
        
        # Remove outputs to free memory, but avoid expensive cache clearing
        del outputs

    adv_suffixes = all_adv_suffixes

    suffixes = [suffix.strip() for suffix in adv_suffixes]

    return suffixes


def generate_all_suffixes(
    samples: List[Dict],
    prompter_model,
    prompter_tokenizer,
    num_suffixes: int,
    device: str,
    batch_size: int = 30,
    save_callback=None,
    logits_control: str = "none",
    delta: float = 10.0,
) -> List[List[str]]:
    """
    Generate suffixes for all samples
    
    Args:
        save_callback: Optional function(suffixes_list, samples_list, current_idx) 
                       called after each sample is generated
        logits_control: Logits control mode: 'none', 'soft', or 'hard'
        delta: Delta value for soft logits control
    """
    all_suffixes = []
    for idx, sample in enumerate(tqdm(samples, desc="Generating suffixes")):
        print(f"\n[Sample {idx+1}/{len(samples)}] Generating suffixes")
        print(f"  Input: {sample['source_input'][:80]}...")
        print(f"  Batch size: {batch_size} suffixes per batch")
        if logits_control != "none":
            print(f"  Logits control: {logits_control} (delta={delta})")

        suffixes = generate_suffixes(
            prompter_model=prompter_model,
            prompter_tokenizer=prompter_tokenizer,
            source_input=sample["source_input"],
            num_suffixes=num_suffixes,
            device=device,
            batch_size=batch_size,
            logits_control=logits_control,
            delta=delta,
        )
        all_suffixes.append(suffixes)
        print(f"  Generated {len(suffixes)} suffixes")
        
        if save_callback:
            save_callback(all_suffixes, samples[:idx+1], idx)
    
    return all_suffixes
