"""
Local replacement for the `llm-attacks` package.
modified by yzx

The original `llm-attacks` (https://github.com/llm-attacks/llm-attacks) pins
transformers==4.28.1 which conflicts with our pair_final environment (4.43.4).
Only three utility functions are actually needed by this project, so we vendor
them here with a more general implementation that works across model families
(Llama-2/3, Qwen2, DeepSeek/Fin-R1, Vicuna, etc.) via the standard
transformers `get_input_embeddings()` API.

Reference: llm_attacks/base/attack_manager.py @ 098262e
"""

import torch


def get_embedding_layer(model):
    """Return the input embedding *module* (nn.Embedding)."""
    return model.get_input_embeddings()


def get_embedding_matrix(model):
    """Return the input embedding weight matrix (vocab_size × hidden_dim)."""
    return model.get_input_embeddings().weight


def get_embeddings(model, input_ids):
    """Return dense embeddings for *input_ids* (batch × seq_len × hidden_dim)."""
    return model.get_input_embeddings()(input_ids)


def get_nonascii_toks(tokenizer, device="cpu"):
    """Return a 1-D int tensor of token IDs that are non-ASCII or non-printable.

    These tokens are typically excluded from the GCG candidate pool to keep
    adversarial suffixes human-readable.
    """
    non_ascii_ids = []
    for i in range(3, tokenizer.vocab_size):
        tok_str = tokenizer.decode([i])
        if not (tok_str.isascii() and tok_str.isprintable()):
            non_ascii_ids.append(i)

    for special_id in (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ):
        # modified by yzx: guard against out-of-range special token IDs
        # (e.g. FinMA sets pad_token_id=32000 but vocab_size=32000, valid range 0-31999)
        if special_id is not None and 0 <= special_id < tokenizer.vocab_size:
            non_ascii_ids.append(special_id)

    return torch.tensor(non_ascii_ids, device=device)
