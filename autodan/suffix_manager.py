"""
Financial Suffix Manager for AutoDAN.

Constructs model-specific prompts from (source_input + adv_suffix + target_label)
and identifies token slices for fitness computation.

Supported models:
  - finma:    plain text (no template)
  - xuanyuan: " Human: {content} Assistant:" format
  - fingpt:   plain text (same as finma, FinGPT uses raw text input)
  - finr1:    Qwen2.5-Instruct chat template
"""

import torch
from typing import Optional, Tuple


class FinancialSuffixManager:
    """
    Manages prompt construction and token-slice identification for the AutoDAN
    financial classification attack.

    For each candidate adversarial suffix, builds:
        full_prompt = format(source_input + " " + adv_suffix) + target_label
    and locates:
        control_slice  – token range of adv_suffix (for PPL computation)
        target_slice   – token range of target_label (for label-flipping loss)
    """

    def __init__(
        self,
        tokenizer,
        model_name: str,
        source_input: str,
        adv_suffix: str,
        target_label: str,
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name.lower()
        self.source_input = source_input
        self.adv_suffix = adv_suffix
        self.target_label = target_label

        # Build prompt and compute slices once
        self._prompt = None
        self._input_ids = None
        self._control_slice = None
        self._target_slice = None
        self._assistant_role_slice = None

    # ------------------------------------------------------------------
    # Prompt formatting (model-specific)
    # ------------------------------------------------------------------
    def _format_user_content(self, content: str) -> str:
        """Wrap user content in model-specific template (WITHOUT target_label)."""
        if self.model_name in ("finma", "fingpt"):
            # FinMA and FinGPT: plain text, no wrapper
            return content
        elif self.model_name == "xuanyuan":
            # XuanYuan-6B: " Human: {content} Assistant:"
            return f" Human: {content} Assistant:"
        elif self.model_name == "finr1":
            # Fin-R1 (Qwen2.5-Instruct): use chat template
            # We'll handle this specially in get_prompt()
            return content
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_prompt(self, adv_suffix: Optional[str] = None) -> str:
        """
        Build the full prompt: formatted(source_input + adv_suffix) + target_label.

        The target_label is appended directly after the assistant role marker
        so that we can compute loss on the model predicting it.
        """
        if adv_suffix is not None:
            self.adv_suffix = adv_suffix

        user_content = f"{self.source_input} {self.adv_suffix}"

        if self.model_name == "finr1":
            # Qwen2.5-Instruct template (generation prompt already ends with \n)
            messages = [{"role": "user", "content": user_content}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Append target label (the model should predict this)
            prompt = formatted + self.target_label
        else:
            formatted = self._format_user_content(user_content)
            # Add space before target label for proper tokenization
            prompt = formatted + " " + self.target_label

        self._prompt = prompt
        return prompt

    def get_input_ids(self, adv_suffix: Optional[str] = None) -> torch.Tensor:
        """
        Tokenize the full prompt and compute control_slice / target_slice.

        Returns:
            input_ids tensor (1-D, on CPU)

        After calling this, access:
            self._control_slice   – slice for adv_suffix tokens
            self._target_slice    – slice for target_label tokens
            self._assistant_role_slice – slice up to (but not including) target_label
        """
        prompt = self.get_prompt(adv_suffix)
        toks = self.tokenizer(prompt, add_special_tokens=True).input_ids
        total_len = len(toks)

        # --- Identify target_slice ---
        # Robust approach: tokenize the prompt WITHOUT target_label and compare
        # token counts. This correctly handles leading-space token merging.
        full_user_content = f"{self.source_input} {self.adv_suffix}"
        if self.model_name == "finr1":
            messages_full = [{"role": "user", "content": full_user_content}]
            prompt_no_target = self.tokenizer.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=True
            )
        elif self.model_name == "xuanyuan":
            prompt_no_target = self._format_user_content(full_user_content)
        else:
            # FinMA/FinGPT: plain text, add trailing space that precedes target
            prompt_no_target = full_user_content + " "

        toks_no_target = self.tokenizer(prompt_no_target, add_special_tokens=True).input_ids
        target_start = len(toks_no_target)
        # Ensure target_slice is not empty (at least 1 token)
        if target_start >= total_len:
            # Fallback: assume target is last 1 token
            target_start = total_len - 1

        self._target_slice = slice(target_start, total_len)
        self._assistant_role_slice = slice(0, target_start)

        # --- Identify control_slice (adv_suffix tokens) ---
        # Tokenize prompt with only source_input (no suffix, no target)
        if self.model_name == "finr1":
            messages_no_suffix = [{"role": "user", "content": self.source_input + " "}]
            prompt_no_suffix = self.tokenizer.apply_chat_template(
                messages_no_suffix, tokenize=False, add_generation_prompt=False
            )
        elif self.model_name == "xuanyuan":
            prompt_no_suffix = self._format_user_content(self.source_input + " ")
        else:
            prompt_no_suffix = self.source_input + " "

        toks_no_suffix = self.tokenizer(prompt_no_suffix, add_special_tokens=True).input_ids
        control_start = len(toks_no_suffix)
        control_end = target_start
        self._control_slice = slice(control_start, max(control_start, control_end))

        self._input_ids = torch.tensor(toks, dtype=torch.long)
        return self._input_ids

    @property
    def control_slice(self):
        return self._control_slice

    @property
    def target_slice(self):
        return self._target_slice

    @property
    def assistant_role_slice(self):
        return self._assistant_role_slice
