"""Hugging Face Transformers integration for CompoSimplex."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from composimplex.sampler import CompoSimplexSampler
from composimplex.config import normalize_sampler_config


class CompoSimplexLogitsProcessor(LogitsProcessor):
    """Convert model logits to CompoSimplex log probabilities."""

    def __init__(
        self,
        sampler_config: dict,
        prompt_length: Optional[int] = None,
        tokenizer=None,
        generation_context: Optional[dict] = None,
    ):
        self.sampler_config = normalize_sampler_config(sampler_config)
        self.sampler = CompoSimplexSampler(self.sampler_config)
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.generation_context = generation_context or {}

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        prompt_length = self.prompt_length if self.prompt_length is not None else input_ids.shape[1]
        rows = []

        for row_idx in range(scores.shape[0]):
            row_ids = input_ids[row_idx]
            state = {
                "input_ids": row_ids.detach().tolist(),
                "prompt_token_ids": row_ids[:prompt_length].detach().tolist(),
                "generated_token_ids": row_ids[prompt_length:].detach().tolist(),
                "tokenizer": self.tokenizer,
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                **self.generation_context,
            }
            rows.append(self.sampler.full_log_probs(scores[row_idx], state=state))

        return torch.stack(rows, dim=0)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str | None = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    sampler_config: dict | None = None,
    max_new_tokens: int | None = None,
    device: str | torch.device | None = None,
    generation_context: Optional[dict] = None,
    **generation_kwargs,
):
    """Generate tokens with Transformers and a CompoSimplex sampler config.

    The return value is the token-ID tensor returned by ``model.generate``.
    Native probability transforms are disabled because the logits processor
    already returns the solved compositional distribution.
    """
    if sampler_config is None:
        raise ValueError("sampler_config is required.")

    cfg = normalize_sampler_config(sampler_config)

    if input_ids is None:
        if prompt is None:
            raise ValueError("Either prompt or input_ids must be provided.")
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    target_device = device or getattr(model, "device", None)
    if target_device is not None:
        input_ids = input_ids.to(target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)

    prompt_length = input_ids.shape[1]
    processor = CompoSimplexLogitsProcessor(
        cfg,
        prompt_length=prompt_length,
        tokenizer=tokenizer,
        generation_context=generation_context,
    )
    chooser_type = cfg.get("chooser", {}).get("type", "multinomial")

    kwargs = dict(generation_kwargs)
    kwargs.setdefault("max_new_tokens", int(max_new_tokens or cfg.get("max_new_tokens", 128)))
    kwargs.setdefault("do_sample", chooser_type != "argmax")
    kwargs.setdefault("temperature", 1.0)
    kwargs.setdefault("top_k", 0)
    kwargs.setdefault("top_p", 1.0)
    kwargs.setdefault("logits_processor", LogitsProcessorList([processor]))
    if attention_mask is not None:
        kwargs.setdefault("attention_mask", attention_mask)

    if tokenizer.pad_token_id is not None:
        kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    return model.generate(input_ids=input_ids, **kwargs)
