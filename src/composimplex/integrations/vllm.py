"""vLLM integration for CompoSimplex."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from composimplex.metrics import DistributionMetricAccumulator
from composimplex.sampler import CompoSimplexSampler
from composimplex.config import normalize_sampler_config


class CompoSimplexVLLMLogitsProcessor:
    """Request-level vLLM logits processor backed by CompoSimplexSampler.

    vLLM accepts request-level processors with either:
      (output_ids, logits) -> logits
      (prompt_token_ids, output_ids, logits) -> logits

    This class supports both signatures.
    """

    def __init__(
        self,
        sampler_config: dict,
        tokenizer=None,
        generation_context: dict[str, Any] | None = None,
        metric_accumulator: DistributionMetricAccumulator | None = None,
    ):
        self.sampler_config = normalize_sampler_config(sampler_config)
        self.sampler = CompoSimplexSampler(self.sampler_config)
        self.tokenizer = tokenizer
        self.generation_context = generation_context or {}
        self.metric_accumulator = metric_accumulator

    @torch.no_grad()
    def __call__(self, *args):
        if len(args) == 2:
            prompt_token_ids = []
            output_ids, logits = args
        elif len(args) == 3:
            prompt_token_ids, output_ids, logits = args
        else:
            raise TypeError(
                "CompoSimplexVLLMLogitsProcessor expects "
                "(output_ids, logits) or (prompt_token_ids, output_ids, logits)."
            )

        state = {
            "input_ids": list(prompt_token_ids) + list(output_ids),
            "prompt_token_ids": list(prompt_token_ids),
            "generated_token_ids": list(output_ids),
            "tokenizer": self.tokenizer,
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
            **self.generation_context,
        }
        if self.metric_accumulator is None:
            return self.sampler.full_log_probs(logits, state=state)

        probabilities, token_ids, metrics = self.sampler.distribution(
            logits,
            state=state,
            return_metrics=True,
        )
        self.metric_accumulator.update(metrics)
        output = torch.full_like(logits.float(), float("-inf"))
        output[token_ids] = torch.log(
            torch.clamp(
                probabilities,
                min=float(self.sampler_config.get("eps", 1e-8)),
            )
        )
        return output


def build_vllm_sampling_params(
    sampler_config: dict,
    tokenizer=None,
    generation_context: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    metric_accumulator: DistributionMetricAccumulator | None = None,
    **sampling_kwargs,
):
    """Return request-level vLLM parameters with a CompoSimplex processor."""
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for build_vllm_sampling_params. "
            "Install with `pip install -e '.[vllm]'`."
        ) from exc

    cfg = normalize_sampler_config(sampler_config)
    chooser_type = cfg.get("chooser", {}).get("type", "multinomial")

    params = dict(sampling_kwargs)
    params.setdefault("max_tokens", int(max_tokens or cfg.get("max_new_tokens", 128)))
    params.setdefault("temperature", 0.0 if chooser_type == "argmax" else 1.0)
    params.setdefault("top_p", 1.0)
    params.setdefault("top_k", -1)
    params.setdefault("min_p", 0.0)
    params.setdefault(
        "logits_processors",
        [
            CompoSimplexVLLMLogitsProcessor(
                sampler_config=cfg,
                tokenizer=tokenizer,
                generation_context=generation_context,
                metric_accumulator=metric_accumulator,
            )
        ],
    )
    return SamplingParams(**params)


def generate_vllm(
    llm,
    sampler_config: dict,
    prompts: str | Sequence[str] | None = None,
    prompt_token_ids: list[int] | Sequence[list[int]] | None = None,
    tokenizer=None,
    generation_context: dict[str, Any] | Sequence[dict[str, Any] | None] | None = None,
    max_tokens: int | None = None,
    use_tqdm: bool = True,
    **sampling_kwargs,
):
    """Generate with vLLM using the same CompoSimplex sampler config.

    For a batch of prompts, pass either one shared generation_context dict or
    one context per prompt. The return value is vLLM's request-output list.
    """
    if prompts is None and prompt_token_ids is None:
        raise ValueError("Either prompts or prompt_token_ids must be provided.")

    prompt_list = None
    token_id_list = None

    if prompt_token_ids is not None:
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt_token_ids cannot be empty.")
        if isinstance(prompt_token_ids[0], int):
            token_id_list = [list(prompt_token_ids)]  # type: ignore[index]
        else:
            token_id_list = [list(ids) for ids in prompt_token_ids]  # type: ignore[union-attr]
        batch_size = len(token_id_list)
    else:
        is_single_prompt = isinstance(prompts, str)
        prompt_list = [prompts] if is_single_prompt else list(prompts or [])
        batch_size = len(prompt_list)

    if generation_context is None or isinstance(generation_context, dict):
        contexts = [generation_context or {} for _ in range(batch_size)]
    else:
        contexts = [ctx or {} for ctx in generation_context]
        if len(contexts) != batch_size:
            raise ValueError("generation_context sequence length must match batch size.")

    sampling_params = [
        build_vllm_sampling_params(
            sampler_config=sampler_config,
            tokenizer=tokenizer,
            generation_context=ctx,
            max_tokens=max_tokens,
            **sampling_kwargs,
        )
        for ctx in contexts
    ]

    if token_id_list is not None:
        return llm.generate(
            prompts=None,
            prompt_token_ids=token_id_list,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )

    return llm.generate(
        prompt_list,
        sampling_params=sampling_params,
        use_tqdm=use_tqdm,
    )
