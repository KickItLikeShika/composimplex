"""Public API for CompoSimplex."""

from composimplex.config import load_sampler_config, normalize_sampler_config

__all__ = [
    "CompoSimplexLogitsProcessor",
    "CompoSimplexSampler",
    "CompoSimplexVLLMLogitsProcessor",
    "build_vllm_sampling_params",
    "generate",
    "generate_vllm",
    "load_sampler_config",
    "normalize_sampler_config",
    "solve_distribution",
]


def __getattr__(name):
    """Load framework integrations only when their API is requested."""
    if name in {"CompoSimplexLogitsProcessor", "generate"}:
        from composimplex.integrations.transformers import (
            CompoSimplexLogitsProcessor,
            generate,
        )

        return {
            "CompoSimplexLogitsProcessor": CompoSimplexLogitsProcessor,
            "generate": generate,
        }[name]

    if name in {"CompoSimplexSampler", "solve_distribution"}:
        from composimplex.sampler import (
            CompoSimplexSampler,
            solve_distribution,
        )

        return {
            "CompoSimplexSampler": CompoSimplexSampler,
            "solve_distribution": solve_distribution,
        }[name]

    if name in {
        "CompoSimplexVLLMLogitsProcessor",
        "build_vllm_sampling_params",
        "generate_vllm",
    }:
        from composimplex.integrations.vllm import (
            CompoSimplexVLLMLogitsProcessor,
            build_vllm_sampling_params,
            generate_vllm,
        )

        return {
            "CompoSimplexVLLMLogitsProcessor": CompoSimplexVLLMLogitsProcessor,
            "build_vllm_sampling_params": build_vllm_sampling_params,
            "generate_vllm": generate_vllm,
        }[name]

    raise AttributeError(f"module 'composimplex' has no attribute {name!r}")
