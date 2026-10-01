#!/usr/bin/env python3
"""Run reasoning or coding benchmarks through Transformers or vLLM."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from composimplex import (  # noqa: E402
    build_vllm_sampling_params,
    normalize_sampler_config,
)
from composimplex.metrics import DistributionMetricAccumulator  # noqa: E402
from composimplex.sampler import CompoSimplexSampler  # noqa: E402
from benchmark.utils import (  # noqa: E402
    BENCHMARK_SEED,
    TASK_TYPES,
    BenchmarkRun,
    PreparedTask,
    load_config,
    print_result,
    resolve_benchmark_config,
)


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    finish_reason: str
    distribution_stats: dict


def _token_id_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    if isinstance(value, Iterable):
        return {int(token_id) for token_id in value}
    return {int(value)}


def resolve_eos_token_ids(model, tokenizer) -> tuple[object, set[int]]:
    """Prefer model generation EOS values, including model-specific lists."""
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    return eos_token_id, _token_id_set(eos_token_id)


def format_prompt(tokenizer, prompt: str, config: dict[str, Any]) -> str:
    """Apply an explicitly configured chat template exactly once."""
    chat_config = config.get("chat_template", {})
    if not chat_config.get("enabled", False):
        if config.get("add_bos_token", False):
            return (tokenizer.bos_token or "") + prompt
        return prompt
    template_kwargs = {}
    if "enable_thinking" in chat_config:
        template_kwargs["enable_thinking"] = bool(chat_config["enable_thinking"])
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


@torch.no_grad()
def generate_samples(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    sampler: CompoSimplexSampler,
    max_new_tokens: int,
    num_samples: int,
    seed: int,
    eos_token_id,
    eos_token_ids: set[int],
    generation_context: dict | None = None,
    stop_token_sequences: tuple[tuple[int, ...], ...] = (),
) -> list[GenerationResult]:
    """Generate samples using one deterministic RNG stream per completion."""
    device = input_ids.device
    prompt_token_ids = input_ids[0].detach().tolist()
    generated_token_ids = [[] for _ in range(num_samples)]
    metric_accumulators = [
        DistributionMetricAccumulator() for _ in range(num_samples)
    ]
    finish_reasons = ["length"] * num_samples
    finished = [False] * num_samples
    context = generation_context or {}
    chooser_type = sampler.cfg.get("chooser", {}).get("type", "multinomial")
    batch_input_ids = input_ids.repeat(num_samples, 1)

    rngs = [torch.Generator(device=device) for _ in range(num_samples)]
    for sample_index, rng in enumerate(rngs):
        rng.manual_seed(seed + sample_index)
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    model_kwargs = {"logits_to_keep": 1} if model_type in ("gemma4", "lfm2") else {}
    outputs = model(input_ids=batch_input_ids, use_cache=True, **model_kwargs)
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    filler_token = tokenizer.pad_token_id
    if filler_token is None:
        filler_token = tokenizer.eos_token_id or 0

    for _ in range(max_new_tokens):
        next_tokens = []
        for sample_index in range(num_samples):
            if finished[sample_index]:
                next_tokens.append(filler_token)
                continue

            generated = generated_token_ids[sample_index]
            state = {
                "input_ids": [*prompt_token_ids, *generated],
                "prompt_token_ids": prompt_token_ids,
                "generated_token_ids": generated,
                "tokenizer": tokenizer,
                "eos_token_id": eos_token_id,
                "eos_token_ids": eos_token_ids,
                **context,
            }
            probabilities, support_token_ids, metrics = sampler.distribution(
                logits=logits[sample_index],
                state=state,
                return_metrics=True,
            )
            metric_accumulators[sample_index].update(metrics)
            if chooser_type == "argmax":
                support_index = torch.argmax(probabilities)
            else:
                support_index = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=rngs[sample_index],
                ).squeeze(0)
            token_id = int(support_token_ids[support_index])
            generated.append(token_id)
            next_tokens.append(token_id)
            if token_id in eos_token_ids:
                finished[sample_index] = True
                finish_reasons[sample_index] = "eos"
            elif any(
                len(generated) >= len(stop) and tuple(generated[-len(stop) :]) == stop
                for stop in stop_token_sequences
                if stop
            ):
                finished[sample_index] = True
                finish_reasons[sample_index] = "stop"

        if all(finished):
            break
        step_input = torch.tensor(
            next_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(1)
        outputs = model(
            input_ids=step_input,
            use_cache=True,
            past_key_values=past_key_values,
            **model_kwargs,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    return [
        GenerationResult(
            token_ids=torch.tensor(token_ids, device=device, dtype=torch.long),
            finish_reason=finish_reason,
            distribution_stats=accumulator.to_dict(),
        )
        for token_ids, finish_reason, accumulator in zip(
            generated_token_ids,
            finish_reasons,
            metric_accumulators,
        )
    ]


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).parent / "configs" / "qwen2_5_7b.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
    )
    parser.add_argument(
        "--benchmark",
        choices=tuple(TASK_TYPES),
        default=None,
        help="Override the config's default benchmark profile.",
    )
    return parser.parse_args()


def _model_dtype(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    mapping = {
        "auto": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def run_transformers(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_name = str(config["model_name"])
    sampler_config = normalize_sampler_config(config["sampler"])
    backend_config = config.get("transformers", {})
    max_model_len = int(backend_config.get("max_model_len", 4096))
    batch_size = int(backend_config.get("batch_size", config["run"]["num_samples"]))
    if max_model_len <= 128:
        raise ValueError("transformers.max_model_len must be greater than 128.")
    if batch_size < 1:
        raise ValueError("transformers.batch_size must be at least 1.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_class = AutoModelForCausalLM
    if model_config.model_type == "gemma4":
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=_model_dtype(str(config.get("dtype", "auto")), device),
        trust_remote_code=True,
    ).to(device)
    model.eval()
    sampler = CompoSimplexSampler(sampler_config)
    eos_token_id, eos_token_ids = resolve_eos_token_ids(model, tokenizer)
    stop_token_sequences = tuple(
        tuple(tokenizer.encode(stop, add_special_tokens=False))
        for stop in config.get("stop_sequences", [])
    )

    with BenchmarkRun(config, tokenizer, backend="transformers") as benchmark:
        for task in benchmark.tasks():
            formatted_prompt = format_prompt(tokenizer, task.prompt, config)
            encoded = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"][:, -(max_model_len - 128) :].to(device)
            prompt_length = int(input_ids.shape[1])
            request_max_tokens = min(
                int(sampler_config["max_new_tokens"]),
                max_model_len - prompt_length,
            )

            for sample_start in range(0, benchmark.num_samples, batch_size):
                current_batch_size = min(
                    batch_size,
                    benchmark.num_samples - sample_start,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                generations = generate_samples(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    sampler=sampler,
                    max_new_tokens=request_max_tokens,
                    num_samples=current_batch_size,
                    seed=benchmark.seed + sample_start,
                    eos_token_id=eos_token_id,
                    eos_token_ids=eos_token_ids,
                    generation_context={
                        "benchmark_name": benchmark.benchmark_name,
                        "reference": task.reference,
                    },
                    stop_token_sequences=stop_token_sequences,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - start) * 1000.0

                for batch_index, generation in enumerate(generations):
                    sample_index = sample_start + batch_index
                    generated_ids = generation.token_ids
                    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    benchmark.record(
                        task=task,
                        sample_index=sample_index,
                        completion=completion,
                        prompt_token_ids=input_ids[0].detach().cpu().tolist(),
                        completion_token_ids=generated_ids.detach().cpu().tolist(),
                        finish_reason=generation.finish_reason,
                        latency_ms=latency_ms,
                        generation_batch_id=f"task_{task.index}_samples_{sample_start}",
                        generation_batch_latency_ms=latency_ms,
                        distribution_stats=generation.distribution_stats,
                    )
        return benchmark.finish()


def _build_vllm(config: dict[str, Any], seed: int):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "The vLLM runner requires `pip install -e '.[benchmark,vllm]'`."
        ) from exc
    backend_config = config.get("vllm", {})
    kwargs: dict[str, Any] = {
        "model": config["model_name"],
        "trust_remote_code": True,
        "dtype": str(config.get("dtype", "auto")),
        "seed": seed,
        "tensor_parallel_size": int(backend_config.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(
            backend_config.get("gpu_memory_utilization", 0.9)
        ),
    }
    for key in (
        "max_model_len",
        "max_num_seqs",
        "enforce_eager",
        "skip_tokenizer_init",
    ):
        if key in backend_config:
            kwargs[key] = backend_config[key]
    return LLM(**kwargs)


def _request_latency_ms(request_output, fallback_ms: float) -> float:
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return fallback_ms
    finished = getattr(metrics, "finished_time", None)
    started = getattr(metrics, "arrival_time", None)
    if started is None:
        started = getattr(metrics, "first_scheduled_time", None)
    if finished is None or started is None:
        return fallback_ms
    return max(0.0, float(finished - started) * 1000.0)


def run_vllm(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        trust_remote_code=True,
    )
    llm = _build_vllm(
        config,
        int(config.get("run", {}).get("seed", BENCHMARK_SEED)),
    )
    backend_config = config.get("vllm", {})
    batch_size = int(backend_config.get("batch_size", 1))
    skip_tokenizer_init = bool(backend_config.get("skip_tokenizer_init", False))
    max_model_len = int(backend_config.get("max_model_len", 4096))
    if batch_size < 1:
        raise ValueError("vllm.batch_size must be at least 1.")
    if max_model_len <= 128:
        raise ValueError("vllm.max_model_len must be greater than 128.")

    with BenchmarkRun(config, tokenizer, backend="vllm") as benchmark:
        task_buffer: list[PreparedTask] = []
        batch_index = 0

        def flush_batch() -> None:
            nonlocal batch_index
            if not task_buffer:
                return
            generation_batch_id = f"batch_{batch_index}"
            prompt_token_ids: list[list[int]] = []
            sampling_params = []
            metadata: list[
                tuple[PreparedTask, int, list[int], DistributionMetricAccumulator]
            ] = []
            for task in task_buffer:
                formatted_prompt = format_prompt(tokenizer, task.prompt, config)
                prompt_ids = tokenizer.encode(
                    formatted_prompt,
                    add_special_tokens=False,
                )[-(max_model_len - 128) :]
                request_max_tokens = min(
                    int(benchmark.sampler_config["max_new_tokens"]),
                    max_model_len - len(prompt_ids),
                )
                for sample_index in range(benchmark.num_samples):
                    metric_accumulator = DistributionMetricAccumulator()
                    prompt_token_ids.append(list(prompt_ids))
                    sampling_params.append(
                        build_vllm_sampling_params(
                            sampler_config=benchmark.sampler_config,
                            tokenizer=tokenizer,
                            generation_context={
                                "benchmark_name": benchmark.benchmark_name,
                                "reference": task.reference,
                            },
                            metric_accumulator=metric_accumulator,
                            max_tokens=request_max_tokens,
                            detokenize=not skip_tokenizer_init,
                            stop_token_ids=(
                                sorted(_token_id_set(tokenizer.eos_token_id))
                                if skip_tokenizer_init
                                else None
                            ),
                            n=1,
                            seed=benchmark.sample_seed(sample_index),
                        )
                    )
                    metadata.append(
                        (task, sample_index, list(prompt_ids), metric_accumulator)
                    )

            start = time.perf_counter()
            outputs = llm.generate(
                prompts=None,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            fallback_latency_ms = elapsed_ms

            for request_output, request_metadata in zip(outputs, metadata):
                (
                    task,
                    sample_index,
                    prompt_ids,
                    metric_accumulator,
                ) = request_metadata
                output_prompt_ids = getattr(request_output, "prompt_token_ids", None)
                latency_ms = _request_latency_ms(
                    request_output,
                    fallback_latency_ms,
                )
                distribution_stats = metric_accumulator.to_dict()
                generated = request_output.outputs[0]
                completion_token_ids = list(getattr(generated, "token_ids", []))
                completion = (
                    tokenizer.decode(completion_token_ids, skip_special_tokens=True)
                    if skip_tokenizer_init
                    else generated.text
                )
                actual_prompt_ids = (
                    list(output_prompt_ids)
                    if output_prompt_ids is not None
                    else prompt_ids
                )
                benchmark.record(
                    task=task,
                    sample_index=sample_index,
                    completion=completion,
                    prompt_token_ids=actual_prompt_ids,
                    completion_token_ids=completion_token_ids,
                    finish_reason=str(
                        getattr(generated, "finish_reason", "unknown")
                    ),
                    latency_ms=latency_ms,
                    generation_batch_id=generation_batch_id,
                    generation_batch_latency_ms=elapsed_ms,
                    distribution_stats=distribution_stats,
                )
            task_buffer.clear()
            batch_index += 1

        for task in benchmark.tasks():
            task_buffer.append(task)
            if len(task_buffer) >= batch_size:
                flush_batch()
        flush_batch()
        return benchmark.finish()


def main() -> None:
    args = parse_args()
    config = resolve_benchmark_config(load_config(args.config), args.benchmark)
    if config["benchmark"] == "IFEVAL":
        from benchmark.grader import load_ifeval_grader

        load_ifeval_grader()
    runner = run_transformers if args.backend == "transformers" else run_vllm
    print_result(runner(config))


if __name__ == "__main__":
    main()
