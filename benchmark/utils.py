"""Dataset, prompt, result, and metric utilities for supported benchmarks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator

import torch
import yaml
from tqdm import tqdm

from benchmark.grader import (
    CODE_BENCHMARKS,
    IFEVAL_REVISION,
    code_prompt,
    grade_outputs,
    load_code_benchmark,
    load_ifeval_benchmark,
    extract_math_answer,
    grade_completion,
    normalize_math_expression,
    parse_gpqa_answer,
)
from composimplex.metrics import DISTRIBUTION_METRIC_NAMES
from composimplex.config import normalize_sampler_config


BENCHMARK_SEED = 0
BENCHMARK_PROTOCOL = "qwen_plain_cot_sample_seed_v1"
SEMANTIC_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EVALUATION_METRICS = (
    "pass_at_k",
    "sc_at_k",
    "semantic_reasoning_diversity",
    "latency",
    "throughput",
    "completion_tokens",
    "kl_q_p",
    "js_q_p",
    "entropy",
    "topm_coverage",
    "diversity_gap",
)


TASK_TYPES = {
    "MATH": "math_reasoning",
    "GPQA": "multiple_choice_reasoning",
    "LIVECODEBENCH_V6": "code_generation",
    "IFEVAL": "instruction_following",
}

# Qwen's plain COT prompts in aakaran/reasoning-with-sampling/llm_experiments.
MATH_PROMPT = "Can you solve the following math problem? "
MATH_COT = (
    " Please reason step by step, and put your final answer within \\boxed{{}}."
)
GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: '\\boxed{{$LETTER}}' (without quotes) "
    "where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by step before "
    "answering.\n\n{question}\n\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}"
)


# ---------------------------------------------------------------------------
# Config, output, and reproducibility
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Benchmark config must contain a mapping: {path}")
    return config


def resolve_benchmark_config(
    config: dict[str, Any],
    benchmark_name: str | None = None,
) -> dict[str, Any]:
    """Merge one named benchmark profile into the shared experiment config."""
    resolved = dict(config)
    selected = str(benchmark_name or config.get("benchmark", "MATH")).upper()
    profiles = config.get("benchmarks", {})
    if profiles:
        profile = profiles.get(selected)
        if not isinstance(profile, dict):
            supported = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown benchmark profile {selected!r}. Supported: {supported}"
            )
        resolved.update(profile)
    resolved["benchmark"] = selected
    resolved.pop("benchmarks", None)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Datasets and prompts
# ---------------------------------------------------------------------------


def infer_task_type(benchmark_name: str) -> str:
    try:
        return TASK_TYPES[benchmark_name.upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_TYPES))
        raise ValueError(
            f"Unsupported benchmark {benchmark_name!r}. Supported: {supported}"
        ) from exc


def load_benchmark(benchmark_name: str, split: str | None = None):
    if benchmark_name.upper() == "IFEVAL":
        if split not in (None, "train"):
            raise ValueError("IFEval has only the official 'train' split (541 evaluation prompts).")
        return load_ifeval_benchmark()
    if benchmark_name.upper() in CODE_BENCHMARKS:
        return load_code_benchmark(benchmark_name.upper())
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Benchmark datasets require `pip install -e '.[benchmark]'`."
        ) from exc

    benchmark_name = benchmark_name.upper()
    if benchmark_name == "MATH":
        return load_dataset("nlile/hendrycks-MATH-benchmark", split=split or "test")
    if benchmark_name == "GPQA":
        return load_dataset(
            "Idavidrein/gpqa",
            "gpqa_diamond",
            split=split or "train",
        )
    infer_task_type(benchmark_name)
    raise AssertionError("unreachable")


def _first_present(item: dict[str, Any], *names: str):
    for name in names:
        value = item.get(name)
        if value is not None:
            return value
    return None


def process_item(
    item: dict[str, Any],
    benchmark_name: str,
    rng: random.Random | None = None,
) -> tuple[str, str, dict[str, Any]]:
    benchmark_name = benchmark_name.upper()
    if benchmark_name == "IFEVAL":
        return item["prompt"], "", {
            "task_id": str(item["key"]),
            "grading_input": {key: item[key] for key in ("key", "instruction_id_list", "kwargs")},
            "dataset_revision": IFEVAL_REVISION,
        }
    if benchmark_name in CODE_BENCHMARKS:
        task_id = str(item.get("task_id", item.get("question_id")))
        return code_prompt(item, benchmark_name), task_id, {"task_id": task_id}
    if benchmark_name == "MATH":
        question = _first_present(item, "problem", "prompt")
        reference = _first_present(item, "answer", "solution")
        if question is None or reference is None:
            raise ValueError(f"MATH item is missing problem/solution fields: {item}")
        prompt = MATH_PROMPT + str(question) + MATH_COT
        return prompt, str(reference), {"question": str(question)}

    if benchmark_name == "GPQA":
        question = _first_present(item, "Question", "question")
        correct = _first_present(item, "Correct Answer", "correct_answer")
        incorrect = [
            _first_present(item, "Incorrect Answer 1", "incorrect_answer_1"),
            _first_present(item, "Incorrect Answer 2", "incorrect_answer_2"),
            _first_present(item, "Incorrect Answer 3", "incorrect_answer_3"),
        ]
        if question is None or correct is None or any(x is None for x in incorrect):
            raise ValueError(f"GPQA item is missing question/answer fields: {item}")
        choices = [str(value) for value in incorrect]
        prompt_rng = rng or random
        prompt_rng.shuffle(choices)
        gold_index = prompt_rng.randint(0, 3)
        choices.insert(gold_index, str(correct))
        query = GPQA_QUERY_TEMPLATE.format(
            question=question,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
        return query, "ABCD"[gold_index], {
            "question": str(question),
            "choices": choices,
        }
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def metric_k_values(num_samples: int) -> list[int]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    values = []
    k = 1
    while k <= num_samples:
        values.append(k)
        k *= 2
    if values[-1] != num_samples:
        values.append(num_samples)
    return values


def evaluation_metric_names(config: dict[str, Any], num_samples: int) -> list[str]:
    """Resolve the configured metric list, expanding pass/sc/all-pass at k."""
    configured = config.get("evaluation", {}).get("metrics")
    names = (
        list(DEFAULT_EVALUATION_METRICS)
        if configured is None
        else list(configured)
    )
    k_values = metric_k_values(num_samples)
    if config.get("benchmark") == "IFEVAL":
        if configured is None:
            names = ["all_pass_at_k" if n == "sc_at_k" else n for n in names]
        elif any(n.startswith("sc_at_") for n in names):
            raise ValueError("SC@k is not defined for IFEval; use all_pass_at_k.")
    if config.get("benchmark") in CODE_BENCHMARKS:
        if configured is None:
            names = [n for n in names if n not in ("sc_at_k", "semantic_reasoning_diversity")]
        elif any(n == "sc_at_k" or n.startswith("sc_at_") for n in names):
            raise ValueError("SC@k is not defined for code-generation benchmarks.")
    if config.get("benchmark") == "LIVECODEBENCH_V6" and configured is None:
        names.append("hard_pass_at_1")
        if num_samples >= 16:
            names.append("all_pass_at_16")
    expanded = []
    for name in names:
        if name == "pass_at_k":
            expanded.extend(f"pass_at_{k}" for k in k_values)
        elif name == "sc_at_k":
            expanded.extend(f"sc_at_{k}" for k in k_values)
        elif name == "all_pass_at_k":
            expanded.extend(f"all_pass_at_{k}" for k in k_values)
        else:
            expanded.append(name)
    return list(dict.fromkeys(expanded))


def _record_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_id = record.get("task_id")
        if task_id is None:
            task_id = str(record["sample_id"]).rsplit("_", 1)[0]
        groups[str(task_id)].append(record)
    return [
        sorted(
            group,
            key=lambda record: int(str(record["sample_id"]).rsplit("_", 1)[1]),
        )
        for group in groups.values()
    ]


def pass_at_k(records: list[dict[str, Any]], k: int) -> float:
    """Fraction of tasks with at least one correct answer among the first k."""
    if any(not isinstance(record.get("correct"), bool) for record in records):
        raise ValueError("Some samples are ungraded; run benchmark/evaluate.py first.")
    groups = _record_groups(records)
    if not groups:
        return 0.0
    return mean(
        float(any(record["correct"] for record in group[:k]))
        for group in groups
    )


def hard_pass_at_1(records: list[dict[str, Any]]) -> float | None:
    """Prefix pass@1 on Hard LiveCodeBench tasks; None if the subset is empty."""
    if any(r.get("benchmark") != "LIVECODEBENCH_V6" for r in records):
        raise ValueError("Hard pass@1 requires LiveCodeBench records.")
    difficulties = {
        str(item["question_id"]): item["difficulty"].lower()
        for item in load_code_benchmark("LIVECODEBENCH_V6")
    }
    hard_records = [r for r in records if difficulties[str(r["task_id"])] == "hard"]
    if any(str(group[0]["sample_id"]).rsplit("_", 1)[1] != "0"
           for group in _record_groups(hard_records)):
        raise ValueError("Hard pass@1 requires sample 0 for every Hard task.")
    return pass_at_k(hard_records, 1) if hard_records else None


def canonical_answer(completion: str, benchmark: str) -> str:
    if benchmark.upper() == "MATH":
        answer = normalize_math_expression(extract_math_answer(completion))
        return answer or "PARSE_ERROR"
    if benchmark.upper() == "GPQA":
        return parse_gpqa_answer(completion)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def all_pass_at_k(records: list[dict[str, Any]], k: int) -> float:
    """Fraction of tasks whose first k samples ALL satisfy the strict grader."""
    if k < 1:
        raise ValueError("k must be positive.")
    if any(not isinstance(record.get("correct"), bool) for record in records):
        raise ValueError("Some samples are ungraded; run benchmark/evaluate.py first.")
    groups = _record_groups(records)
    for group in groups:
        indices = [int(str(r["sample_id"]).rsplit("_", 1)[1]) for r in group[:k]]
        if indices != list(range(k)):
            raise ValueError(f"all-pass@{k} requires samples 0 through {k - 1} for every task.")
    return mean(float(all(r["correct"] for r in group[:k])) for group in groups) if groups else 0.0


def sc_at_k(records: list[dict[str, Any]], k: int) -> float:
    """Majority-vote accuracy over the first k answers; ties use sample order."""
    groups = _record_groups(records)
    if not groups:
        return 0.0
    scores = []
    for group in groups:
        selected = group[:k]
        answers = [str(record["answer"]) for record in selected]
        counts = Counter(answers)
        winner = max(counts, key=lambda answer: (counts[answer], -answers.index(answer)))
        scores.append(
            float(
                next(
                    record["correct"]
                    for record in selected
                    if record["answer"] == winner
                )
            )
        )
    return mean(scores)


def reasoning_trace(completion: str) -> str:
    blocks = re.findall(
        r"<think>(.*?)</think>",
        completion,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if blocks:
        return blocks[-1].strip()
    return re.split(r"<answer>", completion, maxsplit=1, flags=re.IGNORECASE)[0].strip()


@torch.no_grad()
def semantic_reasoning_diversity(
    records: list[dict[str, Any]],
    model_name: str = SEMANTIC_EMBEDDING_MODEL,
    batch_size: int = 32,
    device: str | None = None,
) -> float:
    """Mean pairwise cosine distance between reasoning traces for each task."""
    groups = _record_groups(records)
    if not any(len(group) > 1 for group in groups):
        return 0.0
    from transformers import AutoModel, AutoTokenizer

    texts = [
        reasoning_trace(str(record.get("completion", "")))
        for group in groups
        for record in group
    ]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        embeddings.append(torch.nn.functional.normalize(pooled.float(), dim=1).cpu())
    embeddings = torch.cat(embeddings)

    scores = []
    offset = 0
    for group in groups:
        group_embeddings = embeddings[offset : offset + len(group)]
        offset += len(group)
        if len(group) < 2:
            continue
        similarities = group_embeddings @ group_embeddings.T
        pairs = torch.triu_indices(len(group), len(group), offset=1)
        scores.append(
            float((1.0 - similarities[pairs[0], pairs[1]]).clamp(0.0, 2.0).mean())
        )
    return mean(scores) if scores else 0.0


def distribution_metric_mean(records: list[dict[str, Any]], name: str) -> float:
    """Average a token-level distribution metric per completion."""
    values = []
    for record in records:
        stats = record.get("distribution_stats") or {}
        num_steps = int(stats.get("num_steps", 0))
        metric_sum = stats.get("sums", {}).get(name)
        if num_steps > 0 and metric_sum is not None:
            values.append(float(metric_sum) / num_steps)
    return mean(values) if values else 0.0


def summarize_records(
    records: list[dict[str, Any]],
    requested_metrics: list[str],
    latencies_ms: list[float],
    semantic_model_name: str = SEMANTIC_EMBEDDING_MODEL,
    semantic_batch_size: int = 32,
    semantic_device: str | None = None,
) -> dict[str, Any]:
    metrics: dict[str, float | None] = {}
    for name in requested_metrics:
        if name == "hard_pass_at_1":
            metrics[name] = hard_pass_at_1(records)
        elif name.startswith("all_pass_at_"):
            metrics[name] = all_pass_at_k(records, int(name.removeprefix("all_pass_at_")))
        elif name.startswith("pass_at_"):
            try:
                metrics[name] = pass_at_k(records, int(name.removeprefix("pass_at_")))
            except ValueError:
                pass
        elif name.startswith("sc_at_"):
            try:
                metrics[name] = sc_at_k(records, int(name.removeprefix("sc_at_")))
            except ValueError:
                pass
        elif name in DISTRIBUTION_METRIC_NAMES:
            metrics[name] = distribution_metric_mean(records, name)
        elif name == "semantic_reasoning_diversity":
            metrics[name] = semantic_reasoning_diversity(
                records,
                model_name=semantic_model_name,
                batch_size=semantic_batch_size,
                device=semantic_device,
            )
        elif name == "latency":
            metrics["avg_latency_ms"] = mean(latencies_ms) if latencies_ms else 0.0
            metrics["latency_std_ms"] = (
                pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
            )
        elif name == "throughput":
            batches = {
                str(record["generation_batch_id"]): float(
                    record["generation_batch_latency_ms"]
                )
                for record in records
            }
            elapsed_seconds = sum(batches.values()) / 1000.0
            completion_tokens = sum(record["completion_tokens"] for record in records)
            metrics["output_tokens_per_second"] = (
                completion_tokens / elapsed_seconds if elapsed_seconds else 0.0
            )
        elif name == "completion_tokens":
            lengths = [record["completion_tokens"] for record in records]
            metrics["avg_completion_tokens"] = mean(lengths) if lengths else 0.0
    return {"metrics": metrics}


# ---------------------------------------------------------------------------
# Shared run lifecycle
# ---------------------------------------------------------------------------


@dataclass
class PreparedTask:
    index: int
    prompt: str
    reference: str
    metadata: dict[str, Any]


class BenchmarkRun:
    def __init__(self, config: dict[str, Any], tokenizer, backend: str):
        self.config = config
        self.tokenizer = tokenizer
        self.backend = backend
        self.model_name = str(config["model_name"])
        self.benchmark_name = str(config["benchmark"]).upper()
        self.protocol = (
            "code_plain_completion_sample_seed_v1"
            if self.benchmark_name in CODE_BENCHMARKS else BENCHMARK_PROTOCOL
        )
        if self.benchmark_name in CODE_BENCHMARKS and config.get("chat_template", {}).get("enabled", False):
            self.protocol = "code_chat_completion_sample_seed_v1"
        if self.benchmark_name == "IFEVAL":
            prompt_format = "chat" if config.get("chat_template", {}).get("enabled", False) else "plain"
            self.protocol = f"ifeval_{prompt_format}_strict_sample_seed_v1"
        self.split = config.get("split")
        self.task_type = str(
            config.get("task_type", infer_task_type(self.benchmark_name))
        )
        self.sampler_config = normalize_sampler_config(config["sampler"])

        run_config = config.get("run", {})
        self.seed = int(run_config.get("seed", BENCHMARK_SEED))
        self.num_samples = int(run_config.get("num_samples", 1))
        self.max_examples = run_config.get("max_examples")
        if self.max_examples is not None:
            self.max_examples = max(0, int(self.max_examples))
        if self.num_samples < 1:
            raise ValueError("run.num_samples must be at least 1.")

        set_global_seed(self.seed)
        self._prompt_random = random.Random(self.seed)
        self.dataset = load_benchmark(self.benchmark_name, split=self.split)
        self.dataset_size = len(self.dataset)
        if self.max_examples is not None:
            self.dataset_size = min(self.dataset_size, self.max_examples)

        output_template = str(
            run_config.get("output_dir", "results/{benchmark}/{backend}")
        )
        output_template = f"{output_template.rstrip('/')}/seed_{self.seed}"
        output_dir = str(output_template).format(
            benchmark=self.benchmark_name.lower(),
            backend=backend,
        )
        self.output_dir = ensure_dir(output_dir)
        self.outputs_path = self.output_dir / "outputs.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self._output_file = open(self.outputs_path, "x", encoding="utf-8")
        root = Path(__file__).resolve().parents[1]
        sources = ["benchmark/run.py", "benchmark/utils.py", "benchmark/grader.py"]
        sources += [f"src/composimplex/{name}.py" for name in
                    ("config", "sampler", "regularizers", "solvers", "metrics", "supports")]
        write_json(self.output_dir / "config.json", {
            "protocol": self.protocol, "config": config,
            "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in sources},
        })
        self.records: list[dict[str, Any]] = []
        self.latencies_ms: list[float] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        if not self._output_file.closed:
            self._output_file.close()

    def tasks(self) -> Iterator[PreparedTask]:
        iterator = tqdm(
            enumerate(self.dataset),
            total=self.dataset_size,
            desc=f"{self.benchmark_name}/{self.backend}",
        )
        for index, item in iterator:
            if self.max_examples is not None and index >= self.max_examples:
                break
            prompt, reference, metadata = process_item(
                item=item,
                benchmark_name=self.benchmark_name,
                rng=self._prompt_random,
            )
            yield PreparedTask(index, prompt, reference, metadata)

    def sample_seed(self, sample_index: int) -> int:
        """Return the shared per-completion seed used for every task."""
        return self.seed + int(sample_index)

    def record(
        self,
        task: PreparedTask,
        sample_index: int,
        completion: str,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        finish_reason: str,
        latency_ms: float,
        generation_batch_id: str,
        generation_batch_latency_ms: float,
        distribution_stats: dict | None = None,
    ) -> None:
        deferred_grading = self.benchmark_name in (*CODE_BENCHMARKS, "IFEVAL")
        answer = "" if deferred_grading else canonical_answer(completion, self.benchmark_name)
        correct = None if deferred_grading else grade_completion(
            completion,
            task.reference,
            self.benchmark_name,
        )
        public_record = {
            "protocol": self.protocol,
            "sample_id": f"{task.index}_{sample_index}",
            "task_id": task.metadata.get("task_id", str(task.index)),
            "sample_index": int(sample_index),
            "benchmark": self.benchmark_name,
            "task_type": self.task_type,
            "backend": self.backend,
            "prompt": task.prompt,
            "completion": completion,
            "answer": answer,
            "reference": task.reference,
            "correct": correct,
            "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
            "completion_token_ids": [
                int(token_id) for token_id in completion_token_ids
            ],
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": len(completion_token_ids),
            "finish_reason": finish_reason,
            "seed": self.seed,
            "sample_seed": self.sample_seed(sample_index),
            "distribution_stats": distribution_stats,
            "latency_ms": float(latency_ms),
            "generation_batch_id": str(generation_batch_id),
            "generation_batch_latency_ms": float(generation_batch_latency_ms),
        }
        if self.benchmark_name == "IFEVAL":
            public_record.update(task.metadata)
        self._output_file.write(json.dumps(public_record, ensure_ascii=False) + "\n")
        self._output_file.flush()
        self.records.append(public_record)
        self.latencies_ms.append(float(latency_ms))

    def finish(self) -> dict[str, Any]:
        self.close()
        self.outputs_path = grade_outputs(self.records, self.config, self.outputs_path)
        evaluation_config = self.config.get("evaluation", {})
        requested_metrics = evaluation_metric_names(self.config, self.num_samples)
        semantic_device = evaluation_config.get("semantic_device")
        semantic_model_name = str(
            evaluation_config.get(
                "semantic_embedding_model",
                SEMANTIC_EMBEDDING_MODEL,
            )
        )
        summary = summarize_records(
            self.records,
            requested_metrics,
            self.latencies_ms,
            semantic_model_name=semantic_model_name,
            semantic_batch_size=int(
                evaluation_config.get("semantic_batch_size", 32)
            ),
            semantic_device=str(semantic_device) if semantic_device else None,
        )
        summary.update(
            {
                "model_name": self.model_name,
                "protocol": self.protocol,
                "benchmark": self.benchmark_name,
                "task_type": self.task_type,
                "backend": self.backend,
                "sampler_name": "composimplex",
                "num_examples": len(self.records) // self.num_samples,
                "num_samples_per_example": self.num_samples,
                "k_values": metric_k_values(self.num_samples),
                "semantic_embedding_model": semantic_model_name,
                "seed": self.seed,
                "outputs_path": str(self.outputs_path),
                "sampler": self.sampler_config,
            }
        )
        write_json(self.summary_path, summary)
        return {
            "outputs_path": str(self.outputs_path),
            "summary_path": str(self.summary_path),
            "summary": summary,
        }


def print_result(result: dict[str, Any]) -> None:
    print(f"Saved outputs to {result['outputs_path']}", flush=True)
    print(f"Saved summary to {result['summary_path']}", flush=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
