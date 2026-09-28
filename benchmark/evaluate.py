#!/usr/bin/env python3
"""Recompute benchmark metrics from saved JSONL outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from composimplex import normalize_sampler_config  # noqa: E402
from benchmark.grader import grade_outputs  # noqa: E402
from benchmark.utils import (  # noqa: E402
    BENCHMARK_SEED,
    SEMANTIC_EMBEDDING_MODEL,
    TASK_TYPES,
    evaluation_metric_names,
    infer_task_type,
    load_config,
    metric_k_values,
    print_result,
    resolve_benchmark_config,
    summarize_records,
    write_json,
)


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).parent / "configs" / "qwen2_5_7b.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--summary", default=None)
    parser.add_argument("--regrade", action="store_true", help="Rerun official code or IFEval graders.")
    parser.add_argument(
        "--benchmark",
        choices=tuple(TASK_TYPES),
        default=None,
        help="Override the config's default benchmark profile.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def evaluate(
    outputs_path: Path,
    summary_path: Path,
    config: dict[str, Any],
    regrade: bool = False,
) -> dict[str, Any]:
    records = load_records(outputs_path)
    outputs_path = grade_outputs(records, config, outputs_path, regrade=regrade)
    evaluation_config = config.get("evaluation", {})
    num_samples = int(config.get("run", {}).get("num_samples", 1))
    semantic_model_name = str(
        evaluation_config.get(
            "semantic_embedding_model",
            SEMANTIC_EMBEDDING_MODEL,
        )
    )
    semantic_device = evaluation_config.get("semantic_device")
    summary = summarize_records(
        records,
        evaluation_metric_names(config, num_samples),
        [float(record.get("latency_ms", 0.0)) for record in records],
        semantic_model_name=semantic_model_name,
        semantic_batch_size=int(
            evaluation_config.get("semantic_batch_size", 32)
        ),
        semantic_device=str(semantic_device) if semantic_device else None,
    )
    first_record = records[0] if records else {}
    task_ids = {
        str(record.get("task_id", str(record["sample_id"]).rsplit("_", 1)[0]))
        for record in records
    }
    benchmark = str(config["benchmark"]).upper()
    summary.update(
        {
            "model_name": str(config["model_name"]),
            "protocol": first_record.get("protocol", "legacy_unknown"),
            "benchmark": benchmark,
            "task_type": str(config.get("task_type", infer_task_type(benchmark))),
            "backend": first_record.get("backend", "unknown"),
            "sampler_name": "composimplex",
            "num_examples": len(task_ids),
            "num_samples_per_example": num_samples,
            "k_values": metric_k_values(num_samples),
            "semantic_embedding_model": semantic_model_name,
            "seed": int(first_record.get("seed", BENCHMARK_SEED)),
            "outputs_path": str(outputs_path),
            "sampler": normalize_sampler_config(config["sampler"]),
        }
    )
    write_json(summary_path, summary)
    return {
        "outputs_path": str(outputs_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    outputs_path = Path(args.outputs)
    summary_path = (
        Path(args.summary)
        if args.summary
        else outputs_path.with_name("summary.json")
    )
    config = resolve_benchmark_config(load_config(args.config), args.benchmark)
    print_result(evaluate(outputs_path, summary_path, config, regrade=args.regrade))


if __name__ == "__main__":
    main()
