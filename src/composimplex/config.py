"""Configuration loading and component construction for CompoSimplex."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from composimplex.solvers import MirrorAscentOptimizer
from composimplex.regularizers import (
    CoverageRegularizer,
    DiversityGapRegularizer,
    EntropyRegularizer,
    JSToBaseRegularizer,
    KLToBaseRegularizer,
)


OPTIMIZER_REGISTRY = {
    "mirror_ascent": MirrorAscentOptimizer,
}


def _build_kl_to_base(cfg: dict):
    return KLToBaseRegularizer(alpha=float(cfg["alpha"]))


def _build_js_to_base(cfg: dict):
    return JSToBaseRegularizer(
        alpha=float(cfg["alpha"]),
        eps=float(cfg.get("eps", 1e-8)),
    )


def _build_entropy(cfg: dict):
    return EntropyRegularizer(
        alpha=float(cfg["alpha"]),
        eps=float(cfg.get("eps", 1e-8)),
    )


def _build_coverage(cfg: dict):
    return CoverageRegularizer(
        alpha=float(cfg["alpha"]),
        K=int(cfg.get("K", 16)),
        weight_mode=str(cfg.get("weight_mode", "topm_uniform")),
        top_m=int(cfg.get("top_m", 8)),
        eps=float(cfg.get("eps", 1e-8)),
    )


def _build_diversity_gap(cfg: dict):
    return DiversityGapRegularizer(
        alpha=float(cfg["alpha"]),
        K=int(cfg.get("K", 16)),
        gap_tau=float(cfg.get("gap_tau", 1.0)),
        normalize=bool(cfg.get("normalize", True)),
        eps=float(cfg.get("eps", 1e-8)),
    )


REGULARIZER_REGISTRY = {
    "kl_to_base": _build_kl_to_base,
    "js_to_base": _build_js_to_base,
    "entropy": _build_entropy,
    "coverage": _build_coverage,
    "diversity_gap": _build_diversity_gap,
}

SUPPORTED_REGULARIZERS = set(REGULARIZER_REGISTRY)


def load_sampler_config(path: str | Path, key: str = "sampler") -> dict[str, Any]:
    """Load and normalize a sampler-only or experiment-style YAML mapping."""
    with open(path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    if key and key in data:
        data = data[key]
    return normalize_sampler_config(data)


def normalize_sampler_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a validated sampler config with all runtime defaults filled."""
    if not isinstance(cfg, dict):
        raise ValueError("Sampler config must be a dictionary.")

    normalized = dict(cfg)
    normalized.setdefault("temperature", 1.0)
    normalized.setdefault("max_new_tokens", 128)
    normalized.setdefault("support", {"type": "topm", "value": 200})
    normalized.setdefault("base_distribution", {"type": "softmax", "temperature": 1.0})
    normalized.setdefault("lambda", 1.0)
    normalized.setdefault("solver", "auto")
    normalized.setdefault("optimizer", {"name": "mirror_ascent", "steps": 50, "lr": 0.5})
    normalized.setdefault("chooser", {"type": "multinomial"})
    normalized.setdefault("regularizers", [])
    normalized.setdefault("eps", 1e-8)

    lambda_ = float(normalized["lambda"])
    if not math.isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError("lambda must be a finite non-negative number.")

    solver = str(normalized["solver"])
    if solver not in {"auto", "closed_form", "mirror_ascent", "softmax"}:
        raise ValueError(
            "solver must be one of: auto, closed_form, mirror_ascent, softmax."
        )

    optimizer_cfg = normalized["optimizer"]
    if not isinstance(optimizer_cfg, dict):
        raise ValueError("optimizer must be a mapping.")
    optimizer_name = optimizer_cfg.get("name", "mirror_ascent")
    if optimizer_name not in OPTIMIZER_REGISTRY:
        allowed = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}. Supported: {allowed}")
    steps = int(optimizer_cfg.get("steps", 50))
    lr = float(optimizer_cfg.get("lr", 0.5))
    tol = float(optimizer_cfg.get("tol", 1e-5))
    if steps < 1:
        raise ValueError("optimizer.steps must be at least 1.")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("optimizer.lr must be a finite positive number.")
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("optimizer.tol must be a finite non-negative number.")

    chooser_cfg = normalized["chooser"]
    if not isinstance(chooser_cfg, dict):
        raise ValueError("chooser must be a mapping.")
    chooser_type = chooser_cfg.get("type", "multinomial")
    if chooser_type not in {"argmax", "multinomial"}:
        raise ValueError(f"Unsupported chooser type: {chooser_type!r}.")

    regularizer_configs = normalized["regularizers"]
    if not isinstance(regularizer_configs, list):
        raise ValueError("regularizers must be a list.")
    if solver == "softmax" and regularizer_configs:
        raise ValueError("solver='softmax' requires regularizers: [].")

    seen_types = set()
    alpha_sum = 0.0
    for regularizer_cfg in regularizer_configs:
        if not isinstance(regularizer_cfg, dict) or "type" not in regularizer_cfg:
            raise ValueError(
                f"Each regularizer must be a mapping with a type: {regularizer_cfg!r}"
            )

        regularizer_type = regularizer_cfg["type"]
        if regularizer_type not in SUPPORTED_REGULARIZERS:
            allowed = ", ".join(sorted(SUPPORTED_REGULARIZERS))
            raise ValueError(
                f"Unsupported regularizer {regularizer_type!r}. Supported: {allowed}"
            )
        if regularizer_type in seen_types:
            raise ValueError(
                f"Duplicate regularizer type is not allowed: {regularizer_type!r}"
            )
        seen_types.add(regularizer_type)

        if "weight" in regularizer_cfg:
            raise ValueError(
                "Regularizer 'weight' is no longer supported. "
                "Use sampler-level 'lambda' and per-regularizer 'alpha'."
            )
        if "alpha" not in regularizer_cfg:
            raise ValueError(f"Regularizer {regularizer_type!r} must define alpha.")

        alpha = float(regularizer_cfg["alpha"])
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError(
                "Each regularizer alpha must be a finite non-negative number."
            )
        alpha_sum += alpha

    if regularizer_configs and not math.isclose(
        alpha_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"Regularizer alphas must sum to 1, got {alpha_sum}.")

    return normalized


def build_distribution_optimizer(cfg: dict):
    """Construct the optimizer selected by a normalized optimizer config."""
    name = cfg.get("name", "mirror_ascent")
    optimizer_cls = OPTIMIZER_REGISTRY.get(name)
    if optimizer_cls is None:
        allowed = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise ValueError(f"Unknown distribution optimizer {name!r}. Supported: {allowed}")
    return optimizer_cls(
        steps=int(cfg.get("steps", 50)),
        lr=float(cfg.get("lr", 0.5)),
        eps=float(cfg.get("eps", 1e-8)),
        tol=float(cfg.get("tol", 1e-5)),
    )


def build_regularizers(config_list: list[dict]):
    """Construct ordered regularizer objects from configuration mappings."""
    regularizers = []
    for cfg in config_list:
        regularizer_type = cfg["type"]
        regularizer_builder = REGULARIZER_REGISTRY.get(regularizer_type)

        if regularizer_builder is not None:
            regularizer = regularizer_builder(cfg)
        else:
            raise ValueError(f"Unknown regularizer type: {regularizer_type}")

        regularizers.append(regularizer)

    return regularizers
