"""Objective construction, solution, and stateful compositional sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from composimplex.metrics import compute_distribution_metrics
from composimplex.supports import apply_softmax, apply_support
from composimplex.config import (
    build_distribution_optimizer,
    build_regularizers,
    normalize_sampler_config,
)


@dataclass
class ObjectiveSpec:
    """Support-sized state for one paper objective at one decoding step."""

    score: torch.Tensor
    token_ids: torch.Tensor
    base_distribution: torch.Tensor
    supported_logits: torch.Tensor
    regularizers: list[object]
    lambda_: float
    eps: float


def _build_base_distribution(
    supported_logits: torch.Tensor,
    base_distribution_cfg: dict,
) -> torch.Tensor:
    """Build reference probabilities over the active token support."""
    distribution_type = base_distribution_cfg.get("type", "softmax")
    if distribution_type != "softmax":
        raise ValueError(f"Unsupported base distribution: {distribution_type}")

    temperature = max(float(base_distribution_cfg.get("temperature", 1.0)), 1e-6)
    return apply_softmax(supported_logits / temperature)


def _regularizer_names(regularizers: list[object]) -> list[str]:
    return [regularizer.__class__.__name__ for regularizer in regularizers]


def _infer_solver_mode(regularizers: list[object], lambda_: float) -> str:
    if not regularizers or lambda_ == 0.0:
        return "greedy_closed_form"

    names = _regularizer_names(regularizers)
    if set(names).issubset({"EntropyRegularizer"}):
        return "entropy_closed_form"
    if set(names).issubset({"KLToBaseRegularizer"}):
        return "kl_closed_form"
    return "iterative"


def _resolve_solver_mode(
    requested: str,
    regularizers: list[object],
    lambda_: float,
) -> str:
    if requested == "softmax":
        if regularizers:
            raise ValueError("softmax solver does not accept regularizers.")
        return "softmax_closed_form"
    inferred = _infer_solver_mode(regularizers, lambda_)
    if requested == "auto":
        return inferred
    if requested == "mirror_ascent":
        return "iterative"
    if inferred in {
        "greedy_closed_form",
        "entropy_closed_form",
        "kl_closed_form",
    }:
        return inferred
    raise ValueError("closed_form only supports a single KL or entropy regularizer.")


def _solve_greedy(score: torch.Tensor) -> torch.Tensor:
    index = torch.argmax(score)
    probabilities = torch.zeros_like(score, dtype=torch.float32)
    probabilities[index] = 1.0
    return probabilities


def _solve_entropy_closed_form(
    score: torch.Tensor,
    regularizers: list[object],
    lambda_: float,
    eps: float,
) -> torch.Tensor:
    entropy_alpha = sum(
        float(regularizer.alpha)
        for regularizer in regularizers
        if regularizer.__class__.__name__ == "EntropyRegularizer"
    )
    strength = lambda_ * entropy_alpha
    return apply_softmax(score / max(strength, eps))


def _solve_kl_closed_form(
    score: torch.Tensor,
    base_distribution: torch.Tensor,
    regularizers: list[object],
    lambda_: float,
    eps: float,
) -> torch.Tensor:
    kl_alpha = sum(
        float(regularizer.alpha)
        for regularizer in regularizers
        if regularizer.__class__.__name__ == "KLToBaseRegularizer"
    )
    strength = max(lambda_ * kl_alpha, eps)
    return apply_softmax(
        torch.log(torch.clamp(base_distribution, min=eps)) + score / strength
    )


def _metric_options(regularizers: list[object], eps: float) -> dict:
    del regularizers
    return {"eps": eps}


def _solve_iterative(
    objective: ObjectiveSpec,
    distribution_optimizer,
    context_state: dict | None,
) -> torch.Tensor:
    state = context_state or {}
    initial = torch.clamp(objective.base_distribution.clone(), min=objective.eps)
    initial = initial / initial.sum()

    context = {
        "s": objective.score,
        "p": objective.base_distribution,
        "supported_logits": objective.supported_logits,
        "token_ids": objective.token_ids,
        "state": state,
        "eps": objective.eps,
    }

    def grad_fn(probabilities):
        gradient = objective.score.clone()
        for regularizer in objective.regularizers:
            gradient = gradient - (
                objective.lambda_
                * regularizer.alpha
                * regularizer.grad(probabilities, context)
            )
        return gradient

    return distribution_optimizer.solve(q0=initial, grad_fn=grad_fn)


def solve_distribution(
    logits: torch.Tensor,
    sampler_cfg: dict,
    regularizers: list[object] | None = None,
    distribution_optimizer=None,
    temperature: float | None = None,
    context_state: dict | None = None,
    return_metrics: bool = False,
    metric_kwargs: dict | None = None,
) -> (
    Tuple[torch.Tensor, torch.Tensor]
    | Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
):
    """Return the objective solution on the active token support."""
    sampler_cfg = normalize_sampler_config(sampler_cfg)
    if regularizers is None:
        regularizers = build_regularizers(sampler_cfg["regularizers"])
    if distribution_optimizer is None:
        distribution_optimizer = build_distribution_optimizer(sampler_cfg["optimizer"])

    support_cfg = sampler_cfg.get("support", {"type": "full"})
    base_distribution_cfg = sampler_cfg.get(
        "base_distribution",
        {"type": "softmax", "temperature": 1.0},
    )
    eps = float(sampler_cfg.get("eps", 1e-8))

    supported_logits, token_ids = apply_support(logits, support_cfg)
    selected_temperature = temperature
    if selected_temperature is None:
        selected_temperature = float(sampler_cfg.get("temperature", 1.0))
    selected_temperature = max(float(selected_temperature), 1e-6)

    objective = ObjectiveSpec(
        score=supported_logits / selected_temperature,
        token_ids=token_ids,
        base_distribution=_build_base_distribution(
            supported_logits,
            base_distribution_cfg,
        ),
        supported_logits=supported_logits,
        regularizers=regularizers,
        lambda_=float(sampler_cfg.get("lambda", 1.0)),
        eps=eps,
    )

    solver_mode = _resolve_solver_mode(
        requested=str(sampler_cfg["solver"]),
        regularizers=objective.regularizers,
        lambda_=objective.lambda_,
    )
    if solver_mode == "softmax_closed_form":
        probabilities = apply_softmax(objective.score)
    elif solver_mode == "greedy_closed_form":
        probabilities = _solve_greedy(objective.score)
    elif solver_mode == "entropy_closed_form":
        probabilities = _solve_entropy_closed_form(
            score=objective.score,
            regularizers=objective.regularizers,
            lambda_=objective.lambda_,
            eps=objective.eps,
        )
    elif solver_mode == "kl_closed_form":
        probabilities = _solve_kl_closed_form(
            score=objective.score,
            base_distribution=objective.base_distribution,
            regularizers=objective.regularizers,
            lambda_=objective.lambda_,
            eps=objective.eps,
        )
    else:
        probabilities = _solve_iterative(
            objective,
            distribution_optimizer,
            context_state,
        )
    metric_options = {
        **_metric_options(objective.regularizers, objective.eps),
        **(metric_kwargs or {}),
    }

    if return_metrics:
        metrics = compute_distribution_metrics(
            q=probabilities,
            p=objective.base_distribution,
            supported_logits=objective.supported_logits,
            **metric_options,
        )
        return probabilities, objective.token_ids, metrics
    return probabilities, objective.token_ids


class CompoSimplexSampler:
    """Approximately solve the configured simplex objective at each token step."""

    def __init__(self, cfg: dict):
        self.cfg = normalize_sampler_config(cfg)
        self.optimizer = build_distribution_optimizer(self.cfg["optimizer"])
        self.regularizers = build_regularizers(self.cfg["regularizers"])

    @torch.no_grad()
    def distribution(
        self,
        logits: torch.Tensor,
        state: dict | None = None,
        return_metrics: bool = False,
        metric_kwargs: dict | None = None,
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """Return the supported distribution and optional objective metrics."""
        return solve_distribution(
            logits=logits,
            sampler_cfg=self.cfg,
            regularizers=self.regularizers,
            distribution_optimizer=self.optimizer,
            temperature=float(self.cfg.get("temperature", 1.0)),
            context_state=state,
            return_metrics=return_metrics,
            metric_kwargs=metric_kwargs,
        )

    @torch.no_grad()
    def full_log_probs(
        self,
        logits: torch.Tensor,
        state: dict | None = None,
    ) -> torch.Tensor:
        """Return vocabulary-sized log probabilities with unsupported tokens masked."""
        probabilities, token_ids = self.distribution(logits=logits, state=state)
        output = torch.full_like(logits.float(), float("-inf"))
        output[token_ids] = torch.log(
            torch.clamp(probabilities, min=float(self.cfg.get("eps", 1e-8)))
        )
        return output
