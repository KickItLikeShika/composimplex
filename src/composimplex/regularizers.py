"""Regularizer terms used by compositional probability objectives.

Each ``grad`` method returns the gradient of the paper's regularizer
``Omega_i(q; context)``. The sampler applies the shared coefficient and sign:

    score - lambda * sum_i alpha_i * grad Omega_i.
"""

from __future__ import annotations

import torch

from composimplex.metrics import normalized_topm_weights


class KLToBaseRegularizer:
    """Omega(q) = KL(q || p)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def grad(self, q: torch.Tensor, context: dict) -> torch.Tensor:
        p = context["p"]
        eps = float(context.get("eps", 1e-8))
        q_safe = torch.clamp(q, min=eps, max=1.0)
        p_safe = torch.clamp(p, min=eps, max=1.0)
        return torch.log(q_safe / p_safe) + 1.0


class JSToBaseRegularizer:
    """Omega(q) = JS(q, p)."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        self.alpha = float(alpha)
        self.eps = float(eps)

    def grad(self, q: torch.Tensor, context: dict) -> torch.Tensor:
        p = context["p"]
        q_safe = torch.clamp(q, min=self.eps, max=1.0)
        p_safe = torch.clamp(p, min=self.eps, max=1.0)
        mixture = torch.clamp(0.5 * (p_safe + q_safe), min=self.eps, max=1.0)
        return 0.5 * torch.log(q_safe / mixture)


class EntropyRegularizer:
    """Omega(q) = -H(q) = sum_v q(v) log q(v)."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        self.alpha = float(alpha)
        self.eps = float(eps)

    def grad(self, q: torch.Tensor, context: dict) -> torch.Tensor:
        q_safe = torch.clamp(q, min=self.eps, max=1.0)
        return torch.log(q_safe) + 1.0


class CoverageRegularizer:
    """Negative expected coverage over reference-weighted or Top-M tokens."""

    def __init__(
        self,
        alpha: float = 1.0,
        K: int = 16,
        weight_mode: str = "topm_uniform",
        top_m: int = 8,
        eps: float = 1e-8,
    ):
        self.alpha = float(alpha)
        self.K = int(K)
        self.weight_mode = str(weight_mode)
        self.top_m = int(top_m)
        self.eps = float(eps)
        if self.K < 1:
            raise ValueError("coverage K must be at least 1.")
        if self.weight_mode not in {
            "reference",
            "topm",
            "topm_l2",
            "topm_uniform",
        }:
            raise ValueError(
                "coverage weight_mode must be reference, topm, topm_l2, "
                "or topm_uniform."
            )
        if self.top_m < 1:
            raise ValueError("coverage top_m must be at least 1.")

    def grad(self, q: torch.Tensor, context: dict) -> torch.Tensor:
        reference = context["p"]
        if self.weight_mode == "reference":
            weights = reference
        elif self.weight_mode in {"topm", "topm_l2"}:
            weights = torch.zeros_like(reference)
            m = min(self.top_m, reference.numel())
            indices = torch.topk(reference, k=m).indices
            weights[indices] = 1.0
            if self.weight_mode == "topm_l2":
                weights = weights / float(m) ** 0.5
        else:
            weights = normalized_topm_weights(reference, self.top_m, self.K)
        q_safe = torch.clamp(q, min=self.eps, max=1.0)
        coverage_grad = float(self.K) * torch.pow(
            torch.clamp(1.0 - q_safe, 0.0, 1.0),
            float(self.K - 1),
        )
        return -(weights * coverage_grad).to(dtype=q.dtype)


class DiversityGapRegularizer:
    """Omega(q) is negative K-sample coverage of near-frontier tokens."""

    def __init__(
        self,
        alpha: float = 1.0,
        K: int = 16,
        gap_tau: float = 1.0,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        self.alpha = float(alpha)
        self.K = int(K)
        self.gap_tau = float(gap_tau)
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def grad(self, q: torch.Tensor, context: dict) -> torch.Tensor:
        logits = context["supported_logits"].float()
        gap = torch.max(logits) - logits
        tau = max(self.gap_tau, self.eps)
        weights = gap * torch.exp(-gap / tau)
        if self.normalize:
            total = weights.sum()
            if torch.isfinite(total) and float(total.item()) > 0.0:
                weights = weights / total
        coverage_grad = float(self.K) * torch.pow(
            torch.clamp(1.0 - q, 0.0, 1.0),
            float(self.K - 1),
        )
        return -(weights * coverage_grad).to(dtype=q.dtype)
