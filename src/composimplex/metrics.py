"""Distribution-level metrics for CompoSimplex objectives."""

from __future__ import annotations

import torch

DISTRIBUTION_METRIC_NAMES = (
    "kl_q_p",
    "js_q_p",
    "entropy",
    "unique_token_coverage",
    "topm_coverage",
    "diversity_gap",
)


def normalized_topm_weights(
    reference: torch.Tensor, top_m: int, coverage_k: int
) -> torch.Tensor:
    """Return Top-M weights whose uniform-distribution coverage is one."""
    if top_m < 1:
        raise ValueError("coverage_top_m must be at least 1.")
    m = min(int(top_m), reference.numel())
    max_hit_probability = 1.0 - (1.0 - 1.0 / float(m)) ** coverage_k
    indices = torch.topk(reference, k=m).indices
    weights = torch.zeros_like(reference)
    weights[indices] = 1.0 / (float(m) * max_hit_probability)
    return weights


@torch.no_grad()
def compute_distribution_metrics(
    q: torch.Tensor,
    p: torch.Tensor,
    supported_logits: torch.Tensor,
    *,
    coverage_k: int = 16,
    coverage_top_m: int = 8,
    diversity_k: int = 16,
    diversity_gap_tau: float = 1.0,
    normalize_diversity_gap: bool = True,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute the primary metrics on one supported token distribution.

    q is the optimized distribution and p is the reference distribution on
    the same support. Scalar tensor outputs avoid forced device synchronization.
    Coverage evaluation defaults to Top-8, K=16 for every sampler, independently
    of its objective weights, candidate count, or optimization budget.
    """
    if q.shape != p.shape or q.shape != supported_logits.shape:
        raise ValueError("q, p, and supported_logits must have the same shape.")
    if coverage_k < 1:
        raise ValueError("coverage_k must be at least 1.")
    if diversity_k < 1:
        raise ValueError("diversity_k must be at least 1.")
    if diversity_gap_tau <= 0.0:
        raise ValueError("diversity_gap_tau must be positive.")

    q_safe = torch.clamp(q.float(), min=eps, max=1.0)
    p_safe = torch.clamp(p.float(), min=eps, max=1.0)
    q_float = q.float()
    p_float = p.float()

    log_q = torch.log(q_safe)
    log_p = torch.log(p_safe)
    kl_q_p = torch.sum(q_float * (log_q - log_p))

    mixture = 0.5 * (q_float + p_float)
    log_mixture = torch.log(torch.clamp(mixture, min=eps, max=1.0))
    js_q_p = 0.5 * (
        torch.sum(q_float * (log_q - log_mixture))
        + torch.sum(p_float * (log_p - log_mixture))
    )

    entropy = -torch.sum(q_float * log_q)
    unique_token_coverage = torch.sum(
        1.0
        - torch.pow(
            torch.clamp(1.0 - q_float, min=0.0, max=1.0),
            coverage_k,
        )
    ) / float(coverage_k)
    topm_coverage = torch.sum(
        normalized_topm_weights(p_float, coverage_top_m, coverage_k)
        * (1.0 - torch.pow(
            torch.clamp(1.0 - q_float, min=0.0, max=1.0),
            coverage_k,
        ))
    )

    logits = supported_logits.float()
    gap = torch.max(logits) - logits
    gap_reward = gap * torch.exp(-gap / diversity_gap_tau)
    if normalize_diversity_gap:
        gap_reward = gap_reward / torch.clamp(torch.sum(gap_reward), min=eps)
    diversity_gap = torch.sum(
        gap_reward
        * (
            1.0
            - torch.pow(
                torch.clamp(1.0 - q_float, min=0.0, max=1.0),
                diversity_k,
            )
        )
    )

    return {
        "kl_q_p": kl_q_p,
        "js_q_p": js_q_p,
        "entropy": entropy,
        "unique_token_coverage": unique_token_coverage,
        "topm_coverage": topm_coverage,
        "diversity_gap": diversity_gap,
    }


class DistributionMetricAccumulator:
    """Accumulate per-token metric sums without synchronizing every step."""

    def __init__(self) -> None:
        self.num_steps = 0
        self._sums: torch.Tensor | None = None

    def update(self, metrics: dict[str, torch.Tensor]) -> None:
        values = torch.stack(
            [metrics[name].detach().float() for name in DISTRIBUTION_METRIC_NAMES]
        )
        if self._sums is None:
            self._sums = torch.zeros_like(values)
        self._sums.add_(values)
        self.num_steps += 1

    def to_dict(self) -> dict:
        if self._sums is None:
            sums = [0.0] * len(DISTRIBUTION_METRIC_NAMES)
        else:
            sums = self._sums.cpu().tolist()
        return {
            "num_steps": self.num_steps,
            "sums": dict(zip(DISTRIBUTION_METRIC_NAMES, sums)),
        }
