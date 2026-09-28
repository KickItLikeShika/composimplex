from __future__ import annotations

import torch


class MirrorAscentOptimizer:
    """Approximately maximize an objective with exponentiated mirror ascent."""

    def __init__(
        self,
        steps: int = 50,
        lr: float = 0.5,
        eps: float = 1e-8,
        tol: float = 1e-5,
    ):
        self.steps = int(steps)
        self.lr = float(lr)
        self.eps = float(eps)
        self.tol = float(tol)

    @torch.no_grad()
    def solve(self, q0: torch.Tensor, grad_fn) -> torch.Tensor:
        """Return a mirror-ascent iterate initialized from ``q0``."""
        q = torch.clamp(q0.clone(), min=self.eps)
        q = q / q.sum()

        for _ in range(self.steps):
            gradient = grad_fn(q)
            if not torch.isfinite(gradient).all():
                return self._fallback(q0)

            optimality_gap = torch.max(gradient) - torch.dot(q, gradient)
            if self.tol > 0.0 and float(optimality_gap.item()) <= self.tol:
                break

            gradient = gradient - torch.max(gradient)
            update = torch.exp(self.lr * gradient)
            if not torch.isfinite(update).all():
                return self._fallback(q0)

            q = torch.clamp(q * update, min=self.eps)
            total = torch.sum(q)
            if (not torch.isfinite(total)) or float(total.item()) <= 0.0:
                return self._fallback(q0)
            q = q / total

        if (not torch.isfinite(q).all()) or (q < 0).any() or float(q.sum().item()) <= 0.0:
            return self._fallback(q0)
        return q

    def _fallback(self, q0: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(q0.clone(), min=self.eps)
        return q / q.sum()
