from __future__ import annotations

from typing import Tuple

import torch


def apply_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Return finite probabilities, falling back to uniform on invalid output."""
    logits = logits.float()
    logits = logits - torch.max(logits)
    probabilities = torch.softmax(logits, dim=-1)
    if (not torch.isfinite(probabilities).all()) or float(probabilities.sum().item()) <= 0.0:
        probabilities = torch.full_like(probabilities, 1.0 / probabilities.numel())
    return probabilities


def _support_probs(logits: torch.Tensor, support_cfg: dict) -> torch.Tensor:
    support_temp = max(float(support_cfg.get("temperature", 1.0)), 1e-6)
    return apply_softmax(logits.float() / support_temp)


def _keep_at_least_one(mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return mask
    top_idx = torch.argmax(scores)
    mask[top_idx] = True
    return mask


def _apply_topm_support(logits: torch.Tensor, support_cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    m = int(support_cfg.get("value", 200))
    m = max(1, min(m, logits_f.shape[-1]))
    return torch.topk(logits_f, k=m, dim=-1)


def _apply_topp_support(logits: torch.Tensor, support_cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    probs = _support_probs(logits_f, support_cfg)
    top_p = float(support_cfg.get("top_p", 0.9))

    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    keep_mask = cumsum <= top_p
    keep_mask[0] = True
    first_over = torch.nonzero(cumsum >= top_p, as_tuple=False)
    if first_over.numel() > 0:
        keep_mask[first_over[0].item()] = True

    selected_ids = sorted_ids[keep_mask]
    return logits_f[selected_ids], selected_ids


def _apply_minp_support(logits: torch.Tensor, support_cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    probs = _support_probs(logits_f, support_cfg)
    min_p = float(support_cfg.get("min_p", 0.05))

    keep_mask = probs >= min_p * torch.max(probs)
    keep_mask = _keep_at_least_one(keep_mask, probs)

    selected_ids = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    order = torch.argsort(probs[selected_ids], descending=True)
    selected_ids = selected_ids[order]
    return logits_f[selected_ids], selected_ids


def _apply_eta_support(logits: torch.Tensor, support_cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    probs = _support_probs(logits_f, support_cfg)
    eta_cutoff = float(support_cfg.get("eta_cutoff", 5e-4))

    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum()
    dynamic_cutoff = (eta_cutoff ** 0.5) * torch.exp(-entropy)
    threshold = min(eta_cutoff, float(dynamic_cutoff.item()))

    keep_mask = probs >= threshold
    keep_mask = _keep_at_least_one(keep_mask, probs)

    selected_ids = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    order = torch.argsort(probs[selected_ids], descending=True)
    selected_ids = selected_ids[order]
    return logits_f[selected_ids], selected_ids


def _apply_typical_support(logits: torch.Tensor, support_cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_f = logits.float()
    probs = _support_probs(logits_f, support_cfg)
    typical_p = float(support_cfg.get("typical_p", 0.95))

    log_probs = torch.log(torch.clamp(probs, min=1e-12))
    entropy = -(probs * log_probs).sum()
    deviation = torch.abs(-log_probs - entropy)

    _, sorted_ids = torch.sort(deviation, descending=False)
    sorted_probs = probs[sorted_ids]
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    keep_mask = cumsum <= typical_p
    keep_mask[0] = True
    first_over = torch.nonzero(cumsum >= typical_p, as_tuple=False)
    if first_over.numel() > 0:
        keep_mask[first_over[0].item()] = True

    selected_ids = sorted_ids[keep_mask]
    order = torch.argsort(probs[selected_ids], descending=True)
    selected_ids = selected_ids[order]
    return logits_f[selected_ids], selected_ids


def apply_support(logits: torch.Tensor, support_cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return supported logits and their original vocabulary token IDs."""
    logits_f = logits.float()
    support_type = support_cfg.get("type", "full")

    if support_type == "full":
        token_ids = torch.arange(logits_f.shape[-1], device=logits_f.device)
        return logits_f, token_ids
    if support_type == "topm":
        return _apply_topm_support(logits_f, support_cfg)
    if support_type == "topp":
        return _apply_topp_support(logits_f, support_cfg)
    if support_type == "minp":
        return _apply_minp_support(logits_f, support_cfg)
    if support_type == "eta":
        return _apply_eta_support(logits_f, support_cfg)
    if support_type == "typical":
        return _apply_typical_support(logits_f, support_cfg)

    raise ValueError(f"Unknown support type: {support_type}")
