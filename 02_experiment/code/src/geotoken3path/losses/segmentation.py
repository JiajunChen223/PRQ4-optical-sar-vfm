"""Losses shared across matched training objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def segmentation_cross_entropy(logits: Tensor, target: Tensor, ignore_index: int = 255) -> Tensor:
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("expected logits [B,C,H,W] and target [B,H,W]")
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("logits and target spatial shapes do not match")
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)


def _validate_inputs(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("expected logits [B,C,H,W] and target [B,H,W]")
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("logits and target spatial shapes do not match")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    if target.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("target must use an integer label dtype")
    return logits.float(), target.long()


def _valid_flat(logits: Tensor, target: Tensor, ignore_index: int) -> tuple[Tensor, Tensor]:
    logits, target = _validate_inputs(logits, target)
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    flat_target = target.reshape(-1)
    valid = flat_target != int(ignore_index)
    if not bool(valid.any()):
        raise ValueError("target contains no valid pixels")
    return flat_logits[valid], flat_target[valid]


def per_class_cross_entropy(
    logits: Tensor, target: Tensor, *, ignore_index: int = 255
) -> tuple[Tensor, Tensor]:
    """Return per-class CE and a presence mask, using only classes in target."""

    flat_logits, flat_target = _valid_flat(logits, target, ignore_index)
    classes = flat_logits.shape[1]
    per_pixel = F.cross_entropy(flat_logits, flat_target, reduction="none")
    values = flat_logits.new_zeros(classes)
    present = torch.zeros(classes, dtype=torch.bool, device=flat_logits.device)
    for class_index in range(classes):
        mask = flat_target == class_index
        if bool(mask.any()):
            values[class_index] = per_pixel[mask].mean()
            present[class_index] = True
    return values, present


def macro_class_cross_entropy(
    logits: Tensor, target: Tensor, *, ignore_index: int = 255
) -> Tensor:
    """Average CE equally over semantic classes present in the batch."""

    values, present = per_class_cross_entropy(logits, target, ignore_index=ignore_index)
    return values[present].mean()


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    number = gt_sorted.numel()
    if number == 0:
        return gt_sorted
    intersection = gt_sorted.sum() - gt_sorted.float().cumsum(0)
    union = gt_sorted.sum() + (1.0 - gt_sorted.float()).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1.0)
    if number > 1:
        gradient[1:] = gradient[1:] - gradient[:-1]
    return gradient


def lovasz_softmax_loss(
    logits: Tensor, target: Tensor, *, ignore_index: int = 255
) -> Tensor:
    """Multiclass Lovasz-Softmax surrogate over classes present in target."""

    flat_logits, flat_target = _valid_flat(logits, target, ignore_index)
    probabilities = flat_logits.softmax(dim=-1)
    losses: list[Tensor] = []
    for class_index in range(flat_logits.shape[1]):
        foreground = (flat_target == class_index).to(flat_logits.dtype)
        if not bool(foreground.any()):
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        errors_sorted, order = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground[order])))
    if not losses:
        return flat_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _analytic_class_gradient_contribution(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int,
    macro: bool,
) -> tuple[Tensor, Tensor]:
    """Compute a finite per-class logit-gradient norm without a second backward.

    The vector is an analytic contribution of the CE component.  It is logged
    as a diagnostic, not used as an extra loss or as a claim of exact parameter
    gradient decomposition.
    """

    flat_logits, flat_target = _valid_flat(logits, target, ignore_index)
    probabilities = flat_logits.softmax(dim=-1)
    classes = flat_logits.shape[1]
    contribution = flat_logits.new_zeros(classes)
    present = torch.zeros(classes, dtype=torch.bool, device=flat_logits.device)
    counts = torch.bincount(flat_target, minlength=classes).to(flat_logits.dtype)
    present = counts > 0
    divisor = present.sum().to(flat_logits.dtype).clamp_min(1.0) if macro else flat_logits.new_tensor(float(flat_target.numel()))
    for class_index in range(classes):
        mask = flat_target == class_index
        if bool(mask.any()):
            gradient = probabilities[mask].clone()
            gradient[:, class_index] -= 1.0
            class_count = counts[class_index].clamp_min(1.0)
            weight = 1.0 / (class_count * divisor) if macro else 1.0 / divisor
            contribution[class_index] = gradient.norm() * weight
    return contribution, present


def segmentation_objective(
    logits: Tensor,
    target: Tensor,
    *,
    objective_name: str = "pixel_ce",
    ignore_index: int = 255,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute one frozen V12 objective and auditable class diagnostics."""

    objective = str(objective_name).strip().casefold()
    allowed = {"pixel_ce", "macro_ce", "ce_lovasz", "macro_ce_lovasz"}
    if objective not in allowed:
        raise ValueError(f"unsupported segmentation objective: {objective_name}")
    logits_fp, target_long = _validate_inputs(logits, target)
    per_class_ce, present = per_class_cross_entropy(logits_fp, target_long, ignore_index=ignore_index)
    macro = objective.startswith("macro_")
    ce = macro_ce = None
    if macro:
        ce = per_class_ce[present].mean()
    else:
        ce = F.cross_entropy(logits_fp, target_long, ignore_index=ignore_index)
    lovasz = lovasz_softmax_loss(logits_fp, target_long, ignore_index=ignore_index) if objective.endswith("lovasz") else logits_fp.sum() * 0.0
    loss = ce + lovasz if objective.endswith("lovasz") else ce
    gradient_contribution, gradient_present = _analytic_class_gradient_contribution(
        logits_fp, target_long, ignore_index=ignore_index, macro=macro
    )
    counts = torch.bincount(
        target_long[target_long != int(ignore_index)].reshape(-1), minlength=logits_fp.shape[1]
    ).to(logits_fp.dtype)
    return loss, {
        "per_class_ce": per_class_ce.detach(),
        "per_class_present": present.detach() & gradient_present.detach(),
        "per_class_pixel_count": counts.detach(),
        "per_class_gradient_contribution": gradient_contribution.detach(),
        "ce_component": ce.detach(),
        "lovasz_component": lovasz.detach(),
        "objective_id": logits_fp.new_tensor({"pixel_ce": 0.0, "macro_ce": 1.0, "ce_lovasz": 2.0, "macro_ce_lovasz": 3.0}[objective]),
    }


__all__ = [
    "segmentation_cross_entropy",
    "per_class_cross_entropy",
    "macro_class_cross_entropy",
    "lovasz_softmax_loss",
    "segmentation_objective",
]
