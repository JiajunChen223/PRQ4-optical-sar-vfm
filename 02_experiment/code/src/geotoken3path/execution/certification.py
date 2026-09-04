"""Reusable numerical checks for ICE-Exact certification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn

from .contracts import CromaExecutionContractError


@dataclass(frozen=True)
class TensorEquivalence:
    shape_equal: bool
    dtype_equal: bool
    device_type_equal: bool
    torch_equal: bool
    max_abs_error: float
    mean_abs_error: float
    max_relative_error: float

    @property
    def finite(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (self.max_abs_error, self.mean_abs_error, self.max_relative_error)
        )


def compare_tensors(reference: Tensor, candidate: Tensor) -> TensorEquivalence:
    """Return exact and tolerance-oriented diagnostics without hiding mismatches."""

    if not isinstance(reference, Tensor) or not isinstance(candidate, Tensor):
        raise TypeError("compare_tensors requires torch tensors")
    shape_equal = tuple(reference.shape) == tuple(candidate.shape)
    dtype_equal = reference.dtype == candidate.dtype
    device_type_equal = reference.device.type == candidate.device.type
    if not shape_equal:
        return TensorEquivalence(
            shape_equal=False,
            dtype_equal=dtype_equal,
            device_type_equal=device_type_equal,
            torch_equal=False,
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            max_relative_error=float("inf"),
        )
    left = reference.detach().to(dtype=torch.float64, device="cpu")
    right = candidate.detach().to(dtype=torch.float64, device="cpu")
    diff = (left - right).abs()
    scale = left.abs().clamp_min(torch.finfo(torch.float64).eps)
    return TensorEquivalence(
        shape_equal=True,
        dtype_equal=dtype_equal,
        device_type_equal=device_type_equal,
        torch_equal=bool(torch.equal(reference.detach().cpu(), candidate.detach().cpu())),
        max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
        max_relative_error=float((diff / scale).max().item()) if diff.numel() else 0.0,
    )


def compare_gradients(
    reference: Mapping[str, Tensor | None],
    candidate: Mapping[str, Tensor | None],
) -> dict[str, Any]:
    """Compare named gradients and fail visibly on missing/extra trainable paths."""

    if set(reference) != set(candidate):
        raise CromaExecutionContractError("gradient maps do not contain identical parameter names")
    missing: list[str] = []
    exact = 0
    max_abs = 0.0
    max_rel = 0.0
    per_parameter: dict[str, Any] = {}
    for name in sorted(reference):
        left = reference[name]
        right = candidate[name]
        if left is None or right is None:
            if left is not None or right is not None:
                missing.append(name)
            per_parameter[name] = {"both_none": left is None and right is None}
            continue
        result = compare_tensors(left, right)
        if result.torch_equal:
            exact += 1
        max_abs = max(max_abs, result.max_abs_error)
        max_rel = max(max_rel, result.max_relative_error)
        per_parameter[name] = result.__dict__
    return {
        "parameter_count_compared": len(reference),
        "exact_gradient_count": exact,
        "missing_gradient_names": missing,
        "max_gradient_abs_error": max_abs,
        "max_gradient_relative_error": max_rel,
        "per_parameter": per_parameter,
    }


def named_trainable_gradients(model: nn.Module) -> dict[str, Tensor | None]:
    """Clone the current gradients of every trainable parameter."""

    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
