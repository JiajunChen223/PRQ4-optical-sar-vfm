"""Pure tensor helpers for the D1 dense-gain representation audit."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor


class D1DiagnosticContractError(ValueError):
    """Raised when a D1 attention audit input violates its fixed contract."""


def _check_attention(attention: Tensor) -> tuple[int, int]:
    if not isinstance(attention, Tensor) or attention.ndim != 3:
        raise D1DiagnosticContractError("attention must have shape [B,N,N]")
    if attention.shape[1] != attention.shape[2] or attention.shape[1] < 1:
        raise D1DiagnosticContractError("attention must be square over a non-empty token grid")
    if not bool(torch.isfinite(attention).all()):
        raise D1DiagnosticContractError("attention must be finite")
    if bool((attention < 0).any()):
        raise D1DiagnosticContractError("attention weights must be nonnegative")
    row_sums = attention.sum(dim=-1)
    if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=2e-4, rtol=2e-4)):
        raise D1DiagnosticContractError("attention rows must sum to one")
    return int(attention.shape[0]), int(attention.shape[1])


def token_grid_coordinates(token_count: int, *, device: torch.device | None = None) -> Tensor:
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
        raise D1DiagnosticContractError("token_count must be a positive integer")
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise D1DiagnosticContractError("token_count must be a square")
    y, x = torch.meshgrid(
        torch.arange(side, device=device, dtype=torch.float64),
        torch.arange(side, device=device, dtype=torch.float64),
        indexing="ij",
    )
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)


def valid_token_mask(token_count: int, dx: int, dy: int, *, device: torch.device | None = None) -> Tensor:
    """Return query locations whose source content remains in-bounds after a shift.

    ``shift_token_grid`` moves source content at ``(x,y)`` to
    ``(x+dx,y+dy)``.  The mask therefore describes query locations whose
    original content remains in-bounds after that physical displacement.
    """

    if isinstance(dx, bool) or not isinstance(dx, int) or isinstance(dy, bool) or not isinstance(dy, int):
        raise D1DiagnosticContractError("dx and dy must be integers")
    coordinates = token_grid_coordinates(token_count, device=device)
    side = math.isqrt(token_count)
    x = coordinates[:, 0].to(torch.int64)
    y = coordinates[:, 1].to(torch.int64)
    lower_x, upper_x = max(0, -dx), min(side, side - dx)
    lower_y, upper_y = max(0, -dy), min(side, side - dy)
    if lower_x >= upper_x or lower_y >= upper_y:
        return torch.zeros(token_count, dtype=torch.bool, device=coordinates.device)
    return (x >= lower_x) & (x < upper_x) & (y >= lower_y) & (y < upper_y)


def bilinear_support_valid_mask(token_mask: Tensor, output_size: tuple[int, int]) -> Tensor:
    """Keep pixels whose bilinear logit support is entirely token-valid."""

    if not isinstance(token_mask, Tensor) or token_mask.ndim != 2 or token_mask.shape[0] != token_mask.shape[1]:
        raise D1DiagnosticContractError("token_mask must be a square [side,side] tensor")
    if token_mask.dtype != torch.bool:
        raise D1DiagnosticContractError("token_mask must be boolean")
    if len(output_size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in output_size):
        raise D1DiagnosticContractError("output_size must contain two positive integers")
    side = int(token_mask.shape[0])
    height, width = output_size
    y_position = (torch.arange(height, device=token_mask.device, dtype=torch.float64) + 0.5) * side / height - 0.5
    x_position = (torch.arange(width, device=token_mask.device, dtype=torch.float64) + 0.5) * side / width - 0.5
    y_left = torch.floor(y_position).clamp(0, side - 1).to(torch.long)
    x_left = torch.floor(x_position).clamp(0, side - 1).to(torch.long)
    y_right = (y_left + 1).clamp(0, side - 1)
    x_right = (x_left + 1).clamp(0, side - 1)
    return (
        token_mask[y_left[:, None], x_left[None, :]]
        & token_mask[y_left[:, None], x_right[None, :]]
        & token_mask[y_right[:, None], x_left[None, :]]
        & token_mask[y_right[:, None], x_right[None, :]]
    )


def attention_statistics(attention: Tensor, *, radii: Sequence[int] = (1, 2, 3)) -> dict[str, Tensor]:
    """Return normalized entropy, diagonal mass, and local mass per query."""

    batch, tokens = _check_attention(attention)
    if not radii or any(isinstance(radius, bool) or not isinstance(radius, int) or radius < 0 for radius in radii):
        raise D1DiagnosticContractError("radii must be a non-empty sequence of nonnegative integers")
    coordinates = token_grid_coordinates(tokens, device=attention.device)
    distance = torch.cdist(coordinates, coordinates, p=2)
    entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(tokens) if tokens > 1 else torch.zeros_like(entropy)
    diagonal_mass = attention.diagonal(dim1=-2, dim2=-1)
    result: dict[str, Tensor] = {
        "normalized_entropy": normalized_entropy,
        "same_index_mass": diagonal_mass,
    }
    for radius in radii:
        neighborhood = distance <= float(radius) + 1e-9
        result[f"local_mass_radius_{radius}"] = attention.masked_fill(
            ~neighborhood.unsqueeze(0), 0.0
        ).sum(dim=-1)
    return result


def uniform_local_mass(token_count: int, *, radii: Sequence[int] = (1, 2, 3), device: torch.device | None = None) -> dict[str, Tensor]:
    """Return the per-query local mass expected under uniform attention."""

    if not radii:
        raise D1DiagnosticContractError("radii must be a non-empty sequence")
    coordinates = token_grid_coordinates(token_count, device=device)
    distance = torch.cdist(coordinates, coordinates, p=2)
    result: dict[str, Tensor] = {}
    for radius in radii:
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise D1DiagnosticContractError("radii must contain nonnegative integers")
        neighborhood = distance <= float(radius) + 1e-9
        result[f"local_mass_radius_{radius}"] = neighborhood.to(torch.float64).mean(dim=-1)
    return result


def attention_expected_displacement(attention: Tensor) -> Tensor:
    """Return [B,N,2] expected (x,y) key offset for each query."""

    _, tokens = _check_attention(attention)
    coordinates = token_grid_coordinates(tokens, device=attention.device)
    offsets = coordinates.unsqueeze(0) - coordinates.unsqueeze(1)
    return (attention.unsqueeze(-1) * offsets.unsqueeze(0)).sum(dim=2)


def attention_contract_summary(attention: Tensor) -> dict[str, float]:
    """Return auditable row-sum and nonnegative-weight bounds."""

    _check_attention(attention)
    row_sums = attention.sum(dim=-1)
    return {
        "row_sum_min": float(row_sums.min().item()),
        "row_sum_max": float(row_sums.max().item()),
        "row_sum_max_abs_error": float((row_sums - 1.0).abs().max().item()),
        "minimum_attention_weight": float(attention.min().item()),
    }


def summarize_shift_recovery(
    baseline_expected: Tensor,
    shifted_expected: Tensor,
    *,
    dx: int,
    dy: int,
    valid_mask: Tensor | None = None,
) -> dict[str, Any]:
    if baseline_expected.ndim != 3 or baseline_expected.shape[-1] != 2 or shifted_expected.shape != baseline_expected.shape:
        raise D1DiagnosticContractError("expected displacement tensors must both have shape [B,N,2]")
    if valid_mask is None:
        valid_mask = torch.ones(baseline_expected.shape[1], dtype=torch.bool, device=baseline_expected.device)
    if valid_mask.ndim != 1 or valid_mask.numel() != baseline_expected.shape[1] or not bool(valid_mask.any()):
        raise D1DiagnosticContractError("valid_mask must select at least one token query")
    recovered = (shifted_expected - baseline_expected)[:, valid_mask]
    target = recovered.new_tensor([float(dx), float(dy)])
    error = recovered - target
    magnitude = recovered.norm(dim=-1)
    target_norm = target.norm()
    if float(target_norm) == 0.0:
        directional_accuracy: float | None = None
    else:
        directional = (recovered * target).sum(dim=-1) > 0
        directional_accuracy = float(directional.to(torch.float64).mean().item())
    return {
        "dx": int(dx),
        "dy": int(dy),
        "valid_query_count": int(valid_mask.sum().item()),
        "predicted_displacement_mean": recovered.mean(dim=(0, 1)).detach().cpu().tolist(),
        "displacement_rmse": float(error.square().mean().sqrt().item()),
        "directional_accuracy": directional_accuracy,
        "predicted_magnitude_mean": float(magnitude.mean().item()),
    }
