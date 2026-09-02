"""Pure utilities for checkpoint-level D0 mechanism diagnostics.

The functions in this module do not read files, construct models, access a
device, or alter the training protocol.  They provide deterministic
within-batch/global permutations, square token-grid displacement, and small
statistics used by the cloud-only diagnostic runner.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


class DiagnosticContractError(ValueError):
    """Raised when a D0 diagnostic input violates the fixed contract."""


def fixed_derangement(count: int, seed: int) -> Tensor:
    """Return a deterministic permutation with no fixed points when possible."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise DiagnosticContractError("count must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DiagnosticContractError("seed must be a nonnegative integer")
    identity = torch.arange(count, dtype=torch.long)
    if count < 2:
        return identity
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    if bool((permutation == identity).any()):
        # A cyclic rotation is a guaranteed derangement and remains fully
        # reproducible even when randperm happens to return fixed points.
        shift = 1 + (seed % (count - 1))
        permutation = torch.roll(identity, shifts=shift, dims=0)
    if bool((permutation == identity).any()):
        raise DiagnosticContractError("failed to construct a derangement")
    return permutation


def permutation_sha256(permutation: Tensor) -> str:
    """Hash a CPU int64 permutation for a reproducible diagnostic receipt."""

    import hashlib

    if not isinstance(permutation, Tensor) or permutation.ndim != 1 or permutation.dtype != torch.long:
        raise DiagnosticContractError("permutation must be a one-dimensional int64 tensor")
    return hashlib.sha256(permutation.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def shift_token_grid(tokens: Tensor, dx: int, dy: int) -> Tensor:
    """Displace a square token grid with clamped border sampling.

    The output at target coordinate ``(y, x)`` samples the source at
    ``(clamp(y-dy), clamp(x-dx))``.  This implements a deterministic feature
    displacement without circular wraparound or padding tokens.  Inputs may be
    ``[B,N,D]`` or ``[B,N,G,D]``; all trailing feature dimensions are retained.
    """

    if not isinstance(tokens, Tensor) or tokens.ndim not in {3, 4}:
        raise DiagnosticContractError("tokens must have shape [B,N,D] or [B,N,G,D]")
    if isinstance(dx, bool) or not isinstance(dx, int) or isinstance(dy, bool) or not isinstance(dy, int):
        raise DiagnosticContractError("dx and dy must be integers")
    batch, count = int(tokens.shape[0]), int(tokens.shape[1])
    side = math.isqrt(count)
    if side * side != count or side < 1:
        raise DiagnosticContractError("token count must be a positive square")
    grid = tokens.reshape(batch, side, side, *tokens.shape[2:])
    y = (torch.arange(side, device=tokens.device) - dy).clamp(0, side - 1)
    x = (torch.arange(side, device=tokens.device) - dx).clamp(0, side - 1)
    shifted = grid.index_select(1, y).index_select(2, x)
    return shifted.reshape_as(tokens).contiguous()


def shift_mapping(mapping: Mapping[str, Tensor], dx: int, dy: int) -> dict[str, Tensor]:
    """Apply the same deterministic displacement to every stage mapping."""

    if not isinstance(mapping, Mapping) or not mapping:
        raise DiagnosticContractError("stage mapping must be a non-empty mapping")
    return {str(stage): shift_token_grid(value, dx, dy) for stage, value in mapping.items()}


def flatten_values(values: Iterable[Any], *, label: str) -> Tensor:
    """Convert finite scalar/vector values to a CPU float64 vector."""

    chunks: list[Tensor] = []
    for value in values:
        if isinstance(value, Tensor):
            tensor = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tensor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
        else:
            tensor = torch.as_tensor([value], dtype=torch.float64)
        if tensor.numel() and not bool(torch.isfinite(tensor).all()):
            raise DiagnosticContractError(f"{label} contains non-finite values")
        chunks.append(tensor)
    if not chunks:
        raise DiagnosticContractError(f"{label} must contain at least one value")
    result = torch.cat(chunks)
    if result.numel() == 0:
        raise DiagnosticContractError(f"{label} must contain at least one value")
    return result


def describe_distribution(values: Tensor, *, bins: int = 20, histogram_range: tuple[float, float] | None = None) -> dict[str, Any]:
    """Return auditable scalar distribution statistics."""

    vector = flatten_values([values], label="distribution")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise DiagnosticContractError("bins must be a positive integer")
    lower, upper = histogram_range or (float(vector.min()), float(vector.max()))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        raise DiagnosticContractError("histogram range is invalid")
    if upper == lower:
        histogram = [int(vector.numel())] + [0] * (bins - 1)
    else:
        histogram = torch.histc(vector, bins=bins, min=lower, max=upper).to(torch.int64).tolist()
    std = float(vector.std(unbiased=False).item())
    return {
        "count": int(vector.numel()),
        "mean": float(vector.mean().item()),
        "std": std,
        "min": float(vector.min().item()),
        "max": float(vector.max().item()),
        "histogram": histogram,
        "histogram_range": [lower, upper],
    }


def _average_ranks(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(x: Tensor, y: Tensor) -> float:
    """Compute tie-aware Spearman correlation for two equal-length vectors."""

    left = flatten_values([x], label="spearman x")
    right = flatten_values([y], label="spearman y")
    if left.numel() != right.numel() or left.numel() < 2:
        raise DiagnosticContractError("spearman inputs must have equal length >= 2")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = torch.sqrt(left_centered.square().sum() * right_centered.square().sum())
    if float(denominator) == 0.0:
        return 0.0
    return float((left_centered * right_centered).sum().div(denominator).item())


def cohens_d(correct: Tensor, shuffled: Tensor) -> float:
    """Return standardized mean separation (shuffled minus correct)."""

    left = flatten_values([correct], label="correct")
    right = flatten_values([shuffled], label="shuffled")
    if left.numel() < 2 or right.numel() < 2:
        raise DiagnosticContractError("Cohen's d requires at least two values per group")
    pooled_variance = (
        (left.numel() - 1) * left.var(unbiased=True)
        + (right.numel() - 1) * right.var(unbiased=True)
    ) / (left.numel() + right.numel() - 2)
    denominator = torch.sqrt(pooled_variance)
    difference = right.mean() - left.mean()
    if float(denominator) == 0.0:
        return 0.0 if float(difference) == 0.0 else math.copysign(float("inf"), float(difference))
    return float((difference / denominator).item())


def auroc(correct: Tensor, shuffled: Tensor) -> float:
    """Compute AUROC where shuffled/corrupted is the positive class."""

    negative = flatten_values([correct], label="correct")
    positive = flatten_values([shuffled], label="shuffled")
    if negative.numel() == 0 or positive.numel() == 0:
        raise DiagnosticContractError("AUROC requires both classes")
    scores = torch.cat([negative, positive])
    labels = torch.cat([torch.zeros_like(negative), torch.ones_like(positive)])
    ranks = _average_ranks(scores)
    positive_rank_sum = ranks[labels == 1].sum()
    denominator = positive.numel() * negative.numel()
    return float(((positive_rank_sum - positive.numel() * (positive.numel() + 1) / 2) / denominator).item())


def separation(correct: Tensor, shuffled: Tensor, *, label: str) -> dict[str, Any]:
    """Summarize paired-vs-shuffled separation for one telemetry variable."""

    left = flatten_values([correct], label=f"{label}.correct")
    right = flatten_values([shuffled], label=f"{label}.shuffled")
    return {
        "label": label,
        "correct": describe_distribution(left),
        "shuffled": describe_distribution(right),
        "mean_difference_shuffled_minus_correct": float((right.mean() - left.mean()).item()),
        "cohens_d": cohens_d(left, right),
        "auroc_shuffled_positive": auroc(left, right),
    }


def mean_scalar(values: Iterable[Any], *, label: str) -> float:
    return float(flatten_values(values, label=label).mean().item())

