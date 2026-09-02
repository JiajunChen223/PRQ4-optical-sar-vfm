"""Numerically explicit CROMA input preprocessing.

This module implements the pinned README recipe for a supplied tensor, without
opening files or downloading data.  Statistics are per channel over
``(batch,height,width)`` and therefore depend on the effective micro-batch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import Tensor


_SPLITS = {"training", "validation", "inference"}
_MICRO_BATCH = 16


def _prepare_mask(mask: Tensor | None, batch: int, height: int, width: int) -> Tensor:
    if mask is None:
        return torch.ones((batch, 1, height, width), dtype=torch.bool)
    if not isinstance(mask, Tensor):
        raise ValueError("valid_mask must be a torch.Tensor")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if tuple(mask.shape) != (batch, 1, height, width):
        raise ValueError("valid_mask must have shape [B,H,W] or [B,1,H,W]")
    return mask.to(dtype=torch.bool)


def normalize_croma_dynamic(
    x: Tensor,
    *,
    split: Literal["training", "validation", "inference"],
    valid_mask: Tensor | None = None,
    micro_batch: int = _MICRO_BATCH,
) -> tuple[Tensor, dict[str, Any]]:
    """Apply ``croma_official_dynamic_v1`` to one local micro-batch.

    Training requires exactly 16 samples (the dataloader must use
    ``drop_last=True``).  Validation and inference deterministically repeat the
    last sample to 16 before computing statistics and trim the repeated output
    afterward.  In distributed execution this function is called independently
    on each rank, so no global all-gather statistics are performed.
    """

    if split not in _SPLITS:
        raise ValueError("split must be training, validation or inference")
    if not isinstance(x, Tensor) or x.ndim != 4:
        raise ValueError("x must be a [B,C,H,W] tensor")
    if micro_batch != _MICRO_BATCH:
        raise ValueError("micro_batch must be exactly 16")
    batch, channels, height, width = map(int, x.shape)
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("x must have positive dimensions")
    if batch > micro_batch:
        raise ValueError("one call must contain at most one micro-batch")
    if split == "training" and batch != micro_batch:
        raise ValueError("training requires drop_last and exactly 16 samples")

    work = x.to(dtype=torch.float32)
    mask = _prepare_mask(valid_mask, batch, height, width)
    if batch < micro_batch:
        if split == "training":  # defensive; the check above is the public contract
            raise ValueError("training remainder must be dropped before preprocessing")
        repeat = micro_batch - batch
        work = torch.cat((work, work[-1:].expand(repeat, -1, -1, -1)), dim=0).clone()
        mask = torch.cat((mask, mask[-1:].expand(repeat, -1, -1, -1)), dim=0).clone()

    mask_channels = mask.expand(-1, channels, -1, -1)
    if torch.any(torch.isfinite(work).logical_not() & mask_channels):
        raise ValueError("valid CROMA input contains NaN or infinity")
    values = torch.where(mask_channels, work, torch.zeros_like(work))
    count = mask_channels.sum(dim=(0, 2, 3)).to(dtype=torch.float32)
    if torch.any(count < 2):
        raise ValueError("each channel needs at least two valid values for unbiased std")
    mean = values.sum(dim=(0, 2, 3)) / count
    centered = torch.where(mask_channels, values - mean.view(1, -1, 1, 1), torch.zeros_like(values))
    std = torch.sqrt(centered.square().sum(dim=(0, 2, 3)) / (count - 1.0))
    if torch.any(torch.isfinite(std).logical_not()) or torch.any(std <= 0):
        raise ValueError("CROMA normalization encountered non-finite or zero standard deviation")

    lower = mean - 2.0 * std
    normalized = (values - lower.view(1, -1, 1, 1)) / (4.0 * std.view(1, -1, 1, 1))
    normalized = torch.clamp(normalized, 0.0, 1.0)
    # Invalid/nodata pixels are explicitly filled, so no unresolved sentinel
    # can enter the VFM.  Their mask remains available to the caller.
    normalized = torch.where(mask_channels, normalized, torch.zeros_like(normalized))
    normalized = normalized[:batch]
    padding_count = max(0, micro_batch - batch)
    trace: dict[str, Any] = {
        "scheme": "croma_official_dynamic_v1",
        "statistics_axes": [0, 2, 3],
        "scope": "per_micro_batch_per_channel",
        "micro_batch": micro_batch,
        "input_batch": batch,
        "effective_batch": micro_batch,
        "padding_count": padding_count,
        "last_batch_policy": "drop_last" if split == "training" else "pad_repeat_last_and_trim_outputs",
        "trimmed_outputs": split in {"validation", "inference"} and padding_count > 0,
        "padding_changes_statistics": padding_count > 0,
        "distributed_statistics": "per_rank_micro_batch",
        "mean": mean.detach().cpu(),
        "std": std.detach().cpu(),
    }
    return normalized, trace


def validate_dynamic_preprocessing_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Small runtime guard for callers constructing preprocessing pipelines."""

    if descriptor.get("scheme") != "croma_official_dynamic_v1":
        raise ValueError("unsupported preprocessing scheme")
    if descriptor.get("statistics_axes") != [0, 2, 3] or descriptor.get("micro_batch") != 16:
        raise ValueError("preprocessing descriptor is not CROMA batch-compatible")
    if descriptor.get("distributed_statistics") != "per_rank_micro_batch":
        raise ValueError("preprocessing descriptor must use per-rank micro-batch statistics")
