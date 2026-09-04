"""S2-branch adapter and normalization for the R24 SkySense++ backbone.

R24 runs the SkySense++ S2 semantic-enhanced ViT-L backbone on the SEN12TS
optical branch only.  The SEN12TS optical contract stays untouched: the loader
still emits the full 12-band float32 batch plus paired SAR and labels, and this
module (a) selects the ten S2 bands the pretrained backbone expects, (b) maps
the ignore index into the backbone vocabulary (labels travel through the
backbone annotation channel), and (c) applies the frozen R24 dynamic
normalization over the ten-band subset on the batch side.

The modeling code (which imports the vendor ``transformers`` implementation)
is deliberately NOT imported here: the data layer stays pure torch plus the
existing sen12ts loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .contracts import OPTICAL_BANDS, SAR_CHANNELS
from .sen12ts import SEN12TSLoaderError, build_sen12ts_loader

# SEN12TS 12-band order [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]
# -> SkySense++ S2 10-band order [B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12].
_S2_10_BANDS = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
)
_S2_10_INDICES = tuple(OPTICAL_BANDS.index(band) for band in _S2_10_BANDS)  # noqa: E501

# SkySense++ semantic vocabulary: indices 0..64 are usable annotation labels,
# so the SEN12TS ignore index 255 must be remapped inside the vocabulary.
_IGNORE_TO_VOCAB = 0


def to_skysensepp_optical(batch: Mapping[str, Tensor]) -> Tensor:
    """Select the SkySense++ S2 ten bands from a SEN12TS optical batch.

    The input is the raw loader batch (optical [B,12,H,W] float32); band
    selection happens here on the batch side so the frozen collate pipeline
    and D4 augmentation never change.  The selected order follows the
    canonical SEN12TS 12-band contract minus B01 and B09, matching the
    SkySense++ S2 pretraining band order.
    """
    value = batch["optical"]
    if value.ndim != 4 or value.shape[1] != 12:
        raise SEN12TSLoaderError("skysensepp optical selection requires a [B,12,H,W] batch")
    return value[:, _S2_10_INDICES].contiguous()


def annotation_from_target(target: Tensor) -> Tensor:
    """Map a SEN12TS label map into the SkySense++ semantic-annotation domain.

    The backbone embeds every annotation pixel through its vocabulary token
    table (65 rows), so the ignore index 255 must be folded into the
    vocabulary.  Contract labels 0..10 and the remapped ignore value 0 are all
    in range; loss masking still uses the original target.
    """
    return target.clamp(max=64).where(target != 255, torch.zeros_like(target))


def croma_dynamic_normalize_batch_r24(
    batch: Mapping[str, Tensor],
    *,
    micro_batch: int = 16,
) -> dict[str, Tensor]:
    """Apply the frozen R24 per-micro-batch normalization to the S2 branch.

    Only ``optical`` is processed: it must be float32 [B,12,H,W] with B equal
    to ``micro_batch`` (the loader collate freezes 16).  After selecting the
    ten S2 bands, per-channel statistics are computed over axes (0, 2, 3) and
    values are clipped with the same mean-2std / 4std recipe used by the
    existing CROMA path.  ``target`` and ``valid_count`` pass through
    untouched so downstream masking semantics never change.  The returned
    batch carries exactly ``optical10``, ``target`` and ``valid_count``.

    The check is fail-closed: a non-positive or non-finite per-channel
    standard deviation raises instead of silently emitting garbage.
    """
    optical = batch["optical"]
    if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch <= 0:
        raise SEN12TSLoaderError("R24 micro_batch must be a positive integer")
    if (
        optical.ndim != 4
        or optical.shape[0] != micro_batch
        or optical.shape[1] != 12
        or optical.dtype != torch.float32
    ):
        raise SEN12TSLoaderError("optical batch must be float32 [B,12,H,W] with B==micro_batch")
    value = optical[:, _S2_10_INDICES].contiguous()
    finite = torch.isfinite(value)
    if not bool(finite.all()):
        value = torch.where(finite, value, torch.zeros_like(value))
    mean = value.mean(dim=(0, 2, 3), keepdim=True)
    std = value.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
    if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise SEN12TSLoaderError("R24 optical subset has non-positive/non-finite standard deviation")
    normalized = ((value - (mean - 2.0 * std)) / (4.0 * std)).clamp(0.0, 1.0)
    return {
        "optical10": normalized,
        "target": batch["target"],
        "valid_count": batch["valid_count"],
    }


def build_skysensepp_loader(
    manifest_path: str | Path,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    execution_scale: str,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    augmentation: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> tuple[DataLoader[dict[str, Tensor]], dict[str, Any]]:
    """Build the R24 SEN12TS loader by reusing the audited sen12ts loader.

    The sen12ts loader already performs manifest validation, test-seal
    enforcement, D4 paired augmentation (train only) and the fixed 16-row
    collate.  Band selection to the ten S2 bands happens on the batch side in
    ``croma_dynamic_normalize_batch_r24`` / ``to_skysensepp_optical`` so this
    wrapper only forwards.
    """
    loader, manifest = build_sen12ts_loader(
        manifest_path,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        execution_scale=execution_scale,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        augmentation=augmentation,
        seed=seed,
    )
    return loader, manifest
