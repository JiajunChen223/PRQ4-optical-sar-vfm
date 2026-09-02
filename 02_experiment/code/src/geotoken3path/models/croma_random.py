"""Pinned-source CROMA architecture construction without checkpoint access.

The official ``PretrainedCROMA`` helper unconditionally reads a checkpoint.
This adapter reuses only the architecture primitives from the audited source
snapshot and deliberately exposes explicit weight-disable arguments.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn


class RandomCromaSourceError(RuntimeError):
    """Raised when the pinned source-only CROMA contract is not reproducible."""


def _load_pinned_source(source_path: str, source_sha256: str) -> ModuleType:
    path = Path(source_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RandomCromaSourceError("CROMA source_path must be an existing non-symlink absolute file")
    expected = str(source_sha256).casefold()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RandomCromaSourceError("CROMA source_sha256 is malformed")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RandomCromaSourceError("CROMA source SHA256 differs from the frozen audit")
    spec = importlib.util.spec_from_file_location("geotoken3path_audited_croma_source", path)
    if spec is None or spec.loader is None:
        raise RandomCromaSourceError("cannot create an import spec for the audited CROMA source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("ViT", "BaseTransformerCrossAttn", "get_2dalibi")
    if any(not hasattr(module, name) for name in required):
        raise RandomCromaSourceError("audited CROMA source lacks required architecture primitives")
    return module


class PinnedSourceRandomCROMA(nn.Module):
    """CROMA-base architecture initialized by PyTorch without reading weights."""

    def __init__(
        self,
        *,
        pretrained: bool = False,
        weights: None = None,
        source_path: str,
        source_sha256: str,
        size: str = "base",
        modality: str = "both",
        image_resolution: int = 120,
    ) -> None:
        super().__init__()
        if pretrained is not False or weights is not None:
            raise RandomCromaSourceError("pretrained weights are forbidden for the leakage-blocked route")
        if size != "base" or modality != "both" or image_resolution != 120:
            raise RandomCromaSourceError("formal route freezes CROMA base/both/120")
        source = _load_pinned_source(source_path, source_sha256)
        self.encoder_dim = 768
        self.encoder_depth = 12
        self.num_heads = 16
        self.patch_size = 8
        self.modality = modality
        self.num_patches = int((image_resolution / self.patch_size) ** 2)
        self.s1_channels = 2
        self.s2_channels = 12
        self.attn_bias = source.get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )
        self.s1_encoder = source.ViT(
            dim=self.encoder_dim,
            depth=self.encoder_depth // 2,
            in_channels=self.s1_channels,
        )
        self.GAP_FFN_s1 = self._gap_ffn()
        self.s2_encoder = source.ViT(
            dim=self.encoder_dim,
            depth=self.encoder_depth,
            in_channels=self.s2_channels,
        )
        self.GAP_FFN_s2 = self._gap_ffn()
        self.cross_encoder = source.BaseTransformerCrossAttn(
            dim=self.encoder_dim,
            depth=self.encoder_depth // 2,
            num_heads=self.num_heads,
        )

    def _gap_ffn(self) -> nn.Sequential:
        inner = 4 * self.encoder_dim
        return nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, inner),
            nn.GELU(),
            nn.Linear(inner, self.encoder_dim),
        )

    def forward(
        self, SAR_images: Tensor | None = None, optical_images: Tensor | None = None
    ) -> dict[str, Tensor]:
        if SAR_images is None or optical_images is None:
            raise ValueError("both SAR_images and optical_images are required")
        bias_sar = self.attn_bias.to(SAR_images.device)
        bias_optical = self.attn_bias.to(optical_images.device)
        sar = self.s1_encoder(imgs=SAR_images, attn_bias=bias_sar)
        optical = self.s2_encoder(imgs=optical_images, attn_bias=bias_optical)
        joint = self.cross_encoder(
            x=sar, context=optical, relative_position_bias=bias_optical
        )
        return {
            "SAR_encodings": sar,
            "SAR_GAP": self.GAP_FFN_s1(sar.mean(dim=1)),
            "optical_encodings": optical,
            "optical_GAP": self.GAP_FFN_s2(optical.mean(dim=1)),
            "joint_encodings": joint,
            "joint_GAP": joint.mean(dim=1),
        }

