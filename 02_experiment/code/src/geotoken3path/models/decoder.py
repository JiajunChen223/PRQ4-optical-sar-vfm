"""Spatial decoder used by every baseline and candidate mechanism set."""

from __future__ import annotations

import math

from torch import Tensor, nn
from torch.nn import functional as F


class TokenSegmentationDecoder(nn.Module):
    """Decode a square token grid to dense semantic logits."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, num_classes)

    def forward(self, tokens: Tensor, output_size: tuple[int, int]) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, tokens, dim]")
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError("segmentation decoder requires a square token grid")
        logits = self.projection(tokens).transpose(1, 2)
        logits = logits.reshape(tokens.shape[0], -1, side, side)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
