# -*- coding: utf-8 -*-
"""R7: zero-start residual learned upsample (pixel-domain, rare-class boundary).

Contract (v20 successor, 2026-09-03):
- Placement: logits domain [B, C, H15, W15] after token reshape, before the
  final bilinear interpolation path. The mechanism adds a zero-initialized
  3x3 convolution residual on the interpolated logits, so at optimizer step 0
  the model is exactly identical to the baseline.
- Parameters (router.* namespace): upsample_conv Conv2d(C, C, 3, padding=1,
  bias=False), zero initialized.
- Invariants: identity zero-start; no label/argmax; no second CROMA forward;
  deterministic; training and inference identical; fusion layer untouched
  (always-fuse stage semantics).
- Rationale: baseline pixel recovery is fixed bilinear 8x upsampling; small
  rare-class objects lose boundary detail. The learned residual exposes
  pixel-domain capacity orthogonal to every prior token-domain mechanism.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class R7ResidualUpsample(nn.Module):
    """Zero-start 3x3 residual refinement on the interpolated logits."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.upsample_conv = nn.Conv2d(num_classes, num_classes, 3, padding=1, bias=False)
        nn.init.zeros_(self.upsample_conv.weight)

    def forward(self, up_logits: Tensor) -> Tensor:
        if up_logits.ndim != 4:
            raise ValueError("up_logits must be [B, C, H, W]")
        return up_logits + self.upsample_conv(up_logits)