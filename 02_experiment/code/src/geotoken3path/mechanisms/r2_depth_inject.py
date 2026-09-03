# -*- coding: utf-8 -*-
"""R2: deterministic low-rank injection of the unused SAR depth group.

Contract (v20 plan handoff, 2026-09-03):
- Inputs: depth_features [B, N, 4, D] (stems already applied) and the late
  stage sar_stage [B, N, D].
- Parameters (router.* namespace): layer_weights in R^4 (softmax-normalized,
  zero-initialized -> uniform 1/4) and layer_proj in R^{D x D} (zero
  initialized -> residual injection starts at exactly zero).
- Forward: a = softmax(layer_weights); h = sum_l a_l * depth[:, :, l, :];
  r = layer_proj(h); sar_stage = sar_stage + r (pre-fusion, late stage only).
- Invariants: no spectral/frequency transform, no explicit full projector,
  no second CROMA forward, no label/argmax, training and inference paths
  are identical; gamma-free.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class R2DepthGroupInjector(nn.Module):
    """Zero-start weighted aggregation of the non-spatial SAR depth group."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("dim must be at least 4")
        self.layer_weights = nn.Parameter(torch.zeros(4))
        self.layer_proj = nn.Linear(dim, dim, bias=False)
        # Zero-start: initial injection is exactly zero so the mechanism row
        # equals the baseline at optimizer step 0.
        nn.init.zeros_(self.layer_proj.weight)

    def forward(self, depth_features: Tensor, sar_stage: Tensor) -> Tensor:
        if depth_features.ndim != 4 or depth_features.shape[2] != 4:
            raise ValueError("depth_features must be [B, N, 4, D]")
        if depth_features.shape[:2] != sar_stage.shape[:2]:
            raise ValueError("depth_features and sar_stage must share [B, N]")
        if depth_features.shape[3] != sar_stage.shape[2]:
            raise ValueError("depth features and sar_stage token dims must match")
        a = F.softmax(self.layer_weights, dim=0)
        h = torch.einsum("l,bnld->bnd", a, depth_features)
        r = self.layer_proj(h)
        return sar_stage + r