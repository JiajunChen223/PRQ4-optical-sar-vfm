# -*- coding: utf-8 -*-
"""R3: optical-conditional SAR depth-group selection injection (mid+late).

Contract (v20 successor, 2026-09-03):
- Placement: pre-fusion injection at BOTH mid and late stages; the fusion
  layer itself always executes the verified always-fuse path.
- Condition: per-token layer-selection logits are produced from the optical
  stage carrier (optical-anchored philosophy), softmax over the 4 SAR depth
  layers; the aggregate is passed through a zero-initialized D->D projection
  and added to the SAR stage carrier.
- Parameters (router.* namespace, per stage):
    sel_proj: Linear(D -> 4, bias=False)  zero init -> uniform 1/4 selection
    layer_proj: Linear(D -> D, bias=False) zero init -> zero injection start
- Invariants: zero-start exact-identity vs baseline; no attention, no
  spectral/trajectory transform, no second CROMA forward, no label/argmax,
  training and inference identical.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class R3OpticalConditionalDepthSelect(nn.Module):
    """Per-token optical-conditioned aggregation of the SAR depth group."""

    def __init__(self, dim: int, stages: tuple[str, ...]) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("dim must be at least 4")
        self.stages = tuple(stages)
        self.sel_proj = nn.ModuleDict(
            {stage: nn.Linear(dim, 4, bias=False) for stage in self.stages}
        )
        self.layer_proj = nn.ModuleDict(
            {stage: nn.Linear(dim, dim, bias=False) for stage in self.stages}
        )
        for module in self.sel_proj.values():
            nn.init.zeros_(module.weight)
        for module in self.layer_proj.values():
            nn.init.zeros_(module.weight)

    def forward(self, depth_features: Tensor, sar_stage: Tensor, optical_stage: Tensor, stage: str) -> Tensor:
        if depth_features.ndim != 4 or depth_features.shape[2] != 4:
            raise ValueError("depth_features must be [B, N, 4, D]")
        if depth_features.shape[:2] != sar_stage.shape[:2] or sar_stage.shape[:2] != optical_stage.shape[:2]:
            raise ValueError("depth/sar/optical must share [B, N]")
        if stage not in self.stages:
            raise ValueError(f"stage {stage!r} not declared")
        a = F.softmax(self.sel_proj[stage](optical_stage), dim=-1)  # [B, N, 4]
        h = torch.einsum("bnl,bnld->bnd", a, depth_features)
        r = self.layer_proj[stage](h)
        return sar_stage + r