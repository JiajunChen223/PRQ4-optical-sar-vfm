# -*- coding: utf-8 -*-
"""R6: dual-channel depth-group injection (mid -> optical carrier, late -> SAR).

Contract (v20 successor, 2026-09-03):
- Placement: pre-fusion injection at BOTH stages; the fusion layer itself
  always executes the verified always-fuse path.
- Channel routing: at the mid stage the optical-conditioned depth aggregate
  is injected into the OPTICAL carrier; at the late stage it is injected into
  the SAR carrier. Stage-wise receptor asymmetry exposes complementary depth
  evidence to both modality carriers.
- Condition: per-token layer-selection logits from the optical stage carrier
  (softmax over the 4 SAR depth layers), zero-initialized -> uniform 1/4.
- Parameters (router.* namespace, per stage): sel_proj Linear(D->4) zero,
  layer_proj Linear(D->D) zero  -> exact zero-start identity vs baseline.
- Invariants: zero-start exact identity; no attention; no spectral/trajectory
  transform; no second CROMA forward; no label/argmax; train/infer identical.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class R6DualChannelDepthInject(nn.Module):
    """Stage-asymmetric optical-conditioned SAR depth-group injection."""

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

    def forward(
        self,
        depth_features: Tensor,
        optical_stage: Tensor,
        sar_stage: Tensor,
        stage: str,
    ) -> tuple[Tensor, Tensor]:
        if depth_features.ndim != 4 or depth_features.shape[2] != 4:
            raise ValueError("depth_features must be [B, N, 4, D]")
        if not (depth_features.shape[:2] == optical_stage.shape[:2] == sar_stage.shape[:2]):
            raise ValueError("depth/optical/sar must share [B, N]")
        if stage not in self.stages:
            raise ValueError(f"stage {stage!r} not declared")
        a = F.softmax(self.sel_proj[stage](optical_stage), dim=-1)  # [B, N, 4]
        h = torch.einsum("bnl,bnld->bnd", a, depth_features)
        r = self.layer_proj[stage](h)
        if stage == self.stages[0]:
            optical_stage = optical_stage + r  # mid -> optical carrier
        else:
            sar_stage = sar_stage + r  # late -> SAR carrier
        return optical_stage, sar_stage