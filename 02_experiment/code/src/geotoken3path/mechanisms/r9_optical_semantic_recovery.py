# -*- coding: utf-8 -*-
"""R9: optical semantic recovery for truncated depth rows (D2 recovery).

Contract (V21 design, 2026-09-04):
- Scope: the D2 depth row reads the optical late tap at layer 3. Layers 4-5
  are never executed (ICE plan s2_last=3), so their semantics are missing.
  R9 learns a lightweight token-wise residual g(z3) that recovers the
  task-useful component of the truncated depth without executing layers 4-5.
- Input: the late-stage optical carrier z3 (after the optical stem, before
  the fusion call). Output: an additive residual injected into that carrier.
- Architecture: down 768->r (r=128 default), two hidden rank-r MLP blocks with
  GELU, up r->768; zero-start (only the first down projection is non-zero
  initialized) so the composed model equals the D2 row at optimizer step 0.
- Invariants: no attention, no new tap (ICE plan unchanged), no second CROMA
  forward, no label/argmax, training and inference identical; parameters live
  in the router.* namespace.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class R9OpticalSemanticRecovery(nn.Module):
    """Zero-start lightweight residual that recovers truncated optical depth."""

    def __init__(self, dim: int, rank: int = 128) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("dim must be at least 4")
        if rank < 1:
            raise ValueError("rank must be positive")
        self.down = nn.Linear(dim, rank, bias=False)
        self.hid = nn.Sequential(
            nn.LayerNorm(rank),
            nn.GELU(),
            nn.Linear(rank, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, rank, bias=False),
        )
        self.up = nn.Linear(rank, dim, bias=False)
        # Zero-start: only the down projection is active at step 0; everything
        # after it is zero-initialized so the residual is exactly zero and the
        # mechanism row equals the D2 baseline.
        for module in (self.hid[2], self.hid[4], self.up):
            nn.init.zeros_(module.weight)

    def forward(self, optical_stage: Tensor) -> Tensor:
        if optical_stage.ndim != 3:
            raise ValueError("optical_stage must be [B, N, D]")
        hidden = self.hid(self.down(optical_stage))
        residual = self.up(hidden)
        return optical_stage + residual