# -*- coding: utf-8 -*-
"""R1: classifier-front low-energy channel gain (label-free).

Contract (v20 plan handoff, 2026-09-03):
- Placement: after the stage loop, before self.classifier(fused), acting on
  the final fused carrier [B, N, D].
- Parameter (router.* namespace): gamma in R initialized to 0.0 -> identity.
- Forward (identical in train and inference, no label):
    e_d  = mean_{b,n} |fused[b,n,d]|            per-channel mean abs energy
    e_max = max_d e_d + 1e-6
    scale_d = 1.0 + gamma * (1.0 - e_d / e_max)  low-energy channels amplified
    fused = fused * scale_d[None, None, :]
- Invariants: scale_d >= 1 (amplify-only, no attenuation), single scalar
  step gamma, no normalization of means/variances, no classifier-weight
  geometry, no second CROMA forward, no label/argmax.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class R1LowEnergyChannelGain(nn.Module):
    """Zero-start per-channel amplification of low-energy carrier channels."""

    def __init__(self) -> None:
        super().__init__()
        # relu parameterization keeps gamma >= 0 (amplify-only invariant) while
        # starting at exactly zero (identity at optimizer step 0).
        self.raw_gamma = nn.Parameter(torch.zeros(1))

    @property
    def gamma(self) -> Tensor:
        return torch.relu(self.raw_gamma)

    def forward(self, fused: Tensor) -> Tensor:
        if fused.ndim != 3:
            raise ValueError("fused carrier must be [B, N, D]")
        e = fused.abs().mean(dim=(0, 1))
        e_max = e.max() + 1e-6
        scale = 1.0 + self.gamma * (1.0 - e / e_max)
        return fused * scale[None, None, :]