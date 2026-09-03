# -*- coding: utf-8 -*-
"""R8: composition of R3 (optical-conditional depth-group injection) and R7
(zero-start residual learned upsample).

Contract (v20 successor, 2026-09-03):
- Single internal mechanism delta with two zero-start components:
    1) R3 component: per-token optical-conditioned SAR depth-group selection
       injected pre-fusion at BOTH stages (mid+late -> SAR carrier);
    2) R7 component: zero-init 3x3 residual on the interpolated logits.
- Both components are zero-initialized -> the composed mechanism is exactly
  identical to the baseline at optimizer step 0.
- Invariants: no attention, no spectral transform, no second CROMA forward,
  no label/argmax, deterministic, training and inference identical, fusion
  layer always executes the verified always-fuse path.
"""

from __future__ import annotations

import torch.nn as nn

from geotoken3path.mechanisms.r3_conditional_depth_select import R3OpticalConditionalDepthSelect
from geotoken3path.mechanisms.r7_residual_upsample import R7ResidualUpsample


class R8DepthInjectPlusUpsample(nn.Module):
    """Composed zero-start candidate: depth-group injection + residual upsample."""

    def __init__(self, dim: int, num_classes: int, stages: tuple[str, ...]) -> None:
        super().__init__()
        self.depth_select = R3OpticalConditionalDepthSelect(dim, stages)
        self.upsample = R7ResidualUpsample(num_classes)

    def inject_depth(self, depth_features, sar_stage, optical_stage, stage):
        return self.depth_select(depth_features, sar_stage, optical_stage, stage)

    def refine_logits(self, up_logits):
        return self.upsample(up_logits)