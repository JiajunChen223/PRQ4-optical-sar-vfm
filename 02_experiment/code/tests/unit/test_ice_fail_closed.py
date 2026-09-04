from __future__ import annotations

import pytest
import torch
from torch import nn

from geotoken3path.execution.contracts import CromaExecutionContractError
from geotoken3path.execution.croma_executor import InterfaceCertifiedCromaExecutor
from geotoken3path.execution.croma_plan import CromaExecutionPlan


class _Encoder(nn.Module):
    patch_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.linear_input = nn.Linear(1, 1)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [nn.ModuleList([nn.Identity(), nn.Identity()])]
        )
        self.transformer.norm_out = nn.Identity()


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s1_encoder = _Encoder()
        self.s2_encoder = _Encoder()
        self.GAP_FFN_s1 = nn.Identity()
        self.GAP_FFN_s2 = nn.Identity()
        self.cross_encoder = nn.Identity()
        self.register_buffer("attn_bias", torch.zeros(1), persistent=False)


def _plan() -> CromaExecutionPlan:
    return CromaExecutionPlan(
        required_taps=("s1_encoder.transformer.layers.0.1",),
        s1_last_layer=0,
        s2_last_layer=0,
        require_s1_final_norm=False,
        require_s2_final_norm=False,
        require_joint_encoder=False,
        require_s1_gap=False,
        require_s2_gap=False,
        eliminated_nodes=(
            "s1_encoder.transformer.norm_out",
            "s2_encoder.transformer.norm_out",
            "GAP_FFN_s1",
            "GAP_FFN_s2",
            "cross_encoder",
            "joint_GAP",
        ),
        plan_sha256="0" * 64,
    )


def test_executor_refuses_active_historical_side_channel() -> None:
    backbone = _Backbone()
    backbone.operator_sar_residual = torch.zeros(1)
    executor = InterfaceCertifiedCromaExecutor(_plan())
    with pytest.raises(CromaExecutionContractError, match="side-channel"):
        executor.validate(backbone)


def test_executor_refuses_hook_on_eliminated_module() -> None:
    backbone = _Backbone()
    handle = backbone.cross_encoder.register_forward_hook(lambda *_args: None)
    try:
        executor = InterfaceCertifiedCromaExecutor(_plan())
        with pytest.raises(CromaExecutionContractError, match="forward hooks"):
            executor.validate(backbone)
    finally:
        handle.remove()
