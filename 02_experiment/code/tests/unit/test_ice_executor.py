from __future__ import annotations

import torch
from torch import nn

from geotoken3path.execution.croma_executor import InterfaceCertifiedCromaExecutor
from geotoken3path.execution.croma_plan import CromaExecutionPlan


class _Attention(nn.Module):
    def forward(self, x: torch.Tensor, _bias: torch.Tensor) -> torch.Tensor:
        return x * 0.25


class _FFN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5


class _Encoder(nn.Module):
    patch_size = 1

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.linear_input = nn.Linear(channels, 4, bias=False)
        nn.init.constant_(self.linear_input.weight, 0.125)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [nn.ModuleList([_Attention(), _FFN()]) for _ in range(depth)]
        )
        self.transformer.norm_out = nn.Identity()


class _Cross(nn.Module):
    def forward(self, *, x, context, relative_position_bias):
        del relative_position_bias
        return x + context


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s1_encoder = _Encoder(2, 2)
        self.s2_encoder = _Encoder(3, 3)
        self.GAP_FFN_s1 = nn.Identity()
        self.GAP_FFN_s2 = nn.Identity()
        self.cross_encoder = _Cross()
        self.register_buffer("attn_bias", torch.zeros(1, 1, 4, 4), persistent=False)


def _plan() -> CromaExecutionPlan:
    return CromaExecutionPlan(
        required_taps=(
            "s1_encoder.transformer.layers.0.1",
            "s2_encoder.transformer.layers.0.1",
        ),
        s1_last_layer=0,
        s2_last_layer=0,
        require_s1_final_norm=False,
        require_s2_final_norm=False,
        require_joint_encoder=False,
        require_s1_gap=False,
        require_s2_gap=False,
        eliminated_nodes=(
            "s1_encoder.transformer.layers.1",
            "s2_encoder.transformer.layers.1",
            "s2_encoder.transformer.layers.2",
            "s1_encoder.transformer.norm_out",
            "s2_encoder.transformer.norm_out",
            "GAP_FFN_s1",
            "GAP_FFN_s2",
            "cross_encoder",
            "joint_GAP",
        ),
        plan_sha256="0" * 64,
    )


def test_executor_preserves_ffn_hook_value_and_skips_suffix() -> None:
    backbone = _Backbone()
    captured: dict[str, torch.Tensor] = {}
    handle = backbone.s2_encoder.transformer.layers[0][1].register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("tap", output.detach().clone())
    )
    try:
        executor = InterfaceCertifiedCromaExecutor(_plan())
        sar = torch.ones(1, 2, 2, 2)
        optical = torch.ones(1, 3, 2, 2)
        outputs = executor.execute(backbone, SAR_images=sar, optical_images=optical)
        assert outputs == {}
        assert "tap" in captured
        # linear input -> 0.375; attention residual -> 0.46875; FFN hook sees
        # exactly 0.234375 before that output is added back to the residual.
        assert torch.allclose(captured["tap"], torch.full_like(captured["tap"], 0.234375))
    finally:
        handle.remove()
