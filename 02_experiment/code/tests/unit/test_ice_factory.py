from __future__ import annotations

import torch
from torch import nn

from geotoken3path.models.factory import build_vfm_segmentation_model


class _Attention(nn.Module):
    def forward(self, x: torch.Tensor, _bias: torch.Tensor) -> torch.Tensor:
        return x


class _FFN(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Encoder(nn.Module):
    patch_size = 1

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.linear_input = nn.Linear(channels, 4)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [nn.ModuleList([_Attention(), _FFN()]) for _ in range(depth)]
        )
        self.transformer.norm_out = nn.LayerNorm(4)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s1_encoder = _Encoder(2, 6)
        self.s2_encoder = _Encoder(12, 12)
        self.GAP_FFN_s1 = nn.Linear(4, 4)
        self.GAP_FFN_s2 = nn.Linear(4, 4)
        self.cross_encoder = nn.Sequential(nn.Linear(4, 4))
        self.attn_bias = torch.zeros(1, 1, 4, 4)


def _config() -> dict[str, object]:
    return {
        "model": {
            "token_dim": 4,
            "num_classes": 3,
            "active_budget": 1.0,
            "mechanism_set": "always_fuse",
            "local_window_tokens": 1,
            "stages": ["mid", "late"],
            "depth_taps": {
                "stage": {
                    "optical": {
                        "mid": "s2_encoder.transformer.layers.2.1",
                        "late": "s2_encoder.transformer.layers.5.1",
                    },
                    "sar": {
                        "mid": "s1_encoder.transformer.layers.2.1",
                        "late": "s1_encoder.transformer.layers.5.1",
                    },
                },
                "sar_depth_group": {
                    "mid": [
                        "s1_encoder.transformer.layers.0.1",
                        "s1_encoder.transformer.layers.1.1",
                        "s1_encoder.transformer.layers.2.1",
                        "s1_encoder.transformer.layers.3.1",
                    ],
                    "late": [
                        "s1_encoder.transformer.layers.2.1",
                        "s1_encoder.transformer.layers.3.1",
                        "s1_encoder.transformer.layers.4.1",
                        "s1_encoder.transformer.layers.5.1",
                    ],
                },
            },
        },
        "trainability": {"backbone_policy": "frozen"},
    }


def test_full_and_ice_factory_models_have_identical_state_surfaces() -> None:
    torch.manual_seed(7)
    full_backbone = _Backbone()
    torch.manual_seed(7)
    ice_backbone = _Backbone()
    config = _config()
    # Reseed before each build: building the full model consumes global RNG
    # (token_model initialization), so without a reseed the ice model would
    # initialize its token model on a different random stream and the state
    # comparison below would compare two legitimately different initializations.
    torch.manual_seed(7)
    full = build_vfm_segmentation_model(
        config, audited_croma_backbone=full_backbone, backbone_execution="full"
    )
    torch.manual_seed(7)
    ice = build_vfm_segmentation_model(
        config, audited_croma_backbone=ice_backbone, backbone_execution="ice_exact"
    )
    full_state = full.state_dict()
    ice_state = ice.state_dict()
    assert list(full_state) == list(ice_state)
    assert all(torch.equal(full_state[name], ice_state[name]) for name in full_state)
    assert getattr(full_backbone, "_ice_execution_mode", None) is None
    assert getattr(ice_backbone, "_ice_execution_mode", None) == "ice_exact"
    plan = getattr(ice, "_ice_execution_plan")
    assert plan.s1_last_layer == 5
    assert plan.s2_last_layer == 5
    assert not plan.require_joint_encoder
