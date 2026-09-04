from __future__ import annotations

import torch
from torch import nn

from geotoken3path.execution.certification import compare_gradients, named_trainable_gradients
from geotoken3path.models.factory import build_vfm_segmentation_model


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, _bias: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FFN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class _Encoder(nn.Module):
    patch_size = 1

    def __init__(self, channels: int, depth: int, dim: int = 4) -> None:
        super().__init__()
        self.linear_input = nn.Linear(channels, dim)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [nn.ModuleList([_Attention(dim), _FFN(dim)]) for _ in range(depth)]
        )
        self.transformer.norm_out = nn.LayerNorm(dim)

    def forward(self, imgs: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = imgs.permute(0, 2, 3, 1).reshape(imgs.shape[0], -1, imgs.shape[1])
        x = self.linear_input(x)
        for attention, ffn in self.transformer.layers:
            x = attention(x, attn_bias) + x
            x = ffn(x) + x
        return self.transformer.norm_out(x)


class _CrossEncoder(nn.Module):
    def __init__(self, dim: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(depth)])

    def forward(
        self,
        *,
        x: torch.Tensor,
        context: torch.Tensor,
        relative_position_bias: torch.Tensor,
    ) -> torch.Tensor:
        del relative_position_bias
        value = x + context
        for layer in self.layers:
            value = value + layer(value)
        return value


class _CromaLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s1_encoder = _Encoder(2, 6)
        self.s2_encoder = _Encoder(12, 12)
        self.GAP_FFN_s1 = nn.Linear(4, 4)
        self.GAP_FFN_s2 = nn.Linear(4, 4)
        self.cross_encoder = _CrossEncoder()
        self.attn_bias = torch.zeros(1, 1, 4, 4)

    def forward(self, SAR_images=None, optical_images=None):
        assert SAR_images is not None and optical_images is not None
        sar = self.s1_encoder(SAR_images, self.attn_bias.to(SAR_images.device))
        sar_gap = self.GAP_FFN_s1(sar.mean(dim=1))
        optical = self.s2_encoder(optical_images, self.attn_bias.to(optical_images.device))
        optical_gap = self.GAP_FFN_s2(optical.mean(dim=1))
        joint = self.cross_encoder(
            x=sar,
            context=optical,
            relative_position_bias=self.attn_bias.to(optical_images.device),
        )
        return {
            "SAR_encodings": sar,
            "SAR_GAP": sar_gap,
            "optical_encodings": optical,
            "optical_GAP": optical_gap,
            "joint_encodings": joint,
            "joint_GAP": joint.mean(dim=1),
        }


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
        "trainability": {"backbone_policy": "tap_connected"},
    }


def _pair():
    torch.manual_seed(101)
    full_backbone = _CromaLike()
    torch.manual_seed(101)
    ice_backbone = _CromaLike()
    torch.manual_seed(202)
    full = build_vfm_segmentation_model(
        _config(), audited_croma_backbone=full_backbone, backbone_execution="full"
    )
    torch.manual_seed(202)
    ice = build_vfm_segmentation_model(
        _config(), audited_croma_backbone=ice_backbone, backbone_execution="ice_exact"
    )
    ice.load_state_dict(full.state_dict(), strict=True)
    return full, ice


def test_full_and_ice_match_end_to_end_logits_and_trainable_gradients() -> None:
    full, ice = _pair()
    torch.manual_seed(303)
    optical = torch.randn(2, 12, 2, 2)
    sar = torch.randn(2, 2, 2, 2)

    full.eval()
    ice.eval()
    with torch.no_grad():
        full_logits = full(optical, sar)
        ice_logits = ice(optical, sar)
    assert torch.equal(full_logits, ice_logits)

    full.train()
    ice.train()
    full.zero_grad(set_to_none=True)
    ice.zero_grad(set_to_none=True)
    full_loss = full(optical, sar).square().mean()
    ice_loss = ice(optical, sar).square().mean()
    full_loss.backward()
    ice_loss.backward()
    gradients = compare_gradients(
        named_trainable_gradients(full), named_trainable_gradients(ice)
    )
    assert float(full_loss) == float(ice_loss)
    assert gradients["missing_gradient_names"] == []
    assert gradients["max_gradient_abs_error"] == 0.0
