# -*- coding: utf-8 -*-
"""Protocol assertions for the Task-Depth study (D0/D1/D2).

Dk changes only the optical late tap depth (D0: layer5, D1: layer4,
D2: layer3); the optical mid tap stays at layer 2 and all SAR taps/depth
groups are frozen. These assertions guarantee the depth rows are a clean
single-variable ablation:
  - identical state-dict surface across D0/D1/D2 (same architecture);
  - only the optical late tap layer number differs in resolved config;
  - the ICE compiled plan's s2_last_layer equals the optical late tap layer.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch import nn

from geotoken3path.models.factory import build_model
from geotoken3path.execution import (
    BackboneFeatureContract,
    compile_croma_execution_plan,
)

ROOT = Path(__file__).resolve().parents[2]

# (suffix, optical_late_layer)
DEPTH_ROWS = (("d0_o5", 5), ("d1_o4", 4), ("d2_o3", 3))


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


def _cfg_for(suffix: str, late_layer: int) -> dict:
    yaml_name = "geotoken3path.yaml" if suffix == "d0_o5" else f"geotoken3path_{suffix}.yaml"
    import yaml
    model_doc = yaml.safe_load((ROOT / "configs/model" / yaml_name).read_text(encoding="utf-8"))
    return {
        "model": {
            "token_dim": 4,
            "num_classes": 3,
            "active_budget": 1.0,
            "mechanism_set": "always_fuse",
            "local_window_tokens": 1,
            "stages": ["mid", "late"],
            "depth_taps": model_doc["backbone"]["depth_taps"],
        },
        "trainability": {"backbone_policy": "frozen"},
    }


def test_depth_rows_state_surface_identical() -> None:
    """D0/D1/D2 build the same architecture -> identical state-dict keys."""
    models = {}
    for suffix, late in DEPTH_ROWS:
        torch.manual_seed(7)
        backbone = _Backbone()
        cfg = _cfg_for(suffix, late)
        models[suffix] = build_model(cfg)
    keys0 = list(models["d0_o5"].state_dict())
    for suffix in ("d1_o4", "d2_o3"):
        assert list(models[suffix].state_dict()) == keys0, f"{suffix} state surface differs"


def test_depth_rows_single_variable_is_optical_late_tap() -> None:
    """The resolved depth_taps differ only in the optical late tap layer."""
    taps = {}
    for suffix, late in DEPTH_ROWS:
        cfg = _cfg_for(suffix, late)
        taps[suffix] = cfg["model"]["depth_taps"]
    base = taps["d0_o5"]
    for suffix, late in DEPTH_ROWS[1:]:
        t = taps[suffix]
        assert t["stage"]["sar"] == base["stage"]["sar"], f"{suffix}: SAR taps changed"
        assert t["sar_depth_group"] == base["sar_depth_group"], f"{suffix}: SAR depth group changed"
        assert t["stage"]["optical"]["mid"] == base["stage"]["optical"]["mid"], f"{suffix}: optical mid changed"
        assert t["stage"]["optical"]["late"] == f"s2_encoder.transformer.layers.{late}.1"


def test_depth_rows_ice_plan_s2_last_matches_late_tap() -> None:
    """Compiled ICE plan executes optical prefix up to the late tap layer."""
    for suffix, late in DEPTH_ROWS:
        torch.manual_seed(7)
        backbone = _Backbone()
        cfg = _cfg_for(suffix, late)
        contract = BackboneFeatureContract(
            optical_stages=("mid", "late"),
            sar_stages=("mid", "late"),
            sar_depth_group_stages=("mid", "late"),
            native_joint=False,
            global_optical=False,
            global_sar=False,
        )
        plan = compile_croma_execution_plan(
            model_cfg=cfg["model"], receiver_contract=contract, audited_backbone=backbone
        )
        assert plan.s2_last_layer == late, f"{suffix}: ICE s2_last={plan.s2_last_layer}, expected {late}"
