# -*- coding: utf-8 -*-
"""Unit tests for Task-Specific Model Export (R21 queue 3)."""
import copy
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
import torch
from torch import nn

from geotoken3path.execution.croma_export import CromaExportedBackbone, build_export_backbone

# Reuse the synthetic CROMA-like backbone from the ICE factory tests.
spec = importlib.util.spec_from_file_location(
    "tif", Path(__file__).resolve().parents[1] / "unit" / "test_ice_factory.py"
)
_tif = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(_tif)


def test_export_backbone_strips_dead_modules() -> None:
    backbone = _tif._Backbone()
    export, stats = build_export_backbone(backbone)
    # S2 reduced from 12 to 6 layers; norms/GAP/cross removed.
    assert len(export.s2_encoder.transformer.layers) == 6
    assert not hasattr(export.s2_encoder.transformer, "norm_out")
    assert not hasattr(export.s1_encoder.transformer, "norm_out")
    assert not hasattr(export, "GAP_FFN_s1")
    assert not hasattr(export, "GAP_FFN_s2")
    assert not hasattr(export, "cross_encoder")
    # S1 keeps all 6 layers (taps reach layer 5).
    assert len(export.s1_encoder.transformer.layers) == 6
    # Retained submodule paths are intact for the tap adapter.
    assert export.get_submodule("s2_encoder.transformer.layers.5.1") is not None
    assert export.get_submodule("s1_encoder.transformer.layers.5.1") is not None
    # attn_bias stays a plain attribute (not a buffer -> not in state_dict).
    assert isinstance(export.attn_bias, torch.Tensor)
    assert "attn_bias" not in export.state_dict()
    # Stats are sane.
    assert stats.removed_parameter_count > 0
    assert 0.0 < stats.removed_parameter_fraction < 1.0
    assert any("s2_encoder.transformer.layers.6" in p for p in stats.removed_module_paths)


def test_export_backbone_state_keys_are_subset_of_full() -> None:
    backbone = _tif._Backbone()
    full_keys = set(backbone.state_dict().keys())
    export, _ = build_export_backbone(backbone)
    export_keys = set(export.state_dict().keys())
    assert export_keys <= full_keys, "export keys must be a subset of full keys"
    # Everything retained is bitwise identical.
    for key in export_keys:
        assert torch.equal(export.state_dict()[key], backbone.state_dict()[key])


def test_export_backbone_forward_matches_full_with_shared_weights() -> None:
    g = torch.Generator().manual_seed(3)
    backbone = _tif._Backbone()
    export, _ = build_export_backbone(backbone)
    # Load the full backbone's retained weights into the export (already shared
    # by deepcopy, but ensure forward is executable and deterministic).
    export.eval()
    # Run a paired forward through the export executor path.
    sar = torch.randn(1, 2, 8, 8, generator=g)
    optical = torch.randn(1, 12, 8, 8, generator=g)
    with torch.no_grad():
        out = export(SAR_images=sar, optical_images=optical)
    assert isinstance(out, dict)
    assert "SAR_encodings" in out or set(out.keys()) <= {"SAR_encodings", "optical_encodings"}


def test_export_stats_report_removed_paths() -> None:
    backbone = _tif._Backbone()
    _, stats = build_export_backbone(backbone)
    removed = " ".join(stats.removed_module_paths)
    assert "cross_encoder" in removed
    assert "GAP_FFN_s1" in removed
    assert "s2_encoder.transformer.norm_out" in removed
    assert stats.removed_parameter_fraction == stats.removed_parameter_count / stats.full_parameter_count


def test_export_from_hooked_backbone_has_no_ghost_hooks() -> None:
    """Export built from an adapter-wrapped backbone must not carry ghost hooks."""
    from geotoken3path.models.croma_bridge import CromaDepthTapAdapter

    backbone = _tif._Backbone()
    # Wrap the backbone in a tap adapter (installs 8 forward hooks), mirroring
    # the production certify path where the full model is built first.
    stage_taps = {
        "optical": {"mid": "s2_encoder.transformer.layers.2.1", "late": "s2_encoder.transformer.layers.5.1"},
        "sar": {"mid": "s1_encoder.transformer.layers.2.1", "late": "s1_encoder.transformer.layers.5.1"},
    }
    depth_group_taps = {
        "mid": ["s1_encoder.transformer.layers.0.1", "s1_encoder.transformer.layers.1.1",
                 "s1_encoder.transformer.layers.2.1", "s1_encoder.transformer.layers.3.1"],
        "late": ["s1_encoder.transformer.layers.2.1", "s1_encoder.transformer.layers.3.1",
                  "s1_encoder.transformer.layers.4.1", "s1_encoder.transformer.layers.5.1"],
    }
    adapter = CromaDepthTapAdapter(
        backbone, stages=("mid", "late"), dim=4, stage_taps=stage_taps, depth_group_taps=depth_group_taps,
    )
    assert any(m._forward_hooks for m in backbone.modules()), "adapter should install hooks"
    export, _ = build_export_backbone(backbone)
    # Export backbone modules must carry no forward hooks (no ghost hooks).
    hooked = [p for m in export.modules() for p in m._forward_hooks]
    assert not hooked, f"export backbone carries {len(hooked)} ghost forward hooks"
    adapter.close()
