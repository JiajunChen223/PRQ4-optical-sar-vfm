"""R24 resolution-alignment gates: logits are upsampled to the input grid.

The segmentation losses/metrics require logits at the input resolution while
the SkySense++ S2 backbone emits H/4 x W/4 feature maps (patch 4).  These
tests pin the R24 fix: model and export forwards bilinearly upsample the head
output to the pixel input size (SEN12TS delivers 120x120 crops -> 120x120
logits; the official 16x16 pretraining grid -> 16x16 is also preserved through
the head's optional ``output_size``).  Runs the real 24-layer vendor backbone
with random weights (``drop_path_rate=0``); no safetensors dependency.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "vendor"))

from geotoken3path.execution.skysensepp_export import build_skysensepp_export_model
from geotoken3path.models.skysensepp_seg import (
    Conv1x1SegmentationHead,
    SkySensePPSegmentationModel,
    load_vendor_config,
)


def _pixels_annotation(batch: int, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    pixels = torch.randn(batch, 10, size, size, dtype=torch.float32)
    annotation = torch.randint(0, 65, (batch, size, size), dtype=torch.long)
    return pixels, annotation


@pytest.fixture(scope="module")
def model_a() -> SkySensePPSegmentationModel:
    """Randomly initialized full R24 model, contract 'a' (24 layers executed)."""
    model = SkySensePPSegmentationModel(config_dict=load_vendor_config(), contract="a")
    model.eval()
    return model


@pytest.fixture(scope="module")
def model_b() -> SkySensePPSegmentationModel:
    """Randomly initialized full R24 model, contract 'b' (deepest-map head)."""
    model = SkySensePPSegmentationModel(config_dict=load_vendor_config(), contract="b")
    model.eval()
    return model


def test_model_forward_logits_match_input_grid_at_64(model_a) -> None:
    pixels, annotation = _pixels_annotation(1, 64)
    with torch.no_grad():
        out = model_a(pixel_values=pixels, annotation=annotation)
    assert tuple(out["logits"].shape) == (1, 11, 64, 64)
    for feature in out["feature_maps"]:
        assert tuple(feature.shape) == (1, 1024, 16, 16)  # H/4 unchanged


def test_model_forward_logits_match_input_grid_at_120(model_a) -> None:
    pixels, annotation = _pixels_annotation(1, 120)
    with torch.no_grad():
        out = model_a(pixel_values=pixels, annotation=annotation)
    # SEN12TS crop 120x120 -> backbone grid 30x30 -> logits back at 120x120.
    assert tuple(out["logits"].shape) == (1, 11, 120, 120)
    assert tuple(out["feature_maps"][0].shape) == (1, 1024, 30, 30)


def test_contract_b_model_forward_logits_match_input_grid_at_120(model_b) -> None:
    pixels, annotation = _pixels_annotation(1, 120)
    with torch.no_grad():
        out = model_b(pixel_values=pixels, annotation=annotation)
    assert tuple(out["logits"].shape) == (1, 11, 120, 120)


def test_head_output_size_optional_lowres_contract(model_a) -> None:
    """The raw head still emits H/4 without output_size (backward compatible)."""
    pixels, annotation = _pixels_annotation(1, 120)
    with torch.no_grad():
        out = model_a(pixel_values=pixels, annotation=annotation)
    with torch.no_grad():
        lowres = model_a.head(list(out["feature_maps"]))
    assert tuple(lowres.shape) == (1, 11, 30, 30)


def test_head_upsample_120_from_synthetic_30_grid() -> None:
    """Head upsampling is resolution-agnostic: 30x30 maps -> 120x120 logits."""
    head = Conv1x1SegmentationHead(contract="b", num_classes=11, seed=0)
    head.eval()
    maps = [torch.randn(2, 1024, 30, 30) for _ in range(2)]
    with torch.no_grad():
        logits = head(maps, output_size=(120, 120))
    assert tuple(logits.shape) == (2, 11, 120, 120)
    with torch.no_grad():
        identity = head(maps, output_size=(30, 30))
    with torch.no_grad():
        lowres = head(maps)
    # Same-size interpolate is an exact copy of the raw low-resolution logits.
    assert torch.equal(identity, lowres)
    assert torch.equal(identity, logits) is False


def test_export_contract_b_logits_match_input_grid_at_120(model_b) -> None:
    """Exported ICE forward upsamples too, and matches Full logits bitwise."""
    pixels, annotation = _pixels_annotation(1, 120)
    export, _ = build_skysensepp_export_model(model_b, contract="b")
    export.eval()
    with torch.no_grad():
        reference = model_b(pixel_values=pixels, annotation=annotation)
        exported = export(pixel_values=pixels, annotation=annotation)
    assert tuple(exported["logits"].shape) == (1, 11, 120, 120)
    # Full contract 'b' head reads layer-11 map, exactly what the export serves.
    assert torch.equal(exported["logits"], reference["logits"])
