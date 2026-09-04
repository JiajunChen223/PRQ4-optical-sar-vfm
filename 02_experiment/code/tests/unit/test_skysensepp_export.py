"""Unit tests for the R24 SkySense++ ICE export (physical parameter removal).

Hard gates on randomly initialized models (no safetensors dependency):

  - contract "b" export physically keeps 12 of the 24 backbone layers and
    removes ~half of the parameters (12/24 layers ≈ 49.9%), while contract "a"
    removes nothing;
  - the exported forward is bitwise equal to the full model's head applied to
    the corresponding ICE prefix feature maps;
  - export state-dict keys are a strict subset of the full model's keys with
    identical retained values, and the source model is never mutated (bitwise
    identical parameters before and after export).

The full model fixture is module-scoped; every export deep-copies it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "vendor"))

from geotoken3path.execution.skysensepp_export import (
    SkysenseppExportedModel,
    build_skysensepp_export_model,
)
from geotoken3path.models.skysensepp_seg import (
    SkySensePPSegmentationModel,
    load_vendor_config,
)

_LEN_LAYERS_FULL = 24
_LEN_LAYERS_B = 12


@pytest.fixture(scope="module")
def full_model() -> SkySensePPSegmentationModel:
    """Randomly initialized full R24 model, contract 'b' (shared per module)."""
    model = SkySensePPSegmentationModel(config_dict=load_vendor_config(), contract="b")
    model.eval()
    return model


@pytest.fixture(scope="module")
def full_model_a() -> SkySensePPSegmentationModel:
    """Randomly initialized full R24 model, contract 'a' (shared per module)."""
    model = SkySensePPSegmentationModel(config_dict=load_vendor_config(), contract="a")
    model.eval()
    return model


@pytest.fixture(scope="module")
def pixels_anno() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    pixels = torch.randn(1, 10, 64, 64, dtype=torch.float32)
    annotation = torch.randint(0, 65, (1, 64, 64), dtype=torch.long)
    return pixels, annotation


def test_export_contract_b_slices_backbone_to_12_layers(full_model) -> None:
    export, stats = build_skysensepp_export_model(full_model, contract="b")
    assert isinstance(export, SkysenseppExportedModel)
    assert len(full_model.backbone.layers) == _LEN_LAYERS_FULL  # source untouched
    assert len(export.backbone.layers) == _LEN_LAYERS_B
    assert export.plan.contract == "b"
    assert len(stats.removed_module_paths) == 12


def test_export_contract_b_halves_parameter_count(full_model) -> None:
    _, stats = build_skysensepp_export_model(full_model, contract="b")
    assert stats.full_parameter_count > 0
    assert stats.export_parameter_count == stats.full_parameter_count - stats.removed_parameter_count
    assert 0.48 < stats.removed_parameter_fraction < 0.51  # 12/24 layers ~ 49.9%
    assert stats.removed_parameter_fraction == stats.removed_parameter_count / stats.full_parameter_count


def test_export_contract_a_removes_nothing(full_model_a) -> None:
    export, stats = build_skysensepp_export_model(full_model_a, contract="a")
    assert len(export.backbone.layers) == _LEN_LAYERS_FULL
    assert stats.removed_parameter_count == 0
    assert stats.removed_parameter_fraction == 0.0
    assert stats.removed_module_paths == ()


def test_export_keeps_input_modules_and_head(full_model) -> None:
    export, _ = build_skysensepp_export_model(full_model, contract="b")
    for name in (
        "patch_embed",
        "pos_embed",
        "drop_after_pos",
        "vocabulary_token",
        "vocabulary_weight",
        "mask_token",
    ):
        assert hasattr(export.backbone, name), f"export backbone lost {name}"
    assert export.head is not None


def test_export_removed_module_paths_name_backbone_suffix(full_model) -> None:
    _, stats = build_skysensepp_export_model(full_model, contract="b")
    assert stats.removed_module_paths == tuple(
        f"backbone.layers.{index}" for index in range(_LEN_LAYERS_B, _LEN_LAYERS_FULL)
    )


def test_export_state_dict_is_subset_of_full_with_identical_values(full_model) -> None:
    export, _ = build_skysensepp_export_model(full_model, contract="b")
    full_keys = set(full_model.state_dict().keys())
    export_keys = set(export.state_dict().keys())
    assert export_keys <= full_keys, f"unexpected export keys: {export_keys - full_keys}"
    for key in export_keys:
        assert torch.equal(export.state_dict()[key], full_model.state_dict()[key]), key


def test_export_backbone_stays_frozen_head_stays_trainable(full_model) -> None:
    export, _ = build_skysensepp_export_model(full_model, contract="b")
    assert all(not p.requires_grad for p in export.backbone.parameters())
    assert any(p.requires_grad for p in export.head.parameters())
    trainable = [name for name, p in export.named_parameters() if p.requires_grad]
    assert trainable and all(name.startswith("head.") for name in trainable)


def test_export_forward_b_matches_full_prefix_bitwise(full_model, pixels_anno) -> None:
    """Export 'b' logits == full head over the full model's (5, 11) prefix maps."""
    pixels, annotation = pixels_anno
    export, _ = build_skysensepp_export_model(full_model, contract="b")
    with torch.no_grad():
        reference = full_model(pixel_values=pixels, annotation=annotation)
        exported = export(pixel_values=pixels, annotation=annotation)
    assert tuple(exported["feature_maps"][0].shape) == (1, 1024, 16, 16)
    assert tuple(exported["logits"].shape) == tuple(reference["logits"].shape)
    assert tuple(exported["logits"].shape) == (1, 11, 64, 64)  # input resolution
    prefix = [reference["feature_maps"][0], reference["feature_maps"][1],
              reference["feature_maps"][1], reference["feature_maps"][1]]
    expected = full_model.head(prefix, output_size=(64, 64))
    assert torch.equal(exported["logits"], expected)


def test_export_forward_a_matches_full_forward_bitwise(full_model_a, pixels_anno) -> None:
    """Contract 'a' executes the full 24 layers: logits identical to Full."""
    pixels, annotation = pixels_anno
    export, _ = build_skysensepp_export_model(full_model_a, contract="a")
    with torch.no_grad():
        reference = full_model_a(pixel_values=pixels, annotation=annotation)
        exported = export(pixel_values=pixels, annotation=annotation)
    assert len(exported["feature_maps"]) == 4
    assert exported["executed_layer_count"] == 24
    assert torch.equal(exported["logits"], reference["logits"])


def test_export_does_not_mutate_source_model(full_model_a, pixels_anno) -> None:
    pixels, annotation = pixels_anno
    parameters_before = {name: p.detach().clone() for name, p in full_model_a.named_parameters()}
    with torch.no_grad():
        reference = full_model_a(pixel_values=pixels, annotation=annotation)
    for contract in ("a", "b"):
        if contract == "b":
            # Exporting contract "b" from a contract "a" head is a documented
            # receiver mismatch; the refusal must also leave the source intact.
            try:
                build_skysensepp_export_model(full_model_a, contract=contract)
            except ValueError:
                continue
            raise AssertionError("expected ValueError for head/receiver contract mismatch")
        build_skysensepp_export_model(full_model_a, contract=contract)
    assert len(full_model_a.backbone.layers) == _LEN_LAYERS_FULL
    assert list(full_model_a.state_dict().keys()) == list(parameters_before.keys())
    for name, parameter in parameters_before.items():
        assert torch.equal(full_model_a.get_parameter(name), parameter), name
    with torch.no_grad():
        after = full_model_a(pixel_values=pixels, annotation=annotation)
    assert torch.equal(reference["logits"], after["logits"])


def test_export_rejects_invalid_contract(full_model) -> None:
    with pytest.raises(ValueError):
        build_skysensepp_export_model(full_model, contract="c")


def test_export_rejects_non_skysensepp_source() -> None:
    with pytest.raises(TypeError):
        build_skysensepp_export_model(object(), contract="a")  # type: ignore[arg-type]
