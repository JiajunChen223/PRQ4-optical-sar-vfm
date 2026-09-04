"""R24 export receiver-contract validation gates.

``build_skysensepp_export_model`` must refuse to slice a backbone whose head
was trained for the other receiver contract: exporting contract "a" from a
contract "b" model (or vice versa) raises ``ValueError`` before any physical
removal.  Matched contracts keep working, and the source model is untouched.
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
    SkySensePPSegmentationModel,
    load_vendor_config,
)


@pytest.fixture(scope="module")
def model_a() -> SkySensePPSegmentationModel:
    return SkySensePPSegmentationModel(config_dict=load_vendor_config(), contract="a")


@pytest.fixture(scope="module")
def model_b() -> SkySensePPSegmentationModel:
    return SkySensePPSegmentationModel(config_dict=load_vendor_config(), contract="b")


def test_export_contract_a_from_contract_b_model_raises(model_b) -> None:
    with pytest.raises(ValueError, match="contract"):
        build_skysensepp_export_model(model_b, contract="a")


def test_export_contract_b_from_contract_a_model_raises(model_a) -> None:
    with pytest.raises(ValueError, match="contract"):
        build_skysensepp_export_model(model_a, contract="b")


def test_export_matching_contracts_succeed(model_a, model_b) -> None:
    export_a, stats_a = build_skysensepp_export_model(model_a, contract="a")
    assert export_a.plan.contract == "a"
    assert stats_a.removed_parameter_count == 0
    export_b, stats_b = build_skysensepp_export_model(model_b, contract="b")
    assert export_b.plan.contract == "b"
    assert stats_b.removed_parameter_count > 0


def test_failed_export_leaves_source_and_backbone_untouched(model_b) -> None:
    layers_before = len(model_b.backbone.layers)
    keys_before = list(model_b.state_dict().keys())
    values_before = {k: v.detach().clone() for k, v in model_b.state_dict().items()}
    with pytest.raises(ValueError):
        build_skysensepp_export_model(model_b, contract="a")
    assert len(model_b.backbone.layers) == layers_before
    assert list(model_b.state_dict().keys()) == keys_before
    for key, value in values_before.items():
        assert torch.equal(model_b.state_dict()[key], value), key
