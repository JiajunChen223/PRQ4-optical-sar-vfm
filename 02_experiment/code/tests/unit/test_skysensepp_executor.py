"""Unit tests for the R24 SkySense++ ICE prefix executor (zero-drift gates).

Hard numerical gates on a randomly initialized 24-layer S2 backbone (built
via ``models.skysensepp_seg`` with random weights, ``drop_path_rate=0``):

  1. prefix execution at max_layer=23 is bitwise equal to the official full
     forward (all four grid maps, ``torch.equal``);
  2. prefix execution at max_layer=11 returns the official grid maps for
     layers (5, 11), and map[1] (layer 11) is bitwise equal to the official
     full forward's map at layer 11 (same shared prefix);
  3. plan "b" executes 12 of the 24 layers while plan "a" executes all 24;
  4. the backbone is restored bitwise after every prefix run.

Model construction dominates runtime (~8s CPU), so the randomly initialized
model is a module-scoped fixture shared by every test; forwards run under
``torch.no_grad()``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "vendor"))

from geotoken3path.execution.skysensepp_plan import (
    SkySensePPExecutionPlan,
    compile_skysensepp_plan,
)
from geotoken3path.models.skysensepp_executor import (
    SkySensePPExecutionError,
    SkySensePPPrefixExecutor,
)
from geotoken3path.models.skysensepp_seg import (
    SkySensePPSegmentationModel,
    load_vendor_config,
)

_PLAN_A = compile_skysensepp_plan("a")
_PLAN_B = compile_skysensepp_plan("b")


@pytest.fixture(scope="module")
def backbone() -> torch.nn.Module:
    """Randomly initialized 24-layer S2 backbone (shared across the module)."""
    model = SkySensePPSegmentationModel(config_dict=load_vendor_config(), contract="b")
    model.eval()
    return model.backbone


@pytest.fixture(scope="module")
def pixels_anno() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    pixels = torch.randn(2, 10, 64, 64, dtype=torch.float32)
    annotation = torch.randint(0, 65, (2, 64, 64), dtype=torch.long)
    return pixels, annotation


def _official_full(backbone, pixels, annotation) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        maps = backbone(pixel_values=pixels, annotation=annotation, return_dict=False)
    if isinstance(maps, tuple) and len(maps) == 1 and isinstance(maps[0], tuple):
        maps = maps[0]
    return tuple(maps)


def test_executor_requires_a_plan() -> None:
    with pytest.raises(TypeError):
        SkySensePPPrefixExecutor(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SkySensePPPrefixExecutor(
            SkySensePPExecutionPlan(
                contract="c", max_layer=23, required_output_indices=(23,),
                executed_layer_count=24, eliminated_layers=(), plan_sha256="0" * 64,
            )
        )


def test_executor_rejects_non_backbone_models() -> None:
    executor = SkySensePPPrefixExecutor(_PLAN_B)
    with pytest.raises(SkySensePPExecutionError):
        executor.execute(torch.nn.Linear(4, 4), torch.randn(1, 10, 64, 64), torch.randint(0, 65, (1, 64, 64)))


def test_full_prefix_matches_official_forward_bitwise(
    backbone, pixels_anno,
) -> None:
    """max_layer=23 (contract 'a') reproduces the official full forward exactly."""
    pixels, annotation = pixels_anno
    official = _official_full(backbone, pixels, annotation)
    executor = SkySensePPPrefixExecutor(_PLAN_A)
    result = executor.execute(backbone, pixels, annotation)
    assert tuple(result["layer_indices"]) == (5, 11, 17, 23)
    assert result["executed_layer_count"] == 24
    assert len(result["feature_maps"]) == 4 == len(official)
    for prefix_map, official_map in zip(result["feature_maps"], official):
        assert tuple(prefix_map.shape) == (2, 1024, 16, 16)
        assert torch.equal(prefix_map, official_map)


def test_ice_b_prefix_matches_official_shared_prefix_bitwise(
    backbone, pixels_anno,
) -> None:
    """max_layer=11 emits official grid maps (5, 11), bitwise equal to Full."""
    pixels, annotation = pixels_anno
    official = _official_full(backbone, pixels, annotation)
    executor = SkySensePPPrefixExecutor(_PLAN_B)
    result = executor.execute(backbone, pixels, annotation)
    assert tuple(result["layer_indices"]) == (5, 11)
    assert result["executed_layer_count"] == 12
    maps = result["feature_maps"]
    assert len(maps) == 2
    # Map at layer 5 and the deepest map at layer 11 are the shared prefix.
    assert torch.equal(maps[0], official[0])
    assert torch.equal(maps[1], official[1])


def test_contract_b_executes_fewer_layers_than_contract_a(backbone, pixels_anno) -> None:
    pixels, annotation = pixels_anno
    count_a = SkySensePPPrefixExecutor(_PLAN_A).execute(backbone, pixels, annotation)["executed_layer_count"]
    count_b = SkySensePPPrefixExecutor(_PLAN_B).execute(backbone, pixels, annotation)["executed_layer_count"]
    assert count_a == 24
    assert count_b == 12
    # ICE "b" skips the 12-layer suffix of Full.
    assert count_a - count_b == 12


def test_executor_restores_backbone_after_prefix_runs(backbone, pixels_anno) -> None:
    pixels, annotation = pixels_anno
    state_before = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
    full_keys_before = list(backbone.state_dict().keys())
    # Alternate contract-a (full, no truncation) and contract-b (truncated) runs.
    for _ in range(2):
        SkySensePPPrefixExecutor(_PLAN_A).execute(backbone, pixels, annotation)
        SkySensePPPrefixExecutor(_PLAN_B).execute(backbone, pixels, annotation)
    assert len(backbone.layers) == 24
    assert list(backbone.state_dict().keys()) == full_keys_before
    for key, value in state_before.items():
        assert torch.equal(backbone.state_dict()[key], value)


def test_executor_rejects_truncation_below_required_layer(backbone, pixels_anno) -> None:
    pixels, annotation = pixels_anno
    executor = SkySensePPPrefixExecutor(_PLAN_B)
    # Contract "b" requires layer 11; max_layer=10 would drop it.
    with pytest.raises(SkySensePPExecutionError):
        executor.execute(backbone, pixels, annotation, max_layer=10)
    with pytest.raises(SkySensePPExecutionError):
        executor.execute(backbone, pixels, annotation, max_layer=0)


def test_executor_rejects_max_layer_beyond_plan_and_depth(backbone, pixels_anno) -> None:
    pixels, annotation = pixels_anno
    executor_b = SkySensePPPrefixExecutor(_PLAN_B)
    with pytest.raises(SkySensePPExecutionError):
        executor_b.execute(backbone, pixels, annotation, max_layer=12)
    executor_a = SkySensePPPrefixExecutor(_PLAN_A)
    with pytest.raises(SkySensePPExecutionError):
        executor_a.execute(backbone, pixels, annotation, max_layer=24)


def test_executor_rejects_invalid_max_layer_type(backbone, pixels_anno) -> None:
    pixels, annotation = pixels_anno
    executor = SkySensePPPrefixExecutor(_PLAN_B)
    with pytest.raises(SkySensePPExecutionError):
        executor.execute(backbone, pixels, annotation, max_layer=True)
    with pytest.raises(SkySensePPExecutionError):
        executor.execute(backbone, pixels, annotation, max_layer=11.5)


def test_executor_rejects_malformed_inputs(backbone, pixels_anno) -> None:
    executor = SkySensePPPrefixExecutor(_PLAN_B)
    with pytest.raises(SkySensePPExecutionError):
        executor.execute(backbone, torch.randn(10, 64, 64), torch.randint(0, 65, (1, 64, 64)))
    with pytest.raises(SkySensePPExecutionError):
        executor.execute(backbone, torch.randn(1, 10, 64, 64), torch.randn(1, 64, 64))


def test_unknown_plan_grid_layer_fails_closed(backbone, pixels_anno) -> None:
    """A required grid layer outside the official grid must never be served."""
    pixels, annotation = pixels_anno
    bogus = SkySensePPExecutionPlan(
        contract="b", max_layer=11, required_output_indices=(10,),
        executed_layer_count=12, eliminated_layers=(), plan_sha256="0" * 64,
    )
    with pytest.raises(SkySensePPExecutionError):
        SkySensePPPrefixExecutor(bogus).execute(backbone, pixels, annotation)
