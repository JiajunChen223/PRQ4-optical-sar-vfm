from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from geotoken3path.data.sen12ts import (
    _paired_geometric_orientation,
    _validate_augmentation_spec,
)
from geotoken3path.engine.formal_runner import _formal_loader_optimized
from geotoken3path.utils.config import resolve_approved_config


ROOT = Path(__file__).resolve().parents[2]


def test_paired_geometric_orientation_is_deterministic_and_shared() -> None:
    optical = torch.arange(12 * 4 * 4, dtype=torch.float32).reshape(12, 4, 4)
    sar = optical[:2] + 1000.0
    target = optical[0].to(torch.long)
    first = _paired_geometric_orientation(optical, sar, target, seed=0, index=7)
    second = _paired_geometric_orientation(optical, sar, target, seed=0, index=7)
    assert first[3] in {"identity", "horizontal_flip", "vertical_flip", "rotate_90", "rotate_180", "rotate_270", "transpose", "anti_transpose"}
    assert first[3] == second[3]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[1], first[0][:2] + 1000.0)
    assert torch.equal(first[2], first[0][0].to(torch.long))


def test_augmentation_contract_is_exact() -> None:
    spec = {
        "name": "paired_geometric_v1",
        "enabled": True,
        "train_only": True,
        "deterministic": True,
        "orientation_space": "D4",
        "operations": ["horizontal_flip", "vertical_flip", "rotate_90", "rotate_180", "rotate_270", "transpose", "anti_transpose"],
    }
    assert _validate_augmentation_spec(spec)["name"] == "paired_geometric_v1"


def test_resolved_config_contains_amended_budget_early_stop_and_augmentation() -> None:
    resolved = resolve_approved_config(ROOT, "always_fuse", execution_scale="strengthening")
    runtime = resolved["runtime"]
    assert runtime["max_formal_epochs"] == 24
    assert runtime["early_stopping"]["monitor"] == "validation.mIoU"
    assert runtime["early_stopping"]["burn_in_epochs"] == 8
    assert runtime["early_stopping"]["patience"] == 5
    assert runtime["augmentation"]["name"] == "paired_geometric_v1"
    assert runtime["augmentation"]["train_only"] is True


def test_every_formal_cloud_scale_uses_the_measured_3090_loader_contract() -> None:
    assert _formal_loader_optimized("smoke") is False
    for scale in ("baseline", "screening", "strengthening", "confirmation", "acceptance", "extension"):
        assert _formal_loader_optimized(scale) is True
