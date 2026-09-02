from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from geotoken3path.models.factory import build_model


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(2, 16, 32, generator=generator),
        torch.randn(2, 16, 32, generator=generator),
    )


def test_shared_factory_keeps_parameter_surface_identical() -> None:
    baseline = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    candidate = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    assert list(baseline.state_dict()) == list(candidate.state_dict())
    assert [name for name, p in baseline.named_parameters() if p.requires_grad] == [
        name for name, p in candidate.named_parameters() if p.requires_grad
    ]



