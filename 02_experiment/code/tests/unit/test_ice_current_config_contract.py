from __future__ import annotations

import torch

from geotoken3path.execution.contracts import BackboneFeatureContract
from geotoken3path.models.fusion import OpticalSarTokenModel


def test_verified_baseline_receiver_contract_matches_actual_joint_independence() -> None:
    contract = BackboneFeatureContract(
        optical_stages=("mid", "late"),
        sar_stages=("mid", "late"),
        sar_depth_group_stages=("mid", "late"),
        native_joint=False,
    )
    assert contract.native_joint is False

    torch.manual_seed(17)
    model = OpticalSarTokenModel(
        dim=4,
        num_classes=3,
        active_budget=1.0,
        mechanism_set="always_fuse",
        local_window_size=1,
        stages=("mid", "late"),
        allow_synthetic_depth_group_fallback=False,
    ).eval()
    optical = {
        "mid": torch.randn(2, 4, 4),
        "late": torch.randn(2, 4, 4),
    }
    sar = {
        "mid": torch.randn(2, 4, 4),
        "late": torch.randn(2, 4, 4),
    }
    depth = {
        "mid": torch.randn(2, 4, 4, 4),
        "late": torch.randn(2, 4, 4, 4),
    }
    random_joint = torch.randn(2, 4, 4)
    with torch.no_grad():
        logits_without_joint = model(
            optical,
            sar,
            joint=None,
            depth_group=depth,
            output_size=(2, 2),
        )
        logits_with_joint = model(
            optical,
            sar,
            joint=random_joint,
            depth_group=depth,
            output_size=(2, 2),
        )
    assert torch.equal(logits_without_joint, logits_with_joint)
