from __future__ import annotations

import pytest
from torch import nn

from geotoken3path.execution.contracts import (
    BackboneFeatureContract,
    CromaExecutionContractError,
)
from geotoken3path.execution.croma_plan import compile_croma_execution_plan


class _Encoder(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.linear_input = nn.Identity()
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [nn.ModuleList([nn.Identity(), nn.Identity()]) for _ in range(depth)]
        )


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s1_encoder = _Encoder(6)
        self.s2_encoder = _Encoder(12)


def test_plan_compiler_rejects_unknown_tap_semantics() -> None:
    cfg = {
        "stages": ["late"],
        "depth_taps": {
            "stage": {
                "optical": {"late": "s2_encoder.transformer.layers.5.0"},
                "sar": {"late": "s1_encoder.transformer.layers.5.1"},
            },
            "sar_depth_group": {
                "late": [
                    "s1_encoder.transformer.layers.2.1",
                    "s1_encoder.transformer.layers.3.1",
                    "s1_encoder.transformer.layers.4.1",
                    "s1_encoder.transformer.layers.5.1",
                ]
            },
        },
    }
    contract = BackboneFeatureContract(
        optical_stages=("late",),
        sar_stages=("late",),
        sar_depth_group_stages=("late",),
        native_joint=False,
    )
    with pytest.raises(CromaExecutionContractError, match="supports only audited FFN taps"):
        compile_croma_execution_plan(
            model_cfg=cfg, receiver_contract=contract, audited_backbone=_Backbone()
        )
