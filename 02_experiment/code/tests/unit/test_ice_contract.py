from __future__ import annotations

import pytest

from geotoken3path.execution.contracts import (
    BackboneFeatureContract,
    CromaExecutionContractError,
)


def test_feature_contract_stage_union_is_stable_and_unique() -> None:
    contract = BackboneFeatureContract(
        optical_stages=("mid", "late"),
        sar_stages=("mid", "late"),
        sar_depth_group_stages=("late",),
        native_joint=False,
    )
    assert contract.stage_union == ("mid", "late")


def test_feature_contract_rejects_blank_stage() -> None:
    with pytest.raises(CromaExecutionContractError):
        BackboneFeatureContract(
            optical_stages=("mid", ""),
            sar_stages=("mid",),
            sar_depth_group_stages=("mid",),
            native_joint=False,
        )
