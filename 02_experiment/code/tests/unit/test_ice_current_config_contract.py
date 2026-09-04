from __future__ import annotations

from geotoken3path.execution.contracts import BackboneFeatureContract


def test_verified_baseline_receiver_contract_does_not_consume_native_joint() -> None:
    contract = BackboneFeatureContract(
        optical_stages=("mid", "late"),
        sar_stages=("mid", "late"),
        sar_depth_group_stages=("mid", "late"),
        native_joint=False,
    )
    assert contract.native_joint is False
