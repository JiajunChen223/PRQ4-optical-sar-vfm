from __future__ import annotations

from torch import nn

from geotoken3path.execution.contracts import BackboneFeatureContract
from geotoken3path.execution.croma_plan import compile_croma_execution_plan


class _TinyEncoder(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.linear_input = nn.Linear(4, 4)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [nn.ModuleList([nn.Identity(), nn.Identity()]) for _ in range(depth)]
        )
        self.transformer.norm_out = nn.Identity()


class _TinyCroma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s1_encoder = _TinyEncoder(6)
        self.s2_encoder = _TinyEncoder(12)
        self.GAP_FFN_s1 = nn.Identity()
        self.GAP_FFN_s2 = nn.Identity()
        self.cross_encoder = nn.Identity()


def _model_cfg() -> dict[str, object]:
    return {
        "stages": ["mid", "late"],
        "depth_taps": {
            "stage": {
                "optical": {
                    "mid": "s2_encoder.transformer.layers.2.1",
                    "late": "s2_encoder.transformer.layers.5.1",
                },
                "sar": {
                    "mid": "s1_encoder.transformer.layers.2.1",
                    "late": "s1_encoder.transformer.layers.5.1",
                },
            },
            "sar_depth_group": {
                "mid": [
                    "s1_encoder.transformer.layers.0.1",
                    "s1_encoder.transformer.layers.1.1",
                    "s1_encoder.transformer.layers.2.1",
                    "s1_encoder.transformer.layers.3.1",
                ],
                "late": [
                    "s1_encoder.transformer.layers.2.1",
                    "s1_encoder.transformer.layers.3.1",
                    "s1_encoder.transformer.layers.4.1",
                    "s1_encoder.transformer.layers.5.1",
                ],
            },
        },
    }


def test_current_receiver_contract_compiles_to_expected_minimal_prefix() -> None:
    contract = BackboneFeatureContract(
        optical_stages=("mid", "late"),
        sar_stages=("mid", "late"),
        sar_depth_group_stages=("mid", "late"),
        native_joint=False,
    )
    plan = compile_croma_execution_plan(
        model_cfg=_model_cfg(), receiver_contract=contract, audited_backbone=_TinyCroma()
    )
    assert plan.s1_last_layer == 5
    assert plan.s2_last_layer == 5
    assert not plan.require_joint_encoder
    assert not plan.require_s1_final_norm
    assert not plan.require_s2_final_norm
    assert "s2_encoder.transformer.layers.6" in plan.eliminated_nodes
    assert "cross_encoder" in plan.eliminated_nodes


def test_native_joint_forces_full_modality_encoders() -> None:
    contract = BackboneFeatureContract(
        optical_stages=("mid", "late"),
        sar_stages=("mid", "late"),
        sar_depth_group_stages=("mid", "late"),
        native_joint=True,
    )
    plan = compile_croma_execution_plan(
        model_cfg=_model_cfg(), receiver_contract=contract, audited_backbone=_TinyCroma()
    )
    assert plan.s1_last_layer == 5
    assert plan.s2_last_layer == 11
    assert plan.require_joint_encoder
    assert plan.require_s1_final_norm
    assert plan.require_s2_final_norm
