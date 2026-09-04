from __future__ import annotations

import copy

from geotoken3path.execution.croma_executor import InterfaceCertifiedCromaExecutor
from geotoken3path.execution.croma_plan import CromaExecutionPlan


def test_executor_owns_no_model_state() -> None:
    plan = CromaExecutionPlan(
        required_taps=(),
        s1_last_layer=None,
        s2_last_layer=None,
        require_s1_final_norm=False,
        require_s2_final_norm=False,
        require_joint_encoder=False,
        require_s1_gap=False,
        require_s2_gap=False,
        eliminated_nodes=(),
        plan_sha256="0" * 64,
    )
    executor = InterfaceCertifiedCromaExecutor(plan)
    assert not hasattr(executor, "state_dict")
    assert copy.deepcopy(executor.plan) == plan
