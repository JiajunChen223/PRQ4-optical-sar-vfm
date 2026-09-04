"""Execution backends for audited foundation-model inference/training graphs."""

from .contracts import BackboneFeatureContract, CromaExecutionContractError
from .croma_plan import CromaExecutionPlan, compile_croma_execution_plan
from .croma_executor import InterfaceCertifiedCromaExecutor

__all__ = [
    "BackboneFeatureContract",
    "CromaExecutionContractError",
    "CromaExecutionPlan",
    "InterfaceCertifiedCromaExecutor",
    "compile_croma_execution_plan",
]
