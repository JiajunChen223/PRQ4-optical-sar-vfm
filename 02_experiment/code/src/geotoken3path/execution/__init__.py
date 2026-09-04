"""Execution backends for audited foundation-model inference/training graphs."""

from .certification import (
    TensorEquivalence,
    compare_gradients,
    compare_tensors,
    named_trainable_gradients,
)
from .contracts import BackboneFeatureContract, CromaExecutionContractError
from .croma_executor import (
    InterfaceCertifiedCromaExecutor,
    install_ice_exact_forward,
)
from .croma_plan import CromaExecutionPlan, compile_croma_execution_plan
from .profiling import LatencySummary, profile_cuda_callable

__all__ = [
    "BackboneFeatureContract",
    "CromaExecutionContractError",
    "CromaExecutionPlan",
    "InterfaceCertifiedCromaExecutor",
    "LatencySummary",
    "TensorEquivalence",
    "compare_gradients",
    "compare_tensors",
    "compile_croma_execution_plan",
    "install_ice_exact_forward",
    "named_trainable_gradients",
    "profile_cuda_callable",
]
