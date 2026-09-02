"""Model and fusion components for the verified baseline and D1/D2/D3 diagnostics.

Two-zone cleanup 2026-09-02: rejected route adapters and mechanism exports were
removed; archives live in 20_HISTORY/02_legacy_code_pkgs/rejected_mechanisms_20260902/.
"""

from .croma_bridge import (
    CromaBackboneBridge,
    CromaDepthTapAdapter,
    CromaFeatureContractError,
    CromaGeoTokenSegmentation,
)
from .factory import build_model, build_vfm_segmentation_model
from .fusion import GeoToken3PathFusion, OpticalSarTokenModel
from .initialization import apply_audited_state_dict, validate_pretrained_audit

__all__ = [
    "build_model",
    "build_vfm_segmentation_model",
    "CromaBackboneBridge",
    "CromaDepthTapAdapter",
    "CromaFeatureContractError",
    "CromaGeoTokenSegmentation",
    "GeoToken3PathFusion",
    "OpticalSarTokenModel",
    "apply_audited_state_dict",
    "validate_pretrained_audit",
]