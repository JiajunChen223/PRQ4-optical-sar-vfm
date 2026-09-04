"""Single model factory used by all baseline and candidate entry points."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from torch import nn

from geotoken3path.execution import (
    BackboneFeatureContract,
    InterfaceCertifiedCromaExecutor,
    compile_croma_execution_plan,
    install_ice_exact_forward,
)

from .croma_bridge import (
    CromaBackboneBridge,
    CromaDepthTapAdapter,
    CromaGeoTokenSegmentation,
    freeze_backbone_for_peft,
)
from .fusion import OpticalSarTokenModel


def build_model(config: Mapping[str, Any] | None = None, *, mechanism_set: str | None = None) -> OpticalSarTokenModel:
    """Build a model from a small mapping without touching data or devices."""

    config = config or {}
    model_cfg = config.get("model", config)
    if not isinstance(model_cfg, Mapping):
        raise TypeError("model configuration must be a mapping")
    selected = mechanism_set or str(model_cfg.get("mechanism_set", "always_fuse"))
    local_window_tokens = int(model_cfg.get("local_window_tokens", 49))
    local_window_size = math.isqrt(local_window_tokens)
    if local_window_size * local_window_size != local_window_tokens:
        raise ValueError("local_window_tokens must be an odd square")
    return OpticalSarTokenModel(
        dim=int(model_cfg.get("token_dim", 32)),
        num_classes=int(model_cfg.get("num_classes", 19)),
        active_budget=float(model_cfg.get("active_budget", 0.5)),
        mechanism_set=selected,
        local_window_size=local_window_size,
        stages=tuple(model_cfg.get("stages", ("mid", "late"))),
        allow_synthetic_depth_group_fallback=bool(
            model_cfg.get("allow_synthetic_depth_group_fallback", False)
        ),
    )


def build_vfm_segmentation_model(
    config: Mapping[str, Any],
    *,
    audited_croma_backbone: nn.Module,
    backbone_execution: str = "full",
) -> CromaGeoTokenSegmentation:
    """Build the formal raw-image path around an injected audited backbone.

    ``backbone_execution`` is orthogonal to ``mechanism_set``.  ``full`` keeps
    the verified baseline behavior.  ``ice_exact`` changes only which audited
    CROMA nodes execute; it is currently restricted to the verified
    ``always_fuse`` receiver and registers no new model state.
    """

    execution_mode = str(backbone_execution).strip().casefold()
    if execution_mode not in {"full", "ice_exact"}:
        raise ValueError("backbone_execution must be full or ice_exact")

    token_model = build_model(config)
    # The raw-image VFM lane is always fail-closed: its depth groups must come
    # from the audited CROMA tap adapter, never from the token-only smoke path.
    token_model.allow_synthetic_depth_group_fallback = False
    model_cfg = config.get("model", config)
    if not isinstance(model_cfg, Mapping):
        raise TypeError("formal model configuration must be a mapping")
    depth_taps = model_cfg.get("depth_taps")
    if not isinstance(depth_taps, Mapping):
        raise ValueError("formal CROMA route requires explicit depth_taps")
    stage_taps = depth_taps.get("stage")
    depth_group_taps = depth_taps.get("sar_depth_group")
    if not isinstance(stage_taps, Mapping) or not isinstance(depth_group_taps, Mapping):
        raise ValueError("depth_taps must define stage and sar_depth_group mappings")

    ice_plan = None
    if execution_mode == "ice_exact":
        if token_model.mechanism_set != "always_fuse":
            raise ValueError(
                "R21 ICE exact is certified only for the verified always_fuse receiver"
            )
        stages = tuple(token_model.stages)
        receiver_contract = BackboneFeatureContract(
            optical_stages=stages,
            sar_stages=stages,
            sar_depth_group_stages=stages,
            native_joint=False,
            global_optical=False,
            global_sar=False,
        )
        ice_plan = compile_croma_execution_plan(
            model_cfg=model_cfg,
            receiver_contract=receiver_contract,
            audited_backbone=audited_croma_backbone,
        )
        executor = InterfaceCertifiedCromaExecutor(ice_plan)
        install_ice_exact_forward(audited_croma_backbone, executor)

    tapped_backbone = CromaDepthTapAdapter(
        audited_croma_backbone,
        stages=tuple(model_cfg["stages"]),
        dim=int(model_cfg["token_dim"]),
        stage_taps=stage_taps,
        depth_group_taps=depth_group_taps,
    )
    bridge = CromaBackboneBridge(
        tapped_backbone,
        stages=tuple(model_cfg["stages"]),
        dim=int(model_cfg["token_dim"]),
    )
    model = CromaGeoTokenSegmentation(bridge, token_model)
    # Plain metadata only: neither field is a Module/Parameter/Buffer, so the
    # checkpoint state surface remains identical between full and ICE exact.
    model._backbone_execution = execution_mode
    model._ice_execution_plan = ice_plan

    trainability = config.get("trainability", model_cfg.get("trainability", {}))
    if not isinstance(trainability, Mapping):
        raise ValueError("formal trainability configuration must be a mapping")
    policy = str(trainability.get("backbone_policy", "frozen"))
    freeze_backbone_for_peft(model, policy=policy)
    return model
