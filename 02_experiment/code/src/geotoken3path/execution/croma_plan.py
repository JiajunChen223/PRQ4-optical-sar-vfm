"""Compile downstream representation dependencies into an exact CROMA plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

from torch import nn

from .contracts import BackboneFeatureContract, CromaExecutionContractError


_TAP_RE = re.compile(
    r"^(s1_encoder|s2_encoder)\.transformer\.layers\.(\d+)\.1$"
)


@dataclass(frozen=True)
class CromaExecutionPlan:
    """Minimum audited CROMA subgraph required by one receiver contract."""

    required_taps: tuple[str, ...]
    s1_last_layer: int | None
    s2_last_layer: int | None
    require_s1_final_norm: bool
    require_s2_final_norm: bool
    require_joint_encoder: bool
    require_s1_gap: bool
    require_s2_gap: bool
    eliminated_nodes: tuple[str, ...]
    plan_sha256: str
    ablation_tier: str = "exact"

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _stage_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CromaExecutionContractError(f"{name} must be a mapping")
    return value


def _tap_for_stage(
    stage_taps: Mapping[str, Any], *, modality: str, stage: str
) -> str:
    modality_map = _stage_mapping(stage_taps.get(modality), f"stage_taps.{modality}")
    path = modality_map.get(stage)
    if not isinstance(path, str) or not path.strip():
        raise CromaExecutionContractError(
            f"stage_taps.{modality}.{stage} is missing or invalid"
        )
    return path.strip()


def _parse_layer_tap(path: str) -> tuple[str, int]:
    match = _TAP_RE.fullmatch(path)
    if match is None:
        raise CromaExecutionContractError(
            "ICE exact currently supports only audited FFN taps of the form "
            "s1_encoder|s2_encoder.transformer.layers.<index>.1; "
            f"received {path!r}"
        )
    return match.group(1), int(match.group(2))


def _encoder_depth(backbone: nn.Module, encoder_name: str) -> int:
    encoder = getattr(backbone, encoder_name, None)
    transformer = getattr(encoder, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) < 1:
        raise CromaExecutionContractError(
            f"audited backbone does not expose {encoder_name}.transformer.layers"
        )
    return len(layers)


def _validate_stage_contract(
    contract: BackboneFeatureContract, stages: tuple[str, ...]
) -> None:
    declared = set(stages)
    for name, values in (
        ("optical_stages", contract.optical_stages),
        ("sar_stages", contract.sar_stages),
        ("sar_depth_group_stages", contract.sar_depth_group_stages),
    ):
        unknown = sorted(set(values) - declared)
        if unknown:
            raise CromaExecutionContractError(
                f"receiver {name} contains undeclared stages: {unknown}"
            )


def compile_croma_execution_plan(
    *,
    model_cfg: Mapping[str, Any],
    receiver_contract: BackboneFeatureContract,
    audited_backbone: nn.Module,
    ablation: str = "exact",
) -> CromaExecutionPlan:
    """Derive the minimum exact CROMA prefix from declared receiver taps.

    The current R21 implementation is intentionally conservative.  It only
    compiles the exact FFN hook paths used by the verified baseline.  Native
    joint consumption forces full S1/S2 encoders because the official joint
    encoder consumes their final normalized outputs.

    ``ablation`` selects the execution-cost attribution tier for the paper's
    cost-attribution study.  It does not change the receiver contract; it
    changes only which nodes are retained/eliminated so that each tier's
    marginal cost can be measured.  Tiers (all measured with the same
    checkpoint, all prediction-equal for the always-fuse receiver because
    GAP/joint/optical-suffix are not consumed downstream):
      - "full":      everything (both encoders full depth + norms + GAP + joint);
      - "no_gap":    full depth + joint, but GAP heads eliminated;
      - "no_joint":  full tap-derived depth (no suffix), GAP heads eliminated,
                     joint/cross eliminated  (== current exact for late=5);
      - "exact":     minimum tap-derived plan (== current default behaviour).
    The receiver contract flags remain the source of truth for which outputs
    the caller declares; ablation only widens the executed graph for cost
    attribution and never narrows below the contract.
    """

    if not isinstance(model_cfg, Mapping):
        raise CromaExecutionContractError("model_cfg must be a mapping")
    stages_raw = model_cfg.get("stages")
    if not isinstance(stages_raw, (list, tuple)) or not stages_raw:
        raise CromaExecutionContractError("model_cfg.stages must be a nonempty sequence")
    stages = tuple(str(stage) for stage in stages_raw)
    _validate_stage_contract(receiver_contract, stages)

    depth_taps = _stage_mapping(model_cfg.get("depth_taps"), "depth_taps")
    stage_taps = _stage_mapping(depth_taps.get("stage"), "depth_taps.stage")
    depth_groups = _stage_mapping(
        depth_taps.get("sar_depth_group"), "depth_taps.sar_depth_group"
    )

    required: list[str] = []
    for stage in receiver_contract.optical_stages:
        required.append(_tap_for_stage(stage_taps, modality="optical", stage=stage))
    for stage in receiver_contract.sar_stages:
        required.append(_tap_for_stage(stage_taps, modality="sar", stage=stage))
    for stage in receiver_contract.sar_depth_group_stages:
        paths = depth_groups.get(stage)
        if not isinstance(paths, (list, tuple)) or not paths:
            raise CromaExecutionContractError(
                f"depth_taps.sar_depth_group.{stage} is missing or invalid"
            )
        for path in paths:
            if not isinstance(path, str) or not path.strip():
                raise CromaExecutionContractError(
                    f"depth_taps.sar_depth_group.{stage} contains an invalid path"
                )
            required.append(path.strip())

    required = list(dict.fromkeys(required))
    s1_indices: list[int] = []
    s2_indices: list[int] = []
    for path in required:
        encoder_name, index = _parse_layer_tap(path)
        if encoder_name == "s1_encoder":
            s1_indices.append(index)
        else:
            s2_indices.append(index)

    s1_depth = _encoder_depth(audited_backbone, "s1_encoder")
    s2_depth = _encoder_depth(audited_backbone, "s2_encoder")
    if s1_indices and max(s1_indices) >= s1_depth:
        raise CromaExecutionContractError("required S1 tap exceeds audited encoder depth")
    if s2_indices and max(s2_indices) >= s2_depth:
        raise CromaExecutionContractError("required S2 tap exceeds audited encoder depth")

    require_joint = receiver_contract.native_joint
    require_s1_gap = receiver_contract.global_sar
    require_s2_gap = receiver_contract.global_optical

    tier = str(ablation).strip().casefold()
    if tier not in {"full", "no_gap", "no_joint", "exact"}:
        raise CromaExecutionContractError(f"unknown ablation tier: {ablation}")
    if tier == "full":
        # Widened attribution graph: everything executed. Never narrows below
        # the receiver contract (the contract may itself request joint/GAP).
        require_joint = True
        require_s1_gap = True
        require_s2_gap = True
    elif tier == "no_gap":
        require_joint = True
        require_s1_gap = False
        require_s2_gap = False
    elif tier == "no_joint":
        require_joint = False
        require_s1_gap = False
        require_s2_gap = False
    # tier == "exact": keep contract-derived flags (current behaviour).

    if require_joint:
        s1_last = s1_depth - 1
        s2_last = s2_depth - 1
        require_s1_norm = True
        require_s2_norm = True
    else:
        s1_last = max(s1_indices) if s1_indices else None
        s2_last = max(s2_indices) if s2_indices else None
        require_s1_norm = bool(require_s1_gap)
        require_s2_norm = bool(require_s2_gap)

    if require_s1_gap:
        s1_last = s1_depth - 1
        require_s1_norm = True
    if require_s2_gap:
        s2_last = s2_depth - 1
        require_s2_norm = True

    eliminated: list[str] = []
    if s1_last is not None:
        eliminated.extend(
            f"s1_encoder.transformer.layers.{index}"
            for index in range(s1_last + 1, s1_depth)
        )
    if s2_last is not None:
        eliminated.extend(
            f"s2_encoder.transformer.layers.{index}"
            for index in range(s2_last + 1, s2_depth)
        )
    if not require_s1_norm:
        eliminated.append("s1_encoder.transformer.norm_out")
    if not require_s2_norm:
        eliminated.append("s2_encoder.transformer.norm_out")
    if not require_s1_gap:
        eliminated.append("GAP_FFN_s1")
    if not require_s2_gap:
        eliminated.append("GAP_FFN_s2")
    if not require_joint:
        eliminated.append("cross_encoder")
        eliminated.append("joint_GAP")

    core = {
        "ablation_tier": tier,
        "required_taps": tuple(sorted(required)),
        "s1_last_layer": s1_last,
        "s2_last_layer": s2_last,
        "require_s1_final_norm": require_s1_norm,
        "require_s2_final_norm": require_s2_norm,
        "require_joint_encoder": require_joint,
        "require_s1_gap": require_s1_gap,
        "require_s2_gap": require_s2_gap,
        "eliminated_nodes": tuple(eliminated),
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return CromaExecutionPlan(**core, plan_sha256=digest)
