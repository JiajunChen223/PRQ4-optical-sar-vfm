"""Immutable run-manifest construction shared by all mechanism sets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from typing import Any

from .test_seal import assert_test_access_allowed


_SHA256_LENGTH = 64
_APPROVED_MECHANISMS = {
    "always_fuse",
    "r2_depth_group_inject",
    "r1_low_energy_channel_gain",
}

_EXECUTION_SCALES = {
    "smoke",
    "baseline",
    "screening",
    "strengthening",
    "confirmation",
    "acceptance",
    "extension",
    "final_test",
}

def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _sha256(value: object, name: str) -> str:
    text = _nonempty_text(value, name).casefold()
    if len(text) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a SHA256 digest")
    return text


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum or (maximum is not None and number >= maximum):
        raise ValueError(f"{name} is outside the allowed range")
    if strictly_positive and number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _json_clone(value: object, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON data") from exc


def _validated_resolved_snapshot(resolved: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(resolved, Mapping):
        raise ValueError("resolved configuration must be a mapping")
    model = _mapping(resolved.get("model"), "resolved.model")
    runtime = _mapping(resolved.get("runtime"), "resolved.runtime")
    input_spec = _mapping(resolved.get("input"), "resolved.input")
    objective_spec = resolved.get("objective")
    if objective_spec is not None:
        objective_spec = _mapping(objective_spec, "resolved.objective")
        objective_id = _nonempty_text(objective_spec.get("id"), "resolved.objective.id").casefold()
        if objective_id not in {"pixel_ce", "macro_ce", "ce_lovasz", "macro_ce_lovasz"}:
            raise ValueError("resolved.objective.id is not approved")
        for field in ("ce_weight", "lovasz_weight"):
            if objective_spec.get(field) != 1.0:
                raise ValueError(f"resolved.objective.{field} must be 1.0")
        if objective_spec.get("ignore_index") != 255:
            raise ValueError("resolved.objective.ignore_index must be 255")
        policy = _mapping(objective_spec.get("policy"), "resolved.objective.policy")
        if list(policy.get("allowed_ids", [])) != ["pixel_ce", "macro_ce", "ce_lovasz", "macro_ce_lovasz"]:
            raise ValueError("resolved.objective.policy allowed_ids are not frozen")
        if policy.get("macro_present_class_only") is not True or policy.get("ce_weight") != 1.0 or policy.get("lovasz_weight") != 1.0:
            raise ValueError("resolved.objective.policy is not the frozen V12 contract")
    storage = _mapping(resolved.get("storage"), "resolved.storage")
    trainability = _mapping(resolved.get("trainability"), "resolved.trainability")
    optimizer = _mapping(runtime.get("optimizer"), "resolved.runtime.optimizer")
    scheduler = _mapping(runtime.get("scheduler"), "resolved.runtime.scheduler")

    mechanism = _nonempty_text(model.get("mechanism_set"), "resolved.model.mechanism_set")
    if mechanism not in _APPROVED_MECHANISMS:
        raise ValueError("resolved.model.mechanism_set is not approved")
    # Two-zone cleanup 2026-09-02: rejected route contract validations
    # (tasr/dtsf/rift/mcsl/mcof/jack/ctsp/successor-edge) were removed.
    stages = model.get("stages")
    if not isinstance(stages, list) or not stages or any(not isinstance(stage, str) or not stage for stage in stages):
        raise ValueError("resolved.model.stages must be a nonempty string list")
    depth_taps = _mapping(model.get("depth_taps"), "resolved.model.depth_taps")
    depth_taps = _mapping(model.get("depth_taps"), "resolved.model.depth_taps")
    stage_taps = _mapping(depth_taps.get("stage"), "resolved.model.depth_taps.stage")
    sar_depth_group = _mapping(
        depth_taps.get("sar_depth_group"), "resolved.model.depth_taps.sar_depth_group"
    )
    if set(stage_taps) != {"optical", "sar"} or set(sar_depth_group) != set(stages):
        raise ValueError("resolved.model.depth_taps stage keys do not match the route")
    for modality in ("optical", "sar"):
        modality_taps = _mapping(stage_taps.get(modality), f"resolved.model.depth_taps.stage.{modality}")
        if set(modality_taps) != set(stages) or any(
            not isinstance(modality_taps[stage], str) or not modality_taps[stage].strip()
            for stage in stages
        ):
            raise ValueError(f"resolved.model.depth_taps.stage.{modality} is incomplete")
    for stage in stages:
        paths = sar_depth_group.get(stage)
        if not isinstance(paths, list) or len(paths) != 4 or any(
            not isinstance(path, str) or not path.strip() for path in paths
        ):
            raise ValueError(f"resolved.model.depth_taps.sar_depth_group.{stage} is invalid")

    if _positive_int(input_spec.get("optical_channels"), "resolved.input.optical_channels") != 12:
        raise ValueError("resolved.input.optical_channels must be 12")
    if _positive_int(input_spec.get("sar_channels"), "resolved.input.sar_channels") != 2:
        raise ValueError("resolved.input.sar_channels must be 2")
    if _positive_int(input_spec.get("patch_size"), "resolved.input.patch_size") != 120:
        raise ValueError("resolved.input.patch_size must be 120")
    hard_stop = _positive_int(storage.get("hard_stop_gb"), "resolved.storage.hard_stop_gb")
    total_ceiling = _positive_int(storage.get("total_ceiling_gb"), "resolved.storage.total_ceiling_gb")
    if hard_stop >= total_ceiling:
        raise ValueError("resolved storage hard stop must be below the total ceiling")

    micro_batch = _positive_int(runtime.get("micro_batch"), "resolved.runtime.micro_batch")
    effective_batch = _positive_int(runtime.get("effective_batch"), "resolved.runtime.effective_batch")
    accumulation = _positive_int(
        runtime.get("gradient_accumulation"), "resolved.runtime.gradient_accumulation",
    )
    if effective_batch != micro_batch * accumulation:
        raise ValueError("resolved effective batch does not match accumulation")
    runtime_seed = runtime.get("seed")
    if isinstance(runtime_seed, bool) or not isinstance(runtime_seed, int) or runtime_seed < 0:
        raise ValueError("resolved.runtime.seed must be a nonnegative integer")
    _nonempty_text(runtime.get("test_seal_status"), "resolved.runtime.test_seal_status")
    runtime_objective = runtime.get("objective_name")
    objective_name = _nonempty_text(
        runtime_objective or (objective_spec or {}).get("id", "pixel_ce"),
        "resolved.objective.id",
    ).casefold()
    if objective_name not in {"pixel_ce", "macro_ce", "ce_lovasz", "macro_ce_lovasz"}:
        raise ValueError("resolved.runtime.objective_name is not approved")
    code_sync_ref = resolved.get("code_sync_manifest_ref")
    code_sync_hash = resolved.get("code_sync_manifest_sha256")
    if code_sync_ref is not None:
        _nonempty_text(code_sync_ref, "resolved.code_sync_manifest_ref")
        _sha256(code_sync_hash, "resolved.code_sync_manifest_sha256")

    if _nonempty_text(optimizer.get("name"), "resolved.runtime.optimizer.name").casefold() != "adamw":
        raise ValueError("resolved optimizer must be adamw")
    _finite_number(
        optimizer.get("learning_rate"), "resolved.runtime.optimizer.learning_rate",
        minimum=0.0, strictly_positive=True,
    )
    _finite_number(
        optimizer.get("weight_decay"), "resolved.runtime.optimizer.weight_decay", minimum=0.0,
    )
    betas = optimizer.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("resolved optimizer betas must contain exactly two values")
    for index, beta in enumerate(betas):
        _finite_number(
            beta, f"resolved.runtime.optimizer.betas[{index}]", minimum=0.0, maximum=1.0,
        )
    if _nonempty_text(scheduler.get("name"), "resolved.runtime.scheduler.name").casefold() != "cosine_with_warmup":
        raise ValueError("resolved scheduler must be cosine_with_warmup")
    _finite_number(
        scheduler.get("warmup_fraction"), "resolved.runtime.scheduler.warmup_fraction",
        minimum=0.0, maximum=1.0,
    )
    _finite_number(
        runtime.get("gradient_clip_norm"), "resolved.runtime.gradient_clip_norm",
        minimum=0.0, strictly_positive=True,
    )
    for key in ("trunk", "router", "adapters", "decoder"):
        _nonempty_text(trainability.get(key), f"resolved.trainability.{key}")

    _nonempty_text(resolved.get("route_id"), "resolved.route_id")
    _nonempty_text(resolved.get("candidate_id"), "resolved.candidate_id")
    _nonempty_text(resolved.get("dataset_id"), "resolved.dataset_id")
    _sha256(resolved.get("matched_common_protocol_sha256"), "resolved.matched_common_protocol_sha256")
    _nonempty_text(resolved.get("initialization_ref"), "resolved.initialization_ref")
    return _json_clone(resolved, "resolved configuration")


def verify_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Raise when a returned run manifest was changed after hashing."""

    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be a mapping")
    detached = _json_clone(manifest, "run manifest")
    stored = _sha256(detached.pop("run_contract_sha256", None), "run_contract_sha256")
    canonical = json.dumps(detached, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != stored:
        raise ValueError("run manifest hash mismatch")


def build_run_manifest(
    resolved: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    execution_scale: str,
    candidate_direction_id: str | None = None,
    data_manifest_ref: str | None = None,
    pretrained_audit_ref: str | None = None,
    telemetry_ref: str | None = None,
    telemetry_sha256: str | None = None,
) -> dict[str, Any]:
    snapshot = _validated_resolved_snapshot(resolved)
    model = snapshot["model"]
    mechanism = model["mechanism_set"]
    runtime = snapshot["runtime"]
    objective_spec = snapshot.get("objective") or {}
    objective_name = str(
        runtime.get("objective_name") or objective_spec.get("id", "pixel_ce")
    ).strip().casefold()
    seal = _nonempty_text(runtime.get("test_seal_status"), "resolved.runtime.test_seal_status")
    if execution_scale not in _EXECUTION_SCALES:
        raise ValueError(f"unsupported execution_scale: {execution_scale}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    assert_test_access_allowed(
        {"execution_scale": execution_scale, "test_seal_status": seal},
        split,
    )
    if (telemetry_ref is None) != (telemetry_sha256 is None):
        raise ValueError("telemetry_ref and telemetry_sha256 must be supplied together")
    manifest = {
        "schema_version": "geotoken3path.run.v1",
        "route_id": snapshot["route_id"],
        "candidate_id": snapshot["candidate_id"],
        "dataset_id": snapshot["dataset_id"],
        "mechanism_set": model["mechanism_set"],
        "objective": objective_spec,
        "matched_common_protocol_sha256": snapshot["matched_common_protocol_sha256"],
        # Bind the manifest to the audit actually used by the cloud run.  The
        # model YAML remains a default for local/synthetic construction, but a
        # formal cloud invocation must not silently retain a legacy audit ref.
        "initialization_ref": pretrained_audit_ref or snapshot["initialization_ref"],
        "input": snapshot["input"],
        "storage": snapshot["storage"],
        "trainability": snapshot["trainability"],
        "optimizer": runtime["optimizer"],
        "scheduler": runtime["scheduler"],
        "objective_name": objective_name,
        "gradient_clip_norm": runtime["gradient_clip_norm"],
        "effective_batch": runtime["effective_batch"],
        "data_loader": runtime.get("data_loader"),
        "augmentation": runtime.get("augmentation"),
        "early_stopping": runtime.get("early_stopping"),
        "max_formal_epochs": runtime.get("max_formal_epochs"),
        "rapid_horizon_epochs": runtime.get("rapid_horizon_epochs"),
        "data_manifest_ref": data_manifest_ref or "02_experiment/manifests/cloud_dataset_manifest.json",
        "pretrained_audit_ref": pretrained_audit_ref or snapshot["initialization_ref"],
        # Formal cloud commands export the exact source manifest used for the
        # guarded package.  Keep the legacy default for older local fixtures,
        # but never silently bind a current cloud run to that stale snapshot.
        "code_sync_manifest_ref": str(
            resolved.get("code_sync_manifest_ref")
            or os.environ.get("GEOTOKEN3PATH_CODE_SYNC_MANIFEST_REF")
            or "02_experiment/code/manifests/clean_sync_manifest.json"
        ),
        "code_sync_manifest_sha256": resolved.get("code_sync_manifest_sha256"),
        "seed": int(seed),
        "split": split,
        "execution_scale": execution_scale,
        "test_seal_status": seal,
        "test_accessed": split.strip().casefold() == "test",
    }
    if telemetry_ref is not None and telemetry_sha256 is not None:
        manifest["telemetry_ref"] = _nonempty_text(telemetry_ref, "telemetry_ref")
        manifest["telemetry_sha256"] = _sha256(telemetry_sha256, "telemetry_sha256")
    if candidate_direction_id is not None:
        direction = str(candidate_direction_id).strip()
        # Two-zone cleanup 2026-09-02: only the baseline direction id remains approved.
        if direction != "BASELINE":
            raise ValueError("candidate_direction_id is not an approved formal direction")
        manifest["candidate_direction_id"] = direction
    # Detach every nested object before hashing and returning. Mutating the
    # caller's resolved config cannot mutate this manifest behind its hash.
    manifest = _json_clone(manifest, "run manifest")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["run_contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    verify_run_manifest(manifest)
    return manifest
