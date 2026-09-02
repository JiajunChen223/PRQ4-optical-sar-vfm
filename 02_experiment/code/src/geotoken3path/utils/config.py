"""Fail-closed resolver for the approved ResearchPilot configuration set."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from geotoken3path.data.contracts import (
    DYNAMIC_NORMALIZATION_SCHEME,
    OPTICAL_BANDS,
    SAR_CHANNELS,
    SEN12TS_DATASET_ID,
    SEN12TS_RAW_SAR_CHANNELS,
    SEN12TS_S1_SOURCE_INDICES,
    SEN12TS_S2_SOURCE_INDICES,
)


class ConfigContractError(ValueError):
    """Raised when an approved configuration is incomplete or ambiguous."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigContractError(f"{name} must be a mapping")
    return value


def _require(mapping: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise ConfigContractError(f"{name} missing fields: {', '.join(missing)}")


def _load(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


def _finite_number(value: object, name: str, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum or (maximum is not None and number >= maximum):
        upper = "" if maximum is None else f" and < {maximum}"
        raise ConfigContractError(f"{name} must be finite, >= {minimum}{upper}")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigContractError(f"{name} must be a positive integer")
    return value


def _validate_active_benchmark(benchmark: Mapping[str, Any]) -> None:
    """Fail closed on the approved SEN12TS input/label/split contract."""

    if benchmark["dataset_id"] != SEN12TS_DATASET_ID:
        raise ConfigContractError("the active benchmark must be the approved SEN12TS successor")
    modalities = _mapping(benchmark["modalities"], "benchmark.modalities")
    optical = _mapping(modalities["optical"], "benchmark.modalities.optical")
    sar = _mapping(modalities["sar"], "benchmark.modalities.sar")
    if optical["channels"] != 12 or optical["raw_channels"] != 14 or list(optical["source_indices"]) != SEN12TS_S2_SOURCE_INDICES:
        raise ConfigContractError("SEN12TS optical selector must retain S2 indices 0..11 from 14 channels")
    if sar["channels"] != 2 or sar["raw_channels"] != 19 or list(sar["source_indices"]) != SEN12TS_S1_SOURCE_INDICES:
        raise ConfigContractError("SEN12TS SAR selector must reorder raw VH,VV with indices [1,0]")
    if list(sar["raw_channel_order"]) != SEN12TS_RAW_SAR_CHANNELS or list(sar["canonical_channel_order"]) != SAR_CHANNELS:
        raise ConfigContractError("SEN12TS SAR raw/canonical channel order is inconsistent")
    if list(optical["band_order"]) != OPTICAL_BANDS or list(sar["band_order"]) != SAR_CHANNELS:
        raise ConfigContractError("SEN12TS canonical band order is inconsistent")
    if benchmark["labels"] != 11 or benchmark["label_contract"]["ignore_index"] != 255:
        raise ConfigContractError("SEN12TS must expose 11 classes and ignore_index 255")
    if list(benchmark["parent_shape"]) != [256, 256] or list(benchmark["derived_shape"]) != [120, 120]:
        raise ConfigContractError("SEN12TS parent/derived shapes are inconsistent")
    split = _mapping(benchmark["split_contract"], "benchmark.split_contract")
    if any(split.get(field) is not True for field in ("parent_first", "crop_after_split", "crop_leakage_rejected", "sealed_test")):
        raise ConfigContractError("SEN12TS split must be parent-first, crop-after-split and sealed")
    normalization = _mapping(benchmark["normalization"], "benchmark.normalization")
    if normalization.get("scheme") != DYNAMIC_NORMALIZATION_SCHEME or normalization.get("statistics_axes") != [0, 2, 3]:
        raise ConfigContractError("active benchmark must use CROMA dynamic statistics on axes (0,2,3)")
    if normalization.get("micro_batch") != 16 or normalization.get("distributed_statistics") != "per_rank_micro_batch":
        raise ConfigContractError("active benchmark must freeze micro_batch=16 and per-rank statistics")


def resolve_approved_config(
    code_root: Path,
    mechanism_set: str,
    *,
    execution_scale: str | None = None,
    model_config_name: str = "geotoken3path.yaml",
    route_config_name: str = "approved_route.yaml",
) -> dict[str, Any]:
    """Resolve the four approved YAML documents into one immutable mapping."""

    model = _load(code_root / "configs/model" / model_config_name)
    route = _load(code_root / "configs/experiment" / route_config_name)
    benchmark = _load(code_root / "configs/benchmarks/sen12ts_worldcover.yaml")
    runtime = _load(code_root / "configs/runtime/3090_plan.yaml")
    initialization_doc = _load(code_root / "configs/model/initialization.yaml")
    initialization = _mapping(initialization_doc.get("initialization"), "initialization")

    _require(model, {"route_id", "candidate_id", "backbone_family", "backbone", "decoder", "mechanism", "matched_controls", "input", "initialization_ref", "trainability"}, "model")
    _require(route, {"selected_route_id", "primary_core_candidate_id", "mechanism_sets", "test_seal_status"}, "route")
    objective_contract: Mapping[str, Any] | None = None
    _require(benchmark, {"dataset_id", "modalities", "patch_size", "labels", "official_split", "local_real_data_allowed", "cloud_data_root", "hard_storage_stop_gb", "total_project_ceiling_gb"}, "benchmark")
    _require(runtime, {"target_gpu", "local_gpu_probe_allowed", "cloud_preflight_required", "precision", "micro_batch", "effective_batch", "gradient_accumulation", "real_data_allowed", "real_weights_allowed", "execution_scale", "test_seal_status", "seed", "optimizer", "scheduler", "gradient_clip_norm", "max_formal_epochs", "early_stopping", "augmentation"}, "runtime")
    _validate_active_benchmark(benchmark)
    mode = initialization.get("mode")
    if mode not in {"random_init", "pretrained"}:
        raise ConfigContractError("initialization mode must be pretrained or random_init")
    if mode == "random_init":
        if initialization.get("checkpoint_path") is not None or initialization.get("pretrained_eligible") is not False:
            raise ConfigContractError("random_init cannot declare a checkpoint or pretrained eligibility")
        if initialization.get("fallback_reason") != "pretrained_geography_overlap_confirmed":
            raise ConfigContractError("random-init fallback reason is not the approved leakage exception")
    else:
        cloud_prefix = chr(47) + "root" + chr(47) + "autodl-tmp" + chr(47)
        if not isinstance(initialization.get("checkpoint_path"), str) or not initialization.get("checkpoint_path").startswith(cloud_prefix) or initialization.get("pretrained_eligible") is not True:
            raise ConfigContractError("pretrained mode requires an audited cloud checkpoint")
    if initialization.get("target_test_data_used") is not False:
        raise ConfigContractError("initialization must not use target test data")
    if initialization.get("same_initialization_for_baseline_and_innovation") is not True:
        raise ConfigContractError("baseline and candidate must share initialization")
    if mode == "pretrained" and initialization.get("strict_load") is not True:
        raise ConfigContractError("pretrained initialization requires strict_load=true")
    if initialization.get("constructor_ref") != "geotoken3path.models.croma_random:PinnedSourceRandomCROMA":
        raise ConfigContractError("formal constructor must be the pinned-source adapter")
    constructor_kwargs = _mapping(initialization.get("constructor_kwargs"), "initialization.constructor_kwargs")
    source_path = PurePosixPath(str(constructor_kwargs.get("source_path", "")))
    source_parts = source_path.parts
    expected_tail = ("audits", "prq4-croma-loader-compat-20260822-r1", "use_croma.py")
    source_contract_ok = (
        source_path.is_absolute()
        and len(source_parts) == 6
        and source_parts[1:3] == ("root", "autodl-tmp")
        and source_parts[3:] == expected_tail
        and constructor_kwargs.get("source_sha256") == "a38567beed29eb08108a47cdc97fe98aec50fd4be0bd98a5266bcd18aafb7c5b"
        and constructor_kwargs.get("size") == "base"
        and constructor_kwargs.get("modality") == "both"
        and constructor_kwargs.get("image_resolution") == 120
        and set(constructor_kwargs) == {"source_path", "source_sha256", "size", "modality", "image_resolution"}
    )
    if not source_contract_ok:
        raise ConfigContractError("constructor source contract differs from the frozen audit")
    selected_execution_scale = str(execution_scale or runtime["execution_scale"])
    if selected_execution_scale not in {
        "smoke", "baseline", "screening", "strengthening", "confirmation",
        "acceptance", "extension", "final_test",
    }:
        raise ConfigContractError("unsupported execution_scale")

    if model["route_id"] != route["selected_route_id"] or model["candidate_id"] != route["primary_core_candidate_id"]:
        raise ConfigContractError("model and approved route identifiers do not match")
    if route["test_seal_status"] != "sealed" or runtime["test_seal_status"] != "sealed":
        raise ConfigContractError("test seal must remain sealed")
    if benchmark["local_real_data_allowed"] is not False or runtime["local_gpu_probe_allowed"] is not False:
        raise ConfigContractError("local data and local GPU probing must remain disabled")
    if runtime["real_data_allowed"] is not False or runtime["real_weights_allowed"] is not False:
        raise ConfigContractError("local runtime must remain synthetic-only")

    mechanism_sets = _mapping(route["mechanism_sets"], "mechanism_sets")
    _require(mechanism_sets, {"baseline", "candidate", "controls", "same_model_factory", "single_internal_mechanism_delta"}, "mechanism_sets")
    allowed = {str(mechanism_sets["baseline"]), str(mechanism_sets["candidate"]), *map(str, mechanism_sets["controls"])}
    if mechanism_set not in allowed:
        raise ConfigContractError(f"mechanism_set is not approved: {mechanism_set}")
    if mechanism_sets["same_model_factory"] is not True or mechanism_sets["single_internal_mechanism_delta"] is not True:
        raise ConfigContractError("training-object parity flags must be true")

    mechanism = _mapping(model["mechanism"], "mechanism")
    backbone = _mapping(model["backbone"], "backbone")
    decoder = _mapping(model["decoder"], "decoder")
    trainability = _mapping(model["trainability"], "trainability")
    optimizer = _mapping(runtime["optimizer"], "optimizer")
    scheduler = _mapping(runtime["scheduler"], "scheduler")
    _require(mechanism, {"name", "states", "stages", "local_window_tokens", "expected_active_fraction_budget", "identity_residual_required"}, "mechanism")
    _require(backbone, {"token_dim", "feature_contract", "depth_group_size", "depth_taps"}, "backbone")
    _require(decoder, {"type", "num_classes_from_benchmark", "ignore_index"}, "decoder")
    _require(trainability, {"trunk", "router", "adapters", "decoder", "backbone_policy"}, "trainability")
    if trainability["backbone_policy"] not in {"frozen", "tap_connected"}:
        raise ConfigContractError("trainability.backbone_policy must be frozen or tap_connected")
    _require(optimizer, {"name", "learning_rate", "weight_decay", "betas"}, "optimizer")
    _require(scheduler, {"name", "warmup_fraction"}, "scheduler")
    if mechanism["name"] == "geotoken_3path":
        if len(mechanism["states"]) != 3:
            raise ConfigContractError("the GeoToken mechanism requires exactly three named states")
    else:
        # Two-zone cleanup 2026-09-02: rejected route mechanisms were removed from the codebase.
        raise ConfigContractError(f"only the GeoToken-3Path baseline mechanism is approved: {mechanism['name']}")
    if mechanism["identity_residual_required"] is not True:
        raise ConfigContractError("identity residual invariant is required")
    if decoder["num_classes_from_benchmark"] is not True:
        raise ConfigContractError("class count must be inherited from the benchmark")

    if str(backbone["feature_contract"]) != "croma_depth_tapped_token_groups":
        raise ConfigContractError("backbone.feature_contract must be croma_depth_tapped_token_groups")
    depth_group_size = _positive_int(backbone["depth_group_size"], "backbone.depth_group_size")
    if depth_group_size != 4:
        raise ConfigContractError("backbone.depth_group_size must be exactly four")
    depth_taps = _mapping(backbone["depth_taps"], "backbone.depth_taps")
    stage_taps = _mapping(depth_taps.get("stage"), "backbone.depth_taps.stage")
    sar_depth_group = _mapping(
        depth_taps.get("sar_depth_group"), "backbone.depth_taps.sar_depth_group"
    )
    if set(stage_taps) != {"optical", "sar"}:
        raise ConfigContractError("backbone.depth_taps.stage must define optical and sar")
    stages = tuple(str(stage) for stage in mechanism["stages"])
    for modality in ("optical", "sar"):
        modality_taps = _mapping(stage_taps[modality], f"backbone.depth_taps.stage.{modality}")
        if set(modality_taps) != set(stages) or any(
            not isinstance(modality_taps[stage], str) or not modality_taps[stage].strip()
            for stage in stages
        ):
            raise ConfigContractError(
                f"backbone.depth_taps.stage.{modality} must cover every configured stage"
            )
    if set(sar_depth_group) != set(stages):
        raise ConfigContractError("backbone.depth_taps.sar_depth_group must cover every stage")
    for stage in stages:
        paths = sar_depth_group[stage]
        if not isinstance(paths, list) or len(paths) != depth_group_size or any(
            not isinstance(path, str) or not path.strip() for path in paths
        ):
            raise ConfigContractError(
                f"backbone.depth_taps.sar_depth_group.{stage} must contain four module paths"
            )

    if str(optimizer["name"]).casefold() != "adamw":
        raise ConfigContractError("optimizer.name must be adamw")
    learning_rate = _finite_number(
        optimizer["learning_rate"], "optimizer.learning_rate", minimum=0.0,
    )
    if learning_rate == 0.0:
        raise ConfigContractError("optimizer.learning_rate must be positive")
    weight_decay = _finite_number(
        optimizer["weight_decay"], "optimizer.weight_decay", minimum=0.0,
    )
    betas = optimizer["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ConfigContractError("optimizer.betas must contain exactly two values")
    resolved_betas = [
        _finite_number(value, f"optimizer.betas[{index}]", minimum=0.0, maximum=1.0)
        for index, value in enumerate(betas)
    ]
    if str(scheduler["name"]).casefold() != "cosine_with_warmup":
        raise ConfigContractError("scheduler.name must be cosine_with_warmup")
    warmup_fraction = _finite_number(
        scheduler["warmup_fraction"], "scheduler.warmup_fraction", minimum=0.0, maximum=1.0,
    )
    gradient_clip_norm = _finite_number(
        runtime["gradient_clip_norm"], "gradient_clip_norm", minimum=0.0,
    )
    if gradient_clip_norm == 0.0:
        raise ConfigContractError("gradient_clip_norm must be positive")
    micro_batch = _positive_int(runtime["micro_batch"], "micro_batch")
    effective_batch = _positive_int(runtime["effective_batch"], "effective_batch")
    gradient_accumulation = _positive_int(runtime["gradient_accumulation"], "gradient_accumulation")
    if effective_batch != micro_batch * gradient_accumulation:
        raise ConfigContractError("effective_batch must equal micro_batch * gradient_accumulation")
    seed = runtime["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigContractError("seed must be a nonnegative integer")
    if str(runtime["precision"]).casefold() != "amp":
        raise ConfigContractError("precision must be amp for the approved runtime")
    max_formal_epochs = _positive_int(runtime["max_formal_epochs"], "max_formal_epochs")
    if max_formal_epochs != 24:
        raise ConfigContractError("the amended formal budget must be exactly 24 epochs")
    early_stopping = _mapping(runtime["early_stopping"], "early_stopping")
    _require(
        early_stopping,
        {"enabled", "monitor", "mode", "burn_in_epochs", "patience", "min_delta_percentage_points", "restore_best_checkpoint"},
        "early_stopping",
    )
    if early_stopping["enabled"] is not True or early_stopping["monitor"] != "validation.mIoU" or early_stopping["mode"] != "max":
        raise ConfigContractError("early stopping must monitor validation.mIoU in max mode")
    burn_in_epochs = _positive_int(early_stopping["burn_in_epochs"], "early_stopping.burn_in_epochs")
    patience = _positive_int(early_stopping["patience"], "early_stopping.patience")
    min_delta = _finite_number(
        early_stopping["min_delta_percentage_points"],
        "early_stopping.min_delta_percentage_points",
        minimum=0.0,
    )
    if min_delta == 0.0 or burn_in_epochs >= max_formal_epochs or early_stopping["restore_best_checkpoint"] is not True:
        raise ConfigContractError("early stopping fields are outside the amended contract")
    augmentation = _mapping(runtime["augmentation"], "augmentation")
    if (
        augmentation.get("name") != "paired_geometric_v1"
        or augmentation.get("enabled") is not True
        or augmentation.get("train_only") is not True
        or augmentation.get("deterministic") is not True
        or augmentation.get("orientation_space") != "D4"
        or list(augmentation.get("operations", []))
        != ["horizontal_flip", "vertical_flip", "rotate_90", "rotate_180", "rotate_270", "transpose", "anti_transpose"]
    ):
        raise ConfigContractError("augmentation must be the frozen paired_geometric_v1 D4 contract")

    resolved = {
        "schema_version": "geotoken3path.resolved.v1",
        "route_id": model["route_id"],
        "candidate_id": model["candidate_id"],
        "dataset_id": benchmark["dataset_id"],
        "model": {
            "backbone_family": model["backbone_family"],
            "token_dim": int(backbone["token_dim"]),
            "num_classes": int(benchmark["labels"]),
            "ignore_index": int(decoder["ignore_index"]),
            "mechanism_set": mechanism_set,
            "active_budget": float(mechanism["expected_active_fraction_budget"]),
            "local_window_tokens": int(mechanism["local_window_tokens"]),
            "stages": list(mechanism["stages"]),
            "depth_group_size": depth_group_size,
            "depth_taps": json.loads(json.dumps(depth_taps)),
            "allow_synthetic_depth_group_fallback": selected_execution_scale == "smoke",
        },
        "input": {
            "optical_channels": int(model["input"]["optical_channels"]),
            "sar_channels": int(model["input"]["sar_channels"]),
            "patch_size": int(model["input"]["patch_size"]),
            "optical_band_order": list(benchmark["modalities"]["optical"]["band_order"]),
            "sar_band_order": list(benchmark["modalities"]["sar"]["band_order"]),
            "optical_source_indices": list(benchmark["modalities"]["optical"]["source_indices"]),
            "sar_source_indices": list(benchmark["modalities"]["sar"]["source_indices"]),
            "sar_raw_channel_order": list(benchmark["modalities"]["sar"]["raw_channel_order"]),
            "normalization": json.loads(json.dumps(benchmark["normalization"])),
        },
        "labels": json.loads(json.dumps(benchmark["label_contract"])),
        "split": json.loads(json.dumps(benchmark["split_contract"])),
        "parent_shape": list(benchmark["parent_shape"]),
        "derived_shape": list(benchmark["derived_shape"]),
        "runtime": {
            "target_gpu": runtime["target_gpu"],
            "precision": runtime["precision"],
            "micro_batch": micro_batch,
            "effective_batch": effective_batch,
            "gradient_accumulation": gradient_accumulation,
            "execution_scale": selected_execution_scale,
            "test_seal_status": runtime["test_seal_status"],
            "seed": seed,
            "optimizer": {
                "name": "adamw",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "betas": resolved_betas,
            },
            "scheduler": {
                "name": "cosine_with_warmup",
                "warmup_fraction": warmup_fraction,
            },
            "gradient_clip_norm": gradient_clip_norm,
            "max_formal_epochs": max_formal_epochs,
            "early_stopping": {
                "enabled": True,
                "monitor": "validation.mIoU",
                "mode": "max",
                "burn_in_epochs": burn_in_epochs,
                "patience": patience,
                "min_delta_percentage_points": min_delta,
                "restore_best_checkpoint": True,
            },
            "augmentation": {
                "name": "paired_geometric_v1",
                "enabled": True,
                "train_only": True,
                "deterministic": True,
                "orientation_space": "D4",
                "operations": [
                    "horizontal_flip", "vertical_flip", "rotate_90", "rotate_180",
                    "rotate_270", "transpose", "anti_transpose",
                ],
            },
        },
        "trainability": dict(trainability),
        "initialization_ref": model["initialization_ref"],
        "initialization": {
            "mode": str(initialization["mode"]),
            "audit_ref": str(initialization["audit_report"]),
            "checkpoint_path": initialization.get("checkpoint_path"),
            "target_test_data_used": initialization["target_test_data_used"],
                "fallback_reason": str(initialization.get("fallback_reason", "")),
            "pretrained_eligible": initialization["pretrained_eligible"],
            "same_initialization_for_baseline_and_innovation": initialization["same_initialization_for_baseline_and_innovation"],
            "constructor_ref": initialization["constructor_ref"],
            "constructor_kwargs": dict(constructor_kwargs),
        },
        "cloud_data_root": benchmark["cloud_data_root"],
        "official_split": benchmark["official_split"],
        "objective_policy": json.loads(json.dumps(objective_contract)) if objective_contract is not None else None,
        "code_sync_manifest_ref": None,
        "code_sync_manifest_sha256": None,
        "storage": {
            "hard_stop_gb": int(benchmark["hard_storage_stop_gb"]),
            "total_ceiling_gb": int(benchmark["total_project_ceiling_gb"]),
        },
    }
    common = json.loads(json.dumps(resolved))
    common["model"].pop("mechanism_set")
    resolved["matched_common_protocol_sha256"] = hashlib.sha256(
        json.dumps(common, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return resolved


def single_mechanism_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Return leaf differences; parity allows only model.mechanism_set."""

    differences: list[str] = []

    def walk(a: Any, b: Any, prefix: str) -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), f"{prefix}.{key}" if prefix else str(key))
        elif a != b:
            differences.append(prefix)

    walk(left, right, "")
    return differences
