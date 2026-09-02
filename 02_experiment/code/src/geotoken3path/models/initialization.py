"""Fail-closed validation for cloud-audited pretrained initialization.

This module never opens a checkpoint or performs network/file-system I/O. The
experiment service must load and hash checkpoint bytes on the authorized cloud
host, then pass the already-loaded state dict and its audit record here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any
from urllib.parse import urlparse

from torch import Tensor, nn


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TOP_LEVEL_FIELDS = (
    "status", "execution_context", "initialization_mode", "sha256",
    "source", "compatibility", "comparison_policy",
)
_SOURCE_FIELDS = ("url", "license", "commit")
_COMPATIBILITY_FIELDS = (
    "status", "architecture", "input_spec", "head_replacement",
    "state_dict", "position_resolution_adaptation",
)
_COMPARISON_FIELDS = (
    "same_initialization_for_baseline_and_innovation",
    "same_checkpoint_sha256", "same_input_spec", "target_test_data_used",
)
_MODALITIES = ("optical", "sar")
_DYNAMIC_NORMALIZATION_SCHEME = "croma_official_dynamic_v1"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], path: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{path} missing: {', '.join(missing)}")


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")
    if value.strip().casefold() in {"unknown", "pending", "tbd", "none"}:
        raise ValueError(f"{path} must be resolved before initialization")
    return value.strip()


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value


def _string_list(value: object, path: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    items = [_nonempty_text(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if not allow_empty and not items:
        raise ValueError(f"{path} must not be empty")
    if len(items) != len(set(items)):
        raise ValueError(f"{path} contains duplicate entries")
    return items


def _positive_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{path} must be finite and positive")
    return number


def _pair_of_positive_ints(value: object, path: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must contain exactly two integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{path} must contain exactly two positive integers")
    return int(value[0]), int(value[1])


def _validate_normalization(
    value: object,
    band_order: Mapping[str, list[str]],
    path: str,
) -> object:
    normalization = _mapping(value, path)
    if normalization.get("scheme") == _DYNAMIC_NORMALIZATION_SCHEME:
        required = {
            "scheme", "scope", "statistics_axes", "micro_batch", "last_batch_policy", "batch_semantics",
            "distributed_statistics", "global_batch_statistics", "clip_sigma", "output_range", "encoding",
            "formula", "zero_std_policy", "nodata_policy", "fixed_vectors_declared", "source",
            "source_revision", "readme_sha256", "loader_sha256", "source_evidence_ref", "normalization_locked",
        }
        _require_fields(normalization, tuple(required), path)
        if normalization["scope"] != "per_micro_batch_per_channel" or normalization["statistics_axes"] != [0, 2, 3]:
            raise ValueError(f"{path} must use CROMA axes (0,2,3) over a local micro-batch")
        if normalization["micro_batch"] != 16:
            raise ValueError(f"{path}.micro_batch must be exactly 16")
        if normalization["last_batch_policy"] != {
            "training": "drop_last",
            "validation": "pad_repeat_last_and_trim_outputs",
            "inference": "pad_repeat_last_and_trim_outputs",
        }:
            raise ValueError(f"{path}.last_batch_policy is not frozen")
        if not isinstance(normalization["batch_semantics"], Mapping) or set(normalization["batch_semantics"]) != {"training", "validation", "inference"}:
            raise ValueError(f"{path}.batch_semantics must cover training, validation and inference")
        if normalization["distributed_statistics"] != "per_rank_micro_batch" or normalization["global_batch_statistics"] is not False:
            raise ValueError(f"{path} must use per-rank micro-batch statistics")
        if normalization["clip_sigma"] != 2.0 or normalization["output_range"] != [0.0, 1.0] or normalization["encoding"] != "float32":
            raise ValueError(f"{path} has an unsupported CROMA clip/encoding policy")
        formula = normalization["formula"]
        if not isinstance(formula, str) or "mean_axes_0_2_3" not in formula or "std_axes_0_2_3" not in formula:
            raise ValueError(f"{path}.formula must expose batch/spatial axes")
        if normalization["fixed_vectors_declared"] is not False or normalization["source"] != "official_croma_readme":
            raise ValueError(f"{path} must not replace the official dynamic recipe with fixed vectors")
        if _COMMIT_RE.fullmatch(str(normalization["source_revision"])) is None:
            raise ValueError(f"{path}.source_revision must be a pinned commit")
        for field in ("readme_sha256", "loader_sha256"):
            if _SHA256_RE.fullmatch(str(normalization[field])) is None:
                raise ValueError(f"{path}.{field} must be SHA256")
        for field in ("source_evidence_ref", "zero_std_policy", "nodata_policy"):
            _nonempty_text(normalization[field], f"{path}.{field}")
        if not isinstance(normalization["normalization_locked"], bool):
            raise ValueError(f"{path}.normalization_locked must be boolean")
        return dict(normalization)
    _require_fields(normalization, _MODALITIES, path)
    resolved: dict[str, dict[str, tuple[float, ...]]] = {}
    for modality in _MODALITIES:
        spec = _mapping(normalization[modality], f"{path}.{modality}")
        _require_fields(spec, ("mean", "std"), f"{path}.{modality}")
        mean, std = spec["mean"], spec["std"]
        if not isinstance(mean, list) or not isinstance(std, list):
            raise ValueError(f"{path}.{modality}.mean/std must be lists")
        expected = len(band_order[modality])
        if len(mean) != expected or len(std) != expected:
            raise ValueError(f"{path}.{modality} length must match band_order")
        mean_values: list[float] = []
        std_values: list[float] = []
        for index, item in enumerate(mean):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ValueError(f"{path}.{modality}.mean[{index}] must be finite numeric")
            mean_values.append(float(item))
        for index, item in enumerate(std):
            std_values.append(_positive_number(item, f"{path}.{modality}.std[{index}]"))
        resolved[modality] = {"mean": tuple(mean_values), "std": tuple(std_values)}
    return resolved


def _validate_endpoint_input(value: object, path: str) -> dict[str, object]:
    spec = _mapping(value, path)
    _require_fields(spec, ("band_order", "normalization", "gsd_meters", "patch_size"), path)
    band_mapping = _mapping(spec["band_order"], f"{path}.band_order")
    _require_fields(band_mapping, _MODALITIES, f"{path}.band_order")
    band_order = {
        modality: _string_list(
            band_mapping[modality], f"{path}.band_order.{modality}", allow_empty=False,
        )
        for modality in _MODALITIES
    }
    normalization = _validate_normalization(spec["normalization"], band_order, f"{path}.normalization")
    gsd_mapping = _mapping(spec["gsd_meters"], f"{path}.gsd_meters")
    _require_fields(gsd_mapping, _MODALITIES, f"{path}.gsd_meters")
    gsd = {
        modality: _positive_number(gsd_mapping[modality], f"{path}.gsd_meters.{modality}")
        for modality in _MODALITIES
    }
    patch_size = _pair_of_positive_ints(spec["patch_size"], f"{path}.patch_size")
    return {
        "band_order": band_order, "normalization": normalization,
        "gsd_meters": gsd, "patch_size": patch_size,
    }


def _validate_source(value: object) -> None:
    source = _mapping(value, "source")
    _require_fields(source, _SOURCE_FIELDS, "source")
    url = _nonempty_text(source["url"], "source.url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source.url must be an absolute HTTP(S) URL")
    _nonempty_text(source["license"], "source.license")
    _nonempty_text(source["commit"], "source.commit")


def _validate_compatibility(value: object) -> None:
    compatibility = _mapping(value, "compatibility")
    _require_fields(compatibility, _COMPATIBILITY_FIELDS, "compatibility")
    if compatibility["status"] != "pass":
        raise ValueError("compatibility.status must be pass")

    architecture = _mapping(compatibility["architecture"], "compatibility.architecture")
    _require_fields(
        architecture, ("checkpoint_backbone", "target_backbone", "compatible"),
        "compatibility.architecture",
    )
    _nonempty_text(architecture["checkpoint_backbone"], "compatibility.architecture.checkpoint_backbone")
    _nonempty_text(architecture["target_backbone"], "compatibility.architecture.target_backbone")
    if _bool(architecture["compatible"], "compatibility.architecture.compatible") is not True:
        raise ValueError("checkpoint and target architectures are not compatible")

    input_spec = _mapping(compatibility["input_spec"], "compatibility.input_spec")
    _require_fields(input_spec, ("checkpoint", "target"), "compatibility.input_spec")
    checkpoint_input = _validate_endpoint_input(input_spec["checkpoint"], "compatibility.input_spec.checkpoint")
    target_input = _validate_endpoint_input(input_spec["target"], "compatibility.input_spec.target")
    if checkpoint_input["band_order"] != target_input["band_order"]:
        raise ValueError("checkpoint and target band order must match exactly")
    if checkpoint_input["normalization"] != target_input["normalization"]:
        raise ValueError("checkpoint and target normalization must match exactly")

    head = _mapping(compatibility["head_replacement"], "compatibility.head_replacement")
    _require_fields(
        head, ("required", "configured", "recorded", "checkpoint_head_keys", "target_head_keys"),
        "compatibility.head_replacement",
    )
    required = _bool(head["required"], "compatibility.head_replacement.required")
    if _bool(head["configured"], "compatibility.head_replacement.configured") is not True:
        raise ValueError("head replacement decision must be configured")
    if _bool(head["recorded"], "compatibility.head_replacement.recorded") is not True:
        raise ValueError("head replacement decision must be recorded")
    checkpoint_head_keys = _string_list(
        head["checkpoint_head_keys"], "compatibility.head_replacement.checkpoint_head_keys",
        allow_empty=True,
    )
    target_head_keys = _string_list(
        head["target_head_keys"], "compatibility.head_replacement.target_head_keys",
        allow_empty=True,
    )
    if required and not (checkpoint_head_keys or target_head_keys):
        raise ValueError("required head replacement must declare at least one checkpoint or target key")
    if not required and (checkpoint_head_keys or target_head_keys):
        raise ValueError("head keys require head_replacement.required=true")

    state_dict = _mapping(compatibility["state_dict"], "compatibility.state_dict")
    _require_fields(state_dict, ("missing_keys", "unexpected_keys", "shape_mismatches"), "compatibility.state_dict")
    missing_keys = _string_list(state_dict["missing_keys"], "compatibility.state_dict.missing_keys", allow_empty=True)
    unexpected_keys = _string_list(
        state_dict["unexpected_keys"], "compatibility.state_dict.unexpected_keys", allow_empty=True,
    )
    shape_mismatches = _string_list(
        state_dict["shape_mismatches"], "compatibility.state_dict.shape_mismatches", allow_empty=True,
    )
    if shape_mismatches:
        raise ValueError("unresolved state-dict shape mismatches are forbidden")
    if set(missing_keys) != set(target_head_keys):
        raise ValueError("missing_keys must exactly match recorded target head replacement keys")
    if set(unexpected_keys) != set(checkpoint_head_keys):
        raise ValueError("unexpected_keys must exactly match recorded checkpoint head replacement keys")

    adaptation = _mapping(
        compatibility["position_resolution_adaptation"],
        "compatibility.position_resolution_adaptation",
    )
    _require_fields(
        adaptation, ("required", "status", "method", "source_grid", "target_grid"),
        "compatibility.position_resolution_adaptation",
    )
    adaptation_required = _bool(
        adaptation["required"], "compatibility.position_resolution_adaptation.required",
    )
    method = str(adaptation["method"]).strip().casefold() if isinstance(adaptation["method"], str) else ""
    source_grid = _pair_of_positive_ints(
        adaptation["source_grid"], "compatibility.position_resolution_adaptation.source_grid",
    )
    target_grid = _pair_of_positive_ints(
        adaptation["target_grid"], "compatibility.position_resolution_adaptation.target_grid",
    )
    resolution_changed = (
        checkpoint_input["gsd_meters"] != target_input["gsd_meters"]
        or checkpoint_input["patch_size"] != target_input["patch_size"]
        or source_grid != target_grid
    )
    if adaptation_required:
        if adaptation["status"] != "pass" or method in {"", "none", "not_required"}:
            raise ValueError("required position/resolution adaptation must record a passing method")
    else:
        if adaptation["status"] != "not_required" or method != "none":
            raise ValueError("non-required position/resolution adaptation must be recorded as not_required/none")
        if resolution_changed:
            raise ValueError("resolution/GSD/patch mismatch requires an audited adaptation")


def _validate_comparison_policy(value: object) -> None:
    comparison = _mapping(value, "comparison_policy")
    _require_fields(comparison, _COMPARISON_FIELDS, "comparison_policy")
    if _bool(
        comparison["same_initialization_for_baseline_and_innovation"],
        "comparison_policy.same_initialization_for_baseline_and_innovation",
    ) is not True:
        raise ValueError("baseline and candidate must share initialization")
    if _bool(comparison["same_checkpoint_sha256"], "comparison_policy.same_checkpoint_sha256") is not True:
        raise ValueError("baseline and candidate must share checkpoint SHA256")
    if _bool(comparison["same_input_spec"], "comparison_policy.same_input_spec") is not True:
        raise ValueError("baseline and candidate must share the audited input specification")
    if _bool(comparison["target_test_data_used"], "comparison_policy.target_test_data_used") is not False:
        raise ValueError("target test data must not influence initialization")


def validate_pretrained_audit(audit: Mapping[str, Any]) -> None:
    """Validate the complete cloud-side pretrained compatibility record."""
    if not isinstance(audit, Mapping):
        raise ValueError("pretrained audit must be a mapping")
    _require_fields(audit, _TOP_LEVEL_FIELDS, "pretrained audit")
    if audit["status"] != "pass" or audit["execution_context"] != "cloud":
        raise ValueError("pretrained audit must pass in the cloud execution context")
    if audit["initialization_mode"] != "pretrained":
        raise ValueError("initialization_mode must be pretrained")
    if not isinstance(audit["sha256"], str) or _SHA256_RE.fullmatch(audit["sha256"]) is None:
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    _validate_source(audit["source"])
    _validate_compatibility(audit["compatibility"])
    _validate_comparison_policy(audit["comparison_policy"])


def apply_audited_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    audit: Mapping[str, Any],
    *,
    strict: bool = True,
) -> dict[str, list[str]]:
    """Apply an already-loaded state dict after fail-closed audit validation.

    No checkpoint path is accepted. Key and shape compatibility are checked
    before mutating ``model``. ``strict=False`` is allowed only for the exact
    head-replacement keys declared in the passing audit.
    """
    validate_pretrained_audit(audit)
    if not isinstance(state_dict, Mapping):
        raise ValueError("state_dict must be an already-loaded tensor mapping")
    if any(not isinstance(key, str) or not isinstance(value, Tensor) for key, value in state_dict.items()):
        raise ValueError("state_dict must map string keys to tensors")

    model_state = model.state_dict()
    actual_missing = sorted(set(model_state) - set(state_dict))
    actual_unexpected = sorted(set(state_dict) - set(model_state))
    actual_shape_mismatches = sorted(
        key for key in set(model_state).intersection(state_dict)
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    )
    compatibility = _mapping(audit["compatibility"], "compatibility")
    state_audit = _mapping(compatibility["state_dict"], "compatibility.state_dict")
    recorded_missing = sorted(state_audit["missing_keys"])
    recorded_unexpected = sorted(state_audit["unexpected_keys"])
    if actual_shape_mismatches:
        raise ValueError("state_dict shape mismatch: " + ", ".join(actual_shape_mismatches))
    if actual_missing != recorded_missing or actual_unexpected != recorded_unexpected:
        raise ValueError(
            "loaded state_dict differs from audit: "
            f"missing={actual_missing}, unexpected={actual_unexpected}"
        )
    if strict and (actual_missing or actual_unexpected):
        raise ValueError("strict=True forbids audited head-replacement key differences")

    incompatible = model.load_state_dict(state_dict, strict=strict)
    loaded_missing = sorted(incompatible.missing_keys)
    loaded_unexpected = sorted(incompatible.unexpected_keys)
    if loaded_missing != actual_missing or loaded_unexpected != actual_unexpected:
        raise RuntimeError("PyTorch load result differs from the prevalidated audit")
    return {"missing_keys": loaded_missing, "unexpected_keys": loaded_unexpected}
