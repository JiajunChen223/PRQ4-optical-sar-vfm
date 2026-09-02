"""Audited, cloud-only CROMA constructor and checkpoint loader.

No model class is guessed here.  The cloud runner supplies the pinned official
constructor (``module:attribute``) and this adapter verifies the exact audit,
checkpoint SHA256, and state-dict compatibility before mutating the model.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .initialization import apply_audited_state_dict


class CromaAuditError(RuntimeError):
    """Raised when an audited CROMA initialization cannot be reproduced."""


def _require_cloud_path(value: object, name: str) -> Path:
    if not isinstance(value, str):
        raise CromaAuditError(f"{name} must be under /root/autodl-tmp on the cloud host")
    path = Path(value)
    resolved = path.resolve(strict=False)
    if not resolved.is_absolute() or len(resolved.parts) < 3 or tuple(resolved.parts[1:3]) != ("root", "autodl-tmp") or path.is_symlink():
        raise CromaAuditError(f"{name} escapes the cloud artifact root")
    return path


def _import_symbol(reference: str) -> Any:
    if not isinstance(reference, str) or ":" not in reference:
        raise CromaAuditError("CROMA constructor must be a module:attribute reference")
    module_name, symbol_name = reference.split(":", 1)
    try:
        value = importlib.import_module(module_name)
        for part in symbol_name.split("."):
            value = getattr(value, part)
        return value
    except (ImportError, AttributeError) as exc:
        raise CromaAuditError(f"cannot import audited CROMA constructor {reference}") from exc


def _audit_is_accepted(audit: Mapping[str, Any]) -> None:
    """Validate both the strict bridge audit and the current wrapper audit."""

    if audit.get("status") != "pass" or audit.get("execution_context") != "cloud":
        raise CromaAuditError("CROMA audit must be cloud/pass")
    if audit.get("initialization_mode") != "pretrained":
        raise CromaAuditError("CROMA audit must declare pretrained initialization")
    sha = audit.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha):
        raise CromaAuditError("CROMA audit sha256 is malformed")
    comparison = audit.get("comparison_policy")
    if not isinstance(comparison, Mapping) or comparison.get("same_checkpoint_sha256") is not True or comparison.get("same_initialization_for_baseline_and_innovation") is not True or comparison.get("target_test_data_used") is not False:
        raise CromaAuditError("CROMA baseline/candidate initialization parity is not audited")
    geography = audit.get("geography_overlap_audit")
    if not isinstance(geography, Mapping) or geography.get("status") != "pass" or geography.get("target_test_geographies_excluded") is not True:
        raise CromaAuditError("CROMA geography-overlap exclusion is not closed")
    compatibility = audit.get("compatibility")
    if not isinstance(compatibility, Mapping) or compatibility.get("status") != "pass":
        raise CromaAuditError("CROMA compatibility status is not pass")
    # The old complete schema has explicit shape/missing/unexpected fields;
    # latest wrapper audits expose the same guarantee via shape_mismatches and
    # tensor_key_count.  Never infer success from a missing field.
    if "state_dict" in compatibility:
        state = compatibility["state_dict"]
        if not isinstance(state, Mapping) or state.get("shape_mismatches"):
            raise CromaAuditError("audited CROMA state-dict has unresolved mismatches")
    elif compatibility.get("shape_mismatches"):
        raise CromaAuditError("audited CROMA state-dict has unresolved mismatches")
    else:
        wrapper = compatibility.get("wrapper")
        if not isinstance(wrapper, Mapping) or int(wrapper.get("tensor_key_count", 0)) <= 0:
            raise CromaAuditError("CROMA wrapper key audit is missing")


def _random_init_audit_is_accepted(audit: Mapping[str, Any]) -> None:
    """Validate the explicit leakage-driven random-initialization exception."""

    if audit.get("artifact_type") != "pretrained_alternative_search" or audit.get("schema_version") != "geotoken3path.pretrained_search.v1":
        raise CromaAuditError("random-init audit schema is not recognized")
    if audit.get("status") != "random_init_exception_justified":
        raise CromaAuditError("random-init exception audit is not justified")
    if audit.get("initialization_mode") != "random_init" or audit.get("pretrained_eligible") is not False:
        raise CromaAuditError("random-init audit must reject pretrained eligibility")
    if audit.get("compatible_candidate_found") is not False or audit.get("fallback_justified") is not True:
        raise CromaAuditError("random-init fallback search is incomplete")
    reason = audit.get("fallback_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise CromaAuditError("random-init fallback reason is missing")
    attempts = audit.get("attempts")
    if not isinstance(attempts, list) or not attempts or any(not isinstance(item, Mapping) or not item.get("source") or not item.get("outcome") for item in attempts):
        raise CromaAuditError("random-init search attempts are missing")
    evidence_ref = audit.get("evidence_ref")
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        raise CromaAuditError("random-init leakage evidence reference is missing")
    comparison = audit.get("comparison_policy")
    if not isinstance(comparison, Mapping) or comparison.get("same_initialization_for_baseline_and_innovation") is not True or comparison.get("target_test_data_used") is not False:
        raise CromaAuditError("random-init baseline/candidate parity is not audited")
    for field in ("data_read", "weights_loaded", "gpu_used", "training", "evaluation", "test_accessed"):
        if audit.get(field) is not False:
            raise CromaAuditError(f"random-init audit field {field} must be false")
    if audit.get("constructor_weight_loading_disabled") is not True:
        raise CromaAuditError("random-init constructor must explicitly disable weight loading")


def load_croma_backbone(
    *,
    initialization: Mapping[str, Any],
    audit_path: str | Path,
    constructor_ref: str | None = None,
    constructor: Callable[[], nn.Module] | None = None,
    state_loader: Callable[[str], Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build CROMA with either audited weights or an audited random fallback."""

    mode = initialization.get("mode") if isinstance(initialization, Mapping) else None
    if mode == "pretrained":
        return load_audited_croma_backbone(
            initialization=initialization,
            audit_path=audit_path,
            constructor_ref=constructor_ref,
            constructor=constructor,
            state_loader=state_loader,
        )
    if mode != "random_init":
        raise CromaAuditError("initialization mode must be pretrained or random_init")
    if initialization.get("checkpoint_path") not in {None, ""}:
        raise CromaAuditError("random_init must not declare a checkpoint path")
    audit_file = Path(audit_path)
    if not audit_file.is_absolute():
        raise CromaAuditError("audit_path must be absolute on the cloud host")
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CromaAuditError("cannot read random-init exception audit") from exc
    if not isinstance(audit, Mapping):
        raise CromaAuditError("random-init audit must be a mapping")
    _random_init_audit_is_accepted(audit)
    ctor = constructor if constructor is not None else _import_symbol(
        constructor_ref or initialization.get("constructor_ref", "")
    )
    try:
        signature = inspect.signature(ctor)
        kwargs: dict[str, Any] = {}
        if "pretrained" in signature.parameters:
            kwargs["pretrained"] = False
        if "weights" in signature.parameters:
            kwargs["weights"] = None
        if "pretrained" not in signature.parameters and "weights" not in signature.parameters:
            raise CromaAuditError("random-init constructor has no explicit weight-disable parameter")
        constructor_kwargs = initialization.get("constructor_kwargs", {})
        if not isinstance(constructor_kwargs, Mapping):
            raise CromaAuditError("random-init constructor_kwargs must be a mapping")
        allowed_kwargs = {"source_path", "source_sha256", "size", "modality", "image_resolution"}
        if set(constructor_kwargs) - allowed_kwargs:
            raise CromaAuditError("random-init constructor_kwargs contain an unapproved field")
        for name, value in constructor_kwargs.items():
            if name not in signature.parameters:
                raise CromaAuditError(f"random-init constructor does not accept {name}")
            kwargs[name] = value
        model = ctor(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CromaAuditError("random-init constructor signature cannot be audited") from exc
    if not isinstance(model, nn.Module):
        raise CromaAuditError("CROMA constructor did not return torch.nn.Module")
    return model, {
        "initialization_mode": "random_init",
        "checkpoint_path": None,
        "checkpoint_loaded": False,
        "audit_path": str(audit_file),
        "fallback_reason": audit["fallback_reason"],
        "constructor_weight_loading_disabled": True,
    }


def load_audited_croma_backbone(
    *,
    initialization: Mapping[str, Any],
    audit_path: str | Path,
    constructor_ref: str | None = None,
    constructor: Callable[[], nn.Module] | None = None,
    state_loader: Callable[[str], Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct and load the audited CROMA checkpoint on the cloud host.

    ``constructor`` and ``state_loader`` are injectable solely for cloud-side
    fixture tests; production calls must use the declared import and file.
    """

    if not isinstance(initialization, Mapping):
        raise CromaAuditError("initialization config must be a mapping")
    checkpoint = _require_cloud_path(initialization.get("checkpoint_path"), "checkpoint_path")
    audit_file = Path(audit_path)
    if not audit_file.is_absolute():
        raise CromaAuditError("audit_path must be absolute on the cloud host")
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CromaAuditError("cannot read CROMA audit report") from exc
    if not isinstance(audit, Mapping):
        raise CromaAuditError("CROMA audit report must be a mapping")
    _audit_is_accepted(audit)
    expected_sha = str(audit["sha256"]).casefold()
    ctor_kwargs = initialization.get("constructor_kwargs", {})
    if not isinstance(ctor_kwargs, Mapping):
        raise CromaAuditError("constructor_kwargs must be a mapping")
    source_path = ctor_kwargs.get("source_path")
    source_sha = ctor_kwargs.get("source_sha256")
    if source_path is not None or source_sha is not None:
        if not isinstance(source_path, str) or not isinstance(source_sha, str):
            raise CromaAuditError("pretrained source_path and source_sha256 must be declared together")
        source_file = _require_cloud_path(source_path, "source_path")
        if len(source_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in source_sha):
            raise CromaAuditError("CROMA source_sha256 is malformed")
        source_digest = hashlib.sha256()
        try:
            with source_file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    source_digest.update(block)
        except OSError as exc:
            raise CromaAuditError("audited CROMA source is unavailable") from exc
        if source_digest.hexdigest() != source_sha.casefold():
            raise CromaAuditError("CROMA source SHA256 differs from initialization")
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CromaAuditError("audited CROMA checkpoint is unavailable") from exc
    if digest.hexdigest() != expected_sha:
        raise CromaAuditError("CROMA checkpoint SHA256 differs from audit")

    ctor = constructor if constructor is not None else _import_symbol(
        constructor_ref or initialization.get("constructor_ref", "")
    )
    try:
        signature = inspect.signature(ctor)
        kwargs: dict[str, Any] = {}
        if "pretrained" in signature.parameters:
            kwargs["pretrained"] = False
        if "weights" in signature.parameters:
            kwargs["weights"] = None
        for name, value in ctor_kwargs.items():
            if name not in signature.parameters:
                raise CromaAuditError(f"pretrained constructor does not accept {name}")
            kwargs[name] = value
        model = ctor(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CromaAuditError("pretrained constructor signature cannot be audited") from exc
    if not isinstance(model, nn.Module):
        raise CromaAuditError("CROMA constructor did not return torch.nn.Module")
    try:
        loaded = state_loader(str(checkpoint)) if state_loader else torch.load(
            str(checkpoint), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise CromaAuditError("failed to load audited CROMA checkpoint") from exc
    if not isinstance(loaded, Mapping):
        raise CromaAuditError("checkpoint does not contain a mapping")
    required = {
        "s1_encoder": "s1_encoder",
        "s1_GAP_FFN": "GAP_FFN_s1",
        "s2_encoder": "s2_encoder",
        "s2_GAP_FFN": "GAP_FFN_s2",
        "joint_encoder": "cross_encoder",
    }
    if set(loaded) != set(required):
        raise CromaAuditError("official CROMA checkpoint must contain exactly five nested blocks")
    nested: dict[str, Mapping[str, Tensor]] = {}
    for key in required:
        value = loaded[key]
        if not isinstance(value, Mapping) or any(not isinstance(k, str) or not isinstance(v, Tensor) for k, v in value.items()):
            raise CromaAuditError(f"CROMA nested block {key} is not a tensor mapping")
        nested[key] = value
    try:
        load_report: dict[str, Any] = {"missing_keys": [], "unexpected_keys": [], "blocks": {}}
        for source_key, target_name in required.items():
            target = getattr(model, target_name, None)
            if not isinstance(target, nn.Module):
                raise ValueError(f"constructor missing CROMA block {target_name}")
            expected = target.state_dict()
            actual = nested[source_key]
            if set(actual) != set(expected):
                raise ValueError(f"nested block {source_key} keys differ from constructor")
            if any(tuple(actual[name].shape) != tuple(expected[name].shape) for name in expected):
                raise ValueError(f"nested block {source_key} has shape mismatch")
            incompatible = target.load_state_dict(actual, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f"strict CROMA load returned incompatible keys for {source_key}")
            load_report["blocks"][source_key] = {"target": target_name, "tensor_count": len(actual)}
    except Exception as exc:
        # Current wrapper audits use strict full-state loading; this explicit
        # error preserves the unexplained mismatch instead of strict=False.
        raise CromaAuditError("CROMA state_dict compatibility failed") from exc
    return model, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "audit_path": str(audit_file),
        "load_report": load_report,
    }
