"""Cloud-side certification for R21 ICE-Exact.

The certifier never accesses the sealed test split. It builds two models from
independently audited CROMA instances, loads the same verified downstream
checkpoint into both, and compares full execution with ICE exact at five
levels: state surface, retained taps, FP32 logits, trainable gradients, and the
complete validation confusion matrix.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from ..data.sen12ts import build_sen12ts_loader, croma_dynamic_normalize_batch
from ..execution.certification import compare_gradients, compare_tensors, named_trainable_gradients
from ..losses import segmentation_objective
from ..metrics import confusion_matrix, mean_iou
from ..models.croma_loader import load_croma_backbone
from ..models.factory import build_vfm_segmentation_model
from ..utils.test_seal import assert_test_access_allowed


class IceCertificationError(RuntimeError):
    """Raised when the certification request itself violates a hard contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.flush()
        temp = Path(handle.name)
    temp.replace(path)


def _comparison_payload(result: Any) -> dict[str, Any]:
    raw = dict(result.__dict__)
    for key in ("max_abs_error", "mean_abs_error", "max_relative_error"):
        value = raw.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            raw[key] = None
    return raw


def _checkpoint_state(path: Path) -> Mapping[str, Tensor]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise IceCertificationError("cannot load baseline checkpoint") from exc
    if not isinstance(payload, Mapping):
        raise IceCertificationError("baseline checkpoint must contain a mapping")
    state = payload.get("model", payload)
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise IceCertificationError("baseline checkpoint model state is invalid")
    return state


# Two-zone cleanup 2026-09-02: the verified baseline checkpoint predates the
# mechanism cleanup and carries state keys of rejected mechanisms that no
# longer exist in the model graph (always-fuse never consumed them). ICE
# certification must be able to load that authoritative checkpoint, so these
# keys are stripped before the strict load. The stripped keys are reported in
# the certificate metadata; stripping is limited to the documented rejected
# mechanism surface.
_ARCHIVED_MECHANISM_PREFIXES = (
    "token_model.tasr_redistributor.",
    "token_model.dtsf_adapter.",
    "token_model.rift_adapters.",
    "token_model.mcsl_",
    "token_model.mcof_",
    "token_model.jack_",
    "token_model.ctsp_",
)


def _strip_archived_mechanism_keys(state: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], list[str]]:
    stripped = []
    kept = {}
    for name, value in state.items():
        if any(name.startswith(prefix) for prefix in _ARCHIVED_MECHANISM_PREFIXES):
            stripped.append(name)
            continue
        kept[name] = value
    return kept, stripped  # type: ignore[return-value]


def _state_surface_equal(full: nn.Module, ice: nn.Module) -> tuple[bool, list[str]]:
    full_state = full.state_dict()
    ice_state = ice.state_dict()
    if list(full_state) != list(ice_state):
        names = sorted(set(full_state) ^ set(ice_state))
        return False, names
    unequal = [
        name
        for name in full_state
        if not torch.equal(full_state[name].cpu(), ice_state[name].cpu())
    ]
    return not unequal, unequal


def _capture_paths(
    model: nn.Module, paths: tuple[str, ...]
) -> tuple[dict[str, Tensor], list[Any]]:
    raw_backbone = model.bridge.backbone.backbone
    captures: dict[str, Tensor] = {}
    handles = []
    for path in paths:
        module = raw_backbone.get_submodule(path)

        def _capture(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            name: str = path,
        ) -> None:
            if not isinstance(output, Tensor):
                raise IceCertificationError(f"retained tap {name} did not return a tensor")
            # Certification only needs the value. Move the clone off-GPU so
            # retained-tap evidence does not compete with the later gradient run.
            captures[name] = output.detach().cpu().clone()

        handles.append(module.register_forward_hook(_capture))
    return captures, handles


def _remove_handles(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def build_ice_certification_pair(
    *,
    resolved: Mapping[str, Any],
    initialization: Mapping[str, Any],
    audit_report: str | Path,
    checkpoint: str | Path,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Build full/ICE models with identical audited weights and state surfaces."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        raise IceCertificationError("checkpoint must be an absolute cloud path")
    model_cfg = resolved.get("model")
    if not isinstance(model_cfg, Mapping) or str(model_cfg.get("mechanism_set", "")) != "always_fuse":
        raise IceCertificationError("ICE certification requires the verified always_fuse mechanism")

    full_backbone, full_load_report = load_croma_backbone(
        initialization=initialization,
        audit_path=audit_report,
        constructor_ref=str(initialization.get("constructor_ref", "")),
    )
    ice_backbone, ice_load_report = load_croma_backbone(
        initialization=initialization,
        audit_path=audit_report,
        constructor_ref=str(initialization.get("constructor_ref", "")),
    )
    full_model = build_vfm_segmentation_model(
        resolved,
        audited_croma_backbone=full_backbone,
        backbone_execution="full",
    )
    ice_model = build_vfm_segmentation_model(
        resolved,
        audited_croma_backbone=ice_backbone,
        backbone_execution="ice_exact",
    )
    raw_state = _checkpoint_state(checkpoint_path)
    state, stripped_keys = _strip_archived_mechanism_keys(raw_state)
    full_incompatible = full_model.load_state_dict(state, strict=True)
    ice_incompatible = ice_model.load_state_dict(state, strict=True)
    if (
        full_incompatible.missing_keys
        or full_incompatible.unexpected_keys
        or ice_incompatible.missing_keys
        or ice_incompatible.unexpected_keys
    ):
        raise IceCertificationError("strict baseline state loading returned incompatible keys")
    if stripped_keys:
        metadata_note = {
            "archived_mechanism_keys_stripped": len(stripped_keys),
            "archived_mechanism_keys": stripped_keys,
        }
    state_equal, unequal_names = _state_surface_equal(full_model, ice_model)
    plan = getattr(ice_model, "_ice_execution_plan", None)
    if plan is None:
        raise IceCertificationError("ICE model did not expose a compiled execution plan")
    local_note = locals().get("metadata_note")
    metadata = {
        "checkpoint_sha256": _sha256(checkpoint_path),
        **(local_note or {}),
        "full_croma_load_report": full_load_report,
        "ice_croma_load_report": ice_load_report,
        "state_surface_equal": state_equal,
        "state_surface_unequal_names": unequal_names,
        "execution_plan": plan.payload(),
    }
    return full_model, ice_model, metadata


def _prepare_batch(
    batch: Mapping[str, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    normalized = croma_dynamic_normalize_batch(batch)
    return (
        normalized["optical"].to(device, non_blocking=True),
        normalized["sar"].to(device, non_blocking=True),
        batch["target"].to(device, non_blocking=True),
    )


def run_ice_exact_certification(
    *,
    resolved: Mapping[str, Any],
    initialization: Mapping[str, Any],
    data_manifest: str | Path,
    audit_report: str | Path,
    checkpoint: str | Path,
    output_path: str | Path,
    device: str = "cuda:0",
    execution_scale: str = "acceptance",
    objective_name: str = "ce_lovasz",
    expected_miou_percent: float | None = None,
) -> dict[str, Any]:
    """Run R21 equivalence certification without opening the sealed test split."""

    assert_test_access_allowed(
        {"execution_scale": execution_scale, "test_seal_status": "sealed"},
        "validation",
    )
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise IceCertificationError("formal ICE certification requires CUDA")
    if objective_name != "ce_lovasz":
        raise IceCertificationError("R21 certification inherits the locked CE+Lovasz objective")

    full_model, ice_model, metadata = build_ice_certification_pair(
        resolved=resolved,
        initialization=initialization,
        audit_report=audit_report,
        checkpoint=checkpoint,
    )
    full_model.to(target_device)
    ice_model.to(target_device)

    runtime = resolved.get("runtime")
    model_cfg = resolved.get("model")
    if not isinstance(runtime, Mapping) or not isinstance(model_cfg, Mapping):
        raise IceCertificationError("resolved runtime/model configuration is malformed")
    micro_batch = int(runtime.get("micro_batch", 0))
    if micro_batch != 16:
        raise IceCertificationError("R21 must preserve the frozen micro_batch=16 normalization contract")
    seed = int(runtime.get("seed", 0))
    loader_kwargs = {
        "batch_size": micro_batch,
        "num_workers": 4,
        "execution_scale": execution_scale,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "seed": seed,
    }
    validation_loader, _ = build_sen12ts_loader(
        data_manifest,
        split="validation",
        augmentation=None,
        **loader_kwargs,
    )
    train_loader, _ = build_sen12ts_loader(
        data_manifest,
        split="train",
        augmentation=runtime.get("augmentation"),
        **loader_kwargs,
    )

    # --- FP32 retained-tap and logit certificate on the first validation batch.
    first_validation = next(iter(validation_loader))
    optical, sar, _target = _prepare_batch(first_validation, target_device)
    plan = getattr(ice_model, "_ice_execution_plan")
    paths = tuple(plan.required_taps)
    full_captures, full_handles = _capture_paths(full_model, paths)
    ice_captures, ice_handles = _capture_paths(ice_model, paths)
    full_model.eval()
    ice_model.eval()
    try:
        with torch.no_grad():
            full_logits_fp32 = full_model(optical, sar)
            ice_logits_fp32 = ice_model(optical, sar)
    finally:
        _remove_handles(full_handles)
        _remove_handles(ice_handles)
    if set(full_captures) != set(paths) or set(ice_captures) != set(paths):
        raise IceCertificationError("not every retained tap executed during certification")
    tap_results = {
        path: _comparison_payload(compare_tensors(full_captures[path], ice_captures[path]))
        for path in paths
    }
    fp32_logit_result = _comparison_payload(compare_tensors(full_logits_fp32, ice_logits_fp32))
    del full_captures, ice_captures, full_logits_fp32, ice_logits_fp32, optical, sar
    torch.cuda.empty_cache()

    # --- Gradient certificate on one fixed train batch. Restore RNG before ICE
    # so any stochasticity in the retained graph cannot create a false mismatch.
    torch.manual_seed(seed)
    train_batch = next(iter(train_loader))
    train_optical, train_sar, train_target = _prepare_batch(train_batch, target_device)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()

    full_model.train()
    full_model.zero_grad(set_to_none=True)
    full_logits = full_model(train_optical, train_sar)
    full_loss, _ = segmentation_objective(
        full_logits,
        train_target,
        objective_name=objective_name,
    )
    full_loss.backward()
    full_gradients = named_trainable_gradients(full_model)
    full_loss_value = float(full_loss.detach().cpu())
    full_model.zero_grad(set_to_none=True)
    del full_logits, full_loss
    torch.cuda.empty_cache()

    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    ice_model.train()
    ice_model.zero_grad(set_to_none=True)
    ice_logits = ice_model(train_optical, train_sar)
    ice_loss, _ = segmentation_objective(
        ice_logits,
        train_target,
        objective_name=objective_name,
    )
    ice_loss.backward()
    ice_gradients = named_trainable_gradients(ice_model)
    ice_loss_value = float(ice_loss.detach().cpu())
    ice_model.zero_grad(set_to_none=True)
    del ice_logits, ice_loss, train_optical, train_sar, train_target
    torch.cuda.empty_cache()

    gradient_result = compare_gradients(full_gradients, ice_gradients)
    gradient_result["full_loss"] = full_loss_value
    gradient_result["ice_loss"] = ice_loss_value

    # --- Complete validation under the protocol AMP setting.
    full_model.eval()
    ice_model.eval()
    classes = int(model_cfg["num_classes"])
    full_matrix = torch.zeros(classes, classes, dtype=torch.int64)
    ice_matrix = torch.zeros(classes, classes, dtype=torch.int64)
    amp_enabled = str(runtime.get("precision", "")).casefold() == "amp"
    prediction_pixels = 0
    identical_prediction_pixels = 0
    validation_logit_max_abs = 0.0
    validation_logit_exact_batches = 0
    validation_batches = 0
    with torch.no_grad():
        for batch in validation_loader:
            optical, sar, target = _prepare_batch(batch, target_device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                full_logits = full_model(optical, sar)
                ice_logits = ice_model(optical, sar)
            comparison = compare_tensors(full_logits, ice_logits)
            validation_logit_max_abs = max(
                validation_logit_max_abs,
                comparison.max_abs_error,
            )
            validation_logit_exact_batches += int(comparison.torch_equal)
            validation_batches += 1
            full_pred = full_logits.argmax(dim=1)
            ice_pred = ice_logits.argmax(dim=1)
            prediction_pixels += int(full_pred.numel())
            identical_prediction_pixels += int((full_pred == ice_pred).sum().item())
            full_matrix += confusion_matrix(full_logits, target, classes).cpu()
            ice_matrix += confusion_matrix(ice_logits, target, classes).cpu()

    full_miou = float(mean_iou(full_matrix))
    ice_miou = float(mean_iou(ice_matrix))
    matrix_equal = bool(torch.equal(full_matrix, ice_matrix))
    all_predictions_equal = identical_prediction_pixels == prediction_pixels
    expected_ok = True
    expected_delta_pp = None
    if expected_miou_percent is not None:
        # The archived 49.7808% is the deterministic-replay record value; the
        # certificate recomputes mIoU under its own AMP validation pass, so a
        # small float-level difference is expected. 0.05 pp is far below any
        # scientific effect and only guards against loading the wrong
        # checkpoint (which would differ by >> 0.1 pp).
        expected_delta_pp = abs(full_miou * 100.0 - float(expected_miou_percent))
        expected_ok = expected_delta_pp <= 0.05

    tap_gate = all(
        bool(item["shape_equal"])
        and item["max_abs_error"] is not None
        and float(item["max_abs_error"]) <= 1e-6
        for item in tap_results.values()
    )
    fp32_gate = (
        bool(fp32_logit_result["shape_equal"])
        and fp32_logit_result["max_abs_error"] is not None
        and float(fp32_logit_result["max_abs_error"]) <= 1e-6
    )
    gradient_gate = (
        not gradient_result["missing_gradient_names"]
        and float(gradient_result["max_gradient_abs_error"]) <= 1e-6
        and abs(full_loss_value - ice_loss_value) <= 1e-6
    )
    validation_gate = (
        all_predictions_equal
        and matrix_equal
        and validation_logit_max_abs <= 1e-4
        and abs(full_miou - ice_miou) <= 1e-12
    )
    state_gate = bool(metadata["state_surface_equal"])
    equivalence_pass = all(
        (
            state_gate,
            tap_gate,
            fp32_gate,
            gradient_gate,
            validation_gate,
            expected_ok,
        )
    )
    status = "pass" if equivalence_pass else "fail"

    source_sha = None
    constructor_kwargs = initialization.get("constructor_kwargs")
    if isinstance(constructor_kwargs, Mapping):
        source_sha = constructor_kwargs.get("source_sha256")
    certificate: dict[str, Any] = {
        "schema_version": "prq4.ice_exact_certificate.v1",
        "route": "R21-ICE-VFM-01",
        "status": status,
        "equivalence_certified": equivalence_pass,
        # Equivalence alone is not route-level scientific support. The separate
        # certificate-gated profiling stage must still clear the >=20% latency gate.
        "scientific_route_supported": False,
        "efficiency_evaluated": False,
        "test_accessed": False,
        "backbone_execution_comparison": ["full", "ice_exact"],
        "mechanism_set": "always_fuse",
        "objective_name": objective_name,
        "micro_batch": micro_batch,
        "matched_common_protocol_sha256": resolved.get("matched_common_protocol_sha256"),
        "code_sync_manifest_sha256": resolved.get("code_sync_manifest_sha256"),
        "croma_source_sha256": source_sha,
        **metadata,
        "tap_equivalence": tap_results,
        "fp32_logit_equivalence": fp32_logit_result,
        "gradient_equivalence": gradient_result,
        "validation": {
            "amp_enabled": amp_enabled,
            "batches": validation_batches,
            "logit_exact_batches": validation_logit_exact_batches,
            "max_abs_logit_error": validation_logit_max_abs,
            "prediction_pixels": prediction_pixels,
            "identical_prediction_pixels": identical_prediction_pixels,
            "prediction_pixels_identical": all_predictions_equal,
            "confusion_matrix_identical": matrix_equal,
            "full_mIoU_percent": full_miou * 100.0,
            "ice_mIoU_percent": ice_miou * 100.0,
            "expected_mIoU_percent": expected_miou_percent,
            "expected_delta_pp": expected_delta_pp,
            "expected_baseline_match": expected_ok,
        },
        "gates": {
            "state_surface": state_gate,
            "retained_taps_fp32": tap_gate,
            "logits_fp32": fp32_gate,
            "trainable_gradients_fp32": gradient_gate,
            "full_validation": validation_gate,
            "expected_baseline": expected_ok,
        },
    }
    _atomic_json(Path(output_path), certificate)
    return certificate
