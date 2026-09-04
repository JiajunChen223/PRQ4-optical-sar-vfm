"""Minimal cloud-only baseline/candidate train-and-validation runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import os
import tempfile
from typing import Any

import torch
from torch import Tensor, nn

from ..data.sen12ts import croma_dynamic_normalize_batch, build_sen12ts_loader
from ..data.contracts import cross_validate_dataset_and_pretrained
from ..losses import segmentation_cross_entropy, segmentation_objective
from ..metrics import confusion_matrix, mean_iou
from ..models.croma_loader import load_croma_backbone
from ..models.factory import build_vfm_segmentation_model
from ..utils.run_manifest import build_run_manifest
from ..utils.test_seal import assert_test_access_allowed


class FormalRunnerError(RuntimeError):
    """Raised before any cloud training side effect when a contract is open."""


def _formal_loader_optimized(execution_scale: str) -> bool:
    """Return the approved data/GPU loader mode for a formal cloud scale.

    Only the local synthetic smoke lane is allowed to use a zero-worker CPU
    loader.  Every cloud formal scale, including strengthening, must retain
    the measured 3090 pipeline contract (workers, pinned memory, persistent
    workers, prefetch and AMP).
    """

    if execution_scale == "smoke":
        return False
    if execution_scale not in {
        "baseline", "screening", "strengthening", "confirmation",
        "acceptance", "extension", "cloud",
    }:
        raise FormalRunnerError(f"unsupported formal loader scale: {execution_scale}")
    return True


def _validate_formal_horizon(*, execution_scale: str, epochs: int, rapid_horizon_epochs: int) -> None:
    """Validate formal and rapid horizon parity without silently overtraining screening rows."""

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise FormalRunnerError("epochs must be positive")
    if isinstance(rapid_horizon_epochs, bool) or not isinstance(rapid_horizon_epochs, int) or rapid_horizon_epochs <= 0 or rapid_horizon_epochs > epochs:
        raise FormalRunnerError("rapid_horizon_epochs must be a positive integer no greater than epochs")
    if rapid_horizon_epochs == epochs and execution_scale != "screening":
        raise FormalRunnerError("rapid_horizon_epochs may equal epochs only for screening")


def _cosine_warmup_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def validate_formal_evaluate_paths(
    *,
    data_manifest: str | Path,
    audit_report: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    execution_scale: str = "cloud",
) -> dict[str, str]:
    """Validate an evaluation request without reading data, weights, or devices.

    This is the shared preflight used by ``scripts/evaluate.py``.  It deliberately
    validates only immutable cloud-path shape and test-seal semantics; the formal
    runner remains responsible for opening the artifacts after an authorized
    cloud control has passed.
    """

    if execution_scale not in {"cloud", "baseline", "screening", "strengthening", "confirmation", "acceptance", "extension"}:
        raise FormalRunnerError("formal evaluation requires a non-test cloud execution scale")
    assert_test_access_allowed(
        {"execution_scale": execution_scale, "test_seal_status": "sealed"}, "validation"
    )
    paths = {
        "data_manifest": _cloud_artifact(data_manifest, "data_manifest"),
        "audit_report": _cloud_artifact(audit_report, "audit_report"),
        "checkpoint": _cloud_artifact(checkpoint, "checkpoint"),
        "output_dir": _cloud_artifact(output_dir, "output_dir"),
    }
    if paths["data_manifest"] == paths["audit_report"] or paths["data_manifest"] == paths["checkpoint"]:
        raise FormalRunnerError("data manifest, audit report and checkpoint must be distinct artifacts")
    if paths["output_dir"] in {paths["data_manifest"], paths["audit_report"], paths["checkpoint"]}:
        raise FormalRunnerError("output_dir must be a dedicated cloud directory")
    if paths["data_manifest"].suffix.casefold() not in {chr(46) + "json", chr(46) + "jsonl"}:
        raise FormalRunnerError("data_manifest must be a JSON/JSONL artifact")
    if paths["audit_report"].suffix.casefold() != chr(46) + "json":
        raise FormalRunnerError("audit_report must be a JSON artifact")
    if paths["checkpoint"].suffix.casefold() not in {chr(46) + "pt", chr(46) + "pth", chr(46) + "ckpt"}:
        raise FormalRunnerError("checkpoint must be a PyTorch checkpoint artifact")
    return {name: str(path) for name, path in paths.items()}


def _cloud_artifact(value: str | Path, name: str) -> Path:
    # Validate the remote POSIX spelling lexically.  ``Path`` on
    # a Windows authoring host becomes a drive-qualified path and would incorrectly
    # reject a valid cloud argument before it ever reaches the cloud runner.
    raw = str(value)
    path_posix = PurePosixPath(raw)
    if not raw.startswith(chr(47)) or "\\" in raw or not path_posix.is_absolute():
        raise FormalRunnerError(f"{name} must be an absolute POSIX cloud path")
    parts = path_posix.parts
    if ".." in parts or "." in parts or raw != path_posix.as_posix():
        raise FormalRunnerError(f"{name} must use a canonical POSIX cloud path")
    allowed = len(parts) >= 3 and parts[1] == "root" and parts[2] in {"autodl-tmp", "autodl-workspace"}
    if not allowed:
        raise FormalRunnerError(f"{name} is outside the declared cloud roots")
    return PurePosixPath(raw) if os.name == "nt" else Path(raw)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True)
        handle.flush()
        temp = Path(handle.name)
    temp.replace(path)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        temp = Path(handle.name)
    temp.replace(path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    torch.save(value, temp)
    temp.replace(path)


def _checkpoint_payload(
    *,
    model_state: Mapping[str, Tensor],
    epoch: int,
    run_manifest: Mapping[str, Any],
    checkpoint_role: str,
    optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a checkpoint with truthful epoch and role metadata."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise FormalRunnerError("checkpoint epoch must be a positive integer")
    if checkpoint_role not in {"best_validation", "final", "best_restored_final"}:
        raise FormalRunnerError("checkpoint_role is not recognized")
    payload: dict[str, Any] = {
        "model": dict(model_state),
        "epoch": epoch,
        "run_manifest": dict(run_manifest),
        "checkpoint_role": checkpoint_role,
    }
    if optimizer_state is not None:
        payload["optimizer"] = dict(optimizer_state)
    return payload


def _file_identity(path: Path) -> dict[str, Any]:
    """Return a content identity for a completed checkpoint artifact."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _step(
    model: nn.Module,
    batch: Mapping[str, Tensor],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    clip_norm: float,
    scaler: torch.cuda.amp.GradScaler | None = None,
    backward: bool = False,
    accumulation_divisor: int = 1,
    step_optimizer: bool = True,
    objective_name: str = "pixel_ce",
    collect_loss_details: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    normalized = croma_dynamic_normalize_batch(batch)
    optical = normalized["optical"].to(device, non_blocking=True)
    sar = normalized["sar"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    amp_enabled = scaler is not None and device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        model_output = model(optical, sar)
        logits = model_output
        if objective_name == "pixel_ce":
            loss = segmentation_cross_entropy(logits, target)
            loss_details: dict[str, Tensor] = {}
        else:
            loss, loss_details = segmentation_objective(
                logits, target, objective_name=objective_name
            )
    if optimizer is not None and backward:
        scaled_loss = loss / accumulation_divisor
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
            if step_optimizer:
                scaler.unscale_(optimizer)
        else:
            scaled_loss.backward()
        if step_optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    if collect_loss_details:
        return loss.detach(), logits.detach(), target, loss_details
    return loss.detach(), logits.detach(), target


def _validate(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    classes: int,
    objective_name: str = "pixel_ce",
) -> dict[str, Any]:
    model.eval()
    matrix: Tensor | None = None
    losses: list[Tensor] = []
    per_class_ce_total = torch.zeros(classes, dtype=torch.float64, device=device)
    per_class_grad_total = torch.zeros(classes, dtype=torch.float64, device=device)
    per_class_counts = torch.zeros(classes, dtype=torch.float64, device=device)
    per_class_batch_count = torch.zeros(classes, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            if objective_name == "pixel_ce":
                loss, logits, target = _step(
                    model, batch, device=device, optimizer=None, clip_norm=1.0,
                    objective_name=objective_name,
                )
                details = None
            else:
                loss, logits, target, details = _step(
                    model, batch, device=device, optimizer=None, clip_norm=1.0,
                    objective_name=objective_name, collect_loss_details=True,
                )
            losses.append(loss)
            if details is not None:
                present = details["per_class_present"].to(torch.bool)
                counts = details["per_class_pixel_count"].to(torch.float64)
                per_class_counts += counts
                per_class_batch_count += present.to(torch.float64)
                per_class_ce_total += details["per_class_ce"].to(torch.float64) * present * counts.clamp_min(1.0)
                per_class_grad_total += details["per_class_gradient_contribution"].to(torch.float64) * present
            batch_matrix = confusion_matrix(logits, target, classes)
            matrix = batch_matrix if matrix is None else matrix + batch_matrix
    if matrix is None:
        matrix = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    matrix = matrix.cpu()
    score = mean_iou(matrix)
    total = int(matrix.sum())
    oa = float(matrix.diag().sum().item() / total) if total else float("nan")
    loss_value = float(torch.stack(losses).mean().item()) if losses else float("nan")
    per_class_iou = []
    for class_index in range(classes):
        denominator = matrix[class_index, :].sum() + matrix[:, class_index].sum() - matrix[class_index, class_index]
        per_class_iou.append(float(matrix[class_index, class_index].item() / denominator.item()) if denominator.item() else None)
    present_count = int((matrix.sum(dim=1) + matrix.sum(dim=0) > 0).sum().item())
    valid_classes = [index for index, value in enumerate(per_class_iou) if value is not None]
    ordered = sorted(valid_classes, key=lambda index: int(matrix[index, :].sum().item()), reverse=True)
    frequent = ordered[:4]
    rare = ordered[4:]
    result: dict[str, Any] = {
        "loss": loss_value,
        "mIoU": float(score),
        "OA": oa,
        "per_class_iou": per_class_iou,
        "per_class_pixel_count": [int(value.item()) for value in matrix.sum(dim=1)],
        "valid_class_count": present_count,
        "frequent_macro_IoU": float(sum(per_class_iou[i] for i in frequent) / len(frequent)) if frequent else None,
        "rare_macro_IoU": float(sum(per_class_iou[i] for i in rare) / len(rare)) if rare else None,
    }
    if objective_name != "pixel_ce":
        result.update({
            "per_class_mean_ce": [
                float(per_class_ce_total[i].item() / max(1.0, float(per_class_counts[i].item())))
                if bool(per_class_counts[i] > 0) else None
                for i in range(classes)
            ],
            "per_class_gradient_contribution": [
                float(per_class_grad_total[i].item() / max(1.0, float(per_class_batch_count[i].item())))
                if bool(per_class_batch_count[i] > 0) else None
                for i in range(classes)
            ],
        })
    return result


def run_formal_cloud(
    *,
    code_root: Path,
    resolved: Mapping[str, Any],
    data_manifest: str | Path,
    audit_report: str | Path,
    initialization: Mapping[str, Any],
    output_dir: str | Path,
    mechanism_set: str,
    execution_scale: str,
    epochs: int,
    rapid_horizon_epochs: int = 5,
    device: str = "cuda:0",
    candidate_direction_id: str | None = None,
    model: nn.Module | None = None,
    train_loader: Any | None = None,
    validation_loader: Any | None = None,
    croma_constructor: Callable[[], nn.Module] | None = None,
    allow_injected_fixture: bool = False,
    objective_name: str | None = None,
    backbone_execution: str = "full",
) -> dict[str, Any]:
    """Run one matched baseline/candidate row; test remains sealed."""

    if execution_scale not in {"baseline", "screening", "strengthening", "confirmation", "acceptance", "extension"}:
        raise FormalRunnerError("formal runner requires a non-test approved execution scale")
    if str(backbone_execution).strip().casefold() not in {"full", "ice_exact"}:
        raise FormalRunnerError("backbone_execution must be full or ice_exact")
    _validate_formal_horizon(execution_scale=execution_scale, epochs=epochs, rapid_horizon_epochs=rapid_horizon_epochs)
    assert_test_access_allowed({"execution_scale": execution_scale, "test_seal_status": "sealed"}, "validation")
    data_path = _cloud_artifact(data_manifest, "data_manifest")
    audit_path = _cloud_artifact(audit_report, "audit_report")
    out = _cloud_artifact(output_dir, "output_dir")
    if resolved.get("model", {}).get("mechanism_set") != mechanism_set:
        raise FormalRunnerError("resolved mechanism_set differs from requested row")
    objective_spec = resolved.get("objective", {})
    declared_objective = (
        objective_spec.get("id")
        if isinstance(objective_spec, Mapping)
        else resolved.get("runtime", {}).get("objective_name", "pixel_ce")
    )
    objective = str(objective_name if objective_name is not None else declared_objective).strip().casefold()
    if objective not in {"pixel_ce", "macro_ce", "ce_lovasz", "macro_ce_lovasz"}:
        raise FormalRunnerError("unsupported V12 objective")
    if candidate_direction_id is not None and str(candidate_direction_id).strip() != "BASELINE":
        raise FormalRunnerError("candidate_direction_id is not an approved formal direction")
    resolved_init = resolved.get("initialization")
    if not isinstance(resolved_init, Mapping) or resolved_init.get("mode") != initialization.get("mode"):
        raise FormalRunnerError("resolved and requested initialization modes differ")
    mode = initialization.get("mode")
    if mode not in {"random_init", "pretrained"}:
        raise FormalRunnerError("formal route requires pretrained or random_init initialization")
    if initialization.get("same_initialization_for_baseline_and_innovation") is not True:
        raise FormalRunnerError("baseline and candidate initialization parity is not declared")
    if mode == "random_init" and (initialization.get("checkpoint_path") is not None or initialization.get("pretrained_eligible") is not False):
        raise FormalRunnerError("random_init cannot declare a checkpoint or pretrained eligibility")
    if mode == "pretrained" and (not initialization.get("checkpoint_path") or initialization.get("pretrained_eligible") is not True):
        raise FormalRunnerError("pretrained mode requires an eligible checkpoint")
    if PurePosixPath(str(resolved_init.get("audit_ref", ""))).name != PurePosixPath(str(audit_path)).name:
        raise FormalRunnerError("resolved and requested initialization audit references differ")
    if resolved_init.get("constructor_ref") != initialization.get("constructor_ref") or resolved_init.get("constructor_kwargs") != initialization.get("constructor_kwargs") or resolved_init.get("checkpoint_path") != initialization.get("checkpoint_path"):
        raise FormalRunnerError("resolved and requested initialization identity differs")
    if model is not None and not allow_injected_fixture:
        raise FormalRunnerError("injected models are restricted to explicit synthetic fixtures")
    seed = int(resolved["runtime"]["seed"])
    torch.manual_seed(seed)
    optimized_loader = _formal_loader_optimized(execution_scale)
    loader_contract = {
        "num_workers": 4 if optimized_loader else 0,
        "pin_memory": bool(optimized_loader),
        "persistent_workers": bool(optimized_loader),
        "prefetch_factor": 2,
        "non_blocking_device_copy": True,
        "amp": bool(optimized_loader and str(resolved["runtime"].get("precision", "")).casefold() == "amp"),
    }
    runtime = resolved["runtime"]
    accumulation_steps = int(runtime.get("gradient_accumulation", 1))
    if accumulation_steps <= 0:
        raise FormalRunnerError("gradient_accumulation must be positive")
    augmentation = runtime.get("augmentation")
    augmentation_seed = int(runtime.get("seed", 0))
    if train_loader is None:
        train_loader, manifest = build_sen12ts_loader(
            data_path, split="train", batch_size=int(resolved["runtime"]["micro_batch"]),
            num_workers=loader_contract["num_workers"], execution_scale=execution_scale,
            pin_memory=loader_contract["pin_memory"],
            persistent_workers=loader_contract["persistent_workers"],
            prefetch_factor=loader_contract["prefetch_factor"],
            augmentation=augmentation,
            seed=augmentation_seed,
        )
    else:
        manifest = {"dataset_id": resolved.get("dataset_id"), "test_accessed": False}
    if validation_loader is None:
        validation_loader, _ = build_sen12ts_loader(
            data_path, split="validation", batch_size=int(resolved["runtime"]["micro_batch"]),
            num_workers=loader_contract["num_workers"], execution_scale=execution_scale,
            pin_memory=loader_contract["pin_memory"],
            persistent_workers=loader_contract["persistent_workers"],
            prefetch_factor=loader_contract["prefetch_factor"],
            augmentation=None,
            seed=augmentation_seed,
        )
    if model is None:
        try:
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FormalRunnerError("cannot read pretrained audit for dataset compatibility") from exc
        if not isinstance(audit_payload, Mapping):
            raise FormalRunnerError("pretrained audit must be a mapping")
        try:
            cross_validate_dataset_and_pretrained(manifest, audit_payload)
        except ValueError as exc:
            raise FormalRunnerError("dataset/pretrained compatibility check failed") from exc
    if model is None:
        backbone, load_report = load_croma_backbone(
            initialization=initialization, audit_path=audit_path,
            constructor_ref=str(initialization.get("constructor_ref", "")),
            constructor=croma_constructor,
        )
        model = build_vfm_segmentation_model(
            resolved,
            audited_croma_backbone=backbone,
            backbone_execution=backbone_execution,
        )
    else:
        load_report = {"injected_model": True}
    target_device = torch.device(device)
    model.to(target_device)
    if target_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=loader_contract["amp"] and target_device.type == "cuda")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(resolved["runtime"]["optimizer"]["learning_rate"]),
        weight_decay=float(resolved["runtime"]["optimizer"]["weight_decay"]),
        betas=tuple(float(x) for x in resolved["runtime"]["optimizer"]["betas"]),
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
    total_optimizer_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(math.ceil(
        float(resolved["runtime"]["scheduler"]["warmup_fraction"]) * total_optimizer_steps
    ))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup_multiplier(
            step, total_steps=total_optimizer_steps, warmup_steps=warmup_steps,
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    early_stopping = resolved["runtime"].get("early_stopping", {})
    if not isinstance(early_stopping, Mapping):
        raise FormalRunnerError("early_stopping must be a mapping")
    requested_early_enabled = bool(early_stopping.get("enabled", False))
    # Rapid screening is a fixed-horizon comparison.  It must complete exactly
    # T_rapid and therefore cannot trigger the formal burn-in/patience rule
    # when T_rapid (5) is shorter than the formal burn-in (8).
    early_enabled = requested_early_enabled and execution_scale != "screening"
    if execution_scale == "screening" and requested_early_enabled:
        early_stopping = dict(early_stopping)
        early_stopping["effective_enabled"] = False
        early_stopping["disabled_reason"] = "fixed_rapid_horizon"
    burn_in_epochs = int(early_stopping.get("burn_in_epochs", epochs + 1))
    patience = int(early_stopping.get("patience", epochs + 1))
    min_delta_fraction = float(early_stopping.get("min_delta_percentage_points", 0.0)) / 100.0
    restore_best = bool(early_stopping.get("restore_best_checkpoint", False))
    if early_enabled and (
        early_stopping.get("monitor") != "validation.mIoU"
        or early_stopping.get("mode") != "max"
        or burn_in_epochs <= 0
        or burn_in_epochs >= epochs
        or patience <= 0
        or min_delta_fraction <= 0.0
        or not restore_best
    ):
        raise FormalRunnerError("early stopping configuration is not the amended validation-mIoU contract")
    history: list[dict[str, Any]] = []
    train_class_ce_total = torch.zeros(
        int(resolved["model"]["num_classes"]), dtype=torch.float64, device=target_device
    )
    train_class_grad_total = torch.zeros(
        int(resolved["model"]["num_classes"]), dtype=torch.float64, device=target_device
    )
    train_class_pixel_total = torch.zeros(
        int(resolved["model"]["num_classes"]), dtype=torch.float64, device=target_device
    )
    train_class_batch_total = torch.zeros(
        int(resolved["model"]["num_classes"]), dtype=torch.float64, device=target_device
    )
    rapid_checkpoint: Path | None = None
    best_state: dict[str, Tensor] | None = None
    best_validation: dict[str, float] | None = None
    best_score = float("-inf")
    best_epoch = 0
    bad_epochs = 0
    stopped_epoch = epochs
    for epoch in range(epochs):
        model.train()
        train_losses: list[Tensor] = []
        epoch_mechanism_telemetry: dict[str, float] = {}
        batch_count = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            should_step = ((batch_index + 1) % accumulation_steps == 0) or (batch_index + 1 == batch_count)
            if objective == "pixel_ce":
                loss, _, _ = _step(
                    model, batch, device=target_device, optimizer=optimizer,
                    clip_norm=float(resolved["runtime"]["gradient_clip_norm"]),
                    scaler=scaler,
                    backward=True,
                    accumulation_divisor=accumulation_steps,
                    step_optimizer=should_step,
                    objective_name=objective,
                )
            else:
                loss, _, _, details = _step(
                    model, batch, device=target_device, optimizer=optimizer,
                    clip_norm=float(resolved["runtime"]["gradient_clip_norm"]),
                    scaler=scaler,
                    backward=True,
                    accumulation_divisor=accumulation_steps,
                    step_optimizer=should_step,
                    objective_name=objective,
                    collect_loss_details=True,
                )
                present = details["per_class_present"].to(torch.float64)
                counts = details["per_class_pixel_count"].to(torch.float64)
                train_class_ce_total += details["per_class_ce"].to(torch.float64) * counts * present
                train_class_grad_total += details["per_class_gradient_contribution"].to(torch.float64) * present
                train_class_pixel_total += counts
                train_class_batch_total += present
            train_losses.append(loss)
            if should_step:
                scheduler.step()
        validation = _validate(
            model, validation_loader, device=target_device,
            classes=int(resolved["model"]["num_classes"]),
            objective_name=objective,
        )
        train_loss = float(torch.stack(train_losses).mean().item()) if train_losses else float("nan")
        history_entry: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation": validation,
            "objective_name": objective,
        }
        if epoch_mechanism_telemetry:
            history_entry["mechanism_telemetry"] = epoch_mechanism_telemetry
        history.append(history_entry)
        if epoch + 1 == rapid_horizon_epochs:
            rapid_checkpoint = (out / f"{mechanism_set}_seed{resolved['runtime']['seed']}_rapid_epoch{rapid_horizon_epochs}").with_suffix("".join([chr(46), "pt"]))
            _atomic_torch(rapid_checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": rapid_horizon_epochs})
        score = float(validation["mIoU"])
        improved = best_state is None or score > best_score + min_delta_fraction
        if improved:
            best_score = score
            best_epoch = epoch + 1
            best_validation = dict(validation)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            bad_epochs = 0
        elif early_enabled and epoch + 1 >= burn_in_epochs:
            bad_epochs += 1
        if early_enabled and epoch + 1 >= burn_in_epochs and bad_epochs >= patience:
            stopped_epoch = epoch + 1
            break
    best_was_restored = bool(early_enabled and restore_best and best_state is not None)
    if best_was_restored:
        model.load_state_dict(best_state, strict=True)
    elif best_validation is None and history:
        best_validation = dict(history[-1]["validation"])
    telemetry_path: Path | None = None
    telemetry_identity: dict[str, Any] | None = None
    telemetry_rows = [
        {
            "schema_version": "geotoken3path.baseline_telemetry.v1",
            "epoch": int(entry["epoch"]),
            "split": "train_first_batch",
            "mechanism_set": mechanism_set,
            **dict(entry["mechanism_telemetry"]),
        }
        for entry in history
        if "mechanism_telemetry" in entry
    ]
    if telemetry_rows:
        telemetry_path = out / "baseline_telemetry.jsonl"
        _atomic_jsonl(telemetry_path, telemetry_rows)
        telemetry_identity = _file_identity(telemetry_path)
    resolved_snapshot = json.loads(json.dumps(dict(resolved), sort_keys=True))
    resolved_snapshot["backbone_execution"] = str(backbone_execution)
    resolved_snapshot.setdefault("runtime", {})["data_loader"] = loader_contract
    resolved_snapshot["runtime"]["rapid_horizon_epochs"] = rapid_horizon_epochs
    run_manifest = build_run_manifest(
        resolved_snapshot,
        seed=seed,
        split="validation",
        execution_scale=execution_scale,
        candidate_direction_id=candidate_direction_id,
        data_manifest_ref=str(data_path),
        pretrained_audit_ref=str(audit_path),
        telemetry_ref=str(telemetry_path) if telemetry_path is not None else None,
        telemetry_sha256=(
            str(telemetry_identity["sha256"])
            if telemetry_identity is not None
            else None
        ),
    )
    resolved_snapshot["formal_data_manifest"] = str(data_path)
    resolved_snapshot["formal_audit_report"] = str(audit_path)
    resolved_snapshot["run_manifest"] = run_manifest
    if best_state is None or best_epoch <= 0:
        raise FormalRunnerError("formal run did not produce a best-validation state")
    best_checkpoint = (
        out / f"{mechanism_set}_seed{resolved['runtime']['seed']}_best_epoch{best_epoch}"
    ).with_suffix("".join([chr(46), "pt"]))
    _atomic_torch(
        best_checkpoint,
        _checkpoint_payload(
            model_state=best_state,
            epoch=best_epoch,
            run_manifest=run_manifest,
            checkpoint_role="best_validation",
        ),
    )
    checkpoint = (out / f"{mechanism_set}_seed{resolved['runtime']['seed']}").with_suffix("".join([chr(46), "pt"]))
    final_epoch = best_epoch if best_was_restored else stopped_epoch
    final_role = "best_restored_final" if best_was_restored else "final"
    _atomic_torch(
        checkpoint,
        _checkpoint_payload(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=final_epoch,
            run_manifest=run_manifest,
            checkpoint_role=final_role,
        ),
    )
    best_checkpoint_identity = _file_identity(best_checkpoint)
    final_checkpoint_identity = _file_identity(checkpoint)
    _atomic_json(out / "resolved_config.json", resolved_snapshot)
    result = {
        "status": "formal_cloud_train_validation_complete",
        "scientific_result": True,
        "mechanism_set": mechanism_set,
        "candidate_direction_id": candidate_direction_id,
        "max_epochs": epochs,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "early_stopping": dict(early_stopping),
        "scheduler": {
            "name": "cosine_with_warmup",
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "optimizer_steps_per_epoch": steps_per_epoch,
        },
        "rapid_horizon_epochs": rapid_horizon_epochs,
        "rapid_checkpoint": str(rapid_checkpoint) if rapid_checkpoint else None,
        "best_validation": best_validation,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": final_epoch,
        "checkpoint_role": final_role,
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_epoch": best_epoch,
        "best_checkpoint_role": "best_validation",
        "best_checkpoint_identity": best_checkpoint_identity,
        "final_checkpoint_identity": final_checkpoint_identity,
        "resolved_config": str(out / "resolved_config.json"),
        "run_manifest": run_manifest,
        "croma_load": load_report,
        "dataset_id": manifest.get("dataset_id"),
        "objective_name": objective,
        "mechanism_telemetry": [
            {
                "epoch": entry["epoch"],
                **dict(entry["mechanism_telemetry"]),
            }
            for entry in history
            if "mechanism_telemetry" in entry
        ],
        "telemetry_identity": telemetry_identity,
        "objective_diagnostics": {
            "aggregation": "train frequency averaged per completed epoch; CE and gradient diagnostics weighted/averaged over observed batches",
            "completed_epochs": len(history),
            "train_pixel_frequency": [
                int(round((value / max(1, len(history))).item()))
                for value in train_class_pixel_total
            ],
            "train_per_class_mean_ce": [
                float(train_class_ce_total[i].item() / max(1.0, train_class_pixel_total[i].item()))
                if train_class_pixel_total[i].item() > 0 else None
                for i in range(len(train_class_pixel_total))
            ],
            "train_per_class_gradient_contribution": [
                float(train_class_grad_total[i].item() / max(1.0, train_class_batch_total[i].item()))
                if train_class_batch_total[i].item() > 0 else None
                for i in range(len(train_class_batch_total))
            ],
            "validation": {
                "per_class_pixel_frequency": validation.get("per_class_pixel_count"),
                "per_class_mean_ce": validation.get("per_class_mean_ce"),
                "per_class_gradient_contribution": validation.get("per_class_gradient_contribution"),
                "frequent_macro_IoU": validation.get("frequent_macro_IoU"),
                "rare_macro_IoU": validation.get("rare_macro_IoU"),
            },
        },
    }
    _atomic_json(out / "run_result.json", result)
    return result
