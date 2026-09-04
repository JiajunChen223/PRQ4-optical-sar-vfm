"""Cloud-side certification for R24 SkySense++ S2 ICE-Exact.

The certifier never accesses the sealed test split.  It builds two models from
the same audited SkySense++ S2 safetensors checkpoint -- a Full
``SkySensePPSegmentationModel`` that always runs the official 24-layer
forward, and an ICE receiver that wraps an independent copy of the backbone
behind ``SkySensePPPrefixExecutor`` so its forward is the *official vendor
forward truncated* at the compiled plan bound (contract "b": layers 0..11).
Both models share the same trained segmentation head checkpoint.  Full and
ICE are compared at five levels:

  1. state surface (identical parameter names and values);
  2. retained feature maps in FP32 on the first validation batch (per official
     grid layer, ``compare_tensors`` max abs error <= 1e-6);
  3. FP32 logits on the same batch (<= 1e-6);
  4. trainable gradients on one fixed train batch (head-only, RNG reset
     between the two runs, ``compare_gradients`` <= 1e-6 and loss parity);
  5. the complete validation split under the protocol AMP precision
     (pixel-exact argmax agreement, identical confusion matrices and mIoU
     delta < 1e-12).

Everything downstream consumes the shared SEN12TS loader with the frozen R24
10-band selection and per-micro-batch normalization
(``data/skysensepp.croma_dynamic_normalize_batch_r24``); the annotation
channel is derived from the ground-truth WorldCover target (documented GT
leakage through the semantic-annotation channel, 255 mapped to vocabulary
zero).  The certificate is written as ``prq4.skysensepp_ice_exact_certificate.v1``.
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

from ..data.sen12ts import build_sen12ts_loader
from ..data.skysensepp import (
    annotation_from_target,
    croma_dynamic_normalize_batch_r24,
)
from ..execution.certification import (
    compare_gradients,
    compare_tensors,
    named_trainable_gradients,
)
from ..execution.skysensepp_plan import (
    SkySensePPExecutionPlan,
    compile_skysensepp_plan,
    validate_skysensepp_contract,
)
from ..losses import segmentation_objective
from ..metrics import confusion_matrix, mean_iou
from ..models.skysensepp_executor import SkySensePPExecutionError, SkySensePPPrefixExecutor
from ..models.skysensepp_seg import (
    SkySensePPImportError,
    SkySensePPSegmentationModel,
    build_skysensepp_model,
    load_skysensepp_weights,
    load_vendor_config,
)
from ..utils.test_seal import assert_test_access_allowed

# Official SkySense++ S2 out-grid (layer indices of the four hierarchical maps).
SKYSENSEPP_S2_ROUTE = "R24-SKYSENSEPP-S2-ICE-01"
SKYSENSEPP_S2_OUT_GRID: tuple[int, ...] = (5, 11, 17, 23)


class SkySensePPCertificationError(RuntimeError):
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


def _state_surface_equal(full: nn.Module, ice: nn.Module) -> tuple[bool, list[str]]:
    """Both models must expose identical parameter names with identical values."""
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


def _head_checkpoint_state(path: Path) -> dict[str, Tensor]:
    """Load the trained-head checkpoint (keys relative to the head module)."""
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SkySensePPCertificationError("cannot load skysensepp head checkpoint") from exc
    if not isinstance(payload, Mapping):
        raise SkySensePPCertificationError("skysensepp head checkpoint must contain a mapping")
    state = payload.get("head_state", payload.get("model", payload))
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise SkySensePPCertificationError("skysensepp head checkpoint state is invalid")
    return dict(state)  # type: ignore[return-value]


def _apply_head_checkpoint(model: nn.Module, head_state: Mapping[str, Tensor]) -> None:
    """Strictly load trained-head weights into a model's segmentation head."""
    incompatible = model.head.load_state_dict(head_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SkySensePPCertificationError(
            "head checkpoint did not load strictly: "
            f"{len(incompatible.missing_keys)} missing, {len(incompatible.unexpected_keys)} unexpected"
        )


class SkysenseppIceExecutionModel(nn.Module):
    """ICE receiver over an independent backbone copy behind the prefix executor.

    The module holds its own backbone (audited SkySense++ weights, frozen) and
    segmentation head, and executes the *official vendor forward truncated* at
    ``plan.max_layer`` through ``SkySensePPPrefixExecutor`` -- the same
    arithmetic as the physically compact export artifact.  ``forward`` returns
    logits at the input pixel resolution plus the executed feature maps.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        head: nn.Module,
        plan: SkySensePPExecutionPlan,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch module")
        if not isinstance(head, nn.Module):
            raise TypeError("head must be a torch module")
        if not isinstance(plan, SkySensePPExecutionPlan):
            raise TypeError("plan must be a SkySensePPExecutionPlan")
        self.backbone = backbone
        self.head = head
        self.plan = plan

    def forward(
        self,
        pixel_values: Tensor,
        annotation: Tensor,
        *,
        max_layer: int | None = None,
    ) -> dict[str, Any]:
        try:
            executor = SkySensePPPrefixExecutor(self.plan)
        except Exception as exc:  # pragma: no cover - construction is validated
            raise SkySensePPCertificationError("ICE plan is invalid") from exc
        try:
            result = executor.execute(
                self.backbone,
                pixel_values=pixel_values,
                annotation=annotation,
                max_layer=max_layer,
            )
        except SkySensePPExecutionError as exc:
            raise SkySensePPCertificationError(str(exc)) from exc
        maps = result["feature_maps"]
        head_maps = list(maps)
        if self.plan.contract == "b" and len(maps) == 2:
            # Contract "b" consumes only the deepest executed map; feed the
            # head a full-width map tuple so its index contract is stable
            # (identical to the export artifact's forward).
            head_maps = [maps[0], maps[1], maps[1], maps[1]]
        if not isinstance(pixel_values, Tensor) or pixel_values.ndim != 4:
            raise SkySensePPCertificationError("pixel_values must be [B,C,H,W]")
        logits = self.head(
            head_maps,
            output_size=(int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
        )
        return {
            "logits": logits,
            "feature_maps": tuple(maps),
            "layer_indices": tuple(result["layer_indices"]),
            "executed_layer_count": int(result["executed_layer_count"]),
        }


def build_skysensepp_certification_pair(
    *,
    safetensors_path: str | Path | None = None,
    contract: str = "b",
    resolution: int = 120,
    micro_batch: int = 16,
    head_checkpoint: str | Path | None = None,
) -> tuple[SkySensePPSegmentationModel, SkysenseppIceExecutionModel, dict[str, Any]]:
    """Build the Full/ICE pair from identical audited weights and head state.

    ``safetensors_path=None`` keeps both backbones randomly initialized
    (unit-test surface only); the cloud entry points always pass the audited
    checkpoint.  ``head_checkpoint=None`` keeps the deterministically seeded
    random head.  The returned ICE model exposes ``plan`` and executes the
    official truncated forward; its state surface is asserted equal to Full.
    """
    validate_skysensepp_contract(contract)
    if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch <= 0:
        raise SkySensePPCertificationError("micro_batch must be a positive integer")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise SkySensePPCertificationError("resolution must be a positive integer")
    plan = compile_skysensepp_plan(contract)

    if safetensors_path is None:
        # Random-init lane (unit tests only): vendor post_init draws from the
        # global RNG, so both hosts must be constructed from the same snapshot
        # to share an identical random backbone.  The cloud entry points always
        # pass the audited safetensors checkpoint and never reach this lane.
        random_snapshot = torch.get_rng_state()

        def _random_host() -> SkySensePPSegmentationModel:
            torch.set_rng_state(random_snapshot)
            host = SkySensePPSegmentationModel(
                config_dict=load_vendor_config(),
                contract=contract,
                num_classes=11,
                head_seed=0,
            )
            host.freeze_backbone()
            return host

        full_model = _random_host()
        ice_host = _random_host()
    else:
        weights_path_arg = Path(safetensors_path)
        if not weights_path_arg.is_file():
            raise SkySensePPCertificationError(
                f"skysensepp safetensors checkpoint not found: {weights_path_arg}"
            )
        full_model = build_skysensepp_model(
            contract=contract,
            safetensors_path=str(weights_path_arg),
            num_classes=11,
            seed=0,
        )
        # Second independent instance of the audited backbone + same head recipe.
        ice_host = SkySensePPSegmentationModel(
            config_dict=load_vendor_config(),
            contract=contract,
            num_classes=11,
            head_seed=0,
        )
        weights_report = load_skysensepp_weights(ice_host, str(weights_path_arg))
        if weights_report["missing"] or weights_report["unexpected"]:
            raise SkySensePPCertificationError(
                "skysensepp checkpoint did not load strictly into the ICE host: "
                f"{len(weights_report['missing'])} missing, "
                f"{len(weights_report['unexpected'])} unexpected"
            )
        ice_host.freeze_backbone()

    ice_model = SkysenseppIceExecutionModel(
        backbone=ice_host.backbone,
        head=ice_host.head,
        plan=plan,
    )

    head_checkpoint_sha: str | None = None
    if head_checkpoint is not None:
        checkpoint_path = Path(head_checkpoint)
        if not checkpoint_path.is_absolute():
            raise SkySensePPCertificationError("head checkpoint must be an absolute cloud path")
        if not checkpoint_path.is_file():
            raise SkySensePPCertificationError(
                f"skysensepp head checkpoint not found: {checkpoint_path}"
            )
        head_state = _head_checkpoint_state(checkpoint_path)
        _apply_head_checkpoint(full_model, head_state)
        _apply_head_checkpoint(ice_model, head_state)
        head_checkpoint_sha = _sha256(checkpoint_path)

    state_equal, unequal_names = _state_surface_equal(full_model, ice_model)
    safetensors_sha = None
    if safetensors_path is not None:
        safetensors_sha = _sha256(Path(safetensors_path))
    metadata = {
        "contract": contract,
        "plan_sha256": plan.plan_sha256,
        "max_layer": plan.max_layer,
        "executed_layer_count": plan.executed_layer_count,
        "eliminated_layers": list(plan.eliminated_layers),
        "required_output_indices": list(plan.required_output_indices),
        "annotation_source": "gt_worldcover_leakage_documented",
        "resolution": int(resolution),
        "micro_batch": int(micro_batch),
        "num_classes": int(full_model.num_classes),
        "safetensors_sha256": safetensors_sha,
        "head_checkpoint_sha256": head_checkpoint_sha,
        "head_checkpoint_applied": head_checkpoint is not None,
        "state_surface_equal": state_equal,
        "state_surface_unequal_names": unequal_names,
    }
    return full_model, ice_model, metadata


def _prepare_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
    *,
    micro_batch: int = 16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize the R24 10-band optical batch and derive annotation + target."""
    normalized = croma_dynamic_normalize_batch_r24(batch, micro_batch=micro_batch)
    target = batch["target"].to(device, non_blocking=True)
    optical10 = normalized["optical10"].to(device, non_blocking=True)
    annotation = annotation_from_target(target).to(device, non_blocking=True)
    return optical10, annotation, target


def _loaders(
    data_manifest: str | Path,
    *,
    execution_scale: str,
    micro_batch: int,
    seed: int,
    augmentation: Mapping[str, Any] | None,
    num_workers: int = 4,
):
    loader_kwargs = {
        "batch_size": micro_batch,
        "num_workers": num_workers,
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
        augmentation=augmentation,
        **loader_kwargs,
    )
    return validation_loader, train_loader


def run_skysensepp_ice_certification(
    *,
    resolved: Mapping[str, Any],
    initialization: Mapping[str, Any],
    data_manifest: str | Path,
    safetensors_path: str | Path,
    head_checkpoint: str | Path,
    output_path: str | Path,
    device: str = "cuda:0",
    execution_scale: str = "acceptance",
    objective_name: str = "ce_lovasz",
    expected_miou_percent: float | None = None,
) -> dict[str, Any]:
    """Run the R24 five-level Full/ICE equivalence certification on CUDA."""
    assert_test_access_allowed(
        {"execution_scale": execution_scale, "test_seal_status": "sealed"},
        "validation",
    )
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise SkySensePPCertificationError("formal skysensepp ICE certification requires CUDA")
    if objective_name != "ce_lovasz":
        raise SkySensePPCertificationError("R24 certification inherits the locked CE+Lovasz objective")

    runtime = resolved.get("runtime")
    if not isinstance(runtime, Mapping):
        raise SkySensePPCertificationError("resolved runtime configuration is malformed")
    contract = str(runtime.get("contract", "b"))
    validate_skysensepp_contract(contract)
    micro_batch = int(runtime.get("micro_batch", 0))
    if micro_batch != 16:
        raise SkySensePPCertificationError(
            "R24 must preserve the frozen micro_batch=16 normalization contract"
        )
    seed = int(runtime.get("seed", 0))
    plan = compile_skysensepp_plan(contract)

    full_model, ice_model, metadata = build_skysensepp_certification_pair(
        safetensors_path=safetensors_path,
        contract=contract,
        resolution=int(runtime.get("resolution", 120)),
        micro_batch=micro_batch,
        head_checkpoint=head_checkpoint,
    )
    full_model.to(target_device)
    ice_model.to(target_device)

    validation_loader, train_loader = _loaders(
        data_manifest,
        execution_scale=execution_scale,
        micro_batch=micro_batch,
        seed=seed,
        augmentation=runtime.get("augmentation"),
    )

    # --- Levels 2/3: retained feature maps and logits, FP32, first validation batch.
    first_validation = next(iter(validation_loader))
    optical, annotation, _target = _prepare_batch(first_validation, target_device, micro_batch=micro_batch)
    full_model.eval()
    ice_model.eval()
    with torch.no_grad():
        full_out_fp32 = full_model(optical, annotation)
        ice_out_fp32 = ice_model(optical, annotation)
    full_maps = full_out_fp32["feature_maps"]
    ice_maps = ice_out_fp32["feature_maps"]
    if len(full_maps) != len(SKYSENSEPP_S2_OUT_GRID):
        raise SkySensePPCertificationError(
            f"full forward returned {len(full_maps)} feature maps, expected the official grid"
        )
    full_by_layer = {
        int(layer): value for layer, value in zip(SKYSENSEPP_S2_OUT_GRID, full_maps)
    }
    ice_by_layer = {
        int(layer): value
        for layer, value in zip(ice_out_fp32["layer_indices"], ice_maps)
    }
    if set(ice_by_layer) - set(full_by_layer):
        raise SkySensePPCertificationError(
            "ICE executed grid layers are not a subset of the official full grid"
        )
    feature_results = {
        f"grid_layer_{layer}": _comparison_payload(
            compare_tensors(full_by_layer[layer], ice_by_layer[layer])
        )
        for layer in sorted(ice_by_layer)
    }
    fp32_logit_result = _comparison_payload(
        compare_tensors(full_out_fp32["logits"], ice_out_fp32["logits"])
    )
    del optical, annotation, first_validation, full_out_fp32, ice_out_fp32
    torch.cuda.empty_cache()

    # --- Level 4: head-only trainable gradients on one fixed train batch.
    # Restore RNG before the ICE run so any stochasticity in the retained
    # graph cannot create a false mismatch.
    torch.manual_seed(seed)
    train_batch = next(iter(train_loader))
    train_optical, train_annotation, train_target = _prepare_batch(
        train_batch, target_device, micro_batch=micro_batch
    )
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()

    full_model.train()
    full_model.zero_grad(set_to_none=True)
    full_logits = full_model(train_optical, train_annotation)["logits"]
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
    ice_logits = ice_model(train_optical, train_annotation)["logits"]
    ice_loss, _ = segmentation_objective(
        ice_logits,
        train_target,
        objective_name=objective_name,
    )
    ice_loss.backward()
    ice_gradients = named_trainable_gradients(ice_model)
    ice_loss_value = float(ice_loss.detach().cpu())
    ice_model.zero_grad(set_to_none=True)
    del ice_logits, ice_loss, train_optical, train_annotation, train_target, train_batch
    torch.cuda.empty_cache()

    gradient_result = compare_gradients(full_gradients, ice_gradients)
    gradient_result["full_loss"] = full_loss_value
    gradient_result["ice_loss"] = ice_loss_value

    # --- Level 5: complete validation under the protocol AMP setting.
    full_model.eval()
    ice_model.eval()
    classes = int(full_model.num_classes)
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
            optical, annotation, target = _prepare_batch(batch, target_device, micro_batch=micro_batch)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                full_logits = full_model(optical, annotation)["logits"]
                ice_logits = ice_model(optical, annotation)["logits"]
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
        # The archived best-validation mIoU was recorded by the training run's
        # deterministic validation pass; the certificate recomputes mIoU under
        # its own AMP pass, so a small float-level difference is expected.
        expected_delta_pp = abs(full_miou * 100.0 - float(expected_miou_percent))
        expected_ok = expected_delta_pp <= 0.05

    feature_gate = all(
        bool(item["shape_equal"])
        and item["max_abs_error"] is not None
        and float(item["max_abs_error"]) <= 1e-6
        for item in feature_results.values()
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
        and abs(full_miou - ice_miou) < 1e-12
    )
    state_gate = bool(metadata["state_surface_equal"])
    equivalence_pass = all(
        (
            state_gate,
            feature_gate,
            fp32_gate,
            gradient_gate,
            validation_gate,
            expected_ok,
        )
    )
    status = "pass" if equivalence_pass else "fail"

    initialization_source = str(initialization.get("source_sha256", "")) or None
    certificate: dict[str, Any] = {
        "schema_version": "prq4.skysensepp_ice_exact_certificate.v1",
        "route": SKYSENSEPP_S2_ROUTE,
        "status": status,
        "equivalence_certified": equivalence_pass,
        # Equivalence alone is not route-level scientific support. The
        # certificate-gated profiling stage must still clear the latency gate.
        "scientific_route_supported": False,
        "efficiency_evaluated": False,
        "test_accessed": False,
        "backbone_execution_comparison": ["full", "ice_exact"],
        "objective_name": objective_name,
        "micro_batch": micro_batch,
        "matched_common_protocol_sha256": resolved.get("matched_common_protocol_sha256"),
        "code_sync_manifest_sha256": resolved.get("code_sync_manifest_sha256"),
        "skysensepp_vendor_config_sha256": initialization_source,
        **metadata,
        "retained_feature_maps_fp32": feature_results,
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
            "retained_feature_maps_fp32": feature_gate,
            "logits_fp32": fp32_gate,
            "trainable_gradients_fp32": gradient_gate,
            "full_validation": validation_gate,
            "expected_baseline": expected_ok,
        },
    }
    _atomic_json(Path(output_path), certificate)
    return certificate


__all__ = [
    "SkysenseppIceExecutionModel",
    "SkySensePPCertificationError",
    "SKYSENSEPP_S2_ROUTE",
    "build_skysensepp_certification_pair",
    "run_skysensepp_ice_certification",
]
