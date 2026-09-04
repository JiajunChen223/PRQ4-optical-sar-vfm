"""R24 SkySense++ S2 task-specific export certification entry point (cloud).

Builds the physically compact export artifact (``SkysenseppExportedModel``:
backbone layers sliced to the plan bound, dead modules detached) from the full
audited model plus the trained head checkpoint, then certifies against the
full model over the whole validation split:

  - FP32 logits equality on the first validation batch,
  - whole-validation prediction agreement under the protocol AMP precision
    (pixel-exact argmax + identical confusion matrices + mIoU delta),
  - state-subset correctness (export keys == full keys minus the removed set,
    retained keys bitwise identical),
  - deleted-parameter statistics (count / fraction / removed module paths).

The export artifact is inference-only; no gradient or optimizer claim is made
here (covered by the certification pair's gradient certificate on the
training path).  Output schema: ``prq4.skysensepp_ice_export_certificate.v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch

from geotoken3path.data.skysensepp import (
    annotation_from_target,
    build_skysensepp_loader,
    croma_dynamic_normalize_batch_r24,
)
from geotoken3path.engine.ice_certifier_r24 import SkySensePPCertificationError
from geotoken3path.execution.certification import compare_tensors
from geotoken3path.execution.skysensepp_export import build_skysensepp_export_model
from geotoken3path.metrics import confusion_matrix as cm_fn
from geotoken3path.metrics import mean_iou
from geotoken3path.models.skysensepp_seg import build_skysensepp_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_state(path: Path) -> dict[str, object]:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise SkySensePPCertificationError("head checkpoint must contain a mapping")
    state = payload.get("head_state", payload.get("model", payload))
    if not isinstance(state, dict):
        raise SkySensePPCertificationError("head checkpoint state is invalid")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--weights", required=True, help="Audited SkySense++ S2 safetensors")
    parser.add_argument("--head-checkpoint", required=True, help="Trained head checkpoint")
    parser.add_argument("--contract", choices=("a", "b"), default="b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--execution-scale",
        choices=("smoke", "acceptance"),
        default="acceptance",
    )
    args = parser.parse_args()

    for label, path in (
        ("data-manifest", args.data_manifest),
        ("weights", args.weights),
        ("head-checkpoint", args.head_checkpoint),
        ("output-dir", args.output_dir),
    ):
        if not Path(path).is_absolute():
            parser.error(f"--{label} must be an absolute cloud path")
    weights_path = Path(args.weights)
    head_path = Path(args.head_checkpoint)
    if not weights_path.is_file() or not head_path.is_file():
        parser.error("weights and head checkpoint files must exist")

    device = args.device
    if not torch.cuda.is_available():
        parser.error("R24 export certification requires CUDA")

    # Full audited model + trained head.
    full_model = build_skysensepp_model(
        contract=args.contract,
        safetensors_path=str(weights_path),
        num_classes=11,
        seed=0,
    )
    full_model.head.load_state_dict(_head_state(head_path), strict=True)
    full_model.to(device).eval()

    # Physically compact export (deep copy; contract validated internally).
    export_model, stats = build_skysensepp_export_model(full_model, contract=args.contract)
    export_model.to(device).eval()

    validation_loader, _ = build_skysensepp_loader(
        args.data_manifest,
        split="validation",
        batch_size=16,
        num_workers=4,
        execution_scale=args.execution_scale,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        augmentation=None,
        seed=0,
    )

    first_batch = next(iter(validation_loader))
    first_normalized = croma_dynamic_normalize_batch_r24(first_batch, micro_batch=16)
    fp32_optical = first_normalized["optical10"].to(device)
    fp32_annotation = annotation_from_target(first_batch["target"].to(device))
    with torch.no_grad():
        full_first = full_model(fp32_optical, fp32_annotation)["logits"]
        export_first = export_model(fp32_optical, fp32_annotation)["logits"]
    fp32_result = compare_tensors(full_first, export_first)
    del fp32_optical, fp32_annotation, full_first, export_first
    torch.cuda.empty_cache()

    total_pixels = 0
    identical_pixels = 0
    max_logit_error = 0.0
    batch_count = 0
    matrix_full = None
    matrix_export = None
    with torch.no_grad():
        for batch in validation_loader:
            normalized = croma_dynamic_normalize_batch_r24(batch, micro_batch=16)
            optical = normalized["optical10"].to(device)
            target = batch["target"].to(device)
            annotation = annotation_from_target(target)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                full_logits = full_model(optical, annotation)["logits"]
                export_logits = export_model(optical, annotation)["logits"]
            max_logit_error = max(
                max_logit_error,
                float((full_logits.float() - export_logits.float()).abs().max()),
            )
            total_pixels += target.numel()
            identical_pixels += int((full_logits.argmax(1) == export_logits.argmax(1)).sum())
            classes = int(full_logits.shape[1])
            matrix_full = (
                cm_fn(full_logits, target, classes).cpu()
                if matrix_full is None
                else matrix_full + cm_fn(full_logits, target, classes).cpu()
            )
            matrix_export = (
                cm_fn(export_logits, target, classes).cpu()
                if matrix_export is None
                else matrix_export + cm_fn(export_logits, target, classes).cpu()
            )
            batch_count += 1
    matrix_identical = (
        bool(torch.equal(matrix_full, matrix_export)) if matrix_full is not None else False
    )
    full_miou = float(mean_iou(matrix_full)) if matrix_full is not None else float("nan")
    export_miou = float(mean_iou(matrix_export)) if matrix_export is not None else float("nan")

    full_state = full_model.state_dict()
    export_state = export_model.state_dict()
    full_keys = set(full_state)
    export_keys = set(export_state)
    missing_in_export = sorted(full_keys - export_keys)
    retained_keys_bitwise = all(
        torch.equal(full_state[key], export_state[key]) for key in export_keys
    )
    predictions_identical = identical_pixels == total_pixels
    status = (
        "pass"
        if (
            predictions_identical
            and matrix_identical
            and abs(full_miou - export_miou) < 1e-12
            and max_logit_error <= 1e-4
            and bool(fp32_result.shape_equal)
            and fp32_result.max_abs_error <= 1e-6
            and retained_keys_bitwise
        )
        else "fail"
    )

    certificate = {
        "artifact_type": "task_specific_export_certificate",
        "schema_version": "prq4.skysensepp_ice_export_certificate.v1",
        "status": status,
        "route": "R24-SKYSENSEPP-S2-ICE-01",
        "contract": args.contract,
        "annotation_source": "gt_worldcover_leakage_documented",
        "safetensors_sha256": _sha256(weights_path),
        "head_checkpoint_sha256": _sha256(head_path),
        "device": str(device),
        "validation": {
            "batches": batch_count,
            "prediction_pixels": total_pixels,
            "identical_prediction_pixels": identical_pixels,
            "prediction_identical": predictions_identical,
            "max_abs_logit_error": max_logit_error,
            "confusion_matrix_identical": matrix_identical,
            "full_mIoU_percent": full_miou * 100.0,
            "export_mIoU_percent": export_miou * 100.0,
        },
        "fp32_first_batch_logits": dict(fp32_result.__dict__),
        "state_subset": {
            "full_keys": len(full_keys),
            "export_keys": len(export_keys),
            "removed_keys": missing_in_export,
            "retained_keys_bitwise_identical": retained_keys_bitwise,
        },
        "reduction": {
            "full_parameter_count": stats.full_parameter_count,
            "export_parameter_count": stats.export_parameter_count,
            "removed_parameter_count": stats.removed_parameter_count,
            "removed_parameter_fraction": stats.removed_parameter_fraction,
            "removed_module_paths": list(stats.removed_module_paths),
        },
        "scope": {
            "artifact": "inference_only",
            "training_claim": "not_claimed",
            "note": (
                "Training-path equivalence is covered by the R24 ICE certification "
                "pair gradient certificate."
            ),
        },
        "test_accessed": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "skysensepp_ice_export_certificate.json"
    output_path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(certificate, ensure_ascii=False, indent=2))
    return 0 if certificate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
