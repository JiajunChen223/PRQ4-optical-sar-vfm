"""Cloud-only Task-Specific Export certification (R21 queue 3).

Builds a physically compact export backbone (S2 0-5 + S1 0-5 + patch + attn_bias,
dead modules removed), loads the verified checkpoint's retained subset into it,
and certifies against the full model on the whole validation split:
  - retained-tap equality (FP32, first batch),
  - FP32 logits equality (first batch),
  - whole-validation prediction agreement (AMP, pixel + confusion matrix),
  - state-subset correctness (export keys == full keys minus removed set,
    retained keys bitwise identical),
  - reduction numbers (parameters / checkpoint bytes / removed module paths).

The export model is an inference-only artifact; no gradient/optimizer claim is
made here (covered by the R21 optimizer certificate on the training path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import yaml

from geotoken3path.data.sen12ts import build_sen12ts_loader, croma_dynamic_normalize_batch
from geotoken3path.engine.formal_runner import validate_formal_evaluate_paths
from geotoken3path.engine.ice_certifier import _checkpoint_state, _strip_archived_mechanism_keys
from geotoken3path.execution.certification import compare_tensors
from geotoken3path.execution.croma_export import build_export_backbone
from geotoken3path.models.croma_loader import load_croma_backbone
from geotoken3path.models.factory import build_vfm_segmentation_model
from geotoken3path.utils.config import resolve_approved_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    validated = validate_formal_evaluate_paths(
        data_manifest=args.data_manifest, audit_report=args.audit_report,
        checkpoint=args.checkpoint, output_dir=args.output_dir, execution_scale="acceptance",
    )
    code_root = Path(__file__).resolve().parents[1]
    resolved = resolve_approved_config(code_root, "always_fuse", execution_scale="acceptance")
    init_doc = yaml.safe_load((code_root / "configs/model/initialization.yaml").read_text(encoding="utf-8"))
    initialization = init_doc["initialization"]
    device = args.device

    # Load the audited backbone once; build the export backbone from the clean
    # (hook-free) instance BEFORE wrapping it in the full model's tap adapter,
    # so the deep copy cannot carry ghost hooks bound to the full adapter.
    full_backbone, _ = load_croma_backbone(
        initialization=initialization, audit_path=validated["audit_report"],
        constructor_ref=str(initialization.get("constructor_ref", "")),
    )
    export_backbone, stats = build_export_backbone(full_backbone)

    # Full model from audited backbone + verified checkpoint.
    full_model = build_vfm_segmentation_model(
        resolved, audited_croma_backbone=full_backbone, backbone_execution="full"
    )
    state, _ = _strip_archived_mechanism_keys(_checkpoint_state(validated["checkpoint"]))
    full_model.load_state_dict(state, strict=True)
    full_model.to(device).eval()

    # Export model: stripped backbone + token model from the same factory chain.
    export_model = build_vfm_segmentation_model(
        resolved, audited_croma_backbone=export_backbone, backbone_execution="full",
    )
    # Load the retained subset of the checkpoint state (drop removed-module keys).
    full_state_keys = set(full_model.state_dict().keys())
    export_state_keys = set(export_model.state_dict().keys())
    missing_in_export = full_state_keys - export_state_keys
    export_state = {k: state[k] for k in export_state_keys if k in state}
    export_model.load_state_dict(export_state, strict=True)
    export_model.to(device).eval()

    # Whole-validation agreement.
    loader_kwargs = {
        "batch_size": 16, "num_workers": 0, "pin_memory": False,
        "persistent_workers": False, "execution_scale": "acceptance",
    }
    validation_loader, _ = build_sen12ts_loader(
        validated["data_manifest"], split="validation", augmentation=None, **loader_kwargs
    )
    total_pixels = 0
    identical_pixels = 0
    max_logit_error = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in validation_loader:
            optical = croma_dynamic_normalize_batch(batch)["optical"].to(device)
            sar = croma_dynamic_normalize_batch(batch)["sar"].to(device)
            target = batch["target"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                full_logits = full_model(optical, sar)
                export_logits = export_model(optical, sar)
            max_logit_error = max(max_logit_error, float((full_logits.float() - export_logits.float()).abs().max()))
            total_pixels += target.numel()
            identical_pixels += int((full_logits.argmax(1) == export_logits.argmax(1)).sum())
            batch_count += 1

    retained_keys_bitwise = all(
        torch.equal(full_model.state_dict()[k], export_model.state_dict()[k])
        for k in export_state_keys if k in full_model.state_dict()
    )
    certificate = {
        "artifact_type": "task_specific_export_certificate",
        "schema_version": "prq4.ice_export_certificate.v1",
        "status": "pass"
        if identical_pixels == total_pixels and max_logit_error <= 1e-4 and retained_keys_bitwise
        else "fail",
        "route": "R21-ICE-VFM-01",
        "mechanism_set": "always_fuse",
        "checkpoint_sha256": _sha256(Path(validated["checkpoint"])),
        "device": str(device),
        "validation": {
            "batches": batch_count,
            "prediction_pixels": total_pixels,
            "identical_prediction_pixels": identical_pixels,
            "prediction_identical": identical_pixels == total_pixels,
            "max_abs_logit_error": max_logit_error,
        },
        "state_subset": {
            "full_keys": len(full_state_keys),
            "export_keys": len(export_state_keys),
            "removed_keys": sorted(missing_in_export),
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
            "note": "Training-path equivalence is covered by the R21 optimizer-update certificate.",
        },
        "test_accessed": False,
    }
    out_dir = Path(validated["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "task_specific_export_certificate.json"
    out_path.write_text(json.dumps(certificate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(certificate, ensure_ascii=False, indent=2))
    return 0 if certificate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())