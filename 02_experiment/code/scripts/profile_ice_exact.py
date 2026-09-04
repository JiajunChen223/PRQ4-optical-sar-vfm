"""Certificate-gated CUDA profiling for R21 ICE-Exact."""

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
from geotoken3path.engine.formal_runner import FormalRunnerError, validate_formal_evaluate_paths
from geotoken3path.engine.ice_certifier import build_ice_certification_pair
from geotoken3path.execution.profiling import profile_cuda_pair_abba
from geotoken3path.utils.config import resolve_approved_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_certificate(path: Path, checkpoint: Path) -> dict[str, object]:
    if not path.is_absolute():
        raise ValueError("certificate must be an absolute cloud path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("certificate must contain a JSON object")
    if payload.get("route") != "R21-ICE-VFM-01" or payload.get("status") != "pass":
        raise ValueError("profiling requires a passing R21 ICE certificate")
    if payload.get("test_accessed") is not False:
        raise ValueError("profiling refuses a certificate that accessed the sealed test split")
    if payload.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("certificate and requested baseline checkpoint differ")
    return payload


def _profile_memory(fn) -> dict[str, int]:
    """Measure incremental forward allocation above the two-model resident base."""

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resident = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    return {
        "resident_before_forward_bytes": resident,
        "absolute_peak_allocated_bytes": peak,
        "incremental_forward_peak_bytes": max(0, peak - resident),
    }


def _summary_dict(summary, *, batch_size: int) -> dict[str, float | int]:
    value = dict(summary.__dict__)
    value["images_per_second"] = float(batch_size * 1000.0 / summary.median_ms)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    try:
        validated = validate_formal_evaluate_paths(
            data_manifest=args.data_manifest,
            audit_report=args.audit_report,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            execution_scale="acceptance",
        )
    except FormalRunnerError as exc:
        parser.error(str(exc))
    checkpoint_path = Path(validated["checkpoint"])
    try:
        certificate = _load_certificate(Path(args.certificate), checkpoint_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    code_root = Path(__file__).resolve().parents[1]
    resolved = resolve_approved_config(code_root, "always_fuse", execution_scale="acceptance")
    initialization_doc = yaml.safe_load(
        (code_root / "configs/model/initialization.yaml").read_text(encoding="utf-8")
    )
    initialization = initialization_doc.get("initialization")
    if not isinstance(initialization, dict):
        parser.error("initialization.yaml does not contain an initialization mapping")

    full_model, ice_model, metadata = build_ice_certification_pair(
        resolved=resolved,
        initialization=initialization,
        audit_report=validated["audit_report"],
        checkpoint=checkpoint_path,
    )
    certificate_plan = certificate.get("execution_plan")
    current_plan = metadata.get("execution_plan")
    if (
        not isinstance(certificate_plan, dict)
        or not isinstance(current_plan, dict)
        or certificate_plan.get("plan_sha256") != current_plan.get("plan_sha256")
    ):
        parser.error("certificate execution plan differs from the current ICE plan")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("R21 profiling requires CUDA")
    full_model.to(device).eval()
    ice_model.to(device).eval()

    runtime = resolved["runtime"]
    if int(runtime["micro_batch"]) != 16:
        parser.error("R21 profiling requires the frozen micro_batch=16 normalization contract")
    loader, _ = build_sen12ts_loader(
        validated["data_manifest"],
        split="validation",
        batch_size=16,
        num_workers=4,
        execution_scale="acceptance",
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        augmentation=None,
        seed=int(runtime["seed"]),
    )
    batch = next(iter(loader))
    normalized = croma_dynamic_normalize_batch(batch)
    optical16 = normalized["optical"].to(device, non_blocking=True)
    sar16 = normalized["sar"].to(device, non_blocking=True)
    # B=1 is deployment-reference model latency only. The tensor is sliced
    # after the frozen, protocol-legal B=16 normalization; raw B=1 normalization
    # is intentionally not performed or claimed.
    optical1 = optical16[:1]
    sar1 = sar16[:1]
    amp_enabled = str(runtime.get("precision", "")).casefold() == "amp"

    def make_fn(model, optical, sar):
        def _fn():
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    return model(optical, sar)
        return _fn

    rows: dict[str, dict[str, object]] = {}
    for label, optical, sar, batch_size, protocol_role in (
        ("batch16", optical16, sar16, 16, "protocol_aligned_network_forward"),
        ("batch1", optical1, sar1, 1, "deployment_reference_after_microbatch16_normalization"),
    ):
        full_fn = make_fn(full_model, optical, sar)
        ice_fn = make_fn(ice_model, optical, sar)
        full_summary, ice_summary = profile_cuda_pair_abba(
            full_fn,
            ice_fn,
            warmup=args.warmup,
            iterations_per_mode=args.iterations,
        )
        full_memory = _profile_memory(full_fn)
        ice_memory = _profile_memory(ice_fn)
        speedup_fraction = 1.0 - ice_summary.median_ms / full_summary.median_ms
        activation_reduction = (
            int(full_memory["incremental_forward_peak_bytes"])
            - int(ice_memory["incremental_forward_peak_bytes"])
        )
        rows[label] = {
            "protocol_role": protocol_role,
            "batch_size": batch_size,
            "full": _summary_dict(full_summary, batch_size=batch_size),
            "ice_exact": _summary_dict(ice_summary, batch_size=batch_size),
            "median_latency_reduction_fraction": float(speedup_fraction),
            "memory_scope": "both_full_parameter_sets_resident; report incremental forward allocation",
            "full_memory": full_memory,
            "ice_exact_memory": ice_memory,
            "incremental_forward_peak_reduction_bytes": activation_reduction,
        }

    raw = full_model.bridge.backbone.backbone
    plan = getattr(ice_model, "_ice_execution_plan")
    full_counts = {
        "s1_blocks": len(raw.s1_encoder.transformer.layers),
        "s2_blocks": len(raw.s2_encoder.transformer.layers),
        "joint_blocks": len(raw.cross_encoder.layers),
    }
    ice_counts = {
        "s1_blocks": 0 if plan.s1_last_layer is None else plan.s1_last_layer + 1,
        "s2_blocks": 0 if plan.s2_last_layer is None else plan.s2_last_layer + 1,
        "joint_blocks": len(raw.cross_encoder.layers) if plan.require_joint_encoder else 0,
    }
    full_total = sum(full_counts.values())
    ice_total = sum(ice_counts.values())
    batch16_reduction = float(rows["batch16"]["median_latency_reduction_fraction"])
    result = {
        "schema_version": "prq4.ice_exact_profile.v1",
        "route": "R21-ICE-VFM-01",
        "test_accessed": False,
        "certificate_path": str(Path(args.certificate)),
        "certificate_plan_sha256": certificate_plan["plan_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "amp_enabled": amp_enabled,
        "warmup": args.warmup,
        "iterations_per_mode": args.iterations,
        "measurement_order": "repeated_ABBA",
        "latency": rows,
        "executed_transformer_blocks": {
            "full": full_counts,
            "ice_exact": ice_counts,
            "full_total": full_total,
            "ice_total": ice_total,
            "reduction_fraction": float(1.0 - ice_total / full_total),
        },
        "parameter_reduction_claimed": False,
        "checkpoint_reduction_claimed": False,
        "formal_efficiency_gate": {
            "metric": "batch16_network_forward_median_latency_reduction_fraction",
            "threshold": 0.20,
            "observed": batch16_reduction,
            "pass": batch16_reduction >= 0.20,
        },
    }
    output = Path(validated["output_dir"]) / "ice_exact_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print({"status": "complete", "profile": str(output), "efficiency_gate": result["formal_efficiency_gate"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
