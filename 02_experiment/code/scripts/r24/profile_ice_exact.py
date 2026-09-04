"""Certificate-gated CUDA profiling for R24 SkySense++ S2 ICE-Exact.

Reuses ``execution.profiling.profile_cuda_pair_abba`` (repeated ABBA order) at
B=16 (protocol-aligned network forward) and B=1 (deployment reference after
micro-batch-16 normalization).  The Full and ICE models are the certification
pair (same audited weights and trained head checkpoint).  A passing
``skysensepp_ice_exact_certificate.json`` whose plan sha256 matches the
current plan is required; the certificate's head/weights sha256 are bound
here, so a mismatched checkpoint is refused before any measurement.

Latency tiers (median reduction fraction): >=0.30 strong, >=0.20 supportive,
>=0.10 marginal, below 0.10 weak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
import yaml

from geotoken3path.engine.ice_certifier_r24 import (
    SkySensePPCertificationError,
    build_skysensepp_certification_pair,
    compile_skysensepp_plan,
)
from geotoken3path.execution.profiling import profile_cuda_pair_abba
from geotoken3path.data.skysensepp import (
    annotation_from_target,
    build_skysensepp_loader,
    croma_dynamic_normalize_batch_r24,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_certificate(path: Path, weights: Path, head_checkpoint: Path) -> dict[str, object]:
    if not path.is_absolute():
        raise ValueError("certificate must be an absolute cloud path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("certificate must contain a JSON object")
    if payload.get("route") != "R24-SKYSENSEPP-S2-ICE-01" or payload.get("status") != "pass":
        raise ValueError("profiling requires a passing R24 skysensepp ICE certificate")
    if payload.get("equivalence_certified") is not True:
        raise ValueError("profiling requires equivalence_certified=true")
    if payload.get("test_accessed") is not False:
        raise ValueError("profiling refuses a certificate that accessed the sealed test split")
    if payload.get("safetensors_sha256") != _sha256(weights):
        raise ValueError("certificate and requested skysensepp weights checkpoint differ")
    head_sha = payload.get("head_checkpoint_sha256")
    if payload.get("head_checkpoint_applied") is True and head_sha != _sha256(head_checkpoint):
        raise ValueError("certificate and requested head checkpoint differ")
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


def _tier(fraction: float) -> str:
    if fraction >= 0.30:
        return "strong"
    if fraction >= 0.20:
        return "supportive"
    if fraction >= 0.10:
        return "marginal"
    return "weak"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--weights", required=True, help="Audited SkySense++ S2 safetensors")
    parser.add_argument("--head-checkpoint", required=True, help="Trained head checkpoint")
    parser.add_argument("--certificate", required=True, help="Passing skysensepp ICE certificate")
    parser.add_argument("--contract", choices=("a", "b"), default="b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
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
        ("certificate", args.certificate),
        ("output-dir", args.output_dir),
    ):
        if not Path(path).is_absolute():
            parser.error(f"--{label} must be an absolute cloud path")

    weights_path = Path(args.weights)
    head_path = Path(args.head_checkpoint)
    certificate_path = Path(args.certificate)
    try:
        certificate = _load_certificate(certificate_path, weights_path, head_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    plan = compile_skysensepp_plan(args.contract)
    certificate_plan = certificate.get("execution_plan")
    if not isinstance(certificate_plan, dict):
        certificate_plan = {
            "contract": certificate.get("contract"),
            "max_layer": certificate.get("max_layer"),
            "required_output_indices": certificate.get("required_output_indices"),
            "executed_layer_count": certificate.get("executed_layer_count"),
            "eliminated_layers": certificate.get("eliminated_layers"),
        }
    if certificate_plan.get("plan_sha256") not in (None, plan.plan_sha256):
        parser.error("certificate execution plan differs from the current ICE plan")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("R24 profiling requires CUDA")
    try:
        full_model, ice_model, metadata = build_skysensepp_certification_pair(
            safetensors_path=weights_path,
            contract=args.contract,
            resolution=120,
            micro_batch=16,
            head_checkpoint=head_path,
        )
    except (SkySensePPCertificationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if metadata.get("state_surface_equal") is not True:
        parser.error("certification pair does not share a state surface")
    full_model.to(device).eval()
    ice_model.to(device).eval()

    loader, _ = build_skysensepp_loader(
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
    batch = next(iter(loader))
    normalized = croma_dynamic_normalize_batch_r24(batch, micro_batch=16)
    optical16 = normalized["optical10"].to(device, non_blocking=True)
    target16 = batch["target"].to(device, non_blocking=True)
    annotation16 = annotation_from_target(target16)
    # B=1 is deployment-reference model latency only.  The tensor is sliced
    # after the frozen, protocol-legal B=16 normalization; raw B=1
    # normalization is intentionally not performed or claimed.
    optical1 = optical16[:1]
    annotation1 = annotation16[:1]
    amp_enabled = True  # protocol AMP (R24 acceptance precision: amp)

    def make_fn(model, optical, annotation):
        def _fn():
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    return model(optical, annotation)
        return _fn

    rows: dict[str, dict[str, object]] = {}
    for label, optical, annotation, batch_size, protocol_role in (
        ("batch16", optical16, annotation16, 16, "protocol_aligned_network_forward"),
        ("batch1", optical1, annotation1, 1, "deployment_reference_after_microbatch16_normalization"),
    ):
        full_fn = make_fn(full_model, optical, annotation)
        ice_fn = make_fn(ice_model, optical, annotation)
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

    batch16_reduction = float(rows["batch16"]["median_latency_reduction_fraction"])
    batch1_reduction = float(rows["batch1"]["median_latency_reduction_fraction"])
    tier_verdict = _tier(batch16_reduction)

    full_layers = len(full_model.backbone.layers)
    ice_layers = len(ice_model.backbone.layers)
    result = {
        "schema_version": "prq4.skysensepp_ice_exact_profile.v1",
        "route": "R24-SKYSENSEPP-S2-ICE-01",
        "test_accessed": False,
        "equivalence_certified": True,
        "scientific_route_supported": batch16_reduction >= 0.20,
        "latency_tier": tier_verdict,
        "certificate_path": str(certificate_path),
        "certificate_plan_sha256": certificate_plan.get("plan_sha256") or plan.plan_sha256,
        "safetensors_sha256": metadata["safetensors_sha256"],
        "head_checkpoint_sha256": metadata["head_checkpoint_sha256"],
        "amp_enabled": amp_enabled,
        "warmup": args.warmup,
        "iterations_per_mode": args.iterations,
        "measurement_order": "repeated_ABBA",
        "latency": rows,
        "executed_backbone_layers": {
            "full": full_layers,
            "ice_exact": ice_layers,
            "eliminated": full_layers - ice_layers,
        },
        "formal_efficiency_gate": {
            "metric": "batch16_network_forward_median_latency_reduction_fraction",
            "threshold_supportive": 0.20,
            "threshold_strong": 0.30,
            "observed": batch16_reduction,
            "batch1_observed": batch1_reduction,
            "tier": tier_verdict,
        },
    }
    output = Path(args.output_dir) / "skysensepp_ice_exact_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        {
            "status": "complete",
            "profile": str(output),
            "latency_tier": tier_verdict,
            "efficiency_gate": result["formal_efficiency_gate"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
