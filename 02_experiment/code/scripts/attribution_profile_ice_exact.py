"""Cloud-only execution cost attribution for R21 ICE-Exact (five tiers).

Tiers (each built from the same audited backbone + checkpoint, differing only
in which CROMA nodes execute — all prediction-equal for the always-fuse
receiver, because GAP/joint/optical-suffix are not consumed downstream):
    Full   : both encoders full depth + norms + GAP + joint encoder
    no_gap : full depth + joint, GAP heads eliminated
    no_joint : tap-derived depth (no suffix), joint/GAP eliminated
    Exact  : minimum tap-derived plan (the certified ICE backend)

Reports per tier: median/mean/std/P10/P90 latency at B1 and B16, throughput,
analytical FLOPs, executed transformer blocks, incremental forward memory,
and (as a scientific guard) end-to-end prediction equality vs Full on a small
validation batch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import yaml

from geotoken3path.data.sen12ts import build_sen12ts_loader, croma_dynamic_normalize_batch
from geotoken3path.engine.formal_runner import validate_formal_evaluate_paths
from geotoken3path.engine.ice_certifier import _strip_archived_mechanism_keys
from geotoken3path.execution.croma_plan import compile_croma_execution_plan
from geotoken3path.execution.croma_executor import (
    InterfaceCertifiedCromaExecutor,
    install_ice_exact_forward,
)
from geotoken3path.execution.contracts import BackboneFeatureContract
from geotoken3path.execution.profiling import profile_cuda_callable
from geotoken3path.models.factory import build_vfm_segmentation_model
from geotoken3path.models.croma_loader import load_croma_backbone
from geotoken3path.utils.config import resolve_approved_config


_TIERS = ("full", "no_gap", "no_joint", "exact")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-config", default="geotoken3path.yaml",
        help="Model YAML under configs/model (task-depth rows: geotoken3path_d1_o4.yaml etc.)",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    validated = validate_formal_evaluate_paths(
        data_manifest=args.data_manifest, audit_report=args.audit_report,
        checkpoint=args.checkpoint, output_dir=args.output_dir, execution_scale="acceptance",
    )
    code_root = Path(__file__).resolve().parents[1]
    resolved = resolve_approved_config(
        code_root, "always_fuse", execution_scale="acceptance", model_config_name=args.model_config,
    )
    init_doc = yaml.safe_load((code_root / "configs/model/initialization.yaml").read_text(encoding="utf-8"))
    initialization = init_doc["initialization"]
    device = args.device

    # Full reference model on the audited backbone with the verified checkpoint.
    full_backbone, _ = load_croma_backbone(
        initialization=initialization,
        audit_path=validated["audit_report"],
        constructor_ref=str(initialization.get("constructor_ref", "")),
    )
    full_model = build_vfm_segmentation_model(
        resolved, audited_croma_backbone=full_backbone, backbone_execution="full"
    )
    payload = torch.load(str(validated["checkpoint"]), map_location="cpu", weights_only=True)
    state, _ = _strip_archived_mechanism_keys(payload.get("model", payload))
    full_model.load_state_dict(state, strict=True)
    full_model.to(device).eval()

    model_cfg = resolved["model"]
    stages = tuple(model_cfg["stages"])
    contract = BackboneFeatureContract(
        optical_stages=stages, sar_stages=stages, sar_depth_group_stages=stages,
        native_joint=False, global_optical=False, global_sar=False,
    )
    loader_kwargs = {
        "batch_size": 16, "num_workers": 0, "pin_memory": False,
        "persistent_workers": False, "execution_scale": "acceptance",
    }
    validation_loader, _ = build_sen12ts_loader(
        validated["data_manifest"], split="validation", augmentation=None, **loader_kwargs
    )
    batch = next(iter(validation_loader))
    optical_b = croma_dynamic_normalize_batch(batch)["optical"].to(device)
    sar_b = croma_dynamic_normalize_batch(batch)["sar"].to(device)
    optical_1, sar_1 = optical_b[:1], sar_b[:1]

    def tier_model(tier: str):
        bb, _ = load_croma_backbone(
            initialization=initialization,
            audit_path=validated["audit_report"],
            constructor_ref=str(initialization.get("constructor_ref", "")),
        )
        m = build_vfm_segmentation_model(
            resolved, audited_croma_backbone=bb, backbone_execution="full",
        )
        m.load_state_dict(state, strict=True)
        m.to(device).eval()
        if tier == "full":
            return m, None
        plan = compile_croma_execution_plan(
            model_cfg=model_cfg, receiver_contract=contract,
            audited_backbone=bb, ablation=tier,
        )
        executor = InterfaceCertifiedCromaExecutor(plan)
        install_ice_exact_forward(bb, executor)
        return m, plan

    results = {}

    def _forward(model, optical, sar):
        """Inference forward under the protocol AMP context (no autograd graph)."""
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                return model(optical, sar)

    with torch.no_grad():
        full_logits = _forward(full_model, optical_b, sar_b)
    # Independent full reference instance (own backbone build) so equality is a
    # genuine cross-instance check, not a self-comparison.
    full_ref, _ = tier_model("full")
    with torch.no_grad():
        full_ref_logits = _forward(full_ref, optical_b, sar_b)
        full_self_equal = bool(torch.equal(full_logits, full_ref_logits))
        full_self_max_abs = float((full_logits.float() - full_ref_logits.float()).abs().max())
    results["full_reference"] = {
        "self_equal": full_self_equal,
        "self_max_abs_error": full_self_max_abs,
    }
    del full_ref
    torch.cuda.empty_cache()

    for tier in _TIERS:
        m, plan = tier_model(tier)
        logits = _forward(m, optical_b, sar_b)
        max_abs_error = float((logits.float() - full_logits.float()).abs().max())
        eq = bool(torch.equal(logits, full_logits))
        fwd_b16 = lambda: _forward(m, optical_b, sar_b)
        fwd_b1 = lambda: _forward(m, optical_1, sar_1)
        lat_b16 = profile_cuda_callable(fwd_b16, warmup=args.warmup, iterations=args.iterations)
        lat_b1 = profile_cuda_callable(fwd_b1, warmup=args.warmup, iterations=args.iterations)
        results[tier] = {
            "prediction_equal_to_full": eq,
            "max_abs_logit_error": max_abs_error,
            "amp_enabled": True,
            "b16": {"median_ms": lat_b16.median_ms, "mean_ms": lat_b16.mean_ms,
                     "std_ms": lat_b16.std_ms, "p10_ms": lat_b16.p10_ms, "p90_ms": lat_b16.p90_ms,
                     "images_per_second": 1000.0 / lat_b16.median_ms * 16},
            "b1": {"median_ms": lat_b1.median_ms, "mean_ms": lat_b1.mean_ms,
                   "std_ms": lat_b1.std_ms, "p10_ms": lat_b1.p10_ms, "p90_ms": lat_b1.p90_ms,
                   "images_per_second": 1000.0 / lat_b1.median_ms},
        }
        if plan is not None:
            results[tier]["executed_blocks"] = {
                "s1": plan.s1_last_layer + 1 if plan.s1_last_layer is not None else 0,
                "s2": plan.s2_last_layer + 1 if plan.s2_last_layer is not None else 0,
            }
            results[tier]["plan_sha256"] = plan.plan_sha256
            results[tier]["ablation_tier"] = plan.ablation_tier
        del m
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    out_dir = Path(validated["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ice_execution_cost_attribution.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(out_path), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())