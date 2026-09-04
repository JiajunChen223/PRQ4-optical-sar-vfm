"""Optimizer-update equivalence certificate: Full vs ICE-Exact training step.

Runs 3 AdamW update steps on two identically initialized models (full official
forward vs ICE-Exact minimal execution), comparing at every step:
  - loss
  - gradients (per trainable parameter)
  - post-step parameters
  - AdamW exp_avg / exp_avg_sq

Both models start from the audited CROMA backbone (same as the formal training
protocol) with randomly initialized downstream heads; the same training batch
and RNG state are used for both rows so the comparison isolates the execution
backend. Threshold: max_abs_error <= 1e-6 (AMP-noise level) per the R21
certification convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import yaml

from geotoken3path.engine.formal_runner import validate_formal_evaluate_paths
from geotoken3path.execution.certification import compare_tensors
from geotoken3path.models.factory import build_vfm_segmentation_model
from geotoken3path.models.croma_loader import load_croma_backbone
from geotoken3path.utils.config import resolve_approved_config
from geotoken3path.data.sen12ts import build_sen12ts_loader, croma_dynamic_normalize_batch
from geotoken3path.losses import segmentation_objective


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Cloud checkpoint path (lexical placeholder for path validation; "
             "this certificate starts from the audited CROMA init and does not load it)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
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
    torch.manual_seed(args.seed)

    loader_kwargs = {
        "batch_size": 16, "num_workers": 0, "pin_memory": False,
        "persistent_workers": False, "execution_scale": "acceptance",
    }
    train_loader, _ = build_sen12ts_loader(
        validated["data_manifest"], split="train", augmentation=None, **loader_kwargs
    )

    def build_row(execution: str, init_seed: int):
        torch.manual_seed(init_seed)
        backbone, _ = load_croma_backbone(
            initialization=initialization,
            audit_path=validated["audit_report"],
            constructor_ref=str(initialization.get("constructor_ref", "")),
        )
        model = build_vfm_segmentation_model(
            resolved, audited_croma_backbone=backbone, backbone_execution=execution,
        )
        model.to(device)
        model.train()
        return model

    full_model = build_row("full", args.seed)
    ice_model = build_row("ice_exact", args.seed)

    def make_optimizer(model):
        return torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(resolved["runtime"]["optimizer"]["learning_rate"]),
            weight_decay=float(resolved["runtime"]["optimizer"]["weight_decay"]),
            betas=tuple(float(x) for x in resolved["runtime"]["optimizer"]["betas"]),
        )

    full_opt = make_optimizer(full_model)
    ice_opt = make_optimizer(ice_model)
    assert len(list(full_opt.param_groups[0]["params"])) == len(list(ice_opt.param_groups[0]["params"])), (
        "trainable parameter sets must match between full and ICE rows"
    )
    trainable_names = [n for n, p in full_model.named_parameters() if p.requires_grad]
    ice_trainable_names = [n for n, p in ice_model.named_parameters() if p.requires_grad]
    assert trainable_names == ice_trainable_names, "trainable parameter names must match"

    batches = []
    for _ in range(args.steps):
        batches.append(next(iter(train_loader)))

    # Dedicated final comparison pass (deterministic, same batches, both rows):
    # Re-run the 3 steps while recording per-name states, then compare.
    def run_trace(model, opt, batches, device):
        trace = []
        model.train()
        for step, batch in enumerate(batches):
            optical = croma_dynamic_normalize_batch(batch)["optical"].to(device)
            sar = croma_dynamic_normalize_batch(batch)["sar"].to(device)
            target = batch["target"].to(device)
            torch.manual_seed(1000 + step)
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all()
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_state)
            opt.zero_grad(set_to_none=True)
            logits = model(optical, sar)
            loss, _ = segmentation_objective(logits, target, objective_name="ce_lovasz")
            loss.backward()
            grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
            states = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    st = opt.state[p]
                    states[n] = (
                        p.detach().cpu().clone(),
                        st["exp_avg"].detach().cpu().clone(),
                        st["exp_avg_sq"].detach().cpu().clone(),
                    )
            trace.append({"loss": float(loss.detach()), "grads": grads, "states": states})
        return trace

    full_trace = run_trace(full_model, full_opt, batches, device)
    # ICE row shares the same init seed as the full row; run its trace now.
    ice_trace = run_trace(ice_model, ice_opt, batches, device)

    step_records = []
    max_loss_err = max_grad_err = max_param_err = max_exp_avg_err = max_exp_avg_sq_err = 0.0
    for step, (ft, it) in enumerate(zip(full_trace, ice_trace)):
        loss_err = abs(ft["loss"] - it["loss"])
        max_loss_err = max(max_loss_err, loss_err)
        grad_err = max(float(compare_tensors(ft["grads"][n], it["grads"][n]).max_abs_error) for n in trainable_names)
        param_err = max(float(compare_tensors(ft["states"][n][0], it["states"][n][0]).max_abs_error) for n in trainable_names)
        exp_avg_err = max(float(compare_tensors(ft["states"][n][1], it["states"][n][1]).max_abs_error) for n in trainable_names)
        exp_avg_sq_err = max(float(compare_tensors(ft["states"][n][2], it["states"][n][2]).max_abs_error) for n in trainable_names)
        max_grad_err = max(max_grad_err, grad_err)
        max_param_err = max(max_param_err, param_err)
        max_exp_avg_err = max(max_exp_avg_err, exp_avg_err)
        max_exp_avg_sq_err = max(max_exp_avg_sq_err, exp_avg_sq_err)
        step_records.append({
            "step": step,
            "loss": {"full": ft["loss"], "ice": it["loss"], "abs_error": loss_err},
            "max_grad_abs_error": grad_err,
            "max_param_abs_error": param_err,
            "max_exp_avg_abs_error": exp_avg_err,
            "max_exp_avg_sq_abs_error": exp_avg_sq_err,
        })

    certificate = {
        "artifact_type": "optimizer_update_equivalence_certificate",
        "schema_version": "researchpilot.r21.optimizer_cert.v1",
        "status": "pass" if max(max_loss_err, max_grad_err, max_param_err, max_exp_avg_err, max_exp_avg_sq_err) <= 1e-6 else "fail",
        "route": "R21-ICE-VFM-01",
        "mechanism_set": "always_fuse",
        "objective": "ce_lovasz",
        "seed": args.seed,
        "steps": args.steps,
        "trainable_parameter_count": len(trainable_names),
        "threshold_max_abs_error": 1e-6,
        "step_records": step_records,
        "summary": {
            "max_loss_abs_error": max_loss_err,
            "max_grad_abs_error": max_grad_err,
            "max_param_abs_error": max_param_err,
            "max_exp_avg_abs_error": max_exp_avg_err,
            "max_exp_avg_sq_abs_error": max_exp_avg_sq_err,
        },
        "test_accessed": False,
    }
    out_dir = Path(validated["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optimizer_update_equivalence_certificate.json"
    out_path.write_text(json.dumps(certificate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(certificate, ensure_ascii=False, indent=2))
    return 0 if certificate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())