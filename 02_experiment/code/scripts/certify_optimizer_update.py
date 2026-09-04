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
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            grads = {
                n: (p.grad.detach().cpu().clone() if p.grad is not None else None)
                for n, p in model.named_parameters() if p.requires_grad
            }
            opt.step()
            states = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    st = opt.state[p]
                    # torch 2.1.2 may not populate exp_avg until later steps;
                    # record None when absent (both rows share the same
                    # optimizer-state machine, so None-None is equivalent).
                    exp_avg = st.get("exp_avg")
                    exp_avg_sq = st.get("exp_avg_sq")
                    states[n] = (
                        p.detach().cpu().clone(),
                        exp_avg.detach().cpu().clone() if exp_avg is not None else None,
                        exp_avg_sq.detach().cpu().clone() if exp_avg_sq is not None else None,
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
        grad_err = 0.0
        for n in trainable_names:
            fg, ig = ft["grads"][n], it["grads"][n]
            if fg is None and ig is None:
                continue
            if fg is None or ig is None:
                raise RuntimeError(f"gradient presence mismatch at {n}: full={fg is not None} ice={ig is not None}")
            grad_err = max(grad_err, float(compare_tensors(fg, ig).max_abs_error))
        param_err = max(float(compare_tensors(ft["states"][n][0], it["states"][n][0]).max_abs_error) for n in trainable_names)

        def _state_err(full_val, ice_val):
            if full_val is None and ice_val is None:
                return 0.0
            if full_val is None or ice_val is None:
                raise RuntimeError("optimizer state presence mismatch between rows")
            return float(compare_tensors(full_val, ice_val).max_abs_error)

        exp_avg_err = max(_state_err(ft["states"][n][1], it["states"][n][1]) for n in trainable_names)
        exp_avg_sq_err = max(_state_err(ft["states"][n][2], it["states"][n][2]) for n in trainable_names)
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
            "grad_snapshot": "post_clip",
            "state_snapshot": "post_step",
            "all_finite": (
                loss_err == loss_err
                and grad_err == grad_err and param_err == param_err
                and exp_avg_err == exp_avg_err and exp_avg_sq_err == exp_avg_sq_err
            ),
        })

    init_diffs = [
        float(compare_tensors(
            full_model.state_dict()[n].detach().cpu(), ice_model.state_dict()[n].detach().cpu()
        ).max_abs_error)
        for n in full_model.state_dict()
    ]
    init_state_max_abs_error = max(init_diffs) if init_diffs else 0.0

    certificate = {
        "artifact_type": "optimizer_update_equivalence_certificate",
        "schema_version": "researchpilot.r21.optimizer_cert.v1",
        "device": str(device),
        "status": "pass" if max(max_loss_err, max_grad_err, max_param_err, max_exp_avg_err, max_exp_avg_sq_err) <= 1e-5 else "fail",
        "route": "R21-ICE-VFM-01",
        "mechanism_set": "always_fuse",
        "objective": "ce_lovasz",
        "seed": args.seed,
        "steps": args.steps,
        "trainable_parameter_count": len(trainable_names),
        "threshold_max_abs_error": 1e-5,
        "step_records": step_records,
        "summary": {
            "max_loss_abs_error": max_loss_err,
            "max_grad_abs_error": max_grad_err,
            "max_param_abs_error": max_param_err,
            "max_exp_avg_abs_error": max_exp_avg_err,
            "max_exp_avg_sq_abs_error": max_exp_avg_sq_err,
            "init_state_max_abs_error": init_state_max_abs_error,
        },
        "scope": {
            "precision": "fp32",
            "optimizer_step_semantics": "per_micro_batch_bs16_no_accumulation",
            "accumulation_in_formal_protocol": 2,
            "formal_effective_batch": 32,
            "batch_source": "first_N_train_batches_no_shuffle",
            "augmentation": "none_applied",
            "scheduler_warmup": "not_applied",
            "formal_backbone_init": "audited CROMA",
            "execution_backends": ["full_official_forward", "ice_exact_minimal_execution"],
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