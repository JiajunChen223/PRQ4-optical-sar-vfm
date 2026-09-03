"""Configuration-driven entry point with a synthetic-only local smoke lane.

Real-data training is intentionally refused here until the cloud approval and
preflight controls are satisfied.  The same entry point is used for the
verified baseline and the D1/D2/D3 diagnostic mechanisms on the authorized
cloud host.

Two-zone cleanup 2026-09-02: rejected route variants, mechanism sets and
calibration knobs were removed; archives live in
20_HISTORY/02_legacy_code_pkgs/rejected_mechanisms_20260902/.
"""

import argparse
import math
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from geotoken3path.losses import segmentation_cross_entropy, segmentation_objective
from geotoken3path.models.factory import build_model
from geotoken3path.utils.config import resolve_approved_config
from geotoken3path.utils.run_manifest import build_run_manifest
from geotoken3path.utils.test_seal import assert_test_access_allowed
from geotoken3path.engine.formal_runner import run_formal_cloud


_APPROVED_MECHANISM_SETS = (
    "always_fuse",
    "r2_depth_group_inject",
    "r1_low_energy_channel_gain",
)


def _cosine_warmup_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def run_synthetic_smoke(resolved: dict[str, object]) -> dict[str, object]:
    """Execute the declared CPU smoke semantics without producing science evidence."""

    runtime = resolved["runtime"]
    model_cfg = resolved["model"]
    if not isinstance(runtime, dict) or not isinstance(model_cfg, dict):
        raise ValueError("resolved smoke config is malformed")
    optimizer_cfg = runtime["optimizer"]
    scheduler_cfg = runtime["scheduler"]
    if not isinstance(optimizer_cfg, dict) or not isinstance(scheduler_cfg, dict):
        raise ValueError("resolved optimizer/scheduler config is malformed")

    seed = int(runtime["seed"])
    micro_batch = int(runtime["micro_batch"])
    accumulation_steps = int(runtime["gradient_accumulation"])
    effective_batch = int(runtime["effective_batch"])
    if effective_batch != micro_batch * accumulation_steps:
        raise ValueError("effective batch does not match accumulation semantics")
    torch.manual_seed(seed)
    model = build_model(resolved)
    model.train()
    dim = int(model_cfg["token_dim"])
    classes = int(model_cfg["num_classes"])
    token_count = 16
    token_side = int(math.isqrt(token_count))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_cfg["betas"]),
    )
    total_optimizer_steps = 1
    warmup_steps = int(math.ceil(float(scheduler_cfg["warmup_fraction"]) * total_optimizer_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_warmup_multiplier(
            step,
            total_steps=total_optimizer_steps,
            warmup_steps=warmup_steps,
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    accumulation_counter = 0
    last_logits: torch.Tensor | None = None
    last_aux: dict[str, object] | None = None
    autocast_enabled = str(runtime["precision"]).casefold() == "amp"
    objective_spec = resolved.get("objective", {})
    raw_objective = objective_spec.get("id") if isinstance(objective_spec, dict) else None
    if not raw_objective:
        raw_objective = runtime.get("objective_name") or "pixel_ce"
    objective_name = str(raw_objective).casefold()
    for _ in range(accumulation_steps):
        optical = torch.randn(micro_batch, token_count, dim)
        sar = torch.randn(micro_batch, token_count, dim)
        depth_group = torch.randn(micro_batch, token_count, 4, dim)
        target = torch.randint(0, classes, (micro_batch, token_side, token_side))
        with torch.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits, aux = model(
                optical,
                sar,
                depth_group=depth_group,
                output_size=(token_side, token_side),
                return_aux=True,
            )
            if objective_name == "pixel_ce":
                loss = segmentation_cross_entropy(logits, target)
            else:
                loss, _ = segmentation_objective(logits, target, objective_name=objective_name)
            loss = loss / accumulation_steps
        loss.backward()
        accumulation_counter += 1
        accumulated_loss += float(loss.detach())
        last_logits = logits
        last_aux = aux

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(runtime["gradient_clip_norm"]),
    )
    optimizer.step()
    scheduler.step()
    run_manifest = build_run_manifest(
        resolved,
        seed=seed,
        split="validation",
        execution_scale="smoke",
    )
    assert last_logits is not None and last_aux is not None
    return {
        "status": "synthetic_segmentation_step_pass",
        "scientific_result": False,
        "logits_shape": list(last_logits.shape),
        "active_fraction": float(last_aux["active_fraction"].detach()),
        "loss_is_finite": math.isfinite(accumulated_loss),
        "protocol_sha256": resolved["matched_common_protocol_sha256"],
        "run_contract_sha256": run_manifest["run_contract_sha256"],
        "optimizer_name": optimizer.__class__.__name__.casefold(),
        "optimizer_learning_rate": float(optimizer.defaults["lr"]),
        "optimizer_weight_decay": float(optimizer.defaults["weight_decay"]),
        "optimizer_betas": list(optimizer.defaults["betas"]),
        "optimizer_steps": 1,
        "scheduler_name": "cosine_with_warmup",
        "scheduler_steps": int(scheduler.last_epoch),
        "gradient_accumulation_steps": accumulation_counter,
        "gradient_clip_applied": True,
        "gradient_clip_max_norm": float(runtime["gradient_clip_norm"]),
        "gradient_norm_is_finite": bool(torch.isfinite(gradient_norm).item()),
        "autocast_device": "cpu",
        "autocast_enabled": autocast_enabled,
        "micro_batch": micro_batch,
        "effective_batch": effective_batch,
        "objective_name": objective_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism-set",
        choices=_APPROVED_MECHANISM_SETS,
        default="always_fuse",
    )
    parser.add_argument("--candidate-direction-id", choices=("BASELINE",))
    parser.add_argument("--execution-scale", choices=("smoke", "cloud"), default="smoke")
    parser.add_argument(
        "--formal-scale",
        choices=("baseline", "strengthening", "screening", "confirmation", "acceptance", "extension"),
        default="baseline",
    )
    parser.add_argument("--data-manifest")
    parser.add_argument("--audit-report")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--objective",
        choices=("pixel_ce", "macro_ce", "ce_lovasz", "macro_ce_lovasz"),
        default="ce_lovasz",
        help="Locked baseline objective (CE+Lovasz); resolved config does not declare it",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rapid-horizon-epochs", type=int, default=5)
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Screening/formal seed (protocol seeds [0,1,2]); injected into resolved runtime",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = {"execution_scale": args.execution_scale, "test_seal_status": "sealed"}
    assert_test_access_allowed(manifest, "validation")
    code_root = Path(__file__).resolve().parents[1]
    if args.execution_scale == "cloud":
        if not args.data_manifest or not args.audit_report or not args.output_dir:
            raise RuntimeError("formal cloud execution requires --data-manifest, --audit-report and --output-dir")
        resolved = resolve_approved_config(code_root, args.mechanism_set, execution_scale=args.formal_scale)
        resolved["runtime"] = dict(resolved["runtime"])
        resolved["runtime"]["seed"] = int(args.seed)
        init_cfg = json.loads(json.dumps({}))
        try:
            import yaml
            init_cfg = yaml.safe_load((code_root / "configs/model/initialization.yaml").read_text(encoding="utf-8"))["initialization"]
        except Exception as exc:
            raise RuntimeError("cannot resolve initialization config") from exc
        result = run_formal_cloud(
            code_root=code_root, resolved=resolved, data_manifest=args.data_manifest,
            audit_report=args.audit_report, initialization=init_cfg,
            output_dir=args.output_dir, mechanism_set=args.mechanism_set,
            execution_scale=args.formal_scale, epochs=args.epochs, rapid_horizon_epochs=args.rapid_horizon_epochs,
            device=args.device, candidate_direction_id=args.candidate_direction_id,
            # Resolved config does not declare an objective; the CLI flag
            # carries the locked baseline objective (CE+Lovasz default).
            objective_name=args.objective,
        )
        print(result)
        return 0
    resolved = resolve_approved_config(code_root, args.mechanism_set)
    print(run_synthetic_smoke(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())