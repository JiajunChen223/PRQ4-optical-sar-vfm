"""Validation-only evaluation entry point.

The test split remains sealed.  The ``cloud`` mode performs only a fail-closed
path/control preflight locally; it deliberately does not read data, weights,
checkpoints, construct a model, inspect a device, or execute cloud work.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from geotoken3path.metrics import confusion_matrix, mean_iou
from geotoken3path.engine.formal_runner import FormalRunnerError, validate_formal_evaluate_paths
from geotoken3path.models.factory import build_model
from geotoken3path.utils.config import resolve_approved_config
from geotoken3path.utils.run_manifest import build_run_manifest
from geotoken3path.utils.test_seal import assert_test_access_allowed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism-set",
        choices=("always_fuse", "r2_depth_group_inject", "r1_low_energy_channel_gain", "r3_optical_conditional_depth_select", "r6_depth_dual_channel_inject"),
        default="always_fuse",
    )
    parser.add_argument("--execution-scale", choices=("smoke", "cloud"), default="smoke")
    parser.add_argument("--data-manifest")
    parser.add_argument("--audit-report")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.execution_scale == "cloud":
        required = {
            "--data-manifest": args.data_manifest,
            "--audit-report": args.audit_report,
            "--checkpoint": args.checkpoint,
            "--output": args.output,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error("cloud evaluation requires " + ", ".join(missing))
        try:
            paths = validate_formal_evaluate_paths(
                data_manifest=args.data_manifest,
                audit_report=args.audit_report,
                checkpoint=args.checkpoint,
                output_dir=args.output,
                execution_scale="cloud",
            )
        except FormalRunnerError as exc:
            parser.error(str(exc))
        print({
            "status": "formal_evaluate_request_validated",
            "cloud_execution": False,
            "data_accessed": False,
            "checkpoint_loaded": False,
            "gpu_used": False,
            "test_accessed": False,
            "paths": paths,
        })
        return 0
    assert_test_access_allowed({"execution_scale": "smoke", "test_seal_status": "sealed"}, "validation")
    code_root = Path(__file__).resolve().parents[1]
    resolver = resolve_approved_config
    resolved = resolver(code_root, args.mechanism_set)
    model = build_model(resolved)
    dim = resolved["model"]["token_dim"]
    classes = resolved["model"]["num_classes"]
    with torch.no_grad():
        token_count = 16
        token_side = int(token_count**0.5)
        logits = model(
            torch.zeros(1, token_count, dim),
            torch.zeros(1, token_count, dim),
            depth_group=torch.zeros(1, token_count, 4, dim),
            output_size=(token_side, token_side),
        )
        target = torch.zeros(1, token_side, token_side, dtype=torch.long)
        score = mean_iou(confusion_matrix(logits, target, classes))
    run_manifest = build_run_manifest(resolved, seed=resolved["runtime"]["seed"], split="validation", execution_scale="smoke")
    print({
        "status": "synthetic_validation_contract_pass",
        "scientific_result": False,
        "logits_shape": list(logits.shape),
        "metric_is_finite": bool(torch.isfinite(score).item()),
        "protocol_sha256": resolved["matched_common_protocol_sha256"],
        "run_contract_sha256": run_manifest["run_contract_sha256"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
