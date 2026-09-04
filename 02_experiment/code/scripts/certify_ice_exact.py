"""Cloud-only R21 ICE-Exact equivalence certification entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml

from geotoken3path.engine.formal_runner import FormalRunnerError, validate_formal_evaluate_paths
from geotoken3path.engine.ice_certifier import run_ice_exact_certification
from geotoken3path.utils.config import resolve_approved_config


_VERIFIED_BASELINE_MIOU_PERCENT = 49.78078791964122


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--expected-miou-percent",
        type=float,
        default=_VERIFIED_BASELINE_MIOU_PERCENT,
        help="Verified best-validation comparator; set explicitly only for an audited replacement checkpoint.",
    )
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

    code_root = Path(__file__).resolve().parents[1]
    resolved = resolve_approved_config(
        code_root,
        "always_fuse",
        execution_scale="acceptance",
    )
    initialization_path = code_root / "configs/model/initialization.yaml"
    initialization_doc = yaml.safe_load(initialization_path.read_text(encoding="utf-8"))
    initialization = initialization_doc.get("initialization")
    if not isinstance(initialization, dict):
        parser.error("initialization.yaml does not contain an initialization mapping")

    output_path = Path(validated["output_dir"]) / "ice_exact_equivalence_certificate.json"
    certificate = run_ice_exact_certification(
        resolved=resolved,
        initialization=initialization,
        data_manifest=validated["data_manifest"],
        audit_report=validated["audit_report"],
        checkpoint=validated["checkpoint"],
        output_path=output_path,
        device=args.device,
        execution_scale="acceptance",
        objective_name="ce_lovasz",
        expected_miou_percent=args.expected_miou_percent,
    )
    print(
        {
            "status": certificate["status"],
            "route": certificate["route"],
            "test_accessed": certificate["test_accessed"],
            "certificate": str(output_path),
            "gates": certificate["gates"],
            "validation": certificate["validation"],
        }
    )
    return 0 if certificate["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
