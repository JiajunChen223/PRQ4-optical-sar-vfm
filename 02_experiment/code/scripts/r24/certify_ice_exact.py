"""R24 SkySense++ S2 ICE-Exact equivalence certification entry point (cloud).

Calls ``engine.ice_certifier_r24.run_skysensepp_ice_certification`` and writes
the ``prq4.skysensepp_ice_exact_certificate.v1`` certificate next to the head
checkpoint.  Exit code 0 on a passing certificate, 2 on a failing one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import yaml

from geotoken3path.engine.ice_certifier_r24 import (
    SkySensePPCertificationError,
    run_skysensepp_ice_certification,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--weights", required=True, help="Audited SkySense++ S2 safetensors")
    parser.add_argument("--head-checkpoint", required=True, help="Trained head checkpoint (best/last)")
    parser.add_argument("--contract", choices=("a", "b"), default="b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--execution-scale",
        choices=("smoke", "acceptance"),
        default="acceptance",
    )
    parser.add_argument(
        "--expected-miou-percent",
        type=float,
        default=None,
        help="Best-validation mIoU from the training run; guard against a wrong checkpoint.",
    )
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    resolved: dict[str, object] = {
        "matched_common_protocol_sha256": None,
        "code_sync_manifest_sha256": None,
        "runtime": {
            "contract": args.contract,
            "micro_batch": 16,
            "resolution": 120,
            "seed": 0,
            "precision": "amp",
            "objective_name": "ce_lovasz",
            "augmentation": {
                "name": "paired_geometric_v1",
                "enabled": True,
                "train_only": True,
                "deterministic": True,
                "orientation_space": "D4",
                "operations": [
                    "horizontal_flip", "vertical_flip", "rotate_90", "rotate_180",
                    "rotate_270", "transpose", "anti_transpose",
                ],
            },
        },
    }
    runtime_config = code_root / "configs" / "r24" / "acceptance.yaml"
    if runtime_config.is_file():
        try:
            loaded = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            parser.error(f"cannot parse configs/r24/acceptance.yaml: {exc}")
        if isinstance(loaded, dict) and isinstance(loaded.get("runtime"), dict):
            resolved["runtime"] = {**resolved["runtime"], **loaded["runtime"]}  # type: ignore[operator]
            resolved["runtime"]["contract"] = args.contract  # type: ignore[index]

    initialization: dict[str, object] = {}
    initialization_path = code_root / "configs" / "model" / "initialization.yaml"
    if initialization_path.is_file():
        try:
            doc = yaml.safe_load(initialization_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            doc = None
        if isinstance(doc, dict) and isinstance(doc.get("initialization"), dict):
            initialization = doc["initialization"]

    output_path = Path(args.output_dir) / "skysensepp_ice_exact_certificate.json"
    try:
        certificate = run_skysensepp_ice_certification(
            resolved=resolved,
            initialization=initialization,
            data_manifest=args.data_manifest,
            safetensors_path=args.weights,
            head_checkpoint=args.head_checkpoint,
            output_path=output_path,
            device=args.device,
            execution_scale=args.execution_scale,
            objective_name="ce_lovasz",
            expected_miou_percent=args.expected_miou_percent,
        )
    except SkySensePPCertificationError as exc:
        parser.error(str(exc))
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
