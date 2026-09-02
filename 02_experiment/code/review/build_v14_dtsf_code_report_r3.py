from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(r"F:\PRQ4")
OLD = ROOT / "02_experiment" / "code" / "review" / "CODE_REPORT_V14_DTSF_R2.json"
OUT = ROOT / "02_experiment" / "code" / "review" / "CODE_REPORT_V14_DTSF_R3.json"
MANIFEST = ROOT / "02_experiment" / "code" / "manifests" / "clean_sync_manifest_v14_dtsf_20260830_r5.json"
PACKAGE = ROOT / "02_experiment" / "artifacts" / "geotoken3path_code_v14_dtsf_20260830_r5.tar.gz"
REVIEW = ROOT / "02_experiment" / "code" / "review" / "coordinator_v14_dtsf_code_review_r3_20260830.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    report = json.loads(OLD.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report.update({
        "reviewed_commit_or_sync_manifest": str(MANIFEST),
        "clean_sync_manifest_sha256": sha256(MANIFEST),
        "test_summary": {
            "status": "pass",
            "commands": [
                "pytest F:\\PRQ4\\02_experiment\\code\\tests -q -p no:cacheprovider",
                "validate_code_project.py --project-root F:\\PRQ4",
            ],
            "pytest": "329 passed, 1 warning",
            "validator": "pass; 137 executable/config files; 0 violations",
            "synthetic_only": True,
            "scientific_result": False,
        },
        "review_summary": {
            "status": "pass",
            "review_mode": "coordinator_single_thread_adversarial_repair_review",
            "independent_review": False,
            "review_report": str(REVIEW),
            "finding_count": 0,
            "decision": "PASS_FOR_V14_DTSF_R5_CLOUD_SYNC",
        },
        "repair_delta": {
            "type": "formal_direction_allowlist_repair",
            "failed_run": "PRQ4-V14-DTSF-DTSF01-FORMAL24-SEED0-20260830-R1",
            "root_cause": "run_manifest.py omitted DTSF formal direction IDs",
            "repaired_ids": ["DTSF-01", "DTSF-C2", "DTSF-C3", "DTSF-C4"],
            "scientific_variables_changed": False,
            "failed_run_scientific_result": False,
        },
        "packaging_closure": {
            "status": "pass_local_code_only",
            "clean_sync_manifest": {
                "path": str(MANIFEST),
                "sha256": sha256(MANIFEST),
                "file_count": manifest["file_count"],
            },
            "release_package": {
                "path": str(PACKAGE),
                "sha256": sha256(PACKAGE),
                "bytes": PACKAGE.stat().st_size,
                "member_count": len(tarfile.open(PACKAGE, "r:gz").getnames()),
            },
        },
        "unresolved_issues": [
            "DTSF-01 fresh cloud seed-0 24-epoch validation result is pending after r5 synchronization",
            "DTSF controls remain locked behind the >=+2pp performance gate",
        ],
        "handoff_to_experiment": "After r5 guarded sync and environment reattachment, create a fresh DTSF-01 seed-0 full-horizon run ID; do not reuse the invalid R1 run.",
    })
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "report": str(OUT), "sha256": sha256(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
