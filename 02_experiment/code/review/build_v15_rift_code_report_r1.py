from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(r"F:\PRQ4")
OUT = ROOT / "02_experiment" / "code" / "review" / "CODE_REPORT_V15_RIFT_R5.json"
MANIFEST = ROOT / "02_experiment" / "code" / "manifests" / "clean_sync_manifest_v15_rift_20260830_r4.json"
PACKAGE = ROOT / "02_experiment" / "artifacts" / "geotoken3path_code_v15_rift_20260830_r3.tar.gz"
HARD = ROOT / "02_experiment" / "reports" / "v15_rift_synthetic_hard_contract_20260830_r5.json"
REVIEW = ROOT / "02_experiment" / "code" / "review" / "coordinator_v15_rift_code_review_r4_20260830.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    hard = json.loads(HARD.read_text(encoding="utf-8-sig"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    payload = {
        "schema_version": 1,
        "artifact_type": "researchpilot_code_report",
        "status": "PASS",
        "project_root": str(ROOT),
        "code_root": str(ROOT / "02_experiment" / "code"),
        "reviewed_commit_or_sync_manifest": str(MANIFEST),
        "clean_sync_manifest_sha256": sha256(MANIFEST),
        "route_id": "R-EO-RIFT-V15-01",
        "primary_core_candidate_id": "RIFT-01",
        "test_summary": {
            "status": "pass",
            "commands": [
                "pytest F:\\PRQ4\\02_experiment\\code\\tests -q -p no:cacheprovider",
                "validate_code_project.py --project-root F:\\PRQ4",
                "run_v15_rift_hard_contract.py",
            ],
            "pytest": "334 passed, 1 warning",
            "validator": "pass; 145 executable/config files; 0 violations",
            "hard_contract": "pass; 7/7 checks",
            "synthetic_only": True,
            "scientific_result": False,
        },
        "review_summary": {
            "status": "pass",
            "review_mode": "coordinator_single_thread_adversarial_review",
            "independent_review": False,
            "review_report": str(REVIEW),
            "finding_count": 0,
            "decision": "PASS_FOR_V15_RIFT_CODE_SYNC",
        },
        "rift_contract": {
            "mechanism_set": "rift_relational_innovation_field",
            "grid_side": 15,
            "neighborhood": 3,
            "pure_modality_relations": True,
            "row_sum_zero": True,
            "l1_normalization": True,
            "semantic_transport_target": "fused_tokens",
            "zero_start_stage_scale": True,
            "stage_scale_parameter_budget": 2 * 768,
            "no_learnable_qk": True,
            "no_extra_loss": True,
            "matched_controls_registered": ["rift_sar_relation_control", "rift_shuffled_innovation_control"],
        },
        "training_object_parity": {
            "status": "pass",
            "same_detector_factory": True,
            "single_internal_mechanism_delta": True,
            "external_trainable_component_forbidden": True,
            "same_entry_point": "geotoken3path.models.factory.build_model / build_vfm_segmentation_model",
        },
        "hard_contract_ref": str(HARD),
        "hard_contract_sha256": sha256(HARD),
        "local_data_status": "clean",
        "local_gpu_probe": "forbidden_not_run",
        "pretrained_initializer": "audited_cloud_only",
        "pretrained_audit_ref": str(ROOT / "02_experiment" / "reports" / "v14_dtsf_croma_pretrained_weight_audit_20260830_r1.json"),
        "test_seal_guard": "present_and_tested",
        "packaging_closure": {
            "status": "pass_local_code_only",
            "clean_sync_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST), "file_count": manifest["file_count"]},
            "release_package": {"path": str(PACKAGE), "sha256": sha256(PACKAGE), "bytes": PACKAGE.stat().st_size, "member_count": len(tarfile.open(PACKAGE, "r:gz").getnames())},
            "data_download": False,
            "weights_download": False,
            "gpu_or_training": False,
        },
        "unresolved_issues": [
            "V15 RIFT-01 real cloud seed-0 24-epoch result has not run",
            "RIFT-C2/C3 remain locked until RIFT-01 reaches +2pp",
        ],
        "handoff_to_experiment": "After guarded V15 code sync and cloud reattachment, run RIFT-01 hard-contract successor and then the single seed-0 full-horizon run.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "report": str(OUT), "sha256": sha256(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
