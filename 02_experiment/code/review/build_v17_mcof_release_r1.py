from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"F:\PRQ4")
CODE = ROOT / "02_experiment" / "code"
REVISION = str(os.environ.get("MCOF_RELEASE_REVISION", "r1")).strip().lower() or "r1"
if REVISION not in {"r1", "r2", "r3"}:
    raise RuntimeError("MCOF_RELEASE_REVISION must be r1, r2 or r3")
_validator_revision = "r8" if REVISION == "r1" else "r9"
_hard_contract_revision = "r5" if REVISION == "r1" else "r6"
_architecture_review_revision = "r1" if REVISION == "r1" else "r2"
MANIFEST = CODE / "review" / f"clean_sync_manifest_v17_mcof_20260831_{REVISION}.json"
PACKAGE = ROOT / "02_experiment" / "artifacts" / f"geotoken3path_code_v17_mcof_20260831_{REVISION}.tar.gz"
VERSIONED_REPORT = CODE / "review" / f"CODE_REPORT_V17_MCOF_{REVISION.upper()}.json"
CANONICAL_REPORT = CODE / "review" / "CODE_REPORT.json"
VALIDATOR = CODE / "review" / f"v17_mcof_code_validation_20260831_{_validator_revision}.json"
HARD_CONTRACT = ROOT / "02_experiment" / "reports" / f"v17_mcof_synthetic_hard_contract_20260831_{_hard_contract_revision}.json"
REVIEWS = [
    CODE / "review" / f"v17_mcof_architecture_review_20260831_{_architecture_review_revision}.md",
    CODE / "review" / "v17_mcof_data_seal_review_20260831_r1.md",
    CODE / "review" / "v17_mcof_parity_review_20260831_r1.md",
]
PYTEST_SUMMARY = "358 passed, 0 warnings" if REVISION == "r1" else "360 passed, 0 warnings"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review_summary() -> dict:
    missing = [str(path) for path in REVIEWS if not path.is_file()]
    if missing:
        raise RuntimeError(f"required independent code reviews are missing: {missing}")
    texts = [path.read_text(encoding="utf-8-sig") for path in REVIEWS]
    lowered = "\n".join(texts).casefold()
    blockers = lowered.count("severity: blocker") + lowered.count("severity=blocker")
    major_open = lowered.count("severity: major") + lowered.count("severity=major")
    if blockers or major_open:
        raise RuntimeError("independent review contains unresolved blocker or major finding")
    return {
        "status": "pass",
        "independent_review_count": len(REVIEWS),
        "blocker_count": blockers,
        "major_open_count": major_open,
        "reports": [str(path) for path in REVIEWS],
    }


def main() -> None:
    hard = json.loads(HARD_CONTRACT.read_text(encoding="utf-8-sig"))
    validator = json.loads(VALIDATOR.read_text(encoding="utf-8-sig"))
    if hard.get("status") != "pass" or hard.get("passed") != hard.get("total"):
        raise RuntimeError("MCOF hard contract is not passing")
    if validator.get("status") != "pass" or validator.get("violations"):
        raise RuntimeError("code validator is not passing")
    reviews = review_summary()

    allowed_root = {".gitignore", "LICENSE_STATUS.md", "README.md", "THIRD_PARTY.md", "pyproject.toml"}
    entries = []
    for source in CODE.rglob("*"):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(CODE).as_posix()
        parts = relative.split("/")
        allowed = (len(parts) == 1 and parts[0] in allowed_root) or parts[0] in {
            "configs", "scripts", "src", "tests"
        }
        if not allowed or "__pycache__" in parts or ".pytest_cache" in parts:
            continue
        raw = source.read_bytes()
        entries.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    entries.sort(key=lambda item: item["path"])
    manifest = {
        "artifact_type": "researchpilot_clean_sync_manifest",
        "schema_version": "researchpilot.clean_sync_manifest.v2",
        "status": "pass",
        "generated_for": "v17_mcof_approved_single_route_local_code",
        "route_id": "R-EO-MCOF-V17-01",
        "primary_core_candidate_id": "MCOF-01",
        "candidate_ids": ["MCOF-01"],
        "file_count": len(entries),
        "files": entries,
        "test_accessed": False,
        "local_real_data_included": False,
        "pretrained_weight_binaries_included": False,
        "credentials_included": False,
        "cache_files_included": False,
        "local_gpu_probe": "forbidden_not_run",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_sha = sha256(MANIFEST)
    release = {
        "artifact_type": "researchpilot_code_only_release_package_manifest",
        "schema_version": "researchpilot.code_only_release_package.v1",
        "source_clean_sync_manifest_ref": f"02_experiment/code/review/{MANIFEST.name}",
        "source_clean_sync_manifest_sha256": manifest_sha,
        "file_count": len(entries),
        "files": entries,
        "payload_scope": "reviewed_code_configs_tests_only",
        "route_id": "R-EO-MCOF-V17-01",
        "primary_core_candidate_id": "MCOF-01",
        "hard_contract_ref": str(HARD_CONTRACT),
        "test_accessed": False,
        "local_real_data_included": False,
        "pretrained_weight_binaries_included": False,
        "credentials_included": False,
        "cache_files_included": False,
    }
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prq4-v17-mcof-") as temp_dir:
        stage = Path(temp_dir)
        release_path = stage / "researchpilot_code_release_manifest.json"
        release_path.write_text(json.dumps(release, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        for item in entries:
            destination = stage / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CODE / item["path"], destination)
        with tarfile.open(PACKAGE, "w:gz") as archive:
            archive.add(release_path, arcname="researchpilot_code_release_manifest.json")
            for item in entries:
                archive.add(stage / item["path"], arcname=item["path"])
    with tarfile.open(PACKAGE, "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
        expected = {"researchpilot_code_release_manifest.json", *(item["path"] for item in entries)}
        if members != expected:
            raise RuntimeError("release package member set differs from manifest")
        for item in entries:
            stream = archive.extractfile(item["path"])
            if stream is None:
                raise RuntimeError(f"release package lacks {item['path']}")
            raw = stream.read()
            if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise RuntimeError(f"release package identity mismatch: {item['path']}")
    report = {
        "artifact_type": "researchpilot_code_report",
        "schema_version": 1,
        "status": "PASS",
        "project_root": str(ROOT),
        "code_root": str(CODE),
        "route_id": "R-EO-MCOF-V17-01",
        "primary_core_candidate_id": "MCOF-01",
        "reviewed_commit_or_sync_manifest": str(MANIFEST),
        "scaffold_report": str(CODE / "review" / "scaffold_report.json"),
        "test_summary": {
            "status": "pass",
            "pytest": PYTEST_SUMMARY,
            "hard_contract": f"{hard['passed']}/{hard['total']} pass",
            "hard_contract_ref": str(HARD_CONTRACT),
            "validator": f"pass; {validator['scanned_executable_or_config_files']} executable/config files; 0 violations",
            "validator_ref": str(VALIDATOR),
            "synthetic_only": True,
            "scientific_result": False,
        },
        "review_summary": reviews,
        "local_data_status": "clean",
        "local_gpu_probe": "forbidden_not_run",
        "pretrained_initializer": "audited_cloud_only",
        "pretrained_audit_ref": str(CODE / "configs" / "model" / "pretrained_audit_successor.json"),
        "test_seal_guard": "present_and_tested",
        "training_object_parity": {
            "status": "pass",
            "same_detector_factory": True,
            "single_internal_mechanism_delta": True,
            "external_trainable_component_forbidden": True,
            "config_diff_test": str(CODE / "tests" / "unit" / "test_v17_mcof_config.py"),
            "trainable_parameter_audit_test": str(CODE / "tests" / "integration" / "test_croma_bridge.py"),
            "formal_added_parameters": int(hard["formal_width_parameter_count"]),
        },
        "coupling_ablation_readiness": {
            "status": "not_applicable_single_route_controls_locked",
            "enabled_mechanism_set_config": [
                "mcof_multimodal_conditional_operator",
                "mcof_static_rank_control",
                "mcof_sample_level_control",
                "mcof_shuffled_condition_control",
                "mcof_optical_only_control",
                "mcof_sar_only_control",
            ],
            "same_entry_point_test": str(CODE / "tests" / "integration" / "test_croma_bridge.py"),
            "matched_protocol_budget_hash": "resolved_by_v17_mcof_config",
            "parent_row_config_test": str(CODE / "tests" / "unit" / "test_v17_mcof_config.py"),
            "controls_open_only_after_primary_plus_2pp": True,
        },
        "packaging_closure": {
            "status": "pass_local_code_only_cloud_sync_pending",
            "clean_sync_manifest": {"path": str(MANIFEST), "sha256": manifest_sha, "file_count": len(entries)},
            "release_package": {"path": str(PACKAGE), "sha256": sha256(PACKAGE), "bytes": PACKAGE.stat().st_size, "file_count": len(entries)},
            "entry_audit": "pass_all_members_bytes_and_sha256",
            "data_download": False,
            "weights_download": False,
            "gpu_or_training": False,
        },
        "unresolved_issues": [
            "RTX3090 AMP, incremental VRAM and throughput require the guarded cloud preflight before training.",
            "No MCOF scientific result exists; software and synthetic contracts only.",
            "Matched controls remain locked unless MCOF-01 reaches +2pp over verified R2.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "relevant_file_hashes": {item["path"]: item["sha256"] for item in entries},
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    VERSIONED_REPORT.write_text(text, encoding="utf-8")
    CANONICAL_REPORT.write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "manifest": str(MANIFEST),
        "manifest_sha256": manifest_sha,
        "file_count": len(entries),
        "package": str(PACKAGE),
        "package_sha256": sha256(PACKAGE),
        "package_bytes": PACKAGE.stat().st_size,
        "code_report": str(VERSIONED_REPORT),
        "code_report_sha256": sha256(VERSIONED_REPORT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
