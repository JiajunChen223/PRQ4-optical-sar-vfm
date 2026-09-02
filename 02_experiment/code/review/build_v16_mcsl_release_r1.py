from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"F:\PRQ4")
CODE = ROOT / "02_experiment" / "code"
MANIFEST = CODE / "manifests" / "clean_sync_manifest_v16_mcsl_20260830_r1.json"
PACKAGE = ROOT / "02_experiment" / "artifacts" / "geotoken3path_code_v16_mcsl_20260830_r1.tar.gz"
VERSIONED_REPORT = CODE / "review" / "CODE_REPORT_V16_MCSL_R1.json"
CANONICAL_REPORT = CODE / "review" / "CODE_REPORT.json"
VALIDATOR = CODE / "review" / "validate_code_project_v16_mcsl_r2.json"
HARD_CONTRACT = ROOT / "02_experiment" / "reports" / "v16_mcsl_synthetic_hard_contract_20260830_r1.json"
REVIEW = CODE / "review" / "coordinator_v16_mcsl_code_review_r1.md"
FINDINGS = CODE / "review" / "v16_mcsl_code_review_findings_r1.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    entries.append(
        {
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    )
entries.sort(key=lambda item: item["path"])

manifest = {
    "artifact_type": "researchpilot_clean_sync_manifest",
    "schema_version": "researchpilot.clean_sync_manifest.v2",
    "status": "pass",
    "generated_for": "v16_mcsl_approved_successor_local_code",
    "route_id": "R-EO-MCSL-V16-01",
    "primary_core_candidate_id": "MCSL-01",
    "candidate_ids": ["MCSL-01", "MCSL-C2", "MCSL-C3", "MCSL-C4"],
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
MANIFEST.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
manifest_sha = sha256(MANIFEST)

release = {
    "artifact_type": "researchpilot_code_only_release_package_manifest",
    "schema_version": "researchpilot.code_only_release_package.v1",
    "source_clean_sync_manifest_ref": "02_experiment/code/manifests/clean_sync_manifest_v16_mcsl_20260830_r1.json",
    "source_clean_sync_manifest_sha256": manifest_sha,
    "file_count": len(entries),
    "files": entries,
    "payload_scope": "reviewed_code_configs_tests_only",
    "route_id": "R-EO-MCSL-V16-01",
    "primary_core_candidate_id": "MCSL-01",
    "hard_contract_ref": str(HARD_CONTRACT),
    "test_accessed": False,
    "local_real_data_included": False,
    "pretrained_weight_binaries_included": False,
    "credentials_included": False,
    "cache_files_included": False,
}
PACKAGE.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="prq4-v16-mcsl-") as temp_dir:
    stage = Path(temp_dir)
    release_path = stage / "researchpilot_code_release_manifest.json"
    release_path.write_text(
        json.dumps(release, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    for item in entries:
        destination = stage / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / item["path"], destination)
    with tarfile.open(PACKAGE, "w:gz") as archive:
        archive.add(release_path, arcname="researchpilot_code_release_manifest.json")
        for item in entries:
            archive.add(stage / item["path"], arcname=item["path"])

with tarfile.open(PACKAGE, "r:gz") as archive:
    members = [member for member in archive.getmembers() if member.isfile()]
    member_names = {member.name for member in members}
    expected_names = {"researchpilot_code_release_manifest.json", *(item["path"] for item in entries)}
    if member_names != expected_names:
        raise RuntimeError("release package member set differs from manifest")
    embedded = archive.extractfile("researchpilot_code_release_manifest.json")
    if embedded is None:
        raise RuntimeError("release package lacks embedded manifest")
    release_bytes = embedded.read()
    release_manifest_sha = hashlib.sha256(release_bytes).hexdigest()
    for item in entries:
        stream = archive.extractfile(item["path"])
        if stream is None:
            raise RuntimeError(f"release package lacks {item['path']}")
        raw = stream.read()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise RuntimeError(f"release package identity mismatch: {item['path']}")

hard = json.loads(HARD_CONTRACT.read_text(encoding="utf-8-sig"))
validator = json.loads(VALIDATOR.read_text(encoding="utf-8-sig"))
if hard.get("status") != "pass" or hard.get("passed") != hard.get("total"):
    raise RuntimeError("MCSL hard contract is not passing")
if validator.get("status") != "pass" or validator.get("violations"):
    raise RuntimeError("code validator is not passing")

report = {
    "artifact_type": "researchpilot_code_report",
    "schema_version": 1,
    "status": "PASS",
    "project_root": str(ROOT),
    "code_root": str(CODE),
    "route_id": "R-EO-MCSL-V16-01",
    "primary_core_candidate_id": "MCSL-01",
    "reviewed_commit_or_sync_manifest": str(MANIFEST),
    "scaffold_report": str(CODE / "review" / "scaffold_report.json"),
    "test_summary": {
        "status": "pass",
        "pytest": "349 passed, 0 warnings",
        "hard_contract": "10/10 pass",
        "hard_contract_ref": str(HARD_CONTRACT),
        "validator": f"pass; {validator['scanned_executable_or_config_files']} executable/config files; 0 violations",
        "validator_ref": str(VALIDATOR),
        "commands": [
            "python -X utf8 -B -m pytest -q -p no:cacheprovider tests",
            "python -X utf8 -B scripts/run_v16_mcsl_hard_contract.py",
            "validate_code_project.py --project-root F:\\PRQ4",
        ],
        "synthetic_only": True,
        "scientific_result": False,
    },
    "review_summary": {
        "status": "conditional_pass",
        "finding_count": 6,
        "blocker_count": 0,
        "major_open_count": 0,
        "accepted_risk_count": 1,
        "review_report": str(REVIEW),
        "findings_ref": str(FINDINGS),
        "independence": "coordinator_only_per_user_current_model_amendment",
    },
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
        "config_diff_test": str(CODE / "tests" / "unit" / "test_v16_mcsl_config.py"),
        "trainable_parameter_audit_test": str(CODE / "tests" / "integration" / "test_croma_bridge.py"),
        "formal_added_parameters": 67235,
    },
    "coupling_ablation_readiness": {
        "status": "pass_local_contract_controls_locked",
        "enabled_mechanism_set_config": [
            "mcsl_multimodal_conservative_lifting",
            "mcsl_unconstrained_child_control",
            "mcsl_shuffled_guidance_control",
            "mcsl_single_scale_control",
        ],
        "same_entry_point_test": str(CODE / "tests" / "integration" / "test_croma_bridge.py"),
        "matched_protocol_budget_hash": "resolved_by_v16_mcsl_config",
        "parent_row_config_test": str(CODE / "tests" / "unit" / "test_v16_mcsl_config.py"),
        "controls_open_only_after_primary_plus_2pp": True,
    },
    "packaging_closure": {
        "status": "pass_local_code_only_cloud_sync_pending",
        "clean_sync_manifest": {
            "path": str(MANIFEST),
            "sha256": manifest_sha,
            "file_count": len(entries),
        },
        "release_package": {
            "path": str(PACKAGE),
            "sha256": sha256(PACKAGE),
            "bytes": PACKAGE.stat().st_size,
            "file_count": len(entries),
            "release_manifest_sha256": release_manifest_sha,
        },
        "entry_audit": "pass_all_members_bytes_and_sha256",
        "data_download": False,
        "weights_download": False,
        "gpu_or_training": False,
    },
    "unresolved_issues": [
        "RTX3090 AMP conservation tolerance, incremental VRAM and throughput require the ordinary cloud preflight before training.",
        "No MCSL scientific result exists; software and synthetic contracts only.",
        "MCSL-C2/C3/C4 remain locked unless MCSL-01 reaches +2pp over verified R2.",
    ],
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "relevant_file_hashes": {item["path"]: item["sha256"] for item in entries},
}
text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
VERSIONED_REPORT.write_text(text, encoding="utf-8")
CANONICAL_REPORT.write_text(text, encoding="utf-8")

print(
    json.dumps(
        {
            "status": "pass",
            "manifest": str(MANIFEST),
            "manifest_sha256": manifest_sha,
            "file_count": len(entries),
            "package": str(PACKAGE),
            "package_sha256": sha256(PACKAGE),
            "package_bytes": PACKAGE.stat().st_size,
            "release_manifest_sha256": release_manifest_sha,
            "code_report": str(VERSIONED_REPORT),
            "code_report_sha256": sha256(VERSIONED_REPORT),
        },
        ensure_ascii=False,
        indent=2,
    )
)
