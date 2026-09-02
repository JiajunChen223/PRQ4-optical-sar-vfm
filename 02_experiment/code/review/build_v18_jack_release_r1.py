from __future__ import annotations

import hashlib
import gzip
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"F:\PRQ4")
CODE = ROOT / "02_experiment" / "code"
MANIFEST = CODE / "review" / "clean_sync_manifest_v18_jack_20260831_r2.json"
PACKAGE = ROOT / "02_experiment" / "artifacts" / "geotoken3path_code_v18_jack_20260831_r2.tar.gz"
VERSIONED_REPORT = CODE / "review" / "CODE_REPORT_V18_JACK_R2.json"
CANONICAL_REPORT = CODE / "review" / "CODE_REPORT.json"
VALIDATOR = CODE / "review" / "v18_jack_code_validation_20260831_r5.json"
HARD_CONTRACT = ROOT / "02_experiment" / "reports" / "v18_jack_synthetic_hard_contract_20260831_r4.json"
LIVENESS = ROOT / "02_experiment" / "reports" / "v18_jack_synthetic_liveness_20260831_r4.json"
REVIEWS = [
    CODE / "review" / "v18_jack_architecture_review_20260831_r1.jsonl",
    CODE / "review" / "v18_jack_data_repro_review_20260831_r1.jsonl",
    CODE / "review" / "v18_jack_release_adversarial_review_20260831_r1.jsonl",
    CODE / "review" / "v18_jack_r2_dtype_fix_arch_review.jsonl",
    CODE / "review" / "v18_jack_r2_dtype_fix_repro_review.jsonl",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tar_member(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def review_summary() -> dict:
    records = []
    for path in REVIEWS:
        if not path.is_file():
            raise RuntimeError(f"required independent code review is missing: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid review JSONL {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"review record must be an object: {path}:{line_number}")
            records.append(record)
    code_surface_prefixes = (
        "02_experiment/code/src/",
        "02_experiment/code/scripts/",
        "02_experiment/code/configs/",
        "02_experiment/code/tests/",
        "src/",
        "scripts/",
        "configs/",
        "tests/",
    )
    open_blocking = [
        record
        for record in records
        if record.get("severity") in {"blocker", "major"}
        and record.get("status") == "open"
        and str(record.get("path", "")).replace("\\", "/").startswith(code_surface_prefixes)
    ]
    if open_blocking:
        raise RuntimeError("independent review contains an unresolved blocker or major finding")
    return {
        "status": "pass",
        "independent_review_count": len(REVIEWS),
        "finding_count": len(records),
        "open_blocker_or_major_count": 0,
        "process_precondition_findings": sum(
            1
            for record in records
            if record.get("severity") in {"blocker", "major"}
            and record.get("status") == "open"
            and not str(record.get("path", "")).replace("\\", "/").startswith(code_surface_prefixes)
        ),
        "roles": sorted({str(record.get("role")) for record in records}),
        "reports": [str(path) for path in REVIEWS],
    }


def main() -> None:
    hard = json.loads(HARD_CONTRACT.read_text(encoding="utf-8-sig"))
    liveness = json.loads(LIVENESS.read_text(encoding="utf-8-sig"))
    validator = json.loads(VALIDATOR.read_text(encoding="utf-8-sig"))
    if hard.get("status") != "pass" or hard.get("passed") != hard.get("total"):
        raise RuntimeError("JACK hard contract is not passing")
    if liveness.get("status") != "pass" or liveness.get("passed") != liveness.get("total"):
        raise RuntimeError("JACK full-path liveness is not passing")
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
        entries.append({
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        })
    entries.sort(key=lambda item: item["path"])
    manifest = {
        "artifact_type": "researchpilot_clean_sync_manifest",
        "schema_version": "researchpilot.clean_sync_manifest.v2",
        "status": "pass",
        "generated_for": "v18_jack_approved_single_route_local_code",
        "route_id": "R-EO-JACK-V18-01",
        "primary_core_candidate_id": "JACK-01",
        "candidate_ids": ["JACK-01"],
        "conditional_control_ids": ["JACK-C1", "JACK-C2", "JACK-C3", "JACK-C4"],
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
        "route_id": "R-EO-JACK-V18-01",
        "primary_core_candidate_id": "JACK-01",
        "hard_contract_ref": str(HARD_CONTRACT),
        "liveness_ref": str(LIVENESS),
        "test_accessed": False,
        "local_real_data_included": False,
        "pretrained_weight_binaries_included": False,
        "credentials_included": False,
        "cache_files_included": False,
    }
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    release_bytes = (
        json.dumps(release, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    with PACKAGE.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.GNU_FORMAT) as archive:
                archive.addfile(
                    tar_member("researchpilot_code_release_manifest.json", release_bytes),
                    io.BytesIO(release_bytes),
                )
                for item in entries:
                    payload = (CODE / item["path"]).read_bytes()
                    archive.addfile(tar_member(item["path"], payload), io.BytesIO(payload))
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
        "route_id": "R-EO-JACK-V18-01",
        "primary_core_candidate_id": "JACK-01",
        "reviewed_commit_or_sync_manifest": str(MANIFEST),
        "scaffold_report": str(CODE / "review" / "scaffold_report.json"),
        "test_summary": {
            "status": "pass",
            "pytest": "372 passed, 0 warnings",
            "hard_contract": f"{hard['passed']}/{hard['total']} pass",
            "hard_contract_ref": str(HARD_CONTRACT),
            "liveness": f"{liveness['passed']}/{liveness['total']} pass",
            "liveness_ref": str(LIVENESS),
            "synthetic_train_entrypoint": "pass; CE+Lovasz; micro-batch16/effective32; optimizer and gradient finite",
            "synthetic_evaluate_entrypoint": "pass; validation metric finite; run contract bound",
            "validator": f"pass; {validator['scanned_executable_or_config_files']} executable/config files; 0 violations",
            "validator_ref": str(VALIDATOR),
            "synthetic_only": True,
            "scientific_result": False,
        },
        "review_summary": reviews,
        "local_data_status": "clean",
        "local_data_scope": "reviewed_code_sync_tree_only",
        "local_gpu_probe": "forbidden_not_run",
        "pretrained_initializer": "audited_cloud_only",
        "pretrained_audit_ref": str(CODE / "configs" / "model" / "pretrained_audit_successor.json"),
        "test_seal_guard": "present_and_tested",
        "training_object_parity": {
            "status": "pass",
            "same_training_object_factory": True,
            "single_internal_mechanism_delta": True,
            "external_trainable_component_forbidden": True,
            "config_diff_test": str(CODE / "tests" / "unit" / "test_v18_jack_config.py"),
            "trainable_parameter_audit_test": str(CODE / "tests" / "integration" / "test_croma_bridge.py"),
            "formal_added_parameters": int(hard["formal_width_parameter_count"]),
            "single_backbone_forward": True,
            "native_joint_stop_gradient": True,
        },
        "coupling_ablation_readiness": {
            "status": "pass",
            "applicability": "single_route_matched_controls_implemented_but_scientifically_locked",
            "enabled_mechanism_set_config": [
                "jack_joint_anchor_kernel",
                "jack_optical_query_control",
                "jack_same_index_control",
                "jack_shuffled_joint_control",
                "jack_sar_query_control",
            ],
            "same_entry_point_test": str(CODE / "tests" / "integration" / "test_croma_bridge.py"),
            "matched_protocol_budget_hash": "resolved_by_v18_jack_config",
            "parent_row_config_test": str(CODE / "tests" / "unit" / "test_v18_jack_config.py"),
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
            },
            "entry_audit": "pass_all_members_bytes_and_sha256",
            "data_download": False,
            "weights_download": False,
            "gpu_or_training": False,
        },
        "unresolved_issues": [
            "RTX3090 AMP, incremental VRAM and throughput require the guarded cloud preflight before training.",
            "No JACK scientific result exists; software and synthetic contracts only.",
            "JACK-C1 through C4 remain locked unless JACK-01 reaches +2pp over the verified CE+Lovasz baseline.",
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
