from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from geotoken3path.utils.config import resolve_approved_config
from geotoken3path.utils.run_manifest import build_run_manifest, verify_run_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_run_manifest_is_detached_from_resolved_config_and_hash_verifiable() -> None:
    resolved = resolve_approved_config(ROOT, "always_fuse")
    manifest = build_run_manifest(
        resolved,
        seed=resolved["runtime"]["seed"],
        split="validation",
        execution_scale="smoke",
    )
    original_rate = manifest["optimizer"]["learning_rate"]
    resolved["runtime"]["optimizer"]["learning_rate"] = 9.9
    assert manifest["optimizer"]["learning_rate"] == original_rate
    verify_run_manifest(manifest)

    manifest["optimizer"]["learning_rate"] = 9.9
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_run_manifest(manifest)


@pytest.mark.parametrize(
    "resolved",
    [
        {"model": {}},
        {},
        {"model": {"mechanism_set": "always_fuse"}, "runtime": {}},
    ],
)
def test_incomplete_resolved_configuration_is_rejected(resolved: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_run_manifest(resolved, seed=0, split="validation", execution_scale="smoke")


def test_manifest_rejects_invalid_seed_and_execution_scale() -> None:
    resolved = resolve_approved_config(ROOT, "always_fuse")
    with pytest.raises(ValueError):
        build_run_manifest(resolved, seed=True, split="validation", execution_scale="smoke")
    with pytest.raises(ValueError):
        build_run_manifest(resolved, seed=0, split="validation", execution_scale="unknown")


def test_formal_manifest_paths_bind_to_runtime_artifacts() -> None:
    resolved = resolve_approved_config(ROOT, "always_fuse")
    data_ref = "runtime_data_manifest.json"
    audit_ref = "runtime_pretrained_audit.json"
    manifest = build_run_manifest(
        resolved,
        seed=0,
        split="validation",
        execution_scale="screening",
        candidate_direction_id="BASELINE",
        data_manifest_ref=data_ref,
        pretrained_audit_ref=audit_ref,
    )
    assert manifest["data_manifest_ref"] == data_ref
    assert manifest["pretrained_audit_ref"] == audit_ref
    assert manifest["initialization_ref"] == audit_ref


def test_formal_manifest_does_not_retain_legacy_initialization_ref() -> None:
    resolved = resolve_approved_config(ROOT, "always_fuse")
    manifest = build_run_manifest(
        resolved,
        seed=0,
        split="validation",
        execution_scale="screening",
        candidate_direction_id="BASELINE",
        data_manifest_ref="runtime_successor_data_manifest.json",
        pretrained_audit_ref="runtime_successor_pretrained_audit.json",
    )
    assert manifest["initialization_ref"] == "runtime_successor_pretrained_audit.json"
    assert "pretrained_alternative_search_20260823" not in manifest["initialization_ref"]
