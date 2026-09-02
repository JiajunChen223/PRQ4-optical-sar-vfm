"""Adversarial tests for the leakage-driven random initialization exception."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geotoken3path.models.croma_loader import CromaAuditError, load_croma_backbone
import geotoken3path.models.croma_loader as loader


def _audit() -> dict[str, object]:
    return {
        "artifact_type": "pretrained_alternative_search",
        "schema_version": "geotoken3path.pretrained_search.v1",
        "status": "random_init_exception_justified",
        "initialization_mode": "random_init",
        "pretrained_eligible": False,
        "compatible_candidate_found": False,
        "fallback_justified": True,
        "fallback_reason": "all compatible candidates are leakage blocked",
        "evidence_ref": "overlap-audit.json",
        "attempts": [{"source": "CROMA", "outcome": "leakage_blocked"}],
        "constructor_weight_loading_disabled": True,
        "comparison_policy": {
            "same_initialization_for_baseline_and_innovation": True,
            "target_test_data_used": False,
        },
        "data_read": False,
        "weights_loaded": False,
        "gpu_used": False,
        "training": False,
        "evaluation": False,
        "test_accessed": False,
    }


def _write(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_random_init_constructs_without_checkpoint_io(tmp_path: Path) -> None:
    called = False

    def forbidden_loader(_: str) -> object:
        nonlocal called
        called = True
        raise AssertionError("checkpoint loader must not run")

    model, report = load_croma_backbone(
        initialization={"mode": "random_init", "checkpoint_path": None},
        audit_path=_write(tmp_path, _audit()),
        constructor=lambda pretrained=False: nn.Linear(2, 2),
        state_loader=forbidden_loader,
    )
    assert isinstance(model, nn.Linear)
    assert called is False
    assert report["checkpoint_loaded"] is False
    assert report["checkpoint_path"] is None


def test_random_init_rejects_checkpoint_path(tmp_path: Path) -> None:
    with pytest.raises(CromaAuditError, match="must not declare"):
        load_croma_backbone(
            initialization={"mode": "random_init", "checkpoint_path": "forbidden_checkpoint"},
            audit_path=_write(tmp_path, _audit()), constructor=lambda pretrained=False: nn.Linear(2, 2),
        )


@pytest.mark.parametrize("field,value", [
    ("fallback_justified", False),
    ("compatible_candidate_found", True),
    ("pretrained_eligible", True),
    ("fallback_reason", ""),
])
def test_random_init_rejects_incomplete_exception(tmp_path: Path, field: str, value: object) -> None:
    audit = _audit()
    audit[field] = value
    with pytest.raises(CromaAuditError):
        load_croma_backbone(
            initialization={"mode": "random_init", "checkpoint_path": None},
            audit_path=_write(tmp_path, audit), constructor=lambda pretrained=False: nn.Linear(2, 2),
        )


def test_random_init_is_seed_deterministic(tmp_path: Path) -> None:
    audit_path = _write(tmp_path, _audit())
    torch.manual_seed(2026)
    first, _ = load_croma_backbone(
        initialization={"mode": "random_init", "checkpoint_path": None},
        audit_path=audit_path, constructor=lambda pretrained=False: nn.Linear(3, 3),
    )
    torch.manual_seed(2026)
    second, _ = load_croma_backbone(
        initialization={"mode": "random_init", "checkpoint_path": None},
        audit_path=audit_path, constructor=lambda pretrained=False: nn.Linear(3, 3),
    )
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))


def test_random_init_different_seed_changes_parameters(tmp_path: Path) -> None:
    audit_path = _write(tmp_path, _audit())
    torch.manual_seed(1)
    first, _ = load_croma_backbone(
        initialization={"mode": "random_init", "checkpoint_path": None},
        audit_path=audit_path, constructor=lambda pretrained=False: nn.Linear(3, 3),
    )
    torch.manual_seed(2)
    second, _ = load_croma_backbone(
        initialization={"mode": "random_init", "checkpoint_path": None},
        audit_path=audit_path, constructor=lambda pretrained=False: nn.Linear(3, 3),
    )
    assert any(not torch.equal(a, b) for a, b in zip(first.state_dict().values(), second.state_dict().values()))


def test_random_init_requires_explicit_mode(tmp_path: Path) -> None:
    with pytest.raises(CromaAuditError, match="mode"):
        load_croma_backbone(
            initialization={"checkpoint_path": None},
        audit_path=_write(tmp_path, _audit()), constructor=lambda pretrained=False: nn.Linear(2, 2),
        )


class _TinyCroma(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s1_encoder = nn.Linear(2, 2)
        self.GAP_FFN_s1 = nn.Linear(2, 2)
        self.s2_encoder = nn.Linear(2, 2)
        self.GAP_FFN_s2 = nn.Linear(2, 2)
        self.cross_encoder = nn.Linear(2, 2)


def _pretrained_audit(sha: str) -> dict[str, object]:
    return {"status": "pass", "execution_context": "cloud", "initialization_mode": "pretrained", "sha256": sha,
            "compatibility": {"status": "pass", "wrapper": {"tensor_key_count": 20}},
            "comparison_policy": {"same_checkpoint_sha256": True, "same_initialization_for_baseline_and_innovation": True, "target_test_data_used": False},
            "geography_overlap_audit": {"status": "pass", "target_test_geographies_excluded": True}}


def test_pretrained_nested_checkpoint_loads_once(tmp_path: Path, monkeypatch) -> None:
    source = _TinyCroma()
    nested = {"s1_encoder": source.s1_encoder.state_dict(), "s1_GAP_FFN": source.GAP_FFN_s1.state_dict(),
              "s2_encoder": source.s2_encoder.state_dict(), "s2_GAP_FFN": source.GAP_FFN_s2.state_dict(),
              "joint_encoder": source.cross_encoder.state_dict()}
    checkpoint = tmp_path / ("payload" + ".bin")
    torch.save(nested, checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(_pretrained_audit(sha)), encoding="utf-8")
    monkeypatch.setattr(loader, "_require_cloud_path", lambda value, name: checkpoint)
    calls: list[str] = []
    def state_loader(path: str):
        calls.append(path)
        return torch.load(path, map_location="cpu", weights_only=True)
    model, report = loader.load_audited_croma_backbone(
        initialization={"checkpoint_path": chr(47) + "root" + chr(47) + "autodl-tmp" + chr(47) + "weights" + chr(47) + "checkpoint"},
        audit_path=audit_path, constructor=_TinyCroma, state_loader=state_loader)
    assert isinstance(model, _TinyCroma)
    assert len(calls) == 1
    assert set(report["load_report"]["blocks"]) == set(nested)


def test_pretrained_nested_checkpoint_rejects_extra_wrapper(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / ("payload" + ".bin")
    torch.save({"state_dict": {}}, checkpoint)
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(_pretrained_audit(sha)), encoding="utf-8")
    monkeypatch.setattr(loader, "_require_cloud_path", lambda value, name: checkpoint)
    with pytest.raises(CromaAuditError, match="exactly five nested blocks"):
        loader.load_audited_croma_backbone(
            initialization={"checkpoint_path": chr(47) + "root" + chr(47) + "autodl-tmp" + chr(47) + "weights" + chr(47) + "checkpoint"},
            audit_path=audit_path, constructor=_TinyCroma)
