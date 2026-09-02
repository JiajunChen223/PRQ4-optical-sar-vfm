from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
import torch

from geotoken3path.engine.formal_runner import (
    FormalRunnerError,
    _atomic_torch,
    _checkpoint_payload,
    _file_identity,
)


def _state() -> dict[str, torch.Tensor]:
    return {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}


def test_best_checkpoint_is_explicit_evaluation_state() -> None:
    payload = _checkpoint_payload(
        model_state=_state(),
        epoch=17,
        run_manifest={"split": "validation", "test_accessed": False},
        checkpoint_role="best_validation",
    )
    assert payload["epoch"] == 17
    assert payload["checkpoint_role"] == "best_validation"
    assert "optimizer" not in payload
    assert payload["run_manifest"]["test_accessed"] is False


def test_final_checkpoint_retains_truthful_epoch_and_optimizer() -> None:
    payload = _checkpoint_payload(
        model_state=_state(),
        optimizer_state={"state": {}, "param_groups": []},
        epoch=24,
        run_manifest={"split": "validation", "test_accessed": False},
        checkpoint_role="final",
    )
    assert payload["epoch"] == 24
    assert payload["checkpoint_role"] == "final"
    assert payload["optimizer"] == {"state": {}, "param_groups": []}


def test_checkpoint_file_identity_binds_bytes_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_identity_fixture"
    path.write_bytes(b"best-checkpoint-fixture")
    identity = _file_identity(path)
    assert identity["path"] == str(path)
    assert identity["bytes"] == len(b"best-checkpoint-fixture")
    assert identity["sha256"] == "57bc77fef5ab5743d918b08733180665ed5fdd791f19a636ad5f5f083a583475"


def test_best_and_final_checkpoint_roundtrip_keep_distinct_roles(tmp_path: Path) -> None:
    manifest = {"split": "validation", "test_accessed": False}
    best_path = tmp_path / "best_checkpoint_fixture"
    final_path = tmp_path / "final_checkpoint_fixture"
    _atomic_torch(
        best_path,
        _checkpoint_payload(
            model_state={"weight": torch.tensor([17.0])},
            epoch=17,
            run_manifest=manifest,
            checkpoint_role="best_validation",
        ),
    )
    _atomic_torch(
        final_path,
        _checkpoint_payload(
            model_state={"weight": torch.tensor([24.0])},
            optimizer_state={"state": {}, "param_groups": []},
            epoch=24,
            run_manifest=manifest,
            checkpoint_role="final",
        ),
    )
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    assert best["epoch"] == 17 and best["checkpoint_role"] == "best_validation"
    assert final["epoch"] == 24 and final["checkpoint_role"] == "final"
    assert torch.equal(best["model"]["weight"], torch.tensor([17.0]))
    assert torch.equal(final["model"]["weight"], torch.tensor([24.0]))
    assert "optimizer" not in best and "optimizer" in final
    assert _file_identity(best_path)["sha256"] != _file_identity(final_path)["sha256"]


@pytest.mark.parametrize(
    ("epoch", "role"),
    [(0, "best_validation"), (True, "final"), (17, "mislabeled_best")],
)
def test_checkpoint_payload_rejects_ambiguous_metadata(epoch: int, role: str) -> None:
    with pytest.raises(FormalRunnerError):
        _checkpoint_payload(
            model_state=_state(),
            epoch=epoch,
            run_manifest={"split": "validation", "test_accessed": False},
            checkpoint_role=role,
        )
