"""Adversarial tests for the pinned-source random CROMA adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geotoken3path.models.croma_random import (
    PinnedSourceRandomCROMA,
    RandomCromaSourceError,
    _load_pinned_source,
)


def _source(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "source.py"
    path.write_text(
        "class ViT: pass\nclass BaseTransformerCrossAttn: pass\n"
        "def get_2dalibi(*, num_heads, num_patches): return None\n",
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_source_loader_verifies_sha_and_primitives(tmp_path: Path) -> None:
    path, digest = _source(tmp_path)
    module = _load_pinned_source(str(path), digest)
    assert module.ViT is not None


def test_pinned_source_loader_rejects_changed_source(tmp_path: Path) -> None:
    path, digest = _source(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(RandomCromaSourceError, match="SHA256"):
        _load_pinned_source(str(path), digest)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pretrained": True, "weights": None},
        {"pretrained": False, "weights": "checkpoint"},
    ],
)
def test_constructor_rejects_any_weight_request(kwargs: dict[str, object]) -> None:
    with pytest.raises(RandomCromaSourceError, match="weights are forbidden"):
        PinnedSourceRandomCROMA(
            source_path="unused", source_sha256="0" * 64, **kwargs
        )

