from __future__ import annotations

import torch
from torch.utils.data import Dataset

import geotoken3path.data.sen12ts as sen12ts


class _DummyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "optical": torch.zeros(12, 4, 4),
            "sar": torch.zeros(2, 4, 4),
            "target": torch.zeros(4, 4, dtype=torch.long),
        }


def test_cloud_loader_exposes_prefetch_and_pinned_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        sen12ts,
        "load_sen12ts_manifest",
        lambda *_args, **_kwargs: ({"dataset_id": "fixture", "test_accessed": False}, [{"id": "x"}]),
    )
    monkeypatch.setattr(sen12ts, "SEN12TSDataset", lambda *_args, **_kwargs: _DummyDataset())
    loader, manifest = sen12ts.build_sen12ts_loader(
        "fixture_manifest.json",
        split="train",
        batch_size=16,
        num_workers=2,
        execution_scale="baseline",
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )
    assert manifest["dataset_id"] == "fixture"
    assert loader.pin_memory is True
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
    loader._iterator = None


def test_zero_worker_loader_does_not_request_prefetch(monkeypatch) -> None:
    monkeypatch.setattr(
        sen12ts,
        "load_sen12ts_manifest",
        lambda *_args, **_kwargs: ({"dataset_id": "fixture", "test_accessed": False}, [{"id": "x"}]),
    )
    monkeypatch.setattr(sen12ts, "SEN12TSDataset", lambda *_args, **_kwargs: _DummyDataset())
    loader, _ = sen12ts.build_sen12ts_loader(
        "fixture_manifest.json",
        split="validation",
        batch_size=16,
        num_workers=0,
        execution_scale="smoke",
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    assert loader.num_workers == 0
    assert loader.pin_memory is False
