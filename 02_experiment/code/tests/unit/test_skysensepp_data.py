"""Unit tests for the R24 SkySense++ S2 data adapter."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from geotoken3path.data.contracts import OPTICAL_BANDS
from geotoken3path.data.sen12ts import SEN12TSLoaderError
from geotoken3path.data.skysensepp import (
    _S2_10_BANDS,
    _S2_10_INDICES,
    annotation_from_target,
    build_skysensepp_loader,
    croma_dynamic_normalize_batch_r24,
    to_skysensepp_optical,
)

import geotoken3path.data.sen12ts as sen12ts
import geotoken3path.data.skysensepp as skysensepp


class _DummyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "optical": torch.zeros(12, 4, 4),
            "sar": torch.zeros(2, 4, 4),
            "target": torch.zeros(4, 4, dtype=torch.long),
        }


def _optical(batch: int = 16, size: int = 32, *, dtype: torch.dtype = torch.float32) -> Tensor:
    values = torch.arange(12 * size * size, dtype=dtype).reshape(1, 12, size, size).repeat(batch, 1, 1, 1)
    return (values + torch.arange(batch).reshape(batch, 1, 1, 1) * 1000.0).float() if dtype == torch.float32 else values


def test_s2_10_band_order_matches_contract() -> None:
    """The ten selected bands must be SkySense++ S2 order: all 12 minus B01/B09."""
    assert _S2_10_INDICES == (1, 2, 3, 4, 5, 6, 7, 8, 10, 11)
    assert tuple(OPTICAL_BANDS[index] for index in _S2_10_INDICES) == _S2_10_BANDS
    assert _S2_10_BANDS == ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    assert len(_S2_10_BANDS) == 10
    assert len(set(_S2_10_BANDS)) == 10


def test_to_skysensepp_optical_selects_exact_bands() -> None:
    """Channel labels follow the source index mapping exactly."""
    batch = {"optical": _optical(batch=4)}
    selected = to_skysensepp_optical(batch)
    assert tuple(selected.shape) == (4, 10, 32, 32)
    source = batch["optical"]
    for target_channel, source_channel in enumerate(_S2_10_INDICES):
        assert torch.equal(selected[:, target_channel], source[:, source_channel])


def test_to_skysensepp_optical_rejects_wrong_channel_count() -> None:
    for bad in (torch.randn(16, 11, 8, 8), torch.randn(16, 13, 8, 8), torch.randn(16, 12, 8)):
        try:
            to_skysensepp_optical({"optical": bad})
        except SEN12TSLoaderError:
            continue
        raise AssertionError("expected SEN12TSLoaderError for malformed optical")


def test_annotation_from_target_maps_ignore_to_vocabulary_zero() -> None:
    target = torch.tensor([[0, 1, 255, 5], [10, 255, 2, 255]], dtype=torch.long)
    mapped = annotation_from_target(target)
    expected = torch.tensor([[0, 1, 0, 5], [10, 0, 2, 0]], dtype=torch.long)
    assert torch.equal(mapped, expected)
    assert mapped.dtype == torch.long


def test_annotation_from_target_is_fully_in_vocabulary() -> None:
    values = torch.randint(0, 11, (4, 16, 16), dtype=torch.long)
    noisy = torch.where(torch.rand(4, 16, 16) < 0.1, torch.tensor(255, dtype=torch.long), values)
    mapped = annotation_from_target(noisy)
    assert bool((mapped >= 0).all()) and bool((mapped <= 64).all())


def test_r24_normalize_returns_optical10_target_valid_count() -> None:
    result = croma_dynamic_normalize_batch_r24(
        {"optical": _optical(batch=16, size=24), "target": torch.ones(16, 24, 24, dtype=torch.long) * 255,
         "valid_count": torch.tensor(5)}
    )
    assert set(result) == {"optical10", "target", "valid_count"}
    assert tuple(result["optical10"].shape) == (16, 10, 24, 24)
    assert result["optical10"].dtype == torch.float32
    assert torch.equal(result["target"], torch.ones(16, 24, 24, dtype=torch.long) * 255)
    assert int(result["valid_count"]) == 5


def test_r24_normalize_values_are_clipped_to_unit_range() -> None:
    result = croma_dynamic_normalize_batch_r24(
        {"optical": _optical(batch=16, size=24), "target": torch.zeros(16, 24, 24, dtype=torch.long),
         "valid_count": torch.tensor(16)}
    )
    assert bool((result["optical10"] >= 0.0).all()) and bool((result["optical10"] <= 1.0).all())


def test_r24_normalize_per_channel_recipe_matches_reference() -> None:
    """Formula check: (x-(mean-2std))/(4std) per channel over (0,2,3)."""
    source = _optical(batch=16, size=24)
    value = source[:, _S2_10_INDICES]
    mean = value.mean(dim=(0, 2, 3), keepdim=True)
    std = value.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
    expected = ((value - (mean - 2.0 * std)) / (4.0 * std)).clamp(0.0, 1.0)
    result = croma_dynamic_normalize_batch_r24(
        {"optical": source, "target": torch.zeros(16, 24, 24, dtype=torch.long), "valid_count": torch.tensor(16)}
    )
    assert torch.allclose(result["optical10"], expected)


def test_r24_normalize_is_per_micro_batch_not_per_pixel() -> None:
    """A batch with a constant offset gets identical statistics."""
    batch_a = {"optical": _optical(batch=16, size=24), "target": torch.zeros(16, 24, 24, dtype=torch.long),
               "valid_count": torch.tensor(16)}
    batch_b = {"optical": _optical(batch=16, size=24) + 10.0, "target": batch_a["target"],
               "valid_count": batch_a["valid_count"]}
    result_a = croma_dynamic_normalize_batch_r24(batch_a)
    result_b = croma_dynamic_normalize_batch_r24(batch_b)
    assert torch.allclose(result_a["optical10"], result_b["optical10"])


def test_r24_normalize_raises_on_degenerate_statistics() -> None:
    for payload in (torch.zeros(16, 12, 24, 24, dtype=torch.float32), torch.randn(16, 12, 24, 24) * float("nan")):
        try:
            croma_dynamic_normalize_batch_r24(
                {"optical": payload, "target": torch.zeros(16, 24, 24, dtype=torch.long),
                 "valid_count": torch.tensor(16)}
            )
        except SEN12TSLoaderError as exc:
            assert "standard deviation" in str(exc)
            continue
        raise AssertionError("expected fail-closed SEN12TSLoaderError")


def test_r24_normalize_rejects_bad_shape_dtype_batch() -> None:
    cases = [
        torch.randn(8, 12, 24, 24, dtype=torch.float32),  # B != micro_batch
        torch.randn(16, 12, 24, 24, dtype=torch.float64),  # not float32
        torch.randn(16, 11, 24, 24, dtype=torch.float32),  # wrong channels
        torch.randn(16, 12, 24, dtype=torch.float32),  # not 4D
    ]
    for bad in cases:
        try:
            croma_dynamic_normalize_batch_r24(
                {"optical": bad, "target": torch.zeros(16, 24, 24, dtype=torch.long),
                 "valid_count": torch.tensor(16)}
            )
        except SEN12TSLoaderError:
            continue
        raise AssertionError("expected SEN12TSLoaderError for malformed optical batch")


def test_r24_normalize_rejects_custom_micro_batch_mismatch() -> None:
    try:
        croma_dynamic_normalize_batch_r24(
            {"optical": torch.randn(16, 12, 24, 24), "target": torch.zeros(16, 24, 24, dtype=torch.long),
             "valid_count": torch.tensor(16)},
            micro_batch=8,
        )
    except SEN12TSLoaderError:
        return
    raise AssertionError("expected SEN12TSLoaderError when B differs from micro_batch")


class _RejectingManifestDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        raise AssertionError("manifest records were not validated")


def test_skysensepp_loader_delegates_to_sen12ts_loader(monkeypatch) -> None:
    """The R24 loader wrapper must forward every argument unchanged."""
    captured: dict[str, object] = {}

    def fake_loader(
        manifest_path, *, split, batch_size, num_workers, execution_scale,
        pin_memory=False, augmentation=None, seed=0,
    ):
        captured.update(
            manifest_path=manifest_path, split=split, batch_size=batch_size,
            num_workers=num_workers, execution_scale=execution_scale,
            pin_memory=pin_memory, augmentation=augmentation, seed=seed,
        )
        return (object(), {"dataset_id": "fixture", "test_accessed": False})

    monkeypatch.setattr(skysensepp, "build_sen12ts_loader", fake_loader)
    loader, manifest = build_skysensepp_loader(
        "fixture_manifest.json",
        split="train",
        batch_size=16,
        num_workers=0,
        execution_scale="baseline",
        pin_memory=True,
        augmentation={"name": "paired_geometric_v1", "enabled": True, "train_only": True,
                      "deterministic": True, "orientation_space": "D4",
                      "operations": ["horizontal_flip", "vertical_flip", "rotate_90",
                                     "rotate_180", "rotate_270", "transpose", "anti_transpose"]},
        seed=3,
    )
    assert captured["split"] == "train" and captured["batch_size"] == 16
    assert captured["num_workers"] == 0 and captured["execution_scale"] == "baseline"
    assert captured["pin_memory"] is True and captured["seed"] == 3
    assert isinstance(captured["augmentation"], dict)
    assert loader is not None and manifest["dataset_id"] == "fixture"


def test_skysensepp_loader_reuses_real_sen12ts_loader(monkeypatch, tmp_path) -> None:
    """Sanity: the wrapper returns a genuine sen12ts DataLoader when possible."""
    monkeypatch.setattr(
        sen12ts,
        "load_sen12ts_manifest",
        lambda *_args, **_kwargs: ({"dataset_id": "fixture", "test_accessed": False}, [{"id": "x"}] * 16),
    )
    monkeypatch.setattr(sen12ts, "SEN12TSDataset", lambda *_args, **_kwargs: _DummyDataset())
    loader, manifest = build_skysensepp_loader(
        str(tmp_path / "no_such_manifest.json"),
        split="validation",
        batch_size=16,
        num_workers=0,
        execution_scale="smoke",
    )
    assert isinstance(loader, DataLoader)
    assert manifest["dataset_id"] == "fixture"
    loader._iterator = None
