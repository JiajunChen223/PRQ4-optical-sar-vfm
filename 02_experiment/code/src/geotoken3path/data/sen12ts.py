"""Cloud-only SEN12TS WorldCover manifest and split loader.

The loader is deliberately lazy: constructing it validates metadata and split
membership, but no raster bytes are opened until a sample is requested.  A
manifest may describe TIFF objects (read with rasterio/tifffile on the cloud)
or torch/numpy fixtures used by cloud-side integration tests.  Local paths,
test rows, and unresolved manifests are rejected before a DataLoader exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from .contracts import (
    APPROVED_DATASET_ROOT,
    OPTICAL_BANDS,
    SAR_CHANNELS,
    validate_cloud_dataset_manifest,
)
from ..utils.test_seal import assert_test_access_allowed


class SEN12TSLoaderError(ValueError):
    """Raised when a cloud dataset manifest or sample violates the contract."""


_AUGMENTATION_NAME = "paired_geometric_v1"
_D4_OPERATIONS = (
    "identity",
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "transpose",
    "anti_transpose",
)


def _validate_augmentation_spec(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the only approved train-time augmentation contract.

    The transform is intentionally paired: the same D4 orientation is applied
    to optical, SAR and labels.  Validation never receives this transform.
    """

    if value is None:
        return {
            "name": "disabled",
            "enabled": False,
            "train_only": True,
            "deterministic": True,
            "orientation_space": "identity",
            "operations": ["identity"],
        }
    if not isinstance(value, Mapping):
        raise SEN12TSLoaderError("augmentation must be a mapping")
    if value.get("name") != _AUGMENTATION_NAME:
        raise SEN12TSLoaderError("only paired_geometric_v1 augmentation is approved")
    if value.get("enabled") is not True or value.get("train_only") is not True:
        raise SEN12TSLoaderError("paired augmentation must be enabled on train only")
    if value.get("deterministic") is not True or value.get("orientation_space") != "D4":
        raise SEN12TSLoaderError("paired augmentation must use deterministic D4 orientations")
    operations = value.get("operations")
    if list(operations or []) != list(_D4_OPERATIONS[1:]):
        raise SEN12TSLoaderError("paired augmentation operations must match the frozen D4 contract")
    return {
        "name": _AUGMENTATION_NAME,
        "enabled": True,
        "train_only": True,
        "deterministic": True,
        "orientation_space": "D4",
        "operations": list(_D4_OPERATIONS[1:]),
    }


def _paired_geometric_orientation(
    optical: torch.Tensor,
    sar: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Apply one deterministic D4 orientation to all paired tensors."""

    digest = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()
    operation = _D4_OPERATIONS[digest[0] % len(_D4_OPERATIONS)]

    def transform(value: torch.Tensor) -> torch.Tensor:
        if operation == "identity":
            return value
        if operation == "horizontal_flip":
            return value.flip(-1)
        if operation == "vertical_flip":
            return value.flip(-2)
        if operation == "rotate_90":
            return torch.rot90(value, 1, dims=(-2, -1))
        if operation == "rotate_180":
            return torch.rot90(value, 2, dims=(-2, -1))
        if operation == "rotate_270":
            return torch.rot90(value, 3, dims=(-2, -1))
        transposed = value.transpose(-2, -1)
        return transposed if operation == "transpose" else transposed.flip(-1)

    return transform(optical).contiguous(), transform(sar).contiguous(), transform(target).contiguous(), operation


@dataclass(frozen=True)
class SEN12TSSplitRecord:
    region: str
    parent: str
    split: str
    optical_path: PurePosixPath
    sar_path: PurePosixPath
    label_path: PurePosixPath


def _cloud_path(value: object, *, root: PurePosixPath, name: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise SEN12TSLoaderError(f"{name} must be an absolute POSIX cloud path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value.rstrip(chr(47)):
        raise SEN12TSLoaderError(f"{name} must be normalized")
    if path == root or not path.is_relative_to(root):
        raise SEN12TSLoaderError(f"{name} escapes the dataset root")
    return path


def _object_path(row: Mapping[str, Any], key: str, *, root: PurePosixPath, name: str) -> PurePosixPath:
    item = row.get(key)
    if isinstance(item, Mapping):
        item = item.get("key", item.get("path"))
    if not isinstance(item, str) or not item.strip():
        raise SEN12TSLoaderError(f"split row missing {name} object key")
    # Metadata manifests normally store source-coop object keys relative to the
    # cloud root; accepting an absolute path is useful only after containment
    # has been checked.
    value = item if PurePosixPath(item).is_absolute() else str(root / item)
    return _cloud_path(value, root=root, name=name)


def _rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = manifest.get("records", manifest.get("object_rows"))
    if not isinstance(candidates, list) or not candidates:
        raise SEN12TSLoaderError("manifest must contain non-empty records/object_rows")
    if any(not isinstance(row, Mapping) for row in candidates):
        raise SEN12TSLoaderError("every split row must be a mapping")
    return candidates  # type: ignore[return-value]


def load_sen12ts_manifest(
    manifest_path: str | Path,
    *,
    requested_split: str,
    execution_scale: str,
    configured_root: str = APPROVED_DATASET_ROOT,
) -> tuple[dict[str, Any], list[SEN12TSSplitRecord]]:
    """Validate and select one non-test split from a cloud manifest."""

    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        raise SEN12TSLoaderError("manifest path must be absolute on the cloud host")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SEN12TSLoaderError(f"cannot read cloud dataset manifest: {manifest_file}") from exc
    if not isinstance(manifest, Mapping):
        raise SEN12TSLoaderError("dataset manifest must be a mapping")
    assert_test_access_allowed(
        {"execution_scale": execution_scale, "test_seal_status": manifest.get("test_seal_status", "sealed")},
        requested_split,
    )
    validate_cloud_dataset_manifest(manifest, configured_dataset_root=configured_root)
    root = PurePosixPath(str(manifest["cloud_root"]))
    normalized = requested_split.strip().casefold()
    normalized = "validation" if normalized in {"val", "validation"} else normalized
    if normalized not in {"train", "validation"}:
        raise SEN12TSLoaderError("formal loader accepts train or validation only")

    selected: list[SEN12TSSplitRecord] = []
    all_rows = _rows(manifest)
    parent_splits: dict[tuple[str, str], str] = {}
    for row in all_rows:
        region = str(row.get("region", "")).strip()
        parent = str(row.get("parent", "")).strip()
        split_value = str(row.get("split", "")).strip().casefold()
        split_value = "validation" if split_value in {"val", "validation"} else split_value
        if not region or not parent or not split_value:
            raise SEN12TSLoaderError("every manifest row requires region, parent and split")
        key = (region, parent)
        previous = parent_splits.get(key)
        if previous is not None and previous != split_value:
            raise SEN12TSLoaderError("a parent appears in more than one split")
        parent_splits[key] = split_value
    for row in all_rows:
        split = str(row.get("split", "")).strip().casefold()
        split = "validation" if split in {"val", "validation"} else split
        if split != normalized:
            continue
        if split == "sealed_test" or row.get("test_accessed") is True:
            raise SEN12TSLoaderError("sealed test row cannot enter a formal loader")
        selected.append(
            SEN12TSSplitRecord(
                region=str(row.get("region", "")),
                parent=str(row.get("parent", "")),
                split=normalized,
                optical_path=_object_path(row, "s2", root=root, name="optical"),
                sar_path=_object_path(row, "s1", root=root, name="sar"),
                label_path=_object_path(row, "label", root=root, name="label"),
            )
        )
    if not selected:
        raise SEN12TSLoaderError(f"manifest contains no {normalized} rows")
    return dict(manifest), selected


def _read_array(path: PurePosixPath) -> torch.Tensor:
    """Read one cloud object; imports optional raster readers only on demand."""

    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            return torch.from_numpy(src.read())
    except ImportError as exc:
        raise SEN12TSLoaderError("rasterio is required to read cloud SEN12TS TIFF objects") from exc
    except Exception as exc:
        raise SEN12TSLoaderError(f"failed to read SEN12TS object {path}") from exc


def _to_chw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise SEN12TSLoaderError(f"{name} must be [C,H,W]")
    return value.contiguous()


def _center_crop(value: torch.Tensor, *, size: tuple[int, int], name: str) -> torch.Tensor:
    """Derive the frozen CROMA input window from the parent raster after split."""

    target_h, target_w = size
    height, width = value.shape[-2:]
    if height < target_h or width < target_w:
        raise SEN12TSLoaderError(f"{name} is smaller than the approved derived window")
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return value[..., top : top + target_h, left : left + target_w].contiguous()


class SEN12TSDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy paired optical/SAR/WorldCover samples for one approved split."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        records: list[SEN12TSSplitRecord],
        *,
        split: str = "validation",
        augmentation: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.manifest = manifest
        self.records = records
        self.split = split.strip().casefold()
        if self.split not in {"train", "validation"}:
            raise SEN12TSLoaderError("dataset split must be train or validation")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SEN12TSLoaderError("augmentation seed must be a nonnegative integer")
        self.augmentation = _validate_augmentation_spec(augmentation)
        self.augmentation_seed = seed
        self.optical_indices = tuple(int(x) for x in manifest.get("optical_source_indices", range(12)))
        self.sar_indices = tuple(int(x) for x in manifest.get("sar_source_indices", (1, 0)))
        self.raw_labels = tuple(int(x) for x in manifest.get("label_contract", {}).get("raw_values", ()))
        derived = manifest.get("derived_shape", (120, 120))
        if not isinstance(derived, list) or len(derived) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in derived):
            raise SEN12TSLoaderError("manifest derived_shape must be two positive integers")
        self.derived_shape = (int(derived[0]), int(derived[1]))
        if len(self.optical_indices) != 12 or len(self.sar_indices) != 2 or len(self.raw_labels) != 11:
            raise SEN12TSLoaderError("manifest selectors/label mapping are incomplete")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        optical = _to_chw(_read_array(record.optical_path), name="optical")
        sar = _to_chw(_read_array(record.sar_path), name="sar")
        label = _to_chw(_read_array(record.label_path), name="label")
        if optical.shape[0] < max(self.optical_indices) + 1 or sar.shape[0] < max(self.sar_indices) + 1:
            raise SEN12TSLoaderError("raw modality channels do not satisfy manifest selectors")
        optical = optical[list(self.optical_indices)].to(torch.float32)
        sar = sar[list(self.sar_indices)].to(torch.float32)
        label = label[0].to(torch.long)
        mapped = torch.full_like(label, 255)
        for target, raw in enumerate(self.raw_labels):
            mapped[label == raw] = target
        if optical.shape[-2:] != sar.shape[-2:] or label.shape != optical.shape[-2:]:
            raise SEN12TSLoaderError("optical, SAR and label spatial shapes must match")
        optical = _center_crop(optical, size=self.derived_shape, name="optical")
        sar = _center_crop(sar, size=self.derived_shape, name="sar")
        mapped = _center_crop(mapped, size=self.derived_shape, name="label")
        if self.split == "train" and self.augmentation["enabled"]:
            optical, sar, mapped, _ = _paired_geometric_orientation(
                optical, sar, mapped, seed=self.augmentation_seed, index=index,
            )
        return {"optical": optical, "sar": sar, "target": mapped}


def croma_dynamic_normalize_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the audited per-micro-batch/per-channel CROMA clipping recipe."""

    result = dict(batch)
    for modality, channels in (("optical", 12), ("sar", 2)):
        value = batch[modality]
        if value.ndim != 4 or value.shape[0] != 16 or value.shape[1] != channels or value.dtype != torch.float32:
            raise SEN12TSLoaderError(f"{modality} batch must be float32 [B,{channels},H,W]")
        finite = torch.isfinite(value)
        if not bool(finite.all()):
            value = torch.where(finite, value, torch.zeros_like(value))
        mean = value.mean(dim=(0, 2, 3), keepdim=True)
        std = value.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
        if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
            raise SEN12TSLoaderError(f"{modality} batch has non-positive/non-finite standard deviation")
        result[modality] = ((value - (mean - 2.0 * std)) / (4.0 * std)).clamp(0.0, 1.0)
    return result


def _collate_fixed_croma_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad only the final validation batch to the frozen CROMA micro-batch.

    Repeated rows retain their input statistics, while their targets are set to
    ignore-index so they cannot contribute to validation metrics.
    """
    if not samples:
        raise SEN12TSLoaderError("cannot collate an empty batch")
    if len(samples) > 16:
        raise SEN12TSLoaderError("CROMA collate received more than 16 samples")
    count = len(samples)
    padded = list(samples)
    while len(padded) < 16:
        clone = {key: value.clone() for key, value in samples[-1].items()}
        clone["target"].fill_(255)
        padded.append(clone)
    result = {key: torch.stack([row[key] for row in padded], dim=0) for key in ("optical", "sar", "target")}
    result["valid_count"] = torch.tensor(count, dtype=torch.int64)
    return result


def build_sen12ts_loader(
    manifest_path: str | Path,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    execution_scale: str,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    augmentation: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> tuple[DataLoader[dict[str, torch.Tensor]], dict[str, Any]]:
    """Build a lazy cloud DataLoader after runtime test-seal validation."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise SEN12TSLoaderError("batch_size must be positive")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise SEN12TSLoaderError("num_workers must be nonnegative")
    if not isinstance(pin_memory, bool) or not isinstance(persistent_workers, bool):
        raise SEN12TSLoaderError("pin_memory and persistent_workers must be boolean")
    if isinstance(prefetch_factor, bool) or not isinstance(prefetch_factor, int) or prefetch_factor < 1:
        raise SEN12TSLoaderError("prefetch_factor must be a positive integer")
    manifest, records = load_sen12ts_manifest(
        manifest_path, requested_split=split, execution_scale=execution_scale,
    )
    dataset = SEN12TSDataset(
        manifest,
        records,
        split=split,
        augmentation=augmentation if split.strip().casefold() == "train" else None,
        seed=seed,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": split.strip().casefold() == "train",
        "num_workers": num_workers,
        "drop_last": split.strip().casefold() == "train",
        "collate_fn": _collate_fixed_croma_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs), manifest
