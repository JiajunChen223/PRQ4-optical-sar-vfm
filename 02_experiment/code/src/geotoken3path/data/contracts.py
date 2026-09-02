"""Fail-closed metadata checks for the cloud-only dataset loader.

The active benchmark is the approved SEN12TS WorldCover successor.  The old
BigEarthNet contract remains readable only as an explicitly configured legacy
schema; it is not the active benchmark and is never silently selected.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
APPROVED_ROOT = chr(47) + "root" + chr(47) + "autodl-tmp" + chr(47)
SEN12TS_DATASET_ID = "sen12ts_worldcover_3region_1200"
LEGACY_DATASET_ID = "copernicus_bench_bigearthnet_s1s2_10pct"
APPROVED_DATASET_ROOT = APPROVED_ROOT + "sen12ts_worldcover_3region_1200"
LEGACY_DATASET_ROOT = APPROVED_ROOT + "copernicus_bench"
OPTICAL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
SAR_CHANNELS = ["VV", "VH"]
SEN12TS_RAW_SAR_CHANNELS = ["VH", "VV"]
SEN12TS_S2_SOURCE_INDICES = list(range(12))
SEN12TS_S1_SOURCE_INDICES = [1, 0]
SEN12TS_RAW_LABEL_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
SEN12TS_MODEL_LABEL_VALUES = list(range(11))
DYNAMIC_NORMALIZATION_SCHEME = "croma_official_dynamic_v1"
FIXED_NORMALIZATION_SCHEME = "fixed_vector_v1"
REVISION = re.compile(r"^[0-9a-f]{40}$")


class DatasetManifestError(ValueError):
    """Raised when a cloud dataset manifest violates the frozen protocol."""


def _canonical_cloud_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith(chr(47)) or "\\" in value:
        raise DatasetManifestError(f"{name} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value.rstrip(chr(47)):
        raise DatasetManifestError(f"{name} must be a normalized realpath")
    return path


def _normalization_vector(value: object, expected: int, name: str, *, positive: bool) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise DatasetManifestError(f"{name} must contain exactly {expected} values")
    resolved: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise DatasetManifestError(f"{name}[{index}] must be finite numeric")
        number = float(item)
        if positive and number <= 0:
            raise DatasetManifestError(f"{name}[{index}] must be positive")
        resolved.append(number)
    return resolved


def _nonempty_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value.strip().casefold() in {"pending", "unknown", "tbd"}:
        raise DatasetManifestError(f"{name} must be a resolved non-empty string")


def _validate_fixed_normalization(value: Mapping[str, Any]) -> None:
    if set(value) == {"scheme", "optical", "sar"}:
        if value["scheme"] != FIXED_NORMALIZATION_SCHEME:
            raise DatasetManifestError("unsupported fixed normalization scheme")
    elif set(value) != {"optical", "sar"}:
        raise DatasetManifestError("fixed_vector_v1 normalization must contain exactly optical and sar")
    for modality, expected in (("optical", len(OPTICAL_BANDS)), ("sar", len(SAR_CHANNELS))):
        spec = value[modality]
        if not isinstance(spec, Mapping) or set(spec) != {"mean", "std"}:
            raise DatasetManifestError(f"normalization.{modality} must contain mean and std")
        _normalization_vector(spec["mean"], expected, f"normalization.{modality}.mean", positive=False)
        _normalization_vector(spec["std"], expected, f"normalization.{modality}.std", positive=True)


def _validate_dynamic_normalization(value: Mapping[str, Any]) -> None:
    required = {
        "scheme", "scope", "statistics_axes", "micro_batch", "last_batch_policy", "batch_semantics",
        "distributed_statistics", "global_batch_statistics", "clip_sigma", "output_range", "encoding",
        "formula", "zero_std_policy", "nodata_policy", "fixed_vectors_declared", "source",
        "source_revision", "readme_sha256", "loader_sha256", "source_evidence_ref", "normalization_locked",
    }
    missing = sorted(required - set(value))
    if missing:
        raise DatasetManifestError("dynamic normalization missing: " + ", ".join(missing))
    if value["scheme"] != DYNAMIC_NORMALIZATION_SCHEME:
        raise DatasetManifestError("unsupported dynamic normalization scheme")
    if value["scope"] != "per_micro_batch_per_channel":
        raise DatasetManifestError("CROMA normalization must be per-micro-batch/per-channel")
    if value["statistics_axes"] != [0, 2, 3]:
        raise DatasetManifestError("CROMA normalization statistics axes must be (0,2,3)")
    micro_batch = value["micro_batch"]
    if isinstance(micro_batch, bool) or not isinstance(micro_batch, int) or micro_batch != 16:
        raise DatasetManifestError("CROMA normalization micro_batch must be exactly 16")
    policies = value["last_batch_policy"]
    if not isinstance(policies, Mapping) or dict(policies) != {
        "training": "drop_last",
        "validation": "pad_repeat_last_and_trim_outputs",
        "inference": "pad_repeat_last_and_trim_outputs",
    }:
        raise DatasetManifestError("last-batch policy must freeze training/validation/inference semantics")
    semantics = value["batch_semantics"]
    if not isinstance(semantics, Mapping) or set(semantics) != {"training", "validation", "inference"}:
        raise DatasetManifestError("batch_semantics must describe training, validation and inference")
    for key, item in semantics.items():
        _nonempty_text(item, f"batch_semantics.{key}")
    if value["distributed_statistics"] != "per_rank_micro_batch" or value["global_batch_statistics"] is not False:
        raise DatasetManifestError("distributed CROMA statistics must be per-rank, not implicit global batch")
    clip_sigma = value["clip_sigma"]
    if isinstance(clip_sigma, bool) or not isinstance(clip_sigma, (int, float)) or not math.isfinite(float(clip_sigma)) or float(clip_sigma) != 2.0:
        raise DatasetManifestError("CROMA clip_sigma must be exactly 2.0")
    if value["output_range"] != [0.0, 1.0]:
        raise DatasetManifestError("CROMA output_range must be [0.0, 1.0]")
    if value["encoding"] != "float32":
        raise DatasetManifestError("active CROMA path must freeze float32 encoding")
    _nonempty_text(value["formula"], "normalization.formula")
    formula = str(value["formula"])
    if "mean_axes_0_2_3" not in formula or "std_axes_0_2_3" not in formula or "clip" not in formula:
        raise DatasetManifestError("normalization.formula must expose axes-aware mean/std clipping")
    for field in ("zero_std_policy", "nodata_policy", "source_evidence_ref"):
        _nonempty_text(value[field], f"normalization.{field}")
    if value["fixed_vectors_declared"] is not False:
        raise DatasetManifestError("dynamic CROMA normalization must not declare fixed vectors")
    if value["source"] != "official_croma_readme":
        raise DatasetManifestError("dynamic CROMA normalization must cite the official README")
    if REVISION.fullmatch(str(value["source_revision"]).casefold()) is None:
        raise DatasetManifestError("source_revision must be a 40-character pinned commit")
    for field in ("readme_sha256", "loader_sha256"):
        if SHA256.fullmatch(str(value[field]).casefold()) is None:
            raise DatasetManifestError(f"{field} must be a SHA256 digest")
    if value["normalization_locked"] is not False:
        raise DatasetManifestError("normalization_locked must remain false until value-level parity is recorded")


def _validate_normalization(value: object) -> None:
    if not isinstance(value, Mapping):
        raise DatasetManifestError("normalization must be a mapping")
    if value.get("scheme") == FIXED_NORMALIZATION_SCHEME:
        _validate_fixed_normalization(value)
        return
    if value.get("scheme") == DYNAMIC_NORMALIZATION_SCHEME:
        _validate_dynamic_normalization(value)
        return
    # Legacy manifests used the fixed-vector body without an explicit scheme.
    if set(value) == {"optical", "sar"}:
        _validate_fixed_normalization(value)
        return
    raise DatasetManifestError("normalization must declare fixed_vector_v1 or croma_official_dynamic_v1")


def _exact_int_list(value: object, expected: list[int], name: str) -> None:
    if value != expected:
        raise DatasetManifestError(f"{name} must be exactly {expected}")


def _validate_sen12ts_contract(manifest: Mapping[str, Any]) -> None:
    required = {
        "labels", "optical_source_indices", "sar_source_indices", "sar_raw_channel_order",
        "parent_shape", "derived_shape", "label_contract", "split_contract",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise DatasetManifestError("SEN12TS manifest missing: " + ", ".join(missing))
    if manifest["labels"] != 11:
        raise DatasetManifestError("SEN12TS model label count must be exactly 11")
    _exact_int_list(manifest["optical_source_indices"], SEN12TS_S2_SOURCE_INDICES, "optical_source_indices")
    _exact_int_list(manifest["sar_source_indices"], SEN12TS_S1_SOURCE_INDICES, "sar_source_indices")
    if list(manifest["sar_raw_channel_order"]) != SEN12TS_RAW_SAR_CHANNELS:
        raise DatasetManifestError("SEN12TS raw SAR order must be VH,VV before [1,0] reorder")
    if list(manifest["optical_band_order"]) != OPTICAL_BANDS or list(manifest["sar_channel_order"]) != SAR_CHANNELS:
        raise DatasetManifestError("SEN12TS canonical input bands must be B01..B12 and VV,VH")
    for field, expected in (("parent_shape", [256, 256]), ("derived_shape", [120, 120])):
        _exact_int_list(manifest[field], expected, field)
    labels = manifest["label_contract"]
    if not isinstance(labels, Mapping):
        raise DatasetManifestError("label_contract must be a mapping")
    if list(labels.get("raw_values", [])) != SEN12TS_RAW_LABEL_VALUES:
        raise DatasetManifestError("label_contract.raw_values must be the 11 WorldCover codes")
    if list(labels.get("model_values", [])) != SEN12TS_MODEL_LABEL_VALUES:
        raise DatasetManifestError("label_contract.model_values must be contiguous 0..10")
    if labels.get("ignore_index") != 255 or labels.get("cloud_task") is not False:
        raise DatasetManifestError("SEN12TS labels must use ignore=255 and no cloud task")
    _nonempty_text(labels.get("mapping"), "label_contract.mapping")
    split = manifest["split_contract"]
    if not isinstance(split, Mapping):
        raise DatasetManifestError("split_contract must be a mapping")
    for field in ("parent_first", "crop_after_split", "crop_leakage_rejected", "sealed_test"):
        if split.get(field) is not True:
            raise DatasetManifestError(f"split_contract.{field} must be true")
    _nonempty_text(split.get("region_holdout_axis"), "split_contract.region_holdout_axis")


def _normalization_matches(left: object, right: object) -> bool:
    # Dynamic descriptors are policy objects, not vectors; equality is still
    # required so a checkpoint cannot silently use another batch/statistics
    # semantics. Fixed-vector compatibility remains supported for legacy use.
    return left == right


def validate_cloud_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    configured_dataset_root: str = APPROVED_DATASET_ROOT,
) -> None:
    required = {
        "dataset_id", "cloud_root", "split_manifest_sha256", "payload_sha256", "optical_band_order",
        "sar_channel_order", "normalization", "test_accessed", "storage_bytes", "license_status",
        "component_storage_bytes", "total_active_storage_bytes", "sample_realpaths", "sample_realpaths_resolved",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise DatasetManifestError("dataset manifest missing: " + ", ".join(missing))
    dataset_id = manifest["dataset_id"]
    if dataset_id not in {SEN12TS_DATASET_ID, LEGACY_DATASET_ID}:
        raise DatasetManifestError("unexpected core dataset")
    if dataset_id == SEN12TS_DATASET_ID:
        _validate_sen12ts_contract(manifest)
    elif isinstance(manifest.get("normalization"), Mapping) and manifest["normalization"].get("scheme") == DYNAMIC_NORMALIZATION_SCHEME:
        raise DatasetManifestError("legacy BigEarthNet cannot use the SEN12TS dynamic normalization contract")
    configured_root_path = _canonical_cloud_path(configured_dataset_root, "configured_dataset_root")
    root_path = _canonical_cloud_path(manifest["cloud_root"], "cloud_root")
    if root_path != configured_root_path:
        raise DatasetManifestError("dataset root must exactly match the configured dataset root")
    if manifest["sample_realpaths_resolved"] is not True:
        raise DatasetManifestError("sample paths must be resolved with cloud realpath before validation")
    sample_realpaths = manifest["sample_realpaths"]
    if not isinstance(sample_realpaths, list) or not sample_realpaths:
        raise DatasetManifestError("at least one resolved sample path is required")
    for index, sample in enumerate(sample_realpaths):
        sample_path = _canonical_cloud_path(sample, f"sample_realpaths[{index}]")
        if sample_path == root_path or not sample_path.is_relative_to(root_path):
            raise DatasetManifestError("resolved sample path escapes the configured dataset root")
    for field in ("split_manifest_sha256", "payload_sha256"):
        if not SHA256.fullmatch(str(manifest[field]).casefold()):
            raise DatasetManifestError(f"{field} must be a SHA256 digest")
    if list(manifest["sar_channel_order"]) != SAR_CHANNELS:
        raise DatasetManifestError("SAR channel order must be VV,VH")
    if list(manifest["optical_band_order"]) != OPTICAL_BANDS:
        raise DatasetManifestError("optical manifest must declare the exact approved 12-band order")
    _validate_normalization(manifest["normalization"])
    if manifest["test_accessed"] is not False:
        raise DatasetManifestError("test split must remain sealed")
    storage_bytes = manifest["storage_bytes"]
    components = manifest["component_storage_bytes"]
    total = manifest["total_active_storage_bytes"]
    if isinstance(storage_bytes, bool) or not isinstance(storage_bytes, int) or storage_bytes <= 0:
        raise DatasetManifestError("dataset payload size must be a positive integer")
    if not isinstance(components, Mapping) or not components:
        raise DatasetManifestError("component storage ledger is required")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in components.values()):
        raise DatasetManifestError("component storage ledger values must be nonnegative integers")
    if isinstance(total, bool) or not isinstance(total, int):
        raise DatasetManifestError("total active storage must be an integer")
    if sum(components.values()) != total or components.get("raw_payload") != storage_bytes:
        raise DatasetManifestError("component storage ledger does not reconcile")
    if total >= 45_000_000_000:
        raise DatasetManifestError("total acquisition footprint must remain below the 45GB hard stop")
    if manifest["license_status"] != "verified_before_acquisition":
        raise DatasetManifestError("dataset license inheritance must be verified")


def cross_validate_dataset_and_pretrained(
    manifest: Mapping[str, Any],
    pretrained_audit: Mapping[str, Any],
    *,
    configured_dataset_root: str = APPROVED_DATASET_ROOT,
) -> None:
    """Bind dataset band/normalization metadata to the audited target input."""

    validate_cloud_dataset_manifest(manifest, configured_dataset_root=configured_dataset_root)
    try:
        target = pretrained_audit["compatibility"]["input_spec"]["target"]
    except (KeyError, TypeError) as exc:
        raise DatasetManifestError("pretrained target input spec is missing") from exc
    band_order = target.get("band_order") if isinstance(target, Mapping) else None
    normalization = target.get("normalization") if isinstance(target, Mapping) else None
    if not isinstance(band_order, Mapping):
        raise DatasetManifestError("pretrained target band order is missing")
    if list(band_order.get("optical", [])) != OPTICAL_BANDS or list(band_order.get("sar", [])) != SAR_CHANNELS:
        raise DatasetManifestError("dataset and pretrained band orders do not match")
    if not _normalization_matches(normalization, manifest["normalization"]):
        raise DatasetManifestError("dataset and pretrained normalization policy does not match")
