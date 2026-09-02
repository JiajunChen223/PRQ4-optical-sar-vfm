from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from geotoken3path.data import (
    APPROVED_DATASET_ROOT,
    DYNAMIC_NORMALIZATION_SCHEME,
    FIXED_NORMALIZATION_SCHEME,
    LEGACY_DATASET_ID,
    LEGACY_DATASET_ROOT,
    SEN12TS_DATASET_ID,
    DatasetManifestError,
    cross_validate_dataset_and_pretrained,
    validate_cloud_dataset_manifest,
)


def _dynamic_normalization() -> dict[str, object]:
    return {
        "scheme": DYNAMIC_NORMALIZATION_SCHEME,
        "scope": "per_micro_batch_per_channel",
        "statistics_axes": [0, 2, 3],
        "micro_batch": 16,
        "last_batch_policy": {
            "training": "drop_last",
            "validation": "pad_repeat_last_and_trim_outputs",
            "inference": "pad_repeat_last_and_trim_outputs",
        },
        "batch_semantics": {
            "training": "local_micro_batch_stats_on_exactly_16_samples",
            "validation": "local_micro_batch_stats_with_deterministic_padding",
            "inference": "local_micro_batch_stats_with_deterministic_padding",
        },
        "distributed_statistics": "per_rank_micro_batch",
        "global_batch_statistics": False,
        "clip_sigma": 2.0,
        "output_range": [0.0, 1.0],
        "encoding": "float32",
        "formula": "clip((x - (mean_axes_0_2_3 - 2*std_axes_0_2_3)) / (4*std_axes_0_2_3), 0, 1)",
        "zero_std_policy": "reject_nonfinite_or_nonpositive_std",
        "nodata_policy": "mask_before_statistics_and_fill_invalid_with_zero",
        "fixed_vectors_declared": False,
        "source": "official_croma_readme",
        "source_revision": "59505a6bcadbf36ba20767270154bf9f3067c5e7",
        "readme_sha256": "c" * 64,
        "loader_sha256": "d" * 64,
        "source_evidence_ref": "02_experiment/reports/croma_source_pin_detail_extract_result_20260822_r1.json",
        "normalization_locked": False,
    }


def _manifest() -> dict[str, object]:
    digest = "a" * 64
    return {
        "dataset_id": SEN12TS_DATASET_ID,
        "cloud_root": APPROVED_DATASET_ROOT,
        "sample_realpaths": [APPROVED_DATASET_ROOT + chr(47) + "train" + chr(47) + "sample_0001" + chr(46) + "npy"],
        "sample_realpaths_resolved": True,
        "split_manifest_sha256": digest,
        "payload_sha256": digest,
        "optical_band_order": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        "sar_channel_order": ["VV", "VH"],
        "optical_source_indices": list(range(12)),
        "sar_source_indices": [1, 0],
        "sar_raw_channel_order": ["VH", "VV"],
        "parent_shape": [256, 256],
        "derived_shape": [120, 120],
        "labels": 11,
        "label_contract": {
            "raw_values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
            "model_values": list(range(11)),
            "mapping": "explicit_raw_worldcover_code_to_contiguous_id",
            "ignore_index": 255,
            "cloud_task": False,
        },
        "split_contract": {
            "parent_first": True,
            "crop_after_split": True,
            "crop_leakage_rejected": True,
            "sealed_test": True,
            "region_holdout_axis": "secondary",
        },
        "normalization": _dynamic_normalization(),
        "test_accessed": False,
        "storage_bytes": 20_000_000_000,
        "component_storage_bytes": {
            "raw_payload": 20_000_000_000,
            "extracted": 10_000_000_000,
            "cache": 2_000_000_000,
            "weights": 800_000_000,
            "checkpoints": 2_000_000_000,
        },
        "total_active_storage_bytes": 34_800_000_000,
        "license_status": "verified_before_acquisition",
    }


def _legacy_fixed_vector_manifest() -> dict[str, object]:
    manifest = _manifest()
    manifest["dataset_id"] = LEGACY_DATASET_ID
    manifest["cloud_root"] = LEGACY_DATASET_ROOT
    manifest["sample_realpaths"] = [LEGACY_DATASET_ROOT + chr(47) + "train" + chr(47) + "sample_0001" + chr(46) + "npy"]
    manifest["normalization"] = {
        "scheme": FIXED_NORMALIZATION_SCHEME,
        "optical": {"mean": [0.0] * 12, "std": [1.0] * 12},
        "sar": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }
    return manifest


def test_cloud_manifest_contract_passes_complete_sen12ts_metadata() -> None:
    validate_cloud_dataset_manifest(_manifest())


def test_legacy_fixed_vector_v1_remains_explicitly_supported() -> None:
    validate_cloud_dataset_manifest(_legacy_fixed_vector_manifest(), configured_dataset_root=LEGACY_DATASET_ROOT)


@pytest.mark.parametrize("field", ["payload_sha256", "normalization", "license_status"])
def test_cloud_manifest_contract_fails_closed(field: str) -> None:
    manifest = _manifest()
    manifest.pop(field)
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)


def test_sen12ts_dynamic_normalization_is_axes_and_batch_locked() -> None:
    for field, value in (("statistics_axes", [1, 2, 3]), ("scope", "per_sample_per_channel"), ("micro_batch", 8)):
        broken = _manifest()
        broken["normalization"][field] = value
        with pytest.raises(DatasetManifestError):
            validate_cloud_dataset_manifest(broken)


def test_sen12ts_selectors_labels_and_parent_first_split_fail_closed() -> None:
    for field, value in (
        ("sar_source_indices", [0, 1]),
        ("optical_source_indices", list(range(1, 13))),
        ("labels", 8),
        ("parent_shape", [120, 120]),
    ):
        broken = _manifest()
        broken[field] = value
        with pytest.raises(DatasetManifestError):
            validate_cloud_dataset_manifest(broken)
    broken = _manifest()
    broken["split_contract"]["parent_first"] = False
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(broken)


def test_local_or_test_bearing_manifest_is_rejected() -> None:
    manifest = _manifest()
    manifest["cloud_root"] = "local-data"
    manifest["test_accessed"] = True
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)


def test_dataset_pretrained_input_cross_binding() -> None:
    manifest = _manifest()
    audit = {
        "compatibility": {
            "input_spec": {
                "target": {
                    "band_order": {
                        "optical": manifest["optical_band_order"],
                        "sar": manifest["sar_channel_order"],
                    },
                    "normalization": manifest["normalization"],
                }
            }
        }
    }
    cross_validate_dataset_and_pretrained(manifest, audit)
    audit["compatibility"]["input_spec"]["target"]["band_order"]["sar"] = ["VH", "VV"]
    with pytest.raises(DatasetManifestError):
        cross_validate_dataset_and_pretrained(manifest, audit)


def test_path_escape_and_total_ledger_overflow_are_rejected() -> None:
    manifest = _manifest()
    manifest["cloud_root"] = chr(47) + "root" + chr(47) + "autodl-tmp" + chr(47) + ".." + chr(47) + "escape"
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)
    manifest = _manifest()
    manifest["component_storage_bytes"]["checkpoints"] = 13_000_000_000
    manifest["total_active_storage_bytes"] = 45_800_000_000
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)


def test_boolean_bytes_empty_normalization_and_wrong_exact_root_are_rejected() -> None:
    manifest = _manifest()
    manifest["storage_bytes"] = True
    manifest["component_storage_bytes"]["raw_payload"] = True
    manifest["total_active_storage_bytes"] = 1
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)

    manifest = _manifest()
    manifest["normalization"] = {"scheme": DYNAMIC_NORMALIZATION_SCHEME}
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)

    manifest = _manifest()
    manifest["cloud_root"] = chr(47) + "root" + chr(47) + "autodl-tmp" + chr(47) + "other_dataset"
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)


def test_resolved_sample_realpaths_must_be_inside_exact_dataset_root() -> None:
    manifest = _manifest()
    manifest["sample_realpaths"] = [chr(47) + "root" + chr(47) + "autodl-tmp" + chr(47) + "elsewhere" + chr(47) + "sample" + chr(46) + "npy"]
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)

    manifest = _manifest()
    manifest["sample_realpaths_resolved"] = False
    with pytest.raises(DatasetManifestError):
        validate_cloud_dataset_manifest(manifest)
