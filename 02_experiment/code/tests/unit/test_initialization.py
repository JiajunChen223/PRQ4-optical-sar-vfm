from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
import torch

from geotoken3path.models.factory import build_model
from geotoken3path.models.initialization import apply_audited_state_dict, validate_pretrained_audit


@pytest.fixture(autouse=True)
def _preserve_global_torch_rng() -> object:
    """Keep schema tests from perturbing mechanism tests collected later."""

    state = torch.random.get_rng_state()
    yield
    torch.random.set_rng_state(state)


def _endpoint_input() -> dict[str, object]:
    optical_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    return {
        "band_order": {
            "optical": optical_bands,
            "sar": ["VV", "VH"],
        },
        "normalization": {
            "optical": {"mean": [0.0] * len(optical_bands), "std": [1.0] * len(optical_bands)},
            "sar": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        },
        "gsd_meters": {"optical": 10.0, "sar": 10.0},
        "patch_size": [120, 120],
    }


def _dynamic_normalization() -> dict[str, object]:
    return {
        "scheme": "croma_official_dynamic_v1",
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


def _passing_audit() -> dict[str, object]:
    endpoint = _endpoint_input()
    return {
        "status": "pass",
        "execution_context": "cloud",
        "initialization_mode": "pretrained",
        "sha256": "a" * 64,
        "source": {
            "url": "https://example.org/releases/croma-checkpoint",
            "license": "Apache-2.0",
            "commit": "0123456789abcdef",
        },
        "compatibility": {
            "status": "pass",
            "architecture": {
                "checkpoint_backbone": "croma-radar-optical-vit-b",
                "target_backbone": "croma-radar-optical-vit-b",
                "compatible": True,
            },
            "input_spec": {
                "checkpoint": deepcopy(endpoint),
                "target": deepcopy(endpoint),
            },
            "head_replacement": {
                "required": False,
                "configured": True,
                "recorded": True,
                "checkpoint_head_keys": [],
                "target_head_keys": [],
            },
            "state_dict": {
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatches": [],
            },
            "position_resolution_adaptation": {
                "required": False,
                "status": "not_required",
                "method": "none",
                "source_grid": [12, 12],
                "target_grid": [12, 12],
            },
        },
        "comparison_policy": {
            "same_initialization_for_baseline_and_innovation": True,
            "same_checkpoint_sha256": True,
            "same_input_spec": True,
            "target_test_data_used": False,
        },
    }


def _delete_path(record: dict[str, object], path: str) -> None:
    parts = path.split(".")
    current = record
    for part in parts[:-1]:
        current = current[part]  # type: ignore[assignment,index]
    del current[parts[-1]]


def _set_path(record: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    current = record
    for part in parts[:-1]:
        current = current[part]  # type: ignore[assignment,index]
    current[parts[-1]] = value


def test_complete_audit_passes() -> None:
    validate_pretrained_audit(_passing_audit())


def test_dynamic_normalization_target_is_complete_and_exactly_compared() -> None:
    audit = _passing_audit()
    dynamic = _dynamic_normalization()
    audit["compatibility"]["input_spec"]["checkpoint"]["normalization"] = deepcopy(dynamic)
    audit["compatibility"]["input_spec"]["target"]["normalization"] = deepcopy(dynamic)
    validate_pretrained_audit(audit)

    audit["compatibility"]["input_spec"]["target"]["normalization"]["statistics_axes"] = [1, 2, 3]
    with pytest.raises(ValueError):
        validate_pretrained_audit(audit)


@pytest.mark.parametrize(
    "path",
    [
        "source",
        "source.url",
        "source.license",
        "source.commit",
        "compatibility.architecture",
        "compatibility.architecture.checkpoint_backbone",
        "compatibility.architecture.target_backbone",
        "compatibility.architecture.compatible",
        "compatibility.input_spec",
        "compatibility.input_spec.checkpoint.band_order",
        "compatibility.input_spec.checkpoint.normalization",
        "compatibility.input_spec.checkpoint.gsd_meters",
        "compatibility.input_spec.checkpoint.patch_size",
        "compatibility.input_spec.target",
        "compatibility.head_replacement",
        "compatibility.head_replacement.required",
        "compatibility.head_replacement.configured",
        "compatibility.head_replacement.recorded",
        "compatibility.head_replacement.checkpoint_head_keys",
        "compatibility.head_replacement.target_head_keys",
        "compatibility.state_dict",
        "compatibility.state_dict.missing_keys",
        "compatibility.state_dict.unexpected_keys",
        "compatibility.state_dict.shape_mismatches",
        "compatibility.position_resolution_adaptation",
        "compatibility.position_resolution_adaptation.required",
        "compatibility.position_resolution_adaptation.status",
        "compatibility.position_resolution_adaptation.method",
        "compatibility.position_resolution_adaptation.source_grid",
        "compatibility.position_resolution_adaptation.target_grid",
        "comparison_policy.same_checkpoint_sha256",
        "comparison_policy.same_input_spec",
    ],
)
def test_every_required_nested_field_is_fail_closed(path: str) -> None:
    audit = _passing_audit()
    _delete_path(audit, path)
    with pytest.raises(ValueError):
        validate_pretrained_audit(audit)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("status", "pending"),
        ("execution_context", "local"),
        ("initialization_mode", "random_init"),
        ("sha256", "not-a-64-digit-hex-hash"),
        ("source.url", "ftp://example.org/checkpoint"),
        ("source.license", "pending"),
        ("source.commit", "unknown"),
        ("compatibility.status", "pending"),
        ("compatibility.architecture.compatible", False),
        ("compatibility.input_spec.target.band_order.sar", ["VH", "VV"]),
        ("compatibility.input_spec.target.normalization.sar.std", [1.0, 0.0]),
        ("compatibility.input_spec.target.gsd_meters.sar", -10.0),
        ("compatibility.input_spec.target.patch_size", [120, 0]),
        ("compatibility.head_replacement.configured", False),
        ("compatibility.head_replacement.recorded", False),
        ("compatibility.state_dict.shape_mismatches", ["optical_stem.weight"]),
        ("compatibility.position_resolution_adaptation.status", "pass"),
        ("compatibility.position_resolution_adaptation.method", "bilinear"),
        ("comparison_policy.same_initialization_for_baseline_and_innovation", False),
        ("comparison_policy.same_checkpoint_sha256", False),
        ("comparison_policy.same_input_spec", False),
        ("comparison_policy.target_test_data_used", True),
    ],
)
def test_invalid_audit_values_are_fail_closed(path: str, value: object) -> None:
    audit = _passing_audit()
    _set_path(audit, path, value)
    with pytest.raises(ValueError):
        validate_pretrained_audit(audit)


def test_resolution_difference_requires_a_passing_adaptation() -> None:
    audit = _passing_audit()
    _set_path(audit, "compatibility.input_spec.target.patch_size", [224, 224])
    with pytest.raises(ValueError):
        validate_pretrained_audit(audit)

    _set_path(audit, "compatibility.position_resolution_adaptation.required", True)
    _set_path(audit, "compatibility.position_resolution_adaptation.status", "pass")
    _set_path(audit, "compatibility.position_resolution_adaptation.method", "bicubic_positional_interpolation")
    _set_path(audit, "compatibility.position_resolution_adaptation.target_grid", [14, 14])
    validate_pretrained_audit(audit)


def test_minimal_compatibility_status_is_rejected() -> None:
    audit = _passing_audit()
    audit["compatibility"] = {"status": "pass"}
    with pytest.raises(ValueError):
        validate_pretrained_audit(audit)


def test_same_complete_audit_loads_into_baseline_and_candidate() -> None:
    source = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    baseline = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    candidate = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    audit = _passing_audit()
    assert apply_audited_state_dict(baseline, source.state_dict(), audit) == {
        "missing_keys": [],
        "unexpected_keys": [],
    }
    assert apply_audited_state_dict(candidate, source.state_dict(), audit) == {
        "missing_keys": [],
        "unexpected_keys": [],
    }


def test_unexplained_missing_key_fails_before_model_mutation() -> None:
    source = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    target = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    state = dict(source.state_dict())
    del state["classifier.bias"]
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="differs from audit"):
        apply_audited_state_dict(target, state, _passing_audit(), strict=False)
    assert all(torch.equal(before[key], value) for key, value in target.state_dict().items())


def test_audited_head_replacement_is_exact_and_requires_non_strict_load() -> None:
    source = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    target = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    state = dict(source.state_dict())
    head_keys = ["classifier.weight", "classifier.bias"]
    for key in head_keys:
        del state[key]
    audit = _passing_audit()
    _set_path(audit, "compatibility.head_replacement.required", True)
    _set_path(audit, "compatibility.head_replacement.target_head_keys", head_keys)
    _set_path(audit, "compatibility.state_dict.missing_keys", head_keys)

    with pytest.raises(ValueError, match="strict=True"):
        apply_audited_state_dict(target, state, audit, strict=True)
    assert apply_audited_state_dict(target, state, audit, strict=False) == {
        "missing_keys": sorted(head_keys),
        "unexpected_keys": [],
    }


def test_shape_mismatch_fails_before_model_mutation() -> None:
    source = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    target = build_model({"token_dim": 32}, mechanism_set="always_fuse")
    state = dict(source.state_dict())
    state["classifier.weight"] = state["classifier.weight"][:1]
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_audited_state_dict(target, state, _passing_audit())
    assert all(torch.equal(before[key], value) for key, value in target.state_dict().items())
