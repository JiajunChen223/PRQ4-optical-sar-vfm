import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("geotoken3path_train_script", ROOT / "scripts/train.py")
assert SPEC is not None and SPEC.loader is not None
TRAIN_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SCRIPT)


def _small_resolved() -> dict[str, object]:
    return {
        "route_id": "PRQ4-BASELINE-VERIFIED-R2",
        "candidate_id": "BASELINE-ALWAYS-FUSE",
        "dataset_id": "sen12ts_worldcover_3region_1200",
        "matched_common_protocol_sha256": "a" * 64,
        "initialization_ref": "02_experiment/reports/pretrained_weight_audit.json",
        "model": {
            "token_dim": 16,
            "num_classes": 4,
            "mechanism_set": "always_fuse",
            "active_budget": 0.5,
            "local_window_tokens": 9,
            "stages": ["mid", "late"],
            "depth_group_size": 4,
            "depth_taps": {
                "stage": {
                    "optical": {"mid": "s2.mid", "late": "s2.late"},
                    "sar": {"mid": "s1.mid", "late": "s1.late"},
                },
                "sar_depth_group": {
                    "mid": ["s1.0", "s1.1", "s1.2", "s1.3"],
                    "late": ["s1.2", "s1.3", "s1.4", "s1.5"],
                },
            },
            "allow_synthetic_depth_group_fallback": False,
        },
        "input": {"optical_channels": 12, "sar_channels": 2, "patch_size": 120},
        "storage": {"hard_stop_gb": 45, "total_ceiling_gb": 50},
        "trainability": {
            "trunk": "frozen",
            "router": "trainable",
            "adapters": "trainable",
            "decoder": "trainable",
        },
        "runtime": {
            "precision": "amp",
            "micro_batch": 2,
            "effective_batch": 4,
            "gradient_accumulation": 2,
            "seed": 0,
            "test_seal_status": "sealed",
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.0001,
                "weight_decay": 0.05,
                "betas": [0.9, 0.999],
            },
            "scheduler": {"name": "cosine_with_warmup", "warmup_fraction": 0.05},
            "gradient_clip_norm": 1.0,
        },
    }


def test_synthetic_smoke_executes_declared_runtime_semantics() -> None:
    result = TRAIN_SCRIPT.run_synthetic_smoke(_small_resolved())
    assert result["status"] == "synthetic_segmentation_step_pass"
    assert result["scientific_result"] is False
    assert result["optimizer_name"] == "adamw"
    assert result["optimizer_learning_rate"] == 0.0001
    assert result["optimizer_weight_decay"] == 0.05
    assert result["optimizer_betas"] == [0.9, 0.999]
    assert result["optimizer_steps"] == 1
    assert result["scheduler_name"] == "cosine_with_warmup"
    assert result["scheduler_steps"] == 1
    assert result["gradient_accumulation_steps"] == 2
    assert result["gradient_clip_applied"] is True
    assert result["gradient_clip_max_norm"] == 1.0
    assert result["gradient_norm_is_finite"] is True
    assert result["autocast_device"] == "cpu"
    assert result["autocast_enabled"] is True
    assert result["micro_batch"] == 2
    assert result["effective_batch"] == 4
    assert result["loss_is_finite"] is True
