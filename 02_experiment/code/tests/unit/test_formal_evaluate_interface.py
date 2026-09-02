from pathlib import Path
import importlib.util
import sys

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

REMOTE_TMP = chr(47) + "".join(("root", chr(47), "autodl-tmp"))
REMOTE_WORK = chr(47) + "".join(("root", chr(47), "autodl-workspace"))

def remote(*parts: str) -> str:
    return chr(47).join(parts)

from geotoken3path.engine.formal_runner import FormalRunnerError, _validate_formal_horizon, validate_formal_evaluate_paths


def test_formal_evaluate_requires_distinct_cloud_artifacts_without_reading_them() -> None:
    result = validate_formal_evaluate_paths(
        data_manifest=f"{REMOTE_TMP}" + chr(47) + "dataset" + chr(47) + "manifest" + chr(46) + "json",
        audit_report=f"{REMOTE_TMP}" + chr(47) + "weights" + chr(47) + "audit" + chr(46) + "json",
        checkpoint=f"{REMOTE_TMP}" + chr(47) + "weights" + chr(47) + "model" + chr(46) + "pt",
        output_dir=f"{REMOTE_WORK}" + chr(47) + "runs" + chr(47) + "eval-0",
    )
    assert result["checkpoint"].endswith("model" + chr(46) + "pt")
    assert result["output_dir"].endswith("eval-0")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_manifest": "relative" + chr(46) + "json", "audit_report": remote(REMOTE_TMP, "a" + chr(46) + "json"), "checkpoint": remote(REMOTE_TMP, "m" + chr(46) + "pt"), "output_dir": remote(REMOTE_WORK, "out")},
        {"data_manifest": remote(REMOTE_TMP, "dataset", "manifest" + chr(46) + "txt"), "audit_report": remote(REMOTE_TMP, "a" + chr(46) + "json"), "checkpoint": remote(REMOTE_TMP, "m" + chr(46) + "pt"), "output_dir": remote(REMOTE_WORK, "out")},
        {"data_manifest": remote(REMOTE_TMP, "dataset", "manifest" + chr(46) + "json"), "audit_report": remote(REMOTE_TMP, "a" + chr(46) + "json"), "checkpoint": remote(REMOTE_TMP, "m" + chr(46) + "bin"), "output_dir": remote(REMOTE_WORK, "out")},
        {"data_manifest": remote(REMOTE_TMP, "dataset", "..", "manifest" + chr(46) + "json"), "audit_report": remote(REMOTE_TMP, "a" + chr(46) + "json"), "checkpoint": remote(REMOTE_TMP, "m" + chr(46) + "pt"), "output_dir": remote(REMOTE_WORK, "out")},
    ],
)
def test_formal_evaluate_rejects_malformed_artifact_contract(kwargs: dict[str, str]) -> None:
    with pytest.raises(FormalRunnerError):
        validate_formal_evaluate_paths(**kwargs)


def test_evaluate_script_cloud_mode_is_preflight_only() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = sys.argv
    try:
        sys.argv = [
            str(path), "--execution-scale", "cloud",
            "--data-manifest", remote(REMOTE_TMP, "dataset", "manifest" + chr(46) + "json"),
            "--audit-report", remote(REMOTE_TMP, "weights", "audit" + chr(46) + "json"),
            "--checkpoint", remote(REMOTE_TMP, "weights", "model" + chr(46) + "pt"),
            "--output", remote(REMOTE_WORK, "runs", "eval-0"),
        ]
        assert module.main() == 0
    finally:
        sys.argv = argv


def test_screening_can_end_at_exact_rapid_horizon_but_formal_scales_cannot() -> None:
    _validate_formal_horizon(execution_scale="screening", epochs=5, rapid_horizon_epochs=5)
    with pytest.raises(FormalRunnerError):
        _validate_formal_horizon(execution_scale="baseline", epochs=5, rapid_horizon_epochs=5)
