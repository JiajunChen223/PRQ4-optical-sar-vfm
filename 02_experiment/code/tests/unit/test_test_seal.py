from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from geotoken3path.utils.test_seal import TestSealViolation, assert_test_access_allowed


def test_development_split_is_allowed() -> None:
    assert_test_access_allowed({"execution_scale": "smoke", "test_seal_status": "sealed"}, "validation") is None


def test_test_split_is_rejected_before_final_test() -> None:
    with pytest.raises(TestSealViolation):
        assert_test_access_allowed({"execution_scale": "smoke", "test_seal_status": "sealed"}, "test")


def test_final_test_requires_both_manifest_fields() -> None:
    assert_test_access_allowed({"execution_scale": "final_test", "test_seal_status": "final_test"}, "test") is None
