from __future__ import annotations

import torch

from geotoken3path.execution.certification import compare_gradients, compare_tensors


def test_compare_tensors_reports_exact_identity() -> None:
    value = torch.randn(3, 4)
    result = compare_tensors(value, value.clone())
    assert result.shape_equal
    assert result.dtype_equal
    assert result.torch_equal
    assert result.max_abs_error == 0.0
    assert result.mean_abs_error == 0.0
    assert result.max_relative_error == 0.0


def test_compare_gradients_reports_missing_one_sided_gradient() -> None:
    left = {"a": torch.ones(2), "b": None}
    right = {"a": torch.ones(2), "b": torch.ones(2)}
    result = compare_gradients(left, right)
    assert result["missing_gradient_names"] == ["b"]
    assert result["exact_gradient_count"] == 1
