from __future__ import annotations

import pytest

from geotoken3path.execution.profiling import _percentile


def test_percentile_interpolates_monotonic_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 4.0
    assert _percentile(values, 0.5) == pytest.approx(2.5)
