"""CUDA profiling helpers for ICE-Exact Full/ICE comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Callable

import torch


@dataclass(frozen=True)
class LatencySummary:
    iterations: int
    mean_ms: float
    median_ms: float
    std_ms: float
    p10_ms: float
    p90_ms: float


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def profile_cuda_callable(
    fn: Callable[[], object], *, warmup: int = 50, iterations: int = 200
) -> LatencySummary:
    """Profile one CUDA callable with synchronized CUDA events."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires an available CUDA device")
    if warmup < 1 or iterations < 2:
        raise ValueError("warmup must be >=1 and iterations must be >=2")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    ordered = sorted(values)
    return LatencySummary(
        iterations=iterations,
        mean_ms=float(statistics.fmean(values)),
        median_ms=float(statistics.median(values)),
        std_ms=float(statistics.stdev(values)),
        p10_ms=float(_percentile(ordered, 0.10)),
        p90_ms=float(_percentile(ordered, 0.90)),
    )
