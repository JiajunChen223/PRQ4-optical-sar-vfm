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


def _summary(values: list[float]) -> LatencySummary:
    if len(values) < 2:
        raise ValueError("latency summary requires at least two observations")
    ordered = sorted(values)
    return LatencySummary(
        iterations=len(values),
        mean_ms=float(statistics.fmean(values)),
        median_ms=float(statistics.median(values)),
        std_ms=float(statistics.stdev(values)),
        p10_ms=float(_percentile(ordered, 0.10)),
        p90_ms=float(_percentile(ordered, 0.90)),
    )


def _event_time(fn: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


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
    return _summary([_event_time(fn) for _ in range(iterations)])


def profile_cuda_pair_abba(
    full_fn: Callable[[], object],
    ice_fn: Callable[[], object],
    *,
    warmup: int = 50,
    iterations_per_mode: int = 200,
) -> tuple[LatencySummary, LatencySummary]:
    """Profile Full/ICE in repeated ABBA order to reduce temporal GPU bias.

    Each ABBA cycle records Full, ICE, ICE, Full.  ``iterations_per_mode`` must
    therefore be even; each mode receives exactly that many timed observations.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires an available CUDA device")
    if warmup < 1 or iterations_per_mode < 2 or iterations_per_mode % 2 != 0:
        raise ValueError("warmup must be >=1 and iterations_per_mode must be an even integer >=2")
    for _ in range(warmup):
        full_fn()
        ice_fn()
    torch.cuda.synchronize()
    full_values: list[float] = []
    ice_values: list[float] = []
    for _ in range(iterations_per_mode // 2):
        full_values.append(_event_time(full_fn))
        ice_values.append(_event_time(ice_fn))
        ice_values.append(_event_time(ice_fn))
        full_values.append(_event_time(full_fn))
    return _summary(full_values), _summary(ice_values)
