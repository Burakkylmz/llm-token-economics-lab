"""Pure latency/throughput statistics helpers for Lab08.

Percentiles are computed with linear interpolation between ranks (the
same method used by numpy's default 'linear' interpolation and Excel's
PERCENTILE.INC), so results are reproducible without adding a numpy
dependency for one function.
"""

from __future__ import annotations

from typing import Optional


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (pct / 100)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = rank - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def summarize_batch(latencies_ms: list[float], total_elapsed_ms: float) -> dict[str, Optional[float]]:
    """Aggregate per-call latencies plus the batch's total wall-clock time.

    `total_elapsed_ms` must be measured around the whole batch (sequential
    loop or asyncio.gather), not derived from the individual latencies --
    for concurrent execution, requests_per_second depends on overlap, which
    individual latencies alone cannot tell you.
    """
    if not latencies_ms or total_elapsed_ms <= 0:
        return {"p50_latency_ms": None, "p95_latency_ms": None, "requests_per_second": None, "throughput": None}

    requests_per_second = len(latencies_ms) / (total_elapsed_ms / 1000)
    return {
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "requests_per_second": requests_per_second,
        "throughput": requests_per_second,
    }
