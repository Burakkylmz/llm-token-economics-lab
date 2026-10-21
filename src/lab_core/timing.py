"""Monotonic client-side timing helpers.

Everything measured here is CLIENT-SIDE wall-clock time (via
time.perf_counter_ns), which includes network latency and is distinct from
any provider-reported server-side timing (e.g. Groq's queue_time /
prompt_time / completion_time / total_time). Never merge the two without
labeling which is which.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ElapsedTime:
    elapsed_ms: float


class StopwatchTimer:
    """Context manager measuring client-side elapsed time in milliseconds."""

    def __init__(self) -> None:
        self._start_ns: int | None = None
        self.result: ElapsedTime | None = None

    def __enter__(self) -> "StopwatchTimer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_ns is None:
            return
        elapsed_ns = time.perf_counter_ns() - self._start_ns
        self.result = ElapsedTime(elapsed_ms=elapsed_ns / 1_000_000)
