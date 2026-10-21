"""Generic exponential-backoff-with-jitter helper.

This module intentionally contains no retry *policy* decisions (how many
attempts, whether jitter is on) and is not wired automatically into
groq_client.py. Retry behavior is itself an experiment variable for
Lab07 (Rate Limiting & Retry Strategies), so each lab/experiment decides
explicitly how to use this helper rather than inheriting a hidden default.
"""

from __future__ import annotations

import random


def compute_backoff_delay_seconds(
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    jitter: bool,
) -> float:
    """Compute the delay before retry attempt `attempt` (1-indexed).

    Uses exponential backoff: base_delay * 2^(attempt - 1), capped at
    max_delay_seconds. When jitter is True, returns a uniform random value
    in [0, computed_delay] ("full jitter"), otherwise returns the computed
    delay unmodified.
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")

    exponential_delay = base_delay_seconds * (2 ** (attempt - 1))
    capped_delay = min(exponential_delay, max_delay_seconds)

    if jitter:
        return random.uniform(0, capped_delay)
    return capped_delay
