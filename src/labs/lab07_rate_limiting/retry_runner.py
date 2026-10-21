"""Generic retry loop, built on lab_core.retry's backoff math, used by Lab07's experiments.

Not part of lab_core: unlike the pure math in lab_core.retry, this loop
encodes an actual retry *policy* (when to honor retry-after, when to back
off, when to give up), which is exactly the experiment variable Lab07
studies -- so it stays lab-specific rather than becoming a hidden default
every other lab inherits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from groq import RateLimitError

from lab_core.groq_client import SyncCallResult, call_sync
from lab_core.retry import compute_backoff_delay_seconds


@dataclass
class RetryOutcome:
    call_result: Optional[SyncCallResult]
    succeeded: bool
    retry_count: int
    total_wait_seconds: float
    error: Optional[str]


def call_with_retry(
    client: Any,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    jitter: bool,
    use_backoff: bool,
    respect_retry_after: bool,
    sleep_fn: Callable[[float], None] = time.sleep,
    **create_kwargs: Any,
) -> RetryOutcome:
    retry_count = 0
    total_wait = 0.0

    for attempt in range(1, max_attempts + 1):
        try:
            result = call_sync(client, **create_kwargs)
            return RetryOutcome(
                call_result=result, succeeded=True, retry_count=retry_count, total_wait_seconds=total_wait, error=None
            )
        except RateLimitError as exc:
            retry_count += 1
            if attempt == max_attempts:
                return RetryOutcome(
                    call_result=None,
                    succeeded=False,
                    retry_count=retry_count,
                    total_wait_seconds=total_wait,
                    error=str(exc),
                )

            delay = 0.0
            retry_after_header = exc.response.headers.get("retry-after") if respect_retry_after else None
            if retry_after_header is not None:
                delay = float(retry_after_header)
            elif use_backoff:
                delay = compute_backoff_delay_seconds(attempt, base_delay_seconds, max_delay_seconds, jitter)
            # else ("no_backoff" strategy): retry immediately, delay stays 0.0

            total_wait += delay
            if delay > 0:
                sleep_fn(delay)

    return RetryOutcome(
        call_result=None, succeeded=False, retry_count=retry_count, total_wait_seconds=total_wait, error="max attempts exceeded"
    )
