"""Shared operational retry policy. SDK retries are disabled at client construction."""

from __future__ import annotations

import time
from typing import Callable, Optional

from groq import APIConnectionError, GroqError, Groq, InternalServerError, RateLimitError

from lab_core.config import RetryConfig
from lab_core.groq_client import SyncCallResult, call_sync
from lab_core.retry import compute_backoff_delay_seconds

RETRYABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APIConnectionError)


def call_with_operational_retry(
    client: Groq,
    retry_config: RetryConfig,
    sleep_fn: Callable[[float], None] = time.sleep,
    **create_kwargs,
) -> tuple[Optional[SyncCallResult], int, Optional[str]]:
    """Operational resilience for a long factorial campaign -- NOT Lab07's
    retry-STRATEGY experiment (lab07_rate_limiting/retry_runner.py), which
    exists specifically to compare retry policies against each other as an
    experiment variable. This wrapper applies one fixed, sensible policy
    (this project's existing [retry] config) so a 429/5xx/network blip
    doesn't abort a multi-hundred-call run, using the same backoff math.

    Returns (result, attempts_used, error_message). Exactly one of
    `result`/`error_message` is non-None.

    A non-retryable provider error (bad request, auth, permission, ...) is
    NOT retried, but it also does NOT propagate out of this function: it is
    recorded as this call's single failed attempt and returned like any
    other failure. One rejected parameter combination (e.g. an unsupported
    reasoning_effort/temperature/top_p combination the API itself refuses)
    must not abort the rest of a multi-hundred-call factorial campaign --
    it shows up as one explicit, visible failed measurement instead (see
    labs.lab01_token_anatomy.factorial_models.FactorialMeasurement.error).
    Any exception that is NOT a GroqError (a bug in this project's own code,
    e.g. a wrong keyword argument) is deliberately left to propagate and
    crash loudly rather than being misreported as an "API failure".
    """
    last_error: Optional[str] = None
    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            result = call_sync(client, **create_kwargs)
            return result, attempt, None
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retry_config.max_attempts:
                return None, attempt, last_error

            retry_after_header = None
            if isinstance(exc, RateLimitError):
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", None)
                retry_after_header = headers.get("retry-after") if headers is not None else None

            delay = (
                float(retry_after_header)
                if retry_after_header is not None
                else compute_backoff_delay_seconds(
                    attempt, retry_config.base_delay_seconds, retry_config.max_delay_seconds, retry_config.jitter
                )
            )
            if delay > 0:
                sleep_fn(delay)
        except GroqError as exc:
            return None, attempt, f"{type(exc).__name__}: {exc}"

    return None, retry_config.max_attempts, last_error


