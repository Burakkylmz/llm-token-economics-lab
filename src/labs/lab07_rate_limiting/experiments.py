"""Lab07 experiments: retry/backoff strategies under simulated rate-limit failures.

This lab deliberately never calls the real Groq API. Injecting real 429s
would require either exceeding the live API's actual limits or waiting on
genuine provider load, neither of which is appropriate for a repeatable
classroom exercise. Instead every experiment runs against a self-contained
fake "success" client wrapped by FlakyClient (simulated_client.py), which
raises real groq.RateLimitError instances for a configured number of
leading attempts -- exercising the real exception type and header-parsing
path, with zero real network traffic and zero real cost.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Callable, Optional

from lab_core.config import RetryConfig
from lab_core.metrics import ExecutionMetadata, ExperimentResult, PerformanceMetrics, TokenCostMetrics
from lab_core.timing import StopwatchTimer
from labs.lab07_rate_limiting.retry_runner import call_with_retry
from labs.lab07_rate_limiting.simulated_client import FlakyClient

STRATEGIES: tuple[str, ...] = (
    "no_backoff",
    "exponential_backoff",
    "exponential_backoff_with_jitter",
    "rate_limit_aware",
)


def make_fake_success_client():
    """A minimal fake client whose create() always succeeds, for the FlakyClient to delegate to."""

    class _Completions:
        def create(self, **kwargs):
            message = SimpleNamespace(content="ok")
            choice = SimpleNamespace(message=message)
            usage = SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                total_time=0.1,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )
            return SimpleNamespace(choices=[choice], usage=usage)

    class _Chat:
        def __init__(self, completions):
            self.completions = completions

    class _Client:
        def __init__(self):
            self.chat = _Chat(_Completions())

    return _Client()


def run_retry_strategy(
    strategy: str,
    retry_config: RetryConfig,
    fail_count: int,
    retry_after_seconds: Optional[str] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ExperimentResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Expected one of {STRATEGIES}.")

    use_backoff = strategy != "no_backoff"
    jitter = strategy == "exponential_backoff_with_jitter"
    respect_retry_after = strategy == "rate_limit_aware"

    flaky_client = FlakyClient(
        make_fake_success_client(), fail_count=fail_count, retry_after_seconds=retry_after_seconds
    )

    with StopwatchTimer() as timer:
        outcome = call_with_retry(
            client=flaky_client,
            max_attempts=retry_config.max_attempts,
            base_delay_seconds=retry_config.base_delay_seconds,
            max_delay_seconds=retry_config.max_delay_seconds,
            jitter=jitter,
            use_backoff=use_backoff,
            respect_retry_after=respect_retry_after,
            sleep_fn=sleep_fn,
            model="simulated-model",
            messages=[{"role": "user", "content": "ping"}],
        )

    performance = PerformanceMetrics(client_elapsed_ms=timer.result.elapsed_ms if timer.result else None)
    tokens_and_cost = (
        outcome.call_result.tokens_and_cost
        if outcome.succeeded and outcome.call_result is not None
        else TokenCostMetrics(api_call_count=0)
    )

    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id=f"lab07_{strategy}",
            run_id=str(uuid.uuid4()),
            model_id="simulated-model",
            execution_mode="simulated",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_parameters={
                "fail_count": fail_count,
                "succeeded": outcome.succeeded,
                "total_wait_seconds": outcome.total_wait_seconds,
            },
            error=outcome.error,
            retry_count=outcome.retry_count,
        ),
        tokens_and_cost=tokens_and_cost,
        performance=performance,
    )
