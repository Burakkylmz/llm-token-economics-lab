from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_core.config import RetryConfig
from labs.lab07_rate_limiting.experiments import run_retry_strategy
from labs.lab07_rate_limiting.retry_runner import call_with_retry
from labs.lab07_rate_limiting.simulated_client import FlakyClient, make_rate_limit_error


def _retry_config(**overrides) -> RetryConfig:
    defaults = dict(max_attempts=5, base_delay_seconds=1.0, max_delay_seconds=30.0, jitter=False)
    defaults.update(overrides)
    return RetryConfig(**defaults)


class _FakeCompletions:
    def create(self, **kwargs):
        usage = SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            total_time=0.01,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=usage)


class _FakeSuccessClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_make_rate_limit_error_carries_retry_after_header():
    error = make_rate_limit_error(retry_after_seconds="7")
    assert error.response.status_code == 429
    assert error.response.headers.get("retry-after") == "7"


def test_flaky_client_fails_exactly_fail_count_times_then_succeeds():
    client = FlakyClient(_FakeSuccessClient(), fail_count=2)
    from groq import RateLimitError

    with pytest.raises(RateLimitError):
        client.chat.completions.create(model="x", messages=[])
    with pytest.raises(RateLimitError):
        client.chat.completions.create(model="x", messages=[])
    response = client.chat.completions.create(model="x", messages=[])
    assert response.choices[0].message.content == "ok"


def test_call_with_retry_no_backoff_never_sleeps():
    client = FlakyClient(_FakeSuccessClient(), fail_count=2)
    sleeps = []
    outcome = call_with_retry(
        client=client,
        max_attempts=5,
        base_delay_seconds=1.0,
        max_delay_seconds=30.0,
        jitter=False,
        use_backoff=False,
        respect_retry_after=False,
        sleep_fn=sleeps.append,
        model="x",
        messages=[],
    )
    assert outcome.succeeded is True
    assert outcome.retry_count == 2
    assert sleeps == []  # no_backoff uses a zero delay, so sleep_fn is never invoked


def test_call_with_retry_exponential_backoff_grows_delays():
    client = FlakyClient(_FakeSuccessClient(), fail_count=3)
    sleeps = []
    outcome = call_with_retry(
        client=client,
        max_attempts=5,
        base_delay_seconds=1.0,
        max_delay_seconds=30.0,
        jitter=False,
        use_backoff=True,
        respect_retry_after=False,
        sleep_fn=sleeps.append,
        model="x",
        messages=[],
    )
    assert outcome.succeeded is True
    assert outcome.retry_count == 3
    assert sleeps == [1.0, 2.0, 4.0]


def test_call_with_retry_respects_retry_after_header():
    client = FlakyClient(_FakeSuccessClient(), fail_count=1, retry_after_seconds="9")
    sleeps = []
    outcome = call_with_retry(
        client=client,
        max_attempts=3,
        base_delay_seconds=1.0,
        max_delay_seconds=30.0,
        jitter=False,
        use_backoff=True,
        respect_retry_after=True,
        sleep_fn=sleeps.append,
        model="x",
        messages=[],
    )
    assert sleeps == [9.0]
    assert outcome.succeeded is True


def test_call_with_retry_gives_up_after_max_attempts():
    client = FlakyClient(_FakeSuccessClient(), fail_count=10)
    outcome = call_with_retry(
        client=client,
        max_attempts=3,
        base_delay_seconds=0.01,
        max_delay_seconds=0.02,
        jitter=False,
        use_backoff=True,
        respect_retry_after=False,
        sleep_fn=lambda s: None,
        model="x",
        messages=[],
    )
    assert outcome.succeeded is False
    assert outcome.retry_count == 3
    assert outcome.error is not None


def test_run_retry_strategy_no_backoff():
    result = run_retry_strategy(
        strategy="no_backoff",
        retry_config=_retry_config(),
        fail_count=2,
        sleep_fn=lambda s: None,
    )
    assert result.metadata.model_parameters["succeeded"] is True
    assert result.metadata.retry_count == 2
    assert result.metadata.model_parameters["total_wait_seconds"] == 0.0


def test_run_retry_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        run_retry_strategy(strategy="bogus", retry_config=_retry_config(), fail_count=1)
