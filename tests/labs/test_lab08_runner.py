from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from labs.lab08_concurrency.models import (
    EVENT_PERMIT_ACQUIRED,
    EVENT_QUEUED,
    EVENT_REQUEST_COMPLETED,
    EVENT_REQUEST_STARTED,
    EXECUTION_MODE_CONCURRENT,
    EXECUTION_MODE_SEQUENTIAL,
    STATUS_CANCELLED,
    STATUS_SUCCESS,
    PlannedConcurrencyCall,
)
from labs.lab08_concurrency.runner import run_concurrency_level


def _usage():
    return SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15, total_time=0.01,
        prompt_tokens_details=None, completion_tokens_details=None,
    )


def _response(content="8"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=_usage(), system_fingerprint=None)


class _DelayedCompletions:
    """A fake async completions endpoint with an injectable delay, so
    concurrency enforcement and queue-wait timing are genuinely
    observable (zero-delay mocks would make every task "overlap" trivially)."""

    def __init__(self, delay_seconds: float, response_factory=_response, fail_indices: frozenset[int] = frozenset()):
        self._delay = delay_seconds
        self._response_factory = response_factory
        self._fail_indices = fail_indices
        self.calls: list[dict] = []
        self._call_count = 0

    async def create(self, **kwargs):
        self._call_count += 1
        this_call = self._call_count
        self.calls.append(dict(kwargs))
        await asyncio.sleep(self._delay)
        if this_call in self._fail_indices:
            raise ConnectionError("simulated failure")
        return self._response_factory()


def _fake_client(delay_seconds=0.01, **kwargs):
    completions = _DelayedCompletions(delay_seconds, **kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _plan(max_concurrent, count=4):
    return [
        PlannedConcurrencyCall(execution_index=i + 1, repetition_index=1, max_concurrent=max_concurrent, task_id=f"task_{i+1:03d}", task_index=i)
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_concurrency_one_behaves_sequentially_never_exceeds_one_in_flight():
    client, _ = _fake_client(delay_seconds=0.02)
    measurements, events, wall_clock_ms, peak = await run_concurrency_level(
        client, "m", None, "low", 100, "run-1", _plan(max_concurrent=1, count=4)
    )
    assert peak == 1
    assert all(m.execution_mode == EXECUTION_MODE_SEQUENTIAL for m in measurements)
    assert all(m.request_status == STATUS_SUCCESS for m in measurements)


@pytest.mark.asyncio
async def test_concurrency_limit_enforced_and_execution_mode_concurrent():
    client, _ = _fake_client(delay_seconds=0.03)
    measurements, events, wall_clock_ms, peak = await run_concurrency_level(
        client, "m", None, "low", 100, "run-1", _plan(max_concurrent=3, count=8)
    )
    assert peak <= 3
    assert peak > 1  # genuinely overlapped, not accidentally sequential
    assert all(m.execution_mode == EXECUTION_MODE_CONCURRENT for m in measurements)


@pytest.mark.asyncio
async def test_concurrency_two_enforced():
    client, _ = _fake_client(delay_seconds=0.03)
    _, _, _, peak = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=2, count=6))
    assert peak <= 2
    assert peak > 1


@pytest.mark.asyncio
async def test_task_order_preserved_in_output_despite_concurrent_completion():
    client, _ = _fake_client(delay_seconds=0.01)
    measurements, _, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=5, count=5))
    assert [m.task_id for m in measurements] == [f"task_{i+1:03d}" for i in range(5)]


@pytest.mark.asyncio
async def test_no_semaphore_leak_after_many_batches():
    client, _ = _fake_client(delay_seconds=0.005)
    for _ in range(3):
        _, _, _, peak = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=2, count=4))
        assert peak <= 2  # a leak would let later batches exceed the limit


@pytest.mark.asyncio
async def test_queue_wait_measured_and_excluded_from_service_latency():
    client, _ = _fake_client(delay_seconds=0.03)
    measurements, events, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=1, count=3))
    # The 3rd task must wait for the first two to finish under concurrency=1.
    last = measurements[-1]
    assert last.queue_wait_ms is not None and last.queue_wait_ms > 0
    assert last.request_service_latency_ms is not None
    assert last.task_end_to_end_latency_ms >= last.queue_wait_ms + last.request_service_latency_ms - 5  # small instrumentation overhead tolerance


@pytest.mark.asyncio
async def test_timeline_events_recorded_for_every_task():
    client, _ = _fake_client(delay_seconds=0.01)
    _, events, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=2, count=2))
    event_types = {e.event_type for e in events}
    assert event_types == {EVENT_QUEUED, EVENT_PERMIT_ACQUIRED, EVENT_REQUEST_STARTED, EVENT_REQUEST_COMPLETED}
    assert len(events) == 2 * 4  # 2 tasks x 4 events each


@pytest.mark.asyncio
async def test_one_request_failure_does_not_lose_other_results():
    client, _ = _fake_client(delay_seconds=0.01, fail_indices=frozenset({2}))
    measurements, _, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=4, count=4))
    statuses = [m.request_status for m in measurements]
    assert statuses.count(STATUS_SUCCESS) == 3
    assert "UNKNOWN_ERROR" in statuses
    # the failed task's measurement still carries valid timing fields.
    failed = [m for m in measurements if m.request_status != STATUS_SUCCESS][0]
    assert failed.queue_wait_ms is not None
    assert failed.task_end_to_end_latency_ms is not None
    assert failed.prompt_tokens is None  # never fabricated on failure


@pytest.mark.asyncio
async def test_multiple_failures_all_recorded_independently():
    client, _ = _fake_client(delay_seconds=0.01, fail_indices=frozenset({1, 3}))
    measurements, _, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=4, count=4))
    statuses = [m.request_status for m in measurements]
    assert statuses.count(STATUS_SUCCESS) == 2
    assert statuses.count("UNKNOWN_ERROR") == 2


@pytest.mark.asyncio
async def test_rate_limited_classification(monkeypatch):
    import httpx
    from groq import RateLimitError

    class _RateLimitCompletions:
        def __init__(self):
            self.calls: list[dict] = []

        async def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            response = httpx.Response(status_code=429, headers={}, request=request)
            raise RateLimitError("rate limited", response=response, body=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_RateLimitCompletions()))
    measurements, _, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=2, count=2))
    assert all(m.request_status == "RATE_LIMITED" for m in measurements)


@pytest.mark.asyncio
async def test_connection_error_classification():
    class _ConnErrorCompletions:
        async def create(self, **kwargs):
            import httpx
            from groq import APIConnectionError

            raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))

    client = SimpleNamespace(chat=SimpleNamespace(completions=_ConnErrorCompletions()))
    measurements, _, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=1, count=1))
    assert measurements[0].request_status == "CONNECTION_ERROR"


@pytest.mark.asyncio
async def test_empty_plan_returns_empty_results():
    client, _ = _fake_client()
    measurements, events, wall_clock_ms, peak = await run_concurrency_level(client, "m", None, "low", 100, "run-1", [])
    assert measurements == []
    assert events == []
    assert peak == 0


@pytest.mark.asyncio
async def test_token_and_cost_recorded_on_success(config_with_pricing):
    client, _ = _fake_client(delay_seconds=0.01)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    measurements, _, _, _ = await run_concurrency_level(client, "openai/gpt-oss-20b", pricing, "low", 100, "run-1", _plan(max_concurrent=1, count=1))
    m = measurements[0]
    assert m.prompt_tokens == 10
    assert m.completion_tokens == 5
    assert m.estimated_cost_usd is not None


@pytest.mark.asyncio
async def test_correctness_evaluated_against_task_expected_answer():
    client, _ = _fake_client(delay_seconds=0.01, response_factory=lambda: _response(content="2"))
    measurements, _, _, _ = await run_concurrency_level(client, "m", None, "low", 100, "run-1", _plan(max_concurrent=1, count=1))
    # task_001's expected answer is "2" (1 + 1).
    assert measurements[0].answer_correct is True
