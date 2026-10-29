from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from labs.lab09_batch_processing.datasets import GoldenTask
from labs.lab09_batch_processing.models import (
    CONDITION_CONCURRENT,
    CONDITION_SEQUENTIAL,
    STATUS_SUCCESS,
    STATUS_UNKNOWN_ERROR,
    PlannedRealtimeCall,
)
from labs.lab09_batch_processing.runner import run_realtime_batch


def _usage():
    return SimpleNamespace(
        prompt_tokens=10, completion_tokens=2, total_tokens=12, total_time=0.01,
        prompt_tokens_details=None, completion_tokens_details=None,
    )


def _response(content="POSITIVE"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=_usage(), system_fingerprint=None)


class _DelayedCompletions:
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


def _tasks_by_id(count=4):
    tasks = {}
    for i in range(1, count + 1):
        task_id = f"task_{i:03d}"
        tasks[task_id] = GoldenTask(
            task_id=task_id, category="classification_sentiment", prompt=f"classify {task_id}",
            expected_answer="POSITIVE", evaluation_type="exact_match", metadata={},
        )
    return tasks


def _plan(condition, max_concurrent, task_ids):
    return [
        PlannedRealtimeCall(execution_index=i + 1, condition=condition, max_concurrent=max_concurrent, task_id=tid, task_index=i)
        for i, tid in enumerate(task_ids)
    ]


@pytest.mark.asyncio
async def test_sequential_never_exceeds_one_in_flight():
    client, _ = _fake_client(delay_seconds=0.02)
    by_id = _tasks_by_id(4)
    plan = _plan(CONDITION_SEQUENTIAL, 1, list(by_id.keys()))
    measurements, events, wall_clock_ms, peak = await run_realtime_batch(
        client, "m", None, "low", 100, "run-1", plan, by_id
    )
    assert peak == 1
    assert all(m.condition == CONDITION_SEQUENTIAL for m in measurements)
    assert all(m.request_status == STATUS_SUCCESS for m in measurements)
    assert all(m.answer_correct is True for m in measurements)


@pytest.mark.asyncio
async def test_concurrent_reaches_configured_peak():
    client, _ = _fake_client(delay_seconds=0.05)
    by_id = _tasks_by_id(5)
    plan = _plan(CONDITION_CONCURRENT, 5, list(by_id.keys()))
    measurements, events, wall_clock_ms, peak = await run_realtime_batch(
        client, "m", None, "low", 100, "run-1", plan, by_id
    )
    assert peak == 5
    assert all(m.condition == CONDITION_CONCURRENT for m in measurements)


@pytest.mark.asyncio
async def test_concurrency_never_exceeds_configured_limit():
    client, _ = _fake_client(delay_seconds=0.03)
    by_id = _tasks_by_id(8)
    plan = _plan(CONDITION_CONCURRENT, 3, list(by_id.keys()))
    measurements, events, wall_clock_ms, peak = await run_realtime_batch(
        client, "m", None, "low", 100, "run-1", plan, by_id
    )
    assert peak <= 3


@pytest.mark.asyncio
async def test_request_failure_isolated_other_measurements_preserved():
    client, _ = _fake_client(delay_seconds=0.01, fail_indices=frozenset({2}))
    by_id = _tasks_by_id(3)
    plan = _plan(CONDITION_SEQUENTIAL, 1, list(by_id.keys()))
    measurements, _, _, _ = await run_realtime_batch(client, "m", None, "low", 100, "run-1", plan, by_id)
    statuses = [m.request_status for m in measurements]
    assert statuses.count(STATUS_SUCCESS) == 2
    assert STATUS_UNKNOWN_ERROR in statuses


@pytest.mark.asyncio
async def test_numeric_match_evaluation_applied_correctly():
    client, _ = _fake_client(delay_seconds=0.0, response_factory=lambda: _response("60"))
    by_id = {
        "task_001": GoldenTask(task_id="task_001", category="arithmetic_business", prompt="p", expected_answer="60", evaluation_type="numeric_match", metadata={}),
    }
    plan = _plan(CONDITION_SEQUENTIAL, 1, ["task_001"])
    measurements, _, _, _ = await run_realtime_batch(client, "m", None, "low", 100, "run-1", plan, by_id)
    assert measurements[0].answer_correct is True


@pytest.mark.asyncio
async def test_empty_plan_returns_empty_results():
    client, _ = _fake_client()
    measurements, events, wall_clock_ms, peak = await run_realtime_batch(client, "m", None, "low", 100, "run-1", [], {})
    assert measurements == []
    assert events == []
    assert wall_clock_ms == 0.0
    assert peak == 0


@pytest.mark.asyncio
async def test_never_leaks_expected_answer_into_the_request():
    client, completions = _fake_client(delay_seconds=0.0)
    by_id = {
        "task_001": GoldenTask(task_id="task_001", category="classification_sentiment", prompt="classify this", expected_answer="POSITIVE", evaluation_type="exact_match", metadata={}),
    }
    plan = _plan(CONDITION_SEQUENTIAL, 1, ["task_001"])
    await run_realtime_batch(client, "m", None, "low", 100, "run-1", plan, by_id)
    sent_content = completions.calls[0]["messages"][0]["content"]
    assert "POSITIVE" not in sent_content
