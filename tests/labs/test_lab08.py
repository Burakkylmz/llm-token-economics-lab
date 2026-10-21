from __future__ import annotations

import pytest

from labs.lab08_concurrency.concurrency import percentile, summarize_batch
from labs.lab08_concurrency.experiments import PROMPTS, run_concurrent, run_sequential


def test_percentile_p50_of_sorted_list():
    assert percentile([10, 20, 30, 40, 50], 50) == 30


def test_percentile_p95_interpolates():
    values = list(range(1, 11))  # 1..10
    assert percentile(values, 95) == pytest.approx(9.55)


def test_percentile_rejects_empty_list():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_batch_computes_requests_per_second():
    summary = summarize_batch([100.0, 100.0, 100.0, 100.0], total_elapsed_ms=2000.0)
    assert summary["requests_per_second"] == 2.0
    assert summary["p50_latency_ms"] == 100.0


def test_summarize_batch_handles_empty_input():
    summary = summarize_batch([], total_elapsed_ms=0)
    assert summary["requests_per_second"] is None


@pytest.mark.asyncio
async def test_run_sequential_makes_one_call_per_prompt(fake_async_client_factory, make_response, config_with_pricing):
    client = fake_async_client_factory([make_response() for _ in PROMPTS])
    results, summary = await run_sequential(client, config_with_pricing, "llama-3.1-8b-instant")

    assert len(results) == len(PROMPTS)
    assert len(client.completions.calls) == len(PROMPTS)
    assert summary["requests_per_second"] is not None


@pytest.mark.asyncio
async def test_run_concurrent_makes_one_call_per_prompt_and_respects_limit(fake_async_client_factory, make_response, config_with_pricing):
    client = fake_async_client_factory([make_response() for _ in PROMPTS])
    results, summary = await run_concurrent(client, config_with_pricing, "llama-3.1-8b-instant", max_concurrent=3)

    assert len(results) == len(PROMPTS)
    assert summary["max_concurrent_requests"] == 3
    # results preserve prompt order even though calls may complete out of order
    assert [r.metadata.model_parameters["prompt_index"] for r in results] == list(range(len(PROMPTS)))
