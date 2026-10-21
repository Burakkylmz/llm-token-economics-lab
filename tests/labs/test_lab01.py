from __future__ import annotations

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.experiments import run_experiment


def _pricing(**overrides) -> ModelPricing:
    defaults = dict(
        input_per_1m_usd=1.0,
        output_per_1m_usd=2.0,
        cached_input_per_1m_usd=0.5,
        price_verified_date="2026-10-21",
    )
    defaults.update(overrides)
    return ModelPricing(**defaults)


def test_calculate_cost_basic():
    tokens = TokenCostMetrics(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost = calculate_cost(tokens, _pricing())
    assert cost == 1.0 + 2.0


def test_calculate_cost_applies_cached_discount():
    tokens = TokenCostMetrics(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000)
    cost = calculate_cost(tokens, _pricing())
    assert cost == 0.5  # fully cached: uncached portion is 0, cached portion billed at cached rate


def test_calculate_cost_does_not_double_count_reasoning_tokens():
    # reasoning_tokens is a subset of completion_tokens and must not add extra cost on its own.
    with_reasoning = TokenCostMetrics(prompt_tokens=100, completion_tokens=500, reasoning_tokens=300)
    without_reasoning = TokenCostMetrics(prompt_tokens=100, completion_tokens=500, reasoning_tokens=None)
    pricing = _pricing()
    assert calculate_cost(with_reasoning, pricing) == calculate_cost(without_reasoning, pricing)


def test_calculate_cost_returns_none_when_usage_unavailable():
    tokens = TokenCostMetrics(prompt_tokens=None, completion_tokens=None)
    assert calculate_cost(tokens, _pricing()) is None


def test_run_experiment_sends_correct_prompt_and_computes_cost(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="Paris", prompt_tokens=8, completion_tokens=4, total_tokens=12)
    client = fake_sync_client_factory([response])

    results = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="llama-3.1-8b-instant",
        repetitions=1,
    )

    assert len(results) == 1
    result = results[0]
    assert result.tokens_and_cost.prompt_tokens == 8
    assert result.tokens_and_cost.completion_tokens == 4
    assert result.tokens_and_cost.estimated_cost_usd is not None
    assert client.completions.calls[0]["model"] == "llama-3.1-8b-instant"
    assert "capital of France" in client.completions.calls[0]["messages"][0]["content"]


def test_run_experiment_repeats_the_requested_number_of_times(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response() for _ in range(3)])
    results = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="long_prompt",
        model="llama-3.1-8b-instant",
        repetitions=3,
    )
    assert len(results) == 3
