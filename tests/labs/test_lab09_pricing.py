from __future__ import annotations

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics
from labs.lab09_batch_processing.pricing import (
    BATCH_DISCOUNT_STACKS_WITH_CACHING,
    calculate_batch_cost,
    calculate_standard_cost,
    compare_standard_vs_batch,
)

PRICING = ModelPricing(
    input_per_1m_usd=0.075, output_per_1m_usd=0.30, cached_input_per_1m_usd=0.0375, price_verified_date="2026-10-21"
)


def test_batch_discount_does_not_stack_with_caching():
    assert BATCH_DISCOUNT_STACKS_WITH_CACHING is False


def test_batch_cost_is_half_of_standard_cost_with_no_cached_tokens():
    standard = calculate_standard_cost(TokenCostMetrics(prompt_tokens=1_000_000, completion_tokens=1_000_000), PRICING)
    batch = calculate_batch_cost(1_000_000, 1_000_000, PRICING)
    assert batch == standard / 2


def test_batch_cost_ignores_cached_tokens_never_stacking_discounts():
    """Even when the SAME request would have hit the prompt cache in
    real-time mode, the Batch cost must be computed from the FULL
    prompt_tokens count at the flat batch rate -- never the discounted
    cached rate stacked on top."""
    # A cached rate well below half of the input rate, so the "never
    # stacks" assertion below is not a coincidence of these two numbers
    # happening to match.
    cheap_cache_pricing = ModelPricing(
        input_per_1m_usd=0.075, output_per_1m_usd=0.30, cached_input_per_1m_usd=0.01, price_verified_date="2026-10-21"
    )
    tokens = TokenCostMetrics(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000)
    standard_with_cache = calculate_standard_cost(tokens, cheap_cache_pricing)
    batch = calculate_batch_cost(tokens.prompt_tokens, tokens.completion_tokens, cheap_cache_pricing)

    # Standard cost with a full cache hit is cheap (cached rate)...
    assert standard_with_cache == cheap_cache_pricing.cached_input_per_1m_usd
    # ...but batch cost uses the FULL input rate at 50%, which is MORE
    # expensive than the fully-cached standard price -- proving the two
    # discounts are not stacked.
    assert batch == cheap_cache_pricing.input_per_1m_usd * 0.5
    assert batch > standard_with_cache


def test_calculate_batch_cost_returns_none_for_missing_usage():
    assert calculate_batch_cost(None, 100, PRICING) is None
    assert calculate_batch_cost(100, None, PRICING) is None


def test_calculate_standard_cost_returns_none_for_missing_usage():
    assert calculate_standard_cost(TokenCostMetrics(prompt_tokens=None, completion_tokens=100), PRICING) is None


def test_compare_standard_vs_batch_computes_savings_percent():
    comparison = compare_standard_vs_batch(0.10, 0.05)
    assert comparison.estimated_batch_discount_usd == 0.05
    assert abs(comparison.estimated_savings_percent - 50.0) < 1e-9


def test_compare_standard_vs_batch_guards_none_and_zero_denominator():
    assert compare_standard_vs_batch(None, 0.05).estimated_savings_percent is None
    comparison = compare_standard_vs_batch(0.0, 0.0)
    assert comparison.estimated_batch_discount_usd == 0.0
    assert comparison.estimated_savings_percent is None
