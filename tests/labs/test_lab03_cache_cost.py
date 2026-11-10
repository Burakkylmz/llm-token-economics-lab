from __future__ import annotations

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics
from labs.lab03_prompt_caching.cache_cost import compute_cache_cost_breakdown, compute_uncached_prompt_tokens

PRICING = ModelPricing(input_per_1m_usd=0.075, output_per_1m_usd=0.30, cached_input_per_1m_usd=0.0375, price_verified_date="2026-10-21")


def test_uncached_prompt_tokens_is_prompt_minus_cached():
    assert compute_uncached_prompt_tokens(1000, 400) == 600


def test_uncached_prompt_tokens_none_when_either_missing():
    assert compute_uncached_prompt_tokens(None, 400) is None
    assert compute_uncached_prompt_tokens(1000, None) is None


def test_uncached_prompt_tokens_none_when_cached_exceeds_prompt():
    # Inconsistent provider data -- never silently corrected into a negative number.
    assert compute_uncached_prompt_tokens(100, 400) is None


def test_breakdown_none_when_cached_tokens_unreported():
    tokens = TokenCostMetrics(prompt_tokens=1000, completion_tokens=50, cached_tokens=None)
    breakdown = compute_cache_cost_breakdown(tokens, PRICING)
    assert breakdown.estimated_cost_usd is not None  # total cost still computable
    assert breakdown.estimated_cache_savings_usd is None
    assert breakdown.estimated_input_cost_without_cache_usd is None


def test_breakdown_real_zero_savings_when_cached_tokens_reported_as_zero():
    tokens = TokenCostMetrics(prompt_tokens=1000, completion_tokens=50, cached_tokens=0)
    breakdown = compute_cache_cost_breakdown(tokens, PRICING)
    assert breakdown.estimated_cache_savings_usd == 0.0
    assert breakdown.estimated_cache_savings_percent == 0.0


def test_breakdown_computes_positive_savings_on_a_real_hit():
    tokens = TokenCostMetrics(prompt_tokens=1000, completion_tokens=50, cached_tokens=800)
    breakdown = compute_cache_cost_breakdown(tokens, PRICING)
    expected_savings = 800 / 1_000_000 * (0.075 - 0.0375)
    assert breakdown.estimated_cache_savings_usd == expected_savings
    assert breakdown.estimated_cache_savings_percent > 0


def test_breakdown_savings_percent_matches_counterfactual_ratio():
    tokens = TokenCostMetrics(prompt_tokens=6000, completion_tokens=100, cached_tokens=4800)
    breakdown = compute_cache_cost_breakdown(tokens, PRICING)
    assert round(breakdown.estimated_cache_savings_percent, 1) == 40.0  # 50% discount on 80% of tokens -> 40% of input cost


def test_breakdown_none_when_pricing_missing():
    tokens = TokenCostMetrics(prompt_tokens=1000, completion_tokens=50, cached_tokens=800)
    breakdown = compute_cache_cost_breakdown(tokens, None)
    assert breakdown.estimated_cost_usd is None
    assert breakdown.estimated_cache_savings_usd is None


def test_breakdown_none_when_inconsistent_cache_usage():
    tokens = TokenCostMetrics(prompt_tokens=100, completion_tokens=50, cached_tokens=400)
    breakdown = compute_cache_cost_breakdown(tokens, PRICING)
    assert breakdown.estimated_cache_savings_usd is None
    assert breakdown.estimated_input_cost_without_cache_usd is None
    # Invalid cache counts cannot produce a trustworthy total estimate.
    assert breakdown.estimated_cost_usd is None


def test_completion_cost_is_never_discounted_by_caching():
    no_cache = compute_cache_cost_breakdown(TokenCostMetrics(prompt_tokens=1000, completion_tokens=1000, cached_tokens=0), PRICING)
    full_cache = compute_cache_cost_breakdown(TokenCostMetrics(prompt_tokens=1000, completion_tokens=1000, cached_tokens=1000), PRICING)
    input_only_savings = 1000 / 1_000_000 * (PRICING.input_per_1m_usd - PRICING.cached_input_per_1m_usd)
    assert round(no_cache.estimated_cost_usd - full_cache.estimated_cost_usd, 10) == round(input_only_savings, 10)
