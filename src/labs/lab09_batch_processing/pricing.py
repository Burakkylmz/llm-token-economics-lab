"""Standard vs Batch API pricing for Lab09.

Verified against https://console.groq.com/docs/batch on 2026-10-29:
  - "Pricing is at a 50% cost discount compared to synchronous API pricing."
  - "The batch discount does not stack with prompt caching discounts." --
    every batch-billed token is charged at the flat 50% batch rate,
    whether or not it would have hit the prompt cache in real-time mode.

This module therefore NEVER applies cached_input_per_1m_usd when
computing a Batch cost -- only input_per_1m_usd/output_per_1m_usd, halved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics
from labs.lab01_token_anatomy.cost import calculate_cost

PRICING_SOURCE = "https://console.groq.com/docs/batch"
PRICING_VERIFIED_DATE = "2026-10-29"
PRICING_POLICY_VERSION = "groq-batch-v1-50pct-2026-10-29"
BATCH_DISCOUNT_FRACTION = 0.5
BATCH_DISCOUNT_STACKS_WITH_CACHING = False


@dataclass
class PricingComparison:
    estimated_standard_cost_usd: Optional[float]
    estimated_batch_cost_usd: Optional[float]
    estimated_batch_discount_usd: Optional[float]
    estimated_savings_percent: Optional[float]
    pricing_source: str = PRICING_SOURCE
    pricing_verified_date: str = PRICING_VERIFIED_DATE
    pricing_policy_version: str = PRICING_POLICY_VERSION


def calculate_standard_cost(tokens: TokenCostMetrics, pricing: ModelPricing) -> Optional[float]:
    """Standard (real-time) pricing, including the caching discount when
    cached_tokens is reported -- identical to labs.lab01_token_anatomy.cost.calculate_cost."""
    return calculate_cost(tokens, pricing)


def calculate_batch_cost(prompt_tokens: Optional[int], completion_tokens: Optional[int], pricing: ModelPricing) -> Optional[float]:
    """Batch pricing: the full prompt_tokens count (never reduced for a
    cache hit -- the discount does not stack) at half the standard
    input/output rate."""
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (
        prompt_tokens / 1_000_000 * pricing.input_per_1m_usd * BATCH_DISCOUNT_FRACTION
        + completion_tokens / 1_000_000 * pricing.output_per_1m_usd * BATCH_DISCOUNT_FRACTION
    )


def compare_standard_vs_batch(standard_cost_usd: Optional[float], batch_cost_usd: Optional[float]) -> PricingComparison:
    discount = None
    savings_percent = None
    if standard_cost_usd is not None and batch_cost_usd is not None:
        discount = standard_cost_usd - batch_cost_usd
        if standard_cost_usd > 0:
            savings_percent = (discount / standard_cost_usd) * 100

    return PricingComparison(
        estimated_standard_cost_usd=standard_cost_usd,
        estimated_batch_cost_usd=batch_cost_usd,
        estimated_batch_discount_usd=discount,
        estimated_savings_percent=savings_percent,
    )
