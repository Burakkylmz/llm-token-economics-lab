"""Cache-aware cost economics for Lab03.

`calculate_cost` is reused VERBATIM from labs.lab01_token_anatomy.cost --
it already implements the exact cache-aware billing formula this task
asks for (uncached_tokens * input_price + cached_tokens * cached_price +
completion_tokens * output_price), and every other lab's cost reporting
depends on it staying unchanged. This module only adds the SAVINGS
analysis on top: what the SAME measured usage would have cost with no
caching at all.

That counterfactual is explicitly a re-pricing of one already-measured
call's real token usage -- never a second, actually-executed API call,
and never presented as one. It answers "how much did caching save on
this call's bill", not "how much would an identical uncached call cost"
(those are the same number only because cached/uncached token *usage*
is assumed unchanged by the presence or absence of caching, which is
true for billing purposes but says nothing about completion content).

Missing/inconsistent cache usage never produces a fabricated savings
number:
  - cached_tokens is None (not reported) -> every savings field stays None.
  - cached_tokens == 0 (reported, no hit) -> savings are a real, measured
    $0.00 / 0% -- NOT None, since this is a genuine observation, not a
    missing one.
  - cached_tokens > prompt_tokens (inconsistent provider data) -> this is
    never silently "corrected"; every derived savings field stays None
    and the caller should surface `cache_usage_inconsistent=True` (see
    labs.lab03_prompt_caching.models.CacheMeasurement) rather than
    pretending the inconsistency didn't happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics
from labs.lab01_token_anatomy.cost import calculate_cost  # noqa: F401 -- re-exported for callers


def compute_uncached_prompt_tokens(prompt_tokens: Optional[int], cached_tokens: Optional[int]) -> Optional[int]:
    """prompt_tokens - cached_tokens, only when both are known AND consistent
    (cached_tokens <= prompt_tokens). Cached tokens are already counted
    WITHIN prompt_tokens by Groq's own accounting -- this is a derived
    split, never a correction of the API's own total."""
    if prompt_tokens is None or cached_tokens is None:
        return None
    uncached = prompt_tokens - cached_tokens
    if cached_tokens < 0 or uncached < 0:
        return None
    return uncached


@dataclass
class CacheCostBreakdown:
    estimated_cost_usd: Optional[float]
    estimated_input_cost_without_cache_usd: Optional[float]
    estimated_input_cost_with_cache_usd: Optional[float]
    estimated_cache_savings_usd: Optional[float]
    estimated_cache_savings_percent: Optional[float]


_EMPTY_BREAKDOWN = CacheCostBreakdown(None, None, None, None, None)


def compute_cache_cost_breakdown(tokens: TokenCostMetrics, pricing: Optional[ModelPricing]) -> CacheCostBreakdown:
    if pricing is None or tokens.prompt_tokens is None or tokens.completion_tokens is None:
        return _EMPTY_BREAKDOWN

    total_cost = calculate_cost(tokens, pricing)
    cached = tokens.cached_tokens

    if cached is None:
        # Not reported at all: total cost is still computable (calculate_cost
        # treats a missing cached_tokens as 0 cached, i.e. the whole prompt
        # billed at the uncached rate), but a savings figure would require
        # actually knowing whether/how much was cached -- stays None.
        return CacheCostBreakdown(total_cost, None, None, None, None)

    uncached = compute_uncached_prompt_tokens(tokens.prompt_tokens, cached)
    if uncached is None:
        # cached_tokens outside [0, prompt_tokens]: inconsistent provider data.
        # Shared cost calculation also returns None for invalid usage.
        return CacheCostBreakdown(total_cost, None, None, None, None)

    cost_without_cache = tokens.prompt_tokens / 1_000_000 * pricing.input_per_1m_usd
    cost_with_cache = (
        uncached / 1_000_000 * pricing.input_per_1m_usd + cached / 1_000_000 * pricing.cached_input_per_1m_usd
    )
    savings_usd = cached / 1_000_000 * (pricing.input_per_1m_usd - pricing.cached_input_per_1m_usd)
    savings_percent = (savings_usd / cost_without_cache * 100) if cost_without_cache > 0 else None

    return CacheCostBreakdown(
        estimated_cost_usd=total_cost,
        estimated_input_cost_without_cache_usd=cost_without_cache,
        estimated_input_cost_with_cache_usd=cost_with_cache,
        estimated_cache_savings_usd=savings_usd,
        estimated_cache_savings_percent=savings_percent,
    )
