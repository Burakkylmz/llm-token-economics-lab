"""Input/output cost breakdown for Lab05, in addition to the single total
figure every other lab reports.

`calculate_cost` (labs.lab01_token_anatomy.cost) remains the single source
of truth for the TOTAL figure -- this module only additionally exposes the
input vs. output split for Lab05's per-task reporting (Part I requires
`estimated_input_cost_usd`/`estimated_output_cost_usd` alongside the
total). The two numbers computed here always sum to exactly what
`calculate_cost` would return for the same usage, by construction (same
formula, just not added together).
"""

from __future__ import annotations

from typing import Optional

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics


def cost_breakdown(tokens: TokenCostMetrics, pricing: ModelPricing) -> tuple[Optional[float], Optional[float]]:
    """Returns (estimated_input_cost_usd, estimated_output_cost_usd), or
    (None, None) if usage is unavailable."""
    if tokens.prompt_tokens is None or tokens.completion_tokens is None:
        return None, None

    cached = tokens.cached_tokens or 0
    uncached_prompt_tokens = tokens.prompt_tokens - cached

    input_cost = (
        uncached_prompt_tokens / 1_000_000 * pricing.input_per_1m_usd
        + cached / 1_000_000 * pricing.cached_input_per_1m_usd
    )
    output_cost = tokens.completion_tokens / 1_000_000 * pricing.output_per_1m_usd
    return input_cost, output_cost
