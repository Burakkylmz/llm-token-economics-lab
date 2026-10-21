"""Cost calculation for a single measured API call.

completion_tokens already includes reasoning_tokens when a model reports
reasoning (the OpenAI-compatible usage convention Groq follows): billing
is based on total completion_tokens, so reasoning_tokens must never be
added a second time on top of it — that would double count.
"""

from __future__ import annotations

from typing import Optional

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics


def calculate_cost(tokens: TokenCostMetrics, pricing: ModelPricing) -> Optional[float]:
    """Estimate USD cost for one call's token usage, or None if usage is unavailable."""
    if tokens.prompt_tokens is None or tokens.completion_tokens is None:
        return None

    cached = tokens.cached_tokens or 0
    uncached_prompt_tokens = tokens.prompt_tokens - cached

    return (
        uncached_prompt_tokens / 1_000_000 * pricing.input_per_1m_usd
        + cached / 1_000_000 * pricing.cached_input_per_1m_usd
        + tokens.completion_tokens / 1_000_000 * pricing.output_per_1m_usd
    )
