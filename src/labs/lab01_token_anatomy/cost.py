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

    # Invalid provider usage must not generate negative or otherwise
    # misleading price estimates. A missing cache field is different from
    # an explicitly reported zero; for the total-cost estimate we retain
    # the documented conservative uncached-input pricing assumption.
    if tokens.prompt_tokens < 0 or tokens.completion_tokens < 0:
        return None
    if tokens.cached_tokens is not None and not 0 <= tokens.cached_tokens <= tokens.prompt_tokens:
        return None

    cached = tokens.cached_tokens if tokens.cached_tokens is not None else 0
    uncached_prompt_tokens = tokens.prompt_tokens - cached

    return (
        uncached_prompt_tokens / 1_000_000 * pricing.input_per_1m_usd
        + cached / 1_000_000 * pricing.cached_input_per_1m_usd
        + tokens.completion_tokens / 1_000_000 * pricing.output_per_1m_usd
    )
