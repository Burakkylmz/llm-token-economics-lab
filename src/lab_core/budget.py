"""Shared cost budget and preflight cost estimation for all laboratories.

The prompt-token component is an estimate, not a guaranteed hard bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lab_core.config import ModelPricing

# A crude, well-known heuristic (~4 characters per English token), used ONLY
# to put a rough, pre-flight, approximate figure on prompt-token cost for
# budget estimation -- never written into any measurement record, and never
# presented as a real token count. The real prompt_tokens for every call
# comes from the API's own usage field, same as basic-mode Lab01.
_APPROX_CHARS_PER_TOKEN = 4.0
# Safety margin applied on top of the chars/4 heuristic above, since it can
# understate token count for dense or non-English text; this only affects
# the PRE-FLIGHT estimate used for the budget gate, not any real measurement.
_PROMPT_ESTIMATE_SAFETY_MARGIN = 1.5


def conservative_per_call_cost_ceiling(model_pricing: ModelPricing, max_completion_tokens: int, prompt_text: str) -> float:
    """An upper bound on ONE call's cost, with one caveat stated explicitly:
    the completion-token part is a TRUE ceiling (completion_tokens can never
    exceed max_completion_tokens), but the prompt-token part is only an
    approximation (chars/4 with a 1.5x margin) because real tokenization
    isn't known before the call is made. The combined number is therefore a
    conservative ESTIMATE, not a guaranteed exact upper bound -- unusual
    tokenization (e.g. very non-English text) could in principle exceed it.
    """
    approx_prompt_tokens = (len(prompt_text) / _APPROX_CHARS_PER_TOKEN) * _PROMPT_ESTIMATE_SAFETY_MARGIN
    prompt_cost = approx_prompt_tokens / 1_000_000 * model_pricing.input_per_1m_usd
    completion_cost = max_completion_tokens / 1_000_000 * model_pricing.output_per_1m_usd
    return prompt_cost + completion_cost


@dataclass
class Budget:
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]
    per_call_cost_ceiling: float

    calls_made: int = 0
    accumulated_estimated_cost_usd: float = 0.0

    def can_make_another_call(self) -> Optional[str]:
        """None if another call is allowed; otherwise a human-readable reason
        it must stop, checked using the PER-CALL CEILING so the next call is
        never allowed to exceed the cost cap even in its worst case."""
        if self.max_api_calls is not None and self.calls_made >= self.max_api_calls:
            return f"max_api_calls budget ({self.max_api_calls}) reached"
        if self.max_estimated_cost_usd is not None:
            if self.accumulated_estimated_cost_usd + self.per_call_cost_ceiling > self.max_estimated_cost_usd:
                return (
                    f"max_estimated_cost_usd budget (${self.max_estimated_cost_usd:.4f}) would be "
                    f"exceeded by the next call's worst-case cost ceiling (${self.per_call_cost_ceiling:.6f}) "
                    f"on top of ${self.accumulated_estimated_cost_usd:.6f} already spent"
                )
        return None

    def record_call(self, actual_estimated_cost_usd: Optional[float]) -> None:
        self.calls_made += 1
        if actual_estimated_cost_usd is not None:
            self.accumulated_estimated_cost_usd += actual_estimated_cost_usd
        # NOTE: a FAILED call is still counted toward calls_made (it used up
        # one of the request budget's slots), but contributes $0 to
        # accumulated_estimated_cost_usd since no usage was returned. If a
        # provider still bills for some failed requests, that real cost
        # would NOT show up here -- accumulated_estimated_cost_usd is a
        # lower bound in the presence of failures, not a guarantee of exact
        # spend. The max_api_calls cap and the per-call ceiling check above
        # are what actually bound worst-case exposure regardless.


