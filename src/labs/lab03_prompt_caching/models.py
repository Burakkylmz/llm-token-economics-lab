"""Data shapes for Lab03's prompt-caching experiment.

CacheMeasurement deliberately never stores the model's raw response text
(same data-privacy policy as labs.lab02_context_growth.models) -- only a
derived correctness status/char count. `cached_tokens=None` (unavailable/
not reported) is kept distinct from `cached_tokens=0` (reported, no cache
hit) all the way through this dataclass and everything derived from it --
see labs.lab03_prompt_caching.cache_cost and .cache_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from labs.lab03_prompt_caching.cache_status import CACHE_USAGE_UNAVAILABLE


@dataclass(frozen=True)
class PlannedCacheCall:
    """One planned API call: which condition, which pair (repetition), and
    which of the paired questions this call/pair uses.

    `pair_index` is the shared repetition number BOTH conditions in a pair
    use -- e.g. pair_index=2's repeated_prefix call and pair_index=2's
    cache_unfriendly call are paired: same question, same expected answer,
    comparable prompt length, differing only in prefix reuse.
    """

    execution_index: int
    condition_id: str
    pair_index: int
    question_index: int


@dataclass
class CacheMeasurement:
    """One completed (successful or failed) attempt at a PlannedCacheCall."""

    experiment_run_id: str
    condition_id: str
    execution_index: int
    pair_index: int
    question_index: int
    timestamp: str
    model: str
    max_completion_tokens: int
    succeeded: bool
    attempts_used: int
    error: Optional[str] = None

    prompt_chars: Optional[int] = None
    expected_answer: Optional[str] = None

    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    uncached_prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    non_reasoning_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # True if the provider reported cached_tokens > prompt_tokens (should
    # never happen, but is never silently "corrected" -- see
    # labs.lab03_prompt_caching.cache_cost). When True, the raw fields
    # above are kept exactly as reported, but uncached_prompt_tokens and
    # every cache-savings derived field below are left None rather than
    # computed from inconsistent data.
    cache_usage_inconsistent: bool = False

    finish_reason: Optional[str] = None
    system_fingerprint: Optional[str] = None
    final_answer_char_count: Optional[int] = None

    total_latency_ms: Optional[float] = None
    provider_reported_processing_ms: Optional[float] = None
    # Additional provider-reported timing breakdown (Part E), extracted
    # directly from the raw usage object -- only ever populated when the
    # provider actually reported that specific field (seconds -> ms; None
    # otherwise, never estimated by subtracting other measurements).
    provider_prompt_time_ms: Optional[float] = None
    provider_completion_time_ms: Optional[float] = None
    provider_queue_time_ms: Optional[float] = None

    # Billed cost (cache-aware: reuses labs.lab01_token_anatomy.cost.calculate_cost,
    # so cached tokens are already discounted here exactly like every other lab).
    estimated_cost_usd: Optional[float] = None
    # Counterfactual pricing only -- "what this SAME measured usage would
    # have cost with no caching at all" -- never a second, actually-executed
    # API call. See labs.lab03_prompt_caching.cache_cost module docstring.
    estimated_input_cost_without_cache_usd: Optional[float] = None
    estimated_input_cost_with_cache_usd: Optional[float] = None
    estimated_cache_savings_usd: Optional[float] = None
    estimated_cache_savings_percent: Optional[float] = None

    # Per-call cache observability (Part B). None means "cached_tokens was
    # not reported at all", never conflated with a real, reported miss.
    request_cache_hit: Optional[bool] = None
    request_cache_ratio: Optional[float] = None

    # Canonical 4-way status -- see labs.lab03_prompt_caching.cache_status.
    # CACHE_USAGE_UNAVAILABLE (the default) is NEVER the same thing as
    # CACHE_MISS: the former means the provider never reported
    # prompt_tokens_details at all for this call; the latter means it did,
    # with cached_tokens == 0.
    cache_status: str = CACHE_USAGE_UNAVAILABLE

    # Diagnostic fields (Part A/C of this lab's redesign task) -- bounded,
    # structural facts about the raw provider usage object, never prompt
    # or completion content. See labs.lab03_prompt_caching.diagnostics.
    prompt_tokens_details_present: bool = False
    cached_tokens_field_present: bool = False
    raw_usage_field_names: list = field(default_factory=list)
    sdk_version: Optional[str] = None

    # SHA256 of the exact {model, messages, max_completion_tokens} payload
    # sent to the API -- lets an offline/after-the-fact check confirm two
    # calls were byte-identical requests without ever storing the prompt
    # text itself. See labs.lab03_prompt_caching.diagnostics.compute_request_hash.
    request_hash: Optional[str] = None

    # "correct" | "incorrect" | "format_invalid" | "missing_response" |
    # "unavailable" (legacy data with no stored answer text to evaluate).
    # See labs.lab02_context_growth.evaluation (reused directly here).
    answer_evaluation_status: str = "unavailable"
    answer_correct: Optional[bool] = None
