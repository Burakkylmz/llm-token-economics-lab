"""Canonical 4-way cache status classification for Lab03.

Root-cause context (see the task's diagnostic requirement): the installed
Groq SDK (`groq==1.7.0`, `groq/types/completion_usage.py`) declares
`CompletionUsage.prompt_tokens_details: Optional[PromptTokensDetails] = None`
as a real, typed field -- not an unknown/"extra" one -- and the SDK's
`BaseModel` is configured with `model_config = ConfigDict(extra="allow")`,
so even an undeclared field would be preserved, never silently dropped.
`lab_core.groq_client.normalize_usage` reads exactly
`usage.prompt_tokens_details.cached_tokens`, matching that schema
precisely. So when `cached_tokens` comes back `None`, the cause is never
"our code can't find a field the SDK has" -- it is that the provider's
response did not include a `prompt_tokens_details` object at all for that
call. That is a provider-side signal (CACHE_USAGE_UNAVAILABLE), never a
parsing failure and never a cache miss.

This module gives that signal one canonical, auditable name, used
consistently end to end (measurements, diagnostics, terminal output,
charts, statistics) so "unavailable" is never silently treated as "0".
"""

from __future__ import annotations

from typing import Optional

CACHE_HIT = "CACHE_HIT"
CACHE_MISS = "CACHE_MISS"
CACHE_USAGE_UNAVAILABLE = "CACHE_USAGE_UNAVAILABLE"
INVALID_USAGE = "INVALID_USAGE"

CACHE_STATUS_VALUES = (CACHE_HIT, CACHE_MISS, CACHE_USAGE_UNAVAILABLE, INVALID_USAGE)


def classify_cache_status(cached_tokens: Optional[int], prompt_tokens: Optional[int]) -> str:
    """cached_tokens=None (field absent) -> CACHE_USAGE_UNAVAILABLE, never
    CACHE_MISS. cached_tokens > prompt_tokens (when prompt_tokens is also
    known) -> INVALID_USAGE, checked BEFORE the hit/miss split so
    inconsistent provider data is never misreported as either a hit or a
    miss. cached_tokens == 0 (reported, no hit) -> CACHE_MISS, a real
    observation, not missing data. cached_tokens > 0 -> CACHE_HIT."""
    if cached_tokens is None:
        return CACHE_USAGE_UNAVAILABLE
    if prompt_tokens is not None and cached_tokens > prompt_tokens:
        return INVALID_USAGE
    if cached_tokens > 0:
        return CACHE_HIT
    return CACHE_MISS
