"""Cache observability for Lab03 -- Part B of this lab's design.

Two DIFFERENT, commonly-confused metrics, kept in separate functions with
separate names so they can never accidentally collapse into one number:

  - Metric A, `compute_observed_request_cache_hit_rate`: of the requests
    that actually REPORTED cached_tokens at all, what fraction had
    cached_tokens > 0. A per-REQUEST rate.
  - Metric B, `compute_token_cache_ratio`: of the total prompt tokens
    across requests that reported cached_tokens, what fraction were
    cached. A per-TOKEN ratio, weighted by each call's prompt size --
    NOT the average of each call's own request_cache_ratio.

A denominator in both metrics is "requests/tokens that reported
cached_tokens at all" -- a request with cached_tokens=None (unavailable)
is excluded from both denominators, never treated as a 0 (a miss).
"""

from __future__ import annotations

from typing import Optional, Sequence

from labs.lab03_prompt_caching.models import CacheMeasurement


def compute_request_cache_hit(cached_tokens: Optional[int]) -> Optional[bool]:
    """None if cached_tokens was not reported at all; otherwise cached_tokens > 0."""
    if cached_tokens is None:
        return None
    return cached_tokens > 0


def compute_request_cache_ratio(cached_tokens: Optional[int], prompt_tokens: Optional[int]) -> Optional[float]:
    """cached_tokens / prompt_tokens for ONE request, or None if either value
    is unreported, or prompt_tokens is not a positive number."""
    if cached_tokens is None or prompt_tokens is None or prompt_tokens <= 0:
        return None
    return cached_tokens / prompt_tokens


def compute_observed_request_cache_hit_rate(measurements: Sequence[CacheMeasurement]) -> Optional[float]:
    """Metric A: fraction of successful, cache-usage-REPORTING requests with
    cached_tokens > 0. None if no successful request reported cached_tokens
    at all (never 0.0 in that case -- unavailable is not the same as 0%)."""
    reported = [m for m in measurements if m.succeeded and m.cached_tokens is not None]
    if not reported:
        return None
    hits = sum(1 for m in reported if m.cached_tokens > 0)
    return hits / len(reported)


def compute_token_cache_ratio(measurements: Sequence[CacheMeasurement]) -> Optional[float]:
    """Metric B: total reported cached input tokens / total prompt tokens,
    across successful, cache-usage-reporting, prompt_tokens-known requests.
    None if there is no such request, or the total prompt-token denominator
    is zero."""
    reported = [m for m in measurements if m.succeeded and m.cached_tokens is not None and m.prompt_tokens is not None]
    total_prompt = sum(m.prompt_tokens for m in reported)
    if not reported or total_prompt <= 0:
        return None
    total_cached = sum(m.cached_tokens for m in reported)
    return total_cached / total_prompt
