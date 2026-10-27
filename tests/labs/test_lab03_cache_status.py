from __future__ import annotations

from labs.lab03_prompt_caching.cache_status import (
    CACHE_HIT,
    CACHE_MISS,
    CACHE_USAGE_UNAVAILABLE,
    INVALID_USAGE,
    classify_cache_status,
)


def test_none_is_unavailable_never_miss():
    assert classify_cache_status(None, 1000) == CACHE_USAGE_UNAVAILABLE


def test_none_is_unavailable_even_without_prompt_tokens():
    assert classify_cache_status(None, None) == CACHE_USAGE_UNAVAILABLE


def test_zero_is_a_real_miss():
    assert classify_cache_status(0, 1000) == CACHE_MISS


def test_positive_is_a_hit():
    assert classify_cache_status(500, 1000) == CACHE_HIT


def test_cached_exceeding_prompt_is_invalid():
    assert classify_cache_status(1500, 1000) == INVALID_USAGE


def test_invalid_check_runs_before_hit_so_never_misreported_as_a_hit():
    # 1500 > 1000 AND 1500 > 0 -- must resolve to INVALID_USAGE, not CACHE_HIT.
    status = classify_cache_status(1500, 1000)
    assert status == INVALID_USAGE
    assert status != CACHE_HIT


def test_missing_prompt_tokens_skips_the_invalid_check():
    # Cannot evaluate cached > prompt without a prompt_tokens figure --
    # falls through to the ordinary hit/miss split instead.
    assert classify_cache_status(500, None) == CACHE_HIT
    assert classify_cache_status(0, None) == CACHE_MISS
