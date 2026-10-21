from __future__ import annotations

from lab_core.groq_client import normalize_usage


def test_normalize_usage_none_returns_all_none_except_call_count():
    metrics = normalize_usage(None)
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None
    assert metrics.cached_tokens is None
    assert metrics.reasoning_tokens is None
    assert metrics.api_call_count == 1


def test_normalize_usage_basic_fields(make_usage):
    usage = make_usage(prompt_tokens=12, completion_tokens=34, total_tokens=46)
    metrics = normalize_usage(usage)
    assert metrics.prompt_tokens == 12
    assert metrics.completion_tokens == 34
    assert metrics.total_tokens == 46


def test_normalize_usage_missing_cached_and_reasoning_stay_none(make_usage):
    usage = make_usage(cached_tokens=None, reasoning_tokens=None)
    metrics = normalize_usage(usage)
    assert metrics.cached_tokens is None
    assert metrics.reasoning_tokens is None


def test_normalize_usage_present_cached_and_reasoning_are_captured(make_usage):
    usage = make_usage(cached_tokens=4608, reasoning_tokens=120)
    metrics = normalize_usage(usage)
    assert metrics.cached_tokens == 4608
    assert metrics.reasoning_tokens == 120


def test_normalize_usage_never_defaults_missing_field_to_zero(make_usage):
    usage = make_usage(cached_tokens=None)
    metrics = normalize_usage(usage)
    assert metrics.cached_tokens is None
