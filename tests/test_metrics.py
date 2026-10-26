from __future__ import annotations

from lab_core.groq_client import (
    call_sync,
    extract_finish_reason,
    extract_message_content,
    extract_requested_generation_parameters,
    extract_system_fingerprint,
    normalize_usage,
)


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


def test_extract_finish_reason_reads_first_choice(make_response):
    assert extract_finish_reason(make_response(finish_reason="stop")) == "stop"
    assert extract_finish_reason(make_response(finish_reason="length")) == "length"


def test_extract_finish_reason_none_when_no_choices():
    from types import SimpleNamespace

    assert extract_finish_reason(SimpleNamespace(choices=[])) is None
    assert extract_finish_reason(SimpleNamespace()) is None


def test_extract_message_content_reads_first_choice_text(make_response):
    assert extract_message_content(make_response(content="Paris")) == "Paris"


def test_extract_system_fingerprint_reads_response_field(make_response):
    assert extract_system_fingerprint(make_response(system_fingerprint="fp_abc123")) == "fp_abc123"
    assert extract_system_fingerprint(make_response()) is None


def test_extract_requested_generation_parameters_only_includes_what_was_passed():
    sent = extract_requested_generation_parameters(
        {"model": "openai/gpt-oss-20b", "messages": [], "temperature": 0.2, "max_tokens": 64}
    )
    assert sent == {"temperature": 0.2, "max_tokens": 64}


def test_extract_requested_generation_parameters_empty_when_nothing_explicit():
    sent = extract_requested_generation_parameters({"model": "openai/gpt-oss-20b", "messages": []})
    assert sent == {}


def test_call_sync_populates_finish_reason_and_metadata(fake_sync_client_factory, make_response):
    response = make_response(content="Paris", finish_reason="length", system_fingerprint="fp_xyz")
    client = fake_sync_client_factory([response])

    result = call_sync(client, model="openai/gpt-oss-20b", messages=[{"role": "user", "content": "hi"}])

    assert result.tokens_and_cost.finish_reason == "length"
    assert result.message_content == "Paris"
    assert result.system_fingerprint == "fp_xyz"
    assert result.requested_generation_parameters == {}


def test_call_sync_echoes_back_only_explicitly_sent_generation_parameters(fake_sync_client_factory, make_response):
    client = fake_sync_client_factory([make_response()])

    result = call_sync(
        client,
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        reasoning_effort="low",
    )

    assert result.requested_generation_parameters == {"temperature": 0.0, "reasoning_effort": "low"}
