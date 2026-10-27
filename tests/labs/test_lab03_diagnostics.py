from __future__ import annotations

from types import SimpleNamespace

from labs.lab03_prompt_caching.diagnostics import (
    compute_request_hash,
    groq_sdk_version,
    usage_diagnostic_fields,
)


def test_identical_payloads_hash_identically():
    messages = [{"role": "system", "content": "stable"}, {"role": "user", "content": "question"}]
    h1 = compute_request_hash("model-a", messages, 2048)
    h2 = compute_request_hash("model-a", [dict(m) for m in messages], 2048)  # fresh dict copies, same content
    assert h1 == h2


def test_different_user_message_changes_the_hash():
    base = [{"role": "system", "content": "stable"}, {"role": "user", "content": "question A"}]
    changed = [{"role": "system", "content": "stable"}, {"role": "user", "content": "question B"}]
    assert compute_request_hash("model-a", base, 2048) != compute_request_hash("model-a", changed, 2048)


def test_different_model_changes_the_hash():
    messages = [{"role": "system", "content": "stable"}]
    assert compute_request_hash("model-a", messages, 2048) != compute_request_hash("model-b", messages, 2048)


def test_different_max_completion_tokens_changes_the_hash():
    messages = [{"role": "system", "content": "stable"}]
    assert compute_request_hash("model-a", messages, 2048) != compute_request_hash("model-a", messages, 4096)


def test_hash_is_a_hex_sha256_digest():
    h = compute_request_hash("m", [{"role": "user", "content": "x"}], 10)
    assert len(h) == 64
    int(h, 16)  # must not raise -- confirms it's valid hex


def test_usage_diagnostic_fields_none_usage():
    diag = usage_diagnostic_fields(None)
    assert diag["prompt_tokens_details_present"] is False
    assert diag["cached_tokens_field_present"] is False
    assert diag["raw_usage_field_names"] == []


def test_usage_diagnostic_fields_prompt_tokens_details_absent():
    # Mirrors the SDK's own default: prompt_tokens_details defaults to None
    # when the provider's response did not include it at all.
    usage = SimpleNamespace(prompt_tokens=1454, completion_tokens=50, prompt_tokens_details=None)
    diag = usage_diagnostic_fields(usage)
    assert diag["prompt_tokens_details_present"] is False
    assert diag["cached_tokens_field_present"] is False


def test_usage_diagnostic_fields_prompt_tokens_details_present_with_zero():
    usage = SimpleNamespace(
        prompt_tokens=1454, completion_tokens=50, prompt_tokens_details=SimpleNamespace(cached_tokens=0)
    )
    diag = usage_diagnostic_fields(usage)
    assert diag["prompt_tokens_details_present"] is True
    assert diag["cached_tokens_field_present"] is True


def test_usage_diagnostic_fields_prompt_tokens_details_present_with_positive():
    usage = SimpleNamespace(
        prompt_tokens=1454, completion_tokens=50, prompt_tokens_details=SimpleNamespace(cached_tokens=1200)
    )
    diag = usage_diagnostic_fields(usage)
    assert diag["prompt_tokens_details_present"] is True
    assert diag["cached_tokens_field_present"] is True


def test_usage_diagnostic_fields_only_contains_bounded_structural_keys():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)
    diag = usage_diagnostic_fields(usage)
    assert set(diag.keys()) == {"prompt_tokens_details_present", "cached_tokens_field_present", "raw_usage_field_names"}


def test_usage_diagnostic_fields_picks_up_model_fields_set_and_model_extra():
    class _FakePydanticUsage:
        model_fields_set = {"prompt_tokens", "completion_tokens"}
        model_extra = {"some_new_field": 123}
        prompt_tokens_details = None

    diag = usage_diagnostic_fields(_FakePydanticUsage())
    assert set(diag["raw_usage_field_names"]) == {"prompt_tokens", "completion_tokens", "some_new_field"}


def test_groq_sdk_version_returns_a_string():
    version = groq_sdk_version()
    assert isinstance(version, str)
    assert version  # non-empty
