from __future__ import annotations

from types import SimpleNamespace

from labs.lab06_streaming.stream_events import (
    EVENT_EMPTY_CONTENT,
    EVENT_FINISH_EVENT,
    EVENT_OTHER_METADATA,
    EVENT_REASONING_DELTA,
    EVENT_ROLE_ONLY,
    EVENT_USAGE_ONLY,
    EVENT_VISIBLE_CONTENT,
    classify_chunk,
    extract_chunk_usage,
)


def _choice(content=None, reasoning=None, role=None, finish_reason=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning, role=role)
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _chunk(choices=None, usage=None, x_groq_usage=None, system_fingerprint=None):
    x_groq = SimpleNamespace(usage=x_groq_usage) if x_groq_usage is not None else None
    return SimpleNamespace(choices=choices or [], usage=usage, x_groq=x_groq, system_fingerprint=system_fingerprint)


def test_role_only_chunk():
    chunk = _chunk(choices=[_choice(role="assistant")])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_ROLE_ONLY


def test_visible_content_chunk():
    chunk = _chunk(choices=[_choice(content="Hello")])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_VISIBLE_CONTENT
    assert classified.visible_content == "Hello"


def test_whitespace_only_content_is_still_classified_as_visible_content():
    # Whitespace-only content IS classified VISIBLE_CONTENT (it's real
    # provider content) -- the "meaningful first visible content" policy
    # is applied by the runner's timing logic, not by classification.
    chunk = _chunk(choices=[_choice(content="   ")])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_VISIBLE_CONTENT
    assert classified.visible_content == "   "


def test_empty_string_content_is_empty_content_not_visible():
    chunk = _chunk(choices=[_choice(content="")])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_EMPTY_CONTENT


def test_reasoning_delta_chunk():
    chunk = _chunk(choices=[_choice(reasoning="thinking about the answer")])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_REASONING_DELTA


def test_reasoning_never_classified_as_visible_content():
    chunk = _chunk(choices=[_choice(reasoning="some internal reasoning text")])
    classified = classify_chunk(chunk)
    assert classified.visible_content is None


def test_finish_event_with_no_content_or_reasoning():
    chunk = _chunk(choices=[_choice(finish_reason="stop")])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_FINISH_EVENT
    assert classified.finish_reason == "stop"


def test_finish_reason_alongside_content_is_still_visible_content():
    chunk = _chunk(choices=[_choice(content="last bit.", finish_reason="stop")])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_VISIBLE_CONTENT
    assert classified.finish_reason == "stop"


def test_usage_only_chunk_with_empty_choices():
    chunk = _chunk(choices=[], usage=SimpleNamespace(prompt_tokens=10))
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_USAGE_ONLY
    assert classified.has_usage is True


def test_usage_only_chunk_via_x_groq_location():
    chunk = _chunk(choices=[], x_groq_usage=SimpleNamespace(prompt_tokens=10))
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_USAGE_ONLY
    assert classified.has_usage is True


def test_top_level_usage_preferred_but_both_locations_detected():
    top_level_usage = SimpleNamespace(prompt_tokens=10)
    chunk = _chunk(choices=[], usage=top_level_usage, x_groq_usage=SimpleNamespace(prompt_tokens=99))
    assert extract_chunk_usage(chunk) is top_level_usage


def test_empty_choices_with_no_usage_is_unknown():
    from labs.lab06_streaming.stream_events import EVENT_UNKNOWN

    chunk = _chunk(choices=[])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_UNKNOWN


def test_choices_not_indexed_without_checking_length_first():
    # Must not raise IndexError when choices is an empty list.
    chunk = _chunk(choices=[])
    classify_chunk(chunk)  # must not raise


def test_other_metadata_fallback():
    # No content, no reasoning, no role, no finish_reason, no usage.
    chunk = _chunk(choices=[_choice()])
    classified = classify_chunk(chunk)
    assert classified.event_type == EVENT_OTHER_METADATA
