from __future__ import annotations

from types import SimpleNamespace

from lab_core.config import RetryConfig
from labs.lab06_streaming.models import (
    CONDITION_STREAMING,
    STREAM_STATUS_EMPTY_STREAM,
    STREAM_STATUS_INTERRUPTED,
    STREAM_STATUS_NO_VISIBLE_CONTENT,
    STREAM_STATUS_OK,
    USAGE_AVAILABLE,
    USAGE_UNAVAILABLE,
    PlannedStreamingCall,
)
from labs.lab06_streaming.request_hash import compute_semantic_payload_hash
from labs.lab06_streaming.runner import execute_non_streaming_call, execute_streaming_call

RETRY_CONFIG = RetryConfig(max_attempts=2, base_delay_seconds=0.001, max_delay_seconds=0.002, jitter=False)


def _choice(content=None, reasoning=None, role=None, finish_reason=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning, role=role)
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _chunk(choices=None, usage=None, system_fingerprint=None):
    return SimpleNamespace(choices=choices or [], usage=usage, x_groq=None, system_fingerprint=system_fingerprint)


def _usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, reasoning_tokens=None, cached_tokens=None, completion_time=None):
    prompt_tokens_details = SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None
    completion_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens,
        prompt_tokens_details=prompt_tokens_details, completion_tokens_details=completion_tokens_details,
        completion_time=completion_time,
    )


class _FakeStreamCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._responses:
            raise AssertionError("fake client ran out of canned responses")
        next_response = self._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


def _fake_client(responses):
    completions = _FakeStreamCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _planned_streaming_call(execution_index=1, pair_id=1):
    return PlannedStreamingCall(execution_index=execution_index, pair_id=pair_id, condition=CONDITION_STREAMING)


def test_role_only_chunk_does_not_count_as_first_visible_content():
    chunks = [_chunk([_choice(role="assistant")]), _chunk([_choice(content="Incident Summary: ok")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert m.first_stream_event_ms is not None
    assert m.first_visible_content_ms is not None
    assert m.first_stream_event_ms <= m.first_visible_content_ms


def test_reasoning_only_chunk_does_not_count_as_visible_content():
    chunks = [_chunk([_choice(reasoning="thinking...")]), _chunk([_choice(reasoning="more thinking")]), _chunk([_choice(content="Answer text here")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert m.first_visible_content_ms is not None
    reasoning_events = [e for e in events if e.has_reasoning_delta]
    assert len(reasoning_events) == 2
    assert all(e.visible_content_character_count == 0 for e in reasoning_events)


def test_whitespace_only_content_does_not_advance_first_visible_content():
    chunks = [_chunk([_choice(content="   ")]), _chunk([_choice(content="Real text")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    # The second (non-whitespace) chunk's arrival time is when first_visible_content_ms is set.
    assert m.first_visible_content_ms == events[1].relative_arrival_ms


def test_last_visible_content_tracks_the_final_meaningful_chunk():
    chunks = [_chunk([_choice(content="first")]), _chunk([_choice(content="second")]), _chunk([_choice(content="   ")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    # last whitespace-only chunk must NOT move last_visible_content_ms past the "second" chunk.
    assert m.last_visible_content_ms == events[1].relative_arrival_ms


def test_total_stream_latency_reflects_iterator_termination():
    chunks = [_chunk([_choice(content="a")]), _chunk([_choice(content="b")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert m.total_stream_latency_ms is not None
    assert m.total_stream_latency_ms >= events[-1].relative_arrival_ms


def test_usage_only_final_chunk_does_not_distort_visible_timing_and_is_extracted():
    chunks = [_chunk([_choice(content="hello")]), _chunk(choices=[], usage=_usage(prompt_tokens=80, completion_tokens=20))]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, True, True, "run-1", _planned_streaming_call())
    assert m.usage_status == USAGE_AVAILABLE
    assert m.prompt_tokens == 80
    assert m.completion_tokens == 20
    # the usage-only chunk must not have been misclassified as visible content
    assert m.total_visible_characters == len("hello")


def test_empty_stream_sets_empty_stream_status_and_none_timings():
    client = _fake_client([[]])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert m.stream_status == STREAM_STATUS_EMPTY_STREAM
    assert m.first_stream_event_ms is None
    assert m.first_visible_content_ms is None
    assert events == []


def test_no_visible_content_stream_status():
    chunks = [_chunk([_choice(role="assistant")]), _chunk([_choice(reasoning="internal only")]), _chunk([_choice(finish_reason="stop")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert m.stream_status == STREAM_STATUS_NO_VISIBLE_CONTENT
    assert m.first_visible_content_ms is None
    assert m.evaluation_status == "NO_VISIBLE_CONTENT"


def test_mid_stream_exception_is_reported_as_stream_interrupted_not_successful():
    def _generator():
        yield _chunk([_choice(content="partial")])
        raise ConnectionError("mid-stream network drop")

    client = _fake_client([_generator()])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert m.succeeded is False
    assert m.stream_status == STREAM_STATUS_INTERRUPTED
    assert m.evaluation_status == "STREAM_INTERRUPTED"
    assert m.error is not None
    # partial timing is preserved even though the call did not succeed.
    assert m.first_stream_event_ms is not None


def test_usage_unavailable_never_fabricates_token_values():
    chunks = [_chunk([_choice(content="hello there")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert m.usage_status == USAGE_UNAVAILABLE
    assert m.prompt_tokens is None
    assert m.completion_tokens is None
    assert m.estimated_cost_usd is None


def test_event_timeline_not_recorded_when_record_event_timeline_is_false():
    chunks = [_chunk([_choice(content="hello")])]
    client = _fake_client([chunks])
    m, events = execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, False, "run-1", _planned_streaming_call())
    assert events == []
    # the derived measurement fields are still computed even without the timeline.
    assert m.first_visible_content_ms is not None


def test_stream_options_only_requested_when_usage_flag_is_true():
    chunks = [_chunk([_choice(content="hello")])]
    client = _fake_client([chunks])
    execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    sent = client.chat.completions.calls[0]
    assert "extra_body" not in sent

    chunks2 = [_chunk([_choice(content="hello")])]
    client2 = _fake_client([chunks2])
    execute_streaming_call(client2, RETRY_CONFIG, "m", None, "low", 100, True, True, "run-1", _planned_streaming_call())
    sent2 = client2.chat.completions.calls[0]
    assert sent2["extra_body"] == {"stream_options": {"include_usage": True}}


def test_reasoning_format_parsed_always_requested():
    chunks = [_chunk([_choice(content="hello")])]
    client = _fake_client([chunks])
    execute_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, False, True, "run-1", _planned_streaming_call())
    assert client.chat.completions.calls[0]["reasoning_format"] == "parsed"


def test_non_streaming_and_streaming_share_the_same_request_hash(config_with_pricing):
    hash_a = compute_semantic_payload_hash("m", [{"role": "user", "content": "x"}], 100, "low")
    hash_b = compute_semantic_payload_hash("m", [{"role": "user", "content": "x"}], 100, "low")
    assert hash_a == hash_b


def test_non_streaming_effective_tps_and_cost(fake_sync_client_factory, make_response, config_with_pricing):
    from labs.lab06_streaming.matrix import build_single_condition_plan
    from labs.lab06_streaming.models import CONDITION_NON_STREAMING

    response = make_response(content="Incident Summary: ok", prompt_tokens=100, completion_tokens=50, reasoning_tokens=10)
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    plan_call = build_single_condition_plan(CONDITION_NON_STREAMING, 1)[0]

    m = execute_non_streaming_call(client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", 2048, "run-1", plan_call)
    assert m.succeeded is True
    assert m.non_reasoning_completion_tokens == 40
    assert m.effective_completion_tokens_per_second is not None
    assert m.estimated_cost_usd is not None
    assert m.usage_status == USAGE_AVAILABLE


def test_non_streaming_failed_call_keeps_fields_none(fake_sync_client_factory, config_with_pricing):
    import httpx
    from groq import APIConnectionError

    from labs.lab06_streaming.matrix import build_single_condition_plan
    from labs.lab06_streaming.models import CONDITION_NON_STREAMING

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client = fake_sync_client_factory([APIConnectionError(request=request) for _ in range(RETRY_CONFIG.max_attempts)])
    plan_call = build_single_condition_plan(CONDITION_NON_STREAMING, 1)[0]

    m = execute_non_streaming_call(client, RETRY_CONFIG, "m", None, "low", 100, "run-1", plan_call)
    assert m.succeeded is False
    assert m.prompt_tokens is None
    assert m.evaluation_status == "API_ERROR"
