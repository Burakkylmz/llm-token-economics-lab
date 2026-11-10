"""Executes (or dry-runs) Lab06's paired streaming/non-streaming plan
against the real Groq API.

Investigation finding (Part G, verified by reading the installed
`groq==1.7.0` SDK source rather than guessing):
  - `stream_options` is NOT a typed parameter of `Completions.create()` --
    its full keyword-only parameter list (groq/resources/chat/
    completions.py) has no `stream_options` entry and no catch-all
    `**kwargs`. Requesting streaming usage therefore requires
    `extra_body={"stream_options": {"include_usage": True}}`, which IS a
    real, typed parameter (`Body | None`) that is merged into the raw
    request JSON.
  - `ChatCompletionChunk.usage`'s own docstring (groq/types/chat/
    chat_completion_chunk.py) documents this exact mechanism: "An
    optional field that will only be present when you set
    `stream_options: {"include_usage": true}`...contains a null value
    except for the last chunk". This is provider-documented behavior, not
    an assumption.
  - `ChoiceDelta.reasoning` is a real, separate field, "Only available for
    reasoning models when...reasoning_format has value `parsed`" -- so
    this lab always sets `reasoning_format="parsed"` on BOTH conditions,
    for request-equivalence (Part D) and so reasoning content is never
    confused with visible content (Part F).
  - No verified, distinct provider per-token TTFT source was found beyond
    the first SSE event's own client-measured arrival time -- `ttft_ms`
    therefore stays `None` on every measurement (see models.py).

Reuses, rather than reimplements, this project's existing shared/Lab01
infrastructure: lab_core.groq_client.{call_stream, call_sync,
normalize_usage}, labs.lab01_token_anatomy.factorial_runner.
call_with_operational_retry (non-streaming call; reused verbatim) and
compute_backoff_delay_seconds (a dedicated streaming-create retry loop,
below, since call_with_operational_retry assumes a single atomic
call_sync), labs.lab01_token_anatomy.cost.calculate_cost, labs.
lab01_token_anatomy.observability.compute_non_reasoning_completion_tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from groq import APIConnectionError, Groq, GroqError, InternalServerError, RateLimitError

from lab_core.config import ModelPricing, RetryConfig
from lab_core.groq_client import call_stream, normalize_usage
from lab_core.retry import compute_backoff_delay_seconds
from labs.lab01_token_anatomy.cost import calculate_cost
from lab_core.budget import Budget, conservative_per_call_cost_ceiling  # noqa: F401 -- re-exported
from lab_core.operational_retry import call_with_operational_retry  # noqa: F401 -- re-exported
from labs.lab01_token_anatomy.observability import compute_non_reasoning_completion_tokens
from labs.lab06_streaming.evaluation import STATUS_API_ERROR, STATUS_STREAM_INTERRUPTED, evaluate_incident_response
from labs.lab06_streaming.experiments import build_incident_messages
from labs.lab06_streaming.models import (
    CONDITION_NON_STREAMING,
    CONDITION_STREAMING,
    STREAM_STATUS_EMPTY_STREAM,
    STREAM_STATUS_INTERRUPTED,
    STREAM_STATUS_NO_VISIBLE_CONTENT,
    STREAM_STATUS_OK,
    USAGE_AVAILABLE,
    USAGE_UNAVAILABLE,
    PlannedStreamingCall,
    StreamEventRecord,
    StreamMeasurement,
)
from labs.lab06_streaming.request_hash import compute_semantic_payload_hash
from labs.lab06_streaming.stream_events import EVENT_REASONING_DELTA, EVENT_VISIBLE_CONTENT, classify_chunk, extract_chunk_usage

_RETRYABLE_EXCEPTIONS = (RateLimitError, InternalServerError, APIConnectionError)


def _seconds_to_ms(value: Optional[float]) -> Optional[float]:
    return value * 1000 if value is not None else None


def _rate(count: Optional[float], elapsed_ms: Optional[float]) -> Optional[float]:
    if count is None or not elapsed_ms:
        return None
    return count / (elapsed_ms / 1000)


def _create_stream_with_retry(
    client: Groq, retry_config: RetryConfig, sleep_fn: Callable[[float], None] = time.sleep, **create_kwargs: Any
):
    """The streaming analog of labs.lab01_token_anatomy.factorial_runner.
    call_with_operational_retry: retries only the CONNECTION step
    (`.create()`, before any chunk is read) -- a failure here means no
    content was ever streamed, so retrying is safe. A failure AFTER at
    least one chunk arrived is a mid-stream interruption (see
    _consume_stream), never retried here."""
    last_error: Optional[str] = None
    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            handle = call_stream(client, **create_kwargs)
            return handle, attempt, None
        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retry_config.max_attempts:
                return None, attempt, last_error
            delay = compute_backoff_delay_seconds(attempt, retry_config.base_delay_seconds, retry_config.max_delay_seconds, retry_config.jitter)
            if delay > 0:
                sleep_fn(delay)
        except GroqError as exc:
            return None, attempt, f"{type(exc).__name__}: {exc}"
    return None, retry_config.max_attempts, last_error


@dataclass
class _ConsumeResult:
    events: list[StreamEventRecord]
    first_stream_event_ms: Optional[float]
    first_visible_content_ms: Optional[float]
    last_visible_content_ms: Optional[float]
    total_stream_latency_ms: Optional[float]
    total_chunks: int
    visible_content_chunks: int
    total_visible_characters: int
    stream_status: str
    raw_usage: Optional[Any]
    finish_reason: Optional[str]
    system_fingerprint: Optional[str]
    interrupted_error: Optional[str]
    # In-memory only, for the completeness evaluator -- NEVER written to
    # StreamMeasurement/StreamEventRecord (see this module's docstring and
    # Part L's "raw reasoning traces/response text never persisted" rule).
    visible_text: str = field(default="", repr=False)


def _consume_stream(stream: Any, start_ns: int, execution_index: int, record_events: bool) -> _ConsumeResult:
    events: list[StreamEventRecord] = []
    first_stream_event_ms: Optional[float] = None
    first_visible_content_ms: Optional[float] = None
    last_visible_content_ms: Optional[float] = None
    cumulative_visible_chars = 0
    total_chunks = 0
    visible_content_chunks = 0
    raw_usage: Optional[Any] = None
    finish_reason: Optional[str] = None
    system_fingerprint: Optional[str] = None
    interrupted_error: Optional[str] = None
    last_event_ns = start_ns
    visible_parts: list[str] = []

    try:
        for chunk in stream:
            now_ns = time.perf_counter_ns()
            relative_ms = (now_ns - start_ns) / 1_000_000
            total_chunks += 1
            last_event_ns = now_ns
            if first_stream_event_ms is None:
                first_stream_event_ms = relative_ms

            classified = classify_chunk(chunk)
            chunk_usage = extract_chunk_usage(chunk)
            if chunk_usage is not None:
                raw_usage = chunk_usage
            if classified.finish_reason is not None:
                finish_reason = classified.finish_reason
            fingerprint = getattr(chunk, "system_fingerprint", None)
            if fingerprint:
                system_fingerprint = fingerprint

            char_count = 0
            if classified.event_type == EVENT_VISIBLE_CONTENT:
                visible_content_chunks += 1
                char_count = len(classified.visible_content)
                cumulative_visible_chars += char_count
                visible_parts.append(classified.visible_content)
                if classified.visible_content.strip():
                    if first_visible_content_ms is None:
                        first_visible_content_ms = relative_ms
                    last_visible_content_ms = relative_ms

            if record_events:
                events.append(
                    StreamEventRecord(
                        execution_index=execution_index,
                        event_index=total_chunks,
                        relative_arrival_ms=relative_ms,
                        event_type=classified.event_type,
                        visible_content_character_count=char_count,
                        cumulative_visible_character_count=cumulative_visible_chars,
                        has_reasoning_delta=classified.event_type == EVENT_REASONING_DELTA,
                        has_finish_reason=classified.finish_reason is not None,
                        has_usage=classified.has_usage,
                    )
                )
    except Exception as exc:  # noqa: BLE001 -- any mid-stream transport/provider error
        interrupted_error = f"{type(exc).__name__}: {exc}"
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 -- best-effort cleanup only
                pass

    total_stream_latency_ms = (last_event_ns - start_ns) / 1_000_000 if total_chunks else None

    if interrupted_error is not None:
        stream_status = STREAM_STATUS_INTERRUPTED
    elif total_chunks == 0:
        stream_status = STREAM_STATUS_EMPTY_STREAM
    elif first_visible_content_ms is None:
        stream_status = STREAM_STATUS_NO_VISIBLE_CONTENT
    else:
        stream_status = STREAM_STATUS_OK

    return _ConsumeResult(
        events=events,
        first_stream_event_ms=first_stream_event_ms,
        first_visible_content_ms=first_visible_content_ms,
        last_visible_content_ms=last_visible_content_ms,
        total_stream_latency_ms=total_stream_latency_ms,
        total_chunks=total_chunks,
        visible_content_chunks=visible_content_chunks,
        total_visible_characters=cumulative_visible_chars,
        stream_status=stream_status,
        raw_usage=raw_usage,
        finish_reason=finish_reason,
        system_fingerprint=system_fingerprint,
        interrupted_error=interrupted_error,
        visible_text="".join(visible_parts),
    )


def execute_streaming_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    reasoning_effort: str,
    max_completion_tokens: int,
    request_usage_when_supported: bool,
    record_event_timeline: bool,
    experiment_run_id: str,
    planned_call: PlannedStreamingCall,
) -> tuple[StreamMeasurement, list[StreamEventRecord]]:
    messages = build_incident_messages()
    request_hash = compute_semantic_payload_hash(model, messages, max_completion_tokens, reasoning_effort)
    base = dict(
        experiment_run_id=experiment_run_id,
        execution_index=planned_call.execution_index,
        pair_id=planned_call.pair_id,
        condition=CONDITION_STREAMING,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        request_hash=request_hash,
    )

    create_kwargs: dict = dict(
        model=model, messages=messages, max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort, reasoning_format="parsed",
    )
    if request_usage_when_supported:
        create_kwargs["extra_body"] = {"stream_options": {"include_usage": True}}

    start_ns = time.perf_counter_ns()
    handle, attempts_used, error = _create_stream_with_retry(client, retry_config, **create_kwargs)
    if handle is None:
        return StreamMeasurement(**base, succeeded=False, attempts_used=attempts_used, error=error, evaluation_status=STATUS_API_ERROR), []

    consumed = _consume_stream(handle.stream, start_ns, planned_call.execution_index, record_event_timeline)

    if consumed.interrupted_error is not None:
        measurement = StreamMeasurement(
            **base, succeeded=False, attempts_used=attempts_used, error=consumed.interrupted_error,
            first_stream_event_ms=consumed.first_stream_event_ms, first_visible_content_ms=consumed.first_visible_content_ms,
            last_visible_content_ms=consumed.last_visible_content_ms, total_stream_latency_ms=consumed.total_stream_latency_ms,
            total_chunks=consumed.total_chunks, visible_content_chunks=consumed.visible_content_chunks,
            total_visible_characters=consumed.total_visible_characters, stream_status=consumed.stream_status,
            evaluation_status=STATUS_STREAM_INTERRUPTED,
        )
        return measurement, consumed.events

    tokens = normalize_usage(consumed.raw_usage)
    usage_status = USAGE_AVAILABLE if consumed.raw_usage is not None else USAGE_UNAVAILABLE
    estimated_cost = (
        calculate_cost(tokens, pricing)
        if pricing is not None and tokens.prompt_tokens is not None and tokens.completion_tokens is not None
        else None
    )
    non_reasoning = compute_non_reasoning_completion_tokens(tokens.completion_tokens, tokens.reasoning_tokens)
    provider_completion_time_ms = _seconds_to_ms(getattr(consumed.raw_usage, "completion_time", None))
    evaluation = evaluate_incident_response(consumed.visible_text or None, consumed.finish_reason, stream_status=consumed.stream_status)

    first_event_to_visible_gap_ms = (
        consumed.first_visible_content_ms - consumed.first_stream_event_ms
        if consumed.first_visible_content_ms is not None and consumed.first_stream_event_ms is not None
        else None
    )

    measurement = StreamMeasurement(
        **base,
        succeeded=True,
        attempts_used=attempts_used,
        error=None,
        first_stream_event_ms=consumed.first_stream_event_ms,
        first_visible_content_ms=consumed.first_visible_content_ms,
        last_visible_content_ms=consumed.last_visible_content_ms,
        total_stream_latency_ms=consumed.total_stream_latency_ms,
        first_event_to_visible_gap_ms=first_event_to_visible_gap_ms,
        total_chunks=consumed.total_chunks,
        visible_content_chunks=consumed.visible_content_chunks,
        total_visible_characters=consumed.total_visible_characters,
        stream_status=consumed.stream_status,
        usage_status=usage_status,
        prompt_tokens=tokens.prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        reasoning_tokens=tokens.reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning,
        total_tokens=tokens.total_tokens,
        cached_tokens=tokens.cached_tokens,
        finish_reason=consumed.finish_reason,
        system_fingerprint=consumed.system_fingerprint,
        estimated_cost_usd=estimated_cost,
        provider_completion_time_ms=provider_completion_time_ms,
        effective_completion_tokens_per_second=_rate(tokens.completion_tokens, consumed.total_stream_latency_ms),
        provider_completion_tokens_per_second=_rate(tokens.completion_tokens, provider_completion_time_ms),
        visible_characters_per_second=_rate(consumed.total_visible_characters or None, consumed.total_stream_latency_ms),
        content_chunks_per_second=_rate(consumed.visible_content_chunks or None, consumed.total_stream_latency_ms),
        evaluation_status=evaluation.status,
        answer_complete=evaluation.answer_complete,
        present_sections=evaluation.present_sections,
        required_sections=evaluation.required_sections,
        completeness_ratio=evaluation.completeness_ratio,
    )
    return measurement, consumed.events


def execute_non_streaming_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    reasoning_effort: str,
    max_completion_tokens: int,
    experiment_run_id: str,
    planned_call: PlannedStreamingCall,
) -> StreamMeasurement:
    messages = build_incident_messages()
    request_hash = compute_semantic_payload_hash(model, messages, max_completion_tokens, reasoning_effort)
    base = dict(
        experiment_run_id=experiment_run_id,
        execution_index=planned_call.execution_index,
        pair_id=planned_call.pair_id,
        condition=CONDITION_NON_STREAMING,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        request_hash=request_hash,
    )

    create_kwargs = dict(
        model=model, messages=messages, max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort, reasoning_format="parsed",
    )
    call_result, attempts_used, error = call_with_operational_retry(client, retry_config, **create_kwargs)
    if call_result is None:
        return StreamMeasurement(**base, succeeded=False, attempts_used=attempts_used, error=error, evaluation_status=STATUS_API_ERROR)

    tokens = call_result.tokens_and_cost
    if pricing is not None:
        tokens.estimated_cost_usd = calculate_cost(tokens, pricing)
    non_reasoning = compute_non_reasoning_completion_tokens(tokens.completion_tokens, tokens.reasoning_tokens)
    raw_usage = getattr(call_result.raw_response, "usage", None)
    provider_completion_time_ms = _seconds_to_ms(getattr(raw_usage, "completion_time", None))
    total_latency = call_result.performance.client_elapsed_ms
    usage_status = USAGE_AVAILABLE if tokens.prompt_tokens is not None else USAGE_UNAVAILABLE
    evaluation = evaluate_incident_response(call_result.message_content, tokens.finish_reason)

    return StreamMeasurement(
        **base,
        succeeded=True,
        attempts_used=attempts_used,
        error=None,
        non_streaming_total_latency_ms=total_latency,
        usage_status=usage_status,
        prompt_tokens=tokens.prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        reasoning_tokens=tokens.reasoning_tokens,
        non_reasoning_completion_tokens=non_reasoning,
        total_tokens=tokens.total_tokens,
        cached_tokens=tokens.cached_tokens,
        finish_reason=tokens.finish_reason,
        system_fingerprint=call_result.system_fingerprint,
        estimated_cost_usd=tokens.estimated_cost_usd,
        provider_completion_time_ms=provider_completion_time_ms,
        effective_completion_tokens_per_second=_rate(tokens.completion_tokens, total_latency),
        provider_completion_tokens_per_second=_rate(tokens.completion_tokens, provider_completion_time_ms),
        evaluation_status=evaluation.status,
        answer_complete=evaluation.answer_complete,
        present_sections=evaluation.present_sections,
        required_sections=evaluation.required_sections,
        completeness_ratio=evaluation.completeness_ratio,
    )


def execute_planned_call(
    client: Groq,
    retry_config: RetryConfig,
    model: str,
    pricing: Optional[ModelPricing],
    reasoning_effort: str,
    max_completion_tokens: int,
    request_usage_when_supported: bool,
    record_event_timeline: bool,
    experiment_run_id: str,
    planned_call: PlannedStreamingCall,
) -> tuple[StreamMeasurement, list[StreamEventRecord]]:
    if planned_call.condition == CONDITION_STREAMING:
        return execute_streaming_call(
            client, retry_config, model, pricing, reasoning_effort, max_completion_tokens,
            request_usage_when_supported, record_event_timeline, experiment_run_id, planned_call,
        )
    measurement = execute_non_streaming_call(client, retry_config, model, pricing, reasoning_effort, max_completion_tokens, experiment_run_id, planned_call)
    return measurement, []
