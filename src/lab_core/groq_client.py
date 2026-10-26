"""Thin wrapper around the official Groq SDK.

Responsibilities, and only these:
  - time each call client-side (time.perf_counter_ns), separately from
    whatever the server reports;
  - normalize the SDK's `usage` object into TokenCostMetrics fields,
    defensively, so a missing field becomes None rather than a fabricated 0;
  - normalize the rest of the response's call-level metadata (finish_reason,
    system_fingerprint, the final message text) the same defensive way;
  - echo back only the generation parameters (temperature, top_p, ...) the
    caller actually passed, never a guessed provider default;
  - provide one function per call style (sync, async, streaming).

No retry policy, no cost calculation, no experiment-specific logic lives
here — those are composed by each lab on top of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional

from groq import AsyncGroq, Groq

from lab_core.metrics import PerformanceMetrics, TokenCostMetrics
from lab_core.timing import StopwatchTimer


def _get_nested(obj: Any, *path: str) -> Optional[Any]:
    """Walk an attribute path, returning None as soon as any hop is missing."""
    current = obj
    for attr in path:
        if current is None:
            return None
        current = getattr(current, attr, None)
    return current


def normalize_usage(usage: Any) -> TokenCostMetrics:
    """Convert an SDK `usage` object into TokenCostMetrics.

    Any field not present on `usage` is left as None (unavailable), never
    defaulted to 0. In particular, cached_tokens and reasoning_tokens are
    only populated for models/responses that actually report them; see
    module docstrings in metrics.py for why this distinction matters.
    """
    if usage is None:
        return TokenCostMetrics(api_call_count=1)

    return TokenCostMetrics(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        cached_tokens=_get_nested(usage, "prompt_tokens_details", "cached_tokens"),
        reasoning_tokens=_get_nested(usage, "completion_tokens_details", "reasoning_tokens"),
        api_call_count=1,
    )


def normalize_server_timing(usage: Any) -> PerformanceMetrics:
    """Extract Groq's server-reported timing fields (seconds -> ms), if present."""

    def seconds_to_ms(value: Optional[float]) -> Optional[float]:
        return value * 1000 if value is not None else None

    total_time = getattr(usage, "total_time", None) if usage is not None else None
    return PerformanceMetrics(provider_reported_processing_ms=seconds_to_ms(total_time))


def _first_choice(response: Any) -> Optional[Any]:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    return choices[0]


def extract_finish_reason(response: Any) -> Optional[str]:
    """The reported reason generation stopped ("stop", "length", ...), or None
    if the response carries no choices to read it from."""
    choice = _first_choice(response)
    return getattr(choice, "finish_reason", None) if choice is not None else None


def extract_message_content(response: Any) -> Optional[str]:
    """The final answer text, used only to derive a char count -- never persisted
    verbatim into results (see labs/lab01_token_anatomy's data privacy constraint)."""
    choice = _first_choice(response)
    message = getattr(choice, "message", None) if choice is not None else None
    return getattr(message, "content", None) if message is not None else None


def extract_system_fingerprint(response: Any) -> Optional[str]:
    return getattr(response, "system_fingerprint", None)


# Generation knobs we'll echo back into result metadata, but ONLY when the
# caller actually passed them -- never filled in with an assumed provider
# default, since we don't reliably know what that default is.
GENERATION_PARAMETER_KEYS = (
    "temperature",
    "top_p",
    "reasoning_effort",
    "max_completion_tokens",
    "max_tokens",
)


def extract_requested_generation_parameters(create_kwargs: dict) -> dict:
    """The subset of GENERATION_PARAMETER_KEYS explicitly present in create_kwargs."""
    return {key: create_kwargs[key] for key in GENERATION_PARAMETER_KEYS if key in create_kwargs}


@dataclass
class SyncCallResult:
    raw_response: Any
    tokens_and_cost: TokenCostMetrics
    performance: PerformanceMetrics
    message_content: Optional[str] = None
    system_fingerprint: Optional[str] = None
    requested_generation_parameters: dict = field(default_factory=dict)


def call_sync(client: Groq, **create_kwargs: Any) -> SyncCallResult:
    """Make one synchronous, non-streaming chat completion call, timed client-side."""
    with StopwatchTimer() as timer:
        response = client.chat.completions.create(**create_kwargs, stream=False)

    tokens_and_cost = normalize_usage(getattr(response, "usage", None))
    tokens_and_cost.finish_reason = extract_finish_reason(response)
    performance = normalize_server_timing(getattr(response, "usage", None))
    performance.client_elapsed_ms = timer.result.elapsed_ms if timer.result else None

    return SyncCallResult(
        raw_response=response,
        tokens_and_cost=tokens_and_cost,
        performance=performance,
        message_content=extract_message_content(response),
        system_fingerprint=extract_system_fingerprint(response),
        requested_generation_parameters=extract_requested_generation_parameters(create_kwargs),
    )


async def call_async(client: AsyncGroq, **create_kwargs: Any) -> SyncCallResult:
    """Make one async, non-streaming chat completion call, timed client-side."""
    with StopwatchTimer() as timer:
        response = await client.chat.completions.create(**create_kwargs, stream=False)

    tokens_and_cost = normalize_usage(getattr(response, "usage", None))
    tokens_and_cost.finish_reason = extract_finish_reason(response)
    performance = normalize_server_timing(getattr(response, "usage", None))
    performance.client_elapsed_ms = timer.result.elapsed_ms if timer.result else None

    return SyncCallResult(
        raw_response=response,
        tokens_and_cost=tokens_and_cost,
        performance=performance,
        message_content=extract_message_content(response),
        system_fingerprint=extract_system_fingerprint(response),
        requested_generation_parameters=extract_requested_generation_parameters(create_kwargs),
    )


@dataclass
class StreamHandle:
    """A started stream plus the client-side start timestamp.

    Deliberately NOT consumed here: chunk-by-chunk timing analysis
    (time to first token, first visible content, token velocity) is the
    subject of Lab06 and will be implemented and verified as part of
    that lab's own design/approval cycle, not assumed here.
    """

    stream: Iterator[Any]
    start_perf_counter_ns: int


def call_stream(client: Groq, **create_kwargs: Any) -> StreamHandle:
    import time

    start = time.perf_counter_ns()
    stream = client.chat.completions.create(**create_kwargs, stream=True)
    return StreamHandle(stream=stream, start_perf_counter_ns=start)


@dataclass
class AsyncStreamHandle:
    stream: AsyncIterator[Any]
    start_perf_counter_ns: int


async def call_async_stream(client: AsyncGroq, **create_kwargs: Any) -> AsyncStreamHandle:
    import time

    start = time.perf_counter_ns()
    stream = await client.chat.completions.create(**create_kwargs, stream=True)
    return AsyncStreamHandle(stream=stream, start_perf_counter_ns=start)
