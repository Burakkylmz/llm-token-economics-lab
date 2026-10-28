"""Data shapes for Lab06's streaming/non-streaming perceived-latency
experiment.

One unified `StreamMeasurement` dataclass covers BOTH conditions (not two
separate dataclasses): the two conditions share the large majority of
their fields (token usage, cost, evaluation), and keeping them in one
record makes pairing by `pair_id` trivial. Fields meaningful only to one
condition stay `None` for the other (e.g. `total_stream_latency_ms` is
always `None` for a `non_streaming` row) -- never fabricated or
cross-assigned.

Never stores the model's raw response text or any reasoning trace -- only
derived completeness/status fields and bounded per-chunk metadata (see
StreamEventRecord).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CONDITION_NON_STREAMING = "non_streaming"
CONDITION_STREAMING = "streaming"

STREAM_STATUS_OK = "OK"
STREAM_STATUS_EMPTY_STREAM = "EMPTY_STREAM"
STREAM_STATUS_NO_VISIBLE_CONTENT = "NO_VISIBLE_CONTENT"
STREAM_STATUS_INTERRUPTED = "STREAM_INTERRUPTED"

USAGE_AVAILABLE = "AVAILABLE"
USAGE_UNAVAILABLE = "USAGE_UNAVAILABLE"


@dataclass(frozen=True)
class PlannedStreamingCall:
    execution_index: int
    pair_id: int
    condition: str  # "non_streaming" | "streaming"


@dataclass
class StreamEventRecord:
    """One streaming chunk's bounded, safe metadata -- never the chunk's
    own text content, only its character COUNT."""

    execution_index: int
    event_index: int
    relative_arrival_ms: float
    event_type: str
    visible_content_character_count: int
    cumulative_visible_character_count: int
    has_reasoning_delta: bool
    has_finish_reason: bool
    has_usage: bool


@dataclass
class StreamMeasurement:
    experiment_run_id: str
    execution_index: int
    pair_id: int
    condition: str
    timestamp: str
    model: str
    reasoning_effort: str
    max_completion_tokens: int
    request_hash: str
    succeeded: bool
    attempts_used: int
    error: Optional[str] = None

    # --- Streaming-only timing (Part C); None for non_streaming rows ---
    first_stream_event_ms: Optional[float] = None
    # Always None today: no verified provider per-token timing distinct
    # from the first SSE event's own arrival time exists for Groq (see
    # labs.lab06_streaming.runner's module docstring) -- kept as its own
    # field, never silently equated with first_stream_event_ms, so a
    # future verified provider TTFT source has somewhere to go without a
    # schema change.
    ttft_ms: Optional[float] = None
    first_visible_content_ms: Optional[float] = None
    last_visible_content_ms: Optional[float] = None
    total_stream_latency_ms: Optional[float] = None
    first_event_to_visible_gap_ms: Optional[float] = None
    total_chunks: Optional[int] = None
    visible_content_chunks: Optional[int] = None
    total_visible_characters: Optional[int] = None
    stream_status: Optional[str] = None

    # --- Non-streaming-only timing; None for streaming rows ---
    non_streaming_total_latency_ms: Optional[float] = None

    # --- Shared token usage/cost (both conditions) ---
    usage_status: str = USAGE_UNAVAILABLE
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    non_reasoning_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    system_fingerprint: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    provider_completion_time_ms: Optional[float] = None

    # --- Throughput ---
    effective_completion_tokens_per_second: Optional[float] = None
    provider_completion_tokens_per_second: Optional[float] = None
    visible_characters_per_second: Optional[float] = None
    content_chunks_per_second: Optional[float] = None

    # --- Evaluation (see labs.lab06_streaming.evaluation) ---
    evaluation_status: str = "EVALUATION_UNAVAILABLE"
    answer_complete: Optional[bool] = None
    present_sections: int = 0
    required_sections: int = 5
    completeness_ratio: Optional[float] = None
