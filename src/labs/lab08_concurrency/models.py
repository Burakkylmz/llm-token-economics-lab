"""Data shapes for Lab08's sequential-vs-concurrent async execution
experiment.

`execution_mode` is derived purely from `max_concurrent` ("async_sequential"
for 1, "async_concurrent" for > 1) -- never hardcoded independently of it,
so the two can never silently drift apart (Part A's problem #1: the old
code labeled sequential execution "sync" even though it used the async
client).

Never stores the model's raw response text -- only derived correctness/
completeness fields, consistent with this project's data-privacy policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

EXECUTION_MODE_SEQUENTIAL = "async_sequential"
EXECUTION_MODE_CONCURRENT = "async_concurrent"

STATUS_SUCCESS = "SUCCESS"
STATUS_RATE_LIMITED = "RATE_LIMITED"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_CONNECTION_ERROR = "CONNECTION_ERROR"
STATUS_PROVIDER_ERROR = "PROVIDER_ERROR"
STATUS_CANCELLED = "CANCELLED"
STATUS_UNKNOWN_ERROR = "UNKNOWN_ERROR"

REQUEST_STATUSES = (
    STATUS_SUCCESS, STATUS_RATE_LIMITED, STATUS_TIMEOUT, STATUS_CONNECTION_ERROR,
    STATUS_PROVIDER_ERROR, STATUS_CANCELLED, STATUS_UNKNOWN_ERROR,
)

EVENT_QUEUED = "QUEUED"
EVENT_PERMIT_ACQUIRED = "PERMIT_ACQUIRED"
EVENT_REQUEST_STARTED = "REQUEST_STARTED"
EVENT_REQUEST_COMPLETED = "REQUEST_COMPLETED"


def execution_mode_for(max_concurrent: int) -> str:
    return EXECUTION_MODE_SEQUENTIAL if max_concurrent == 1 else EXECUTION_MODE_CONCURRENT


@dataclass(frozen=True)
class PlannedConcurrencyCall:
    execution_index: int
    repetition_index: int
    max_concurrent: int
    task_id: str
    task_index: int


@dataclass
class RequestTimelineEvent:
    execution_index: int
    repetition_index: int
    max_concurrent: int
    task_id: str
    event_type: str
    relative_ms: float


@dataclass
class ConcurrencyMeasurement:
    experiment_run_id: str
    execution_index: int
    repetition_index: int
    task_id: str
    max_concurrent: int
    execution_mode: str
    timestamp: str
    model: str
    reasoning_effort: str
    max_completion_tokens: int
    request_status: str
    error: Optional[str] = None

    queue_wait_ms: Optional[float] = None
    request_service_latency_ms: Optional[float] = None
    task_end_to_end_latency_ms: Optional[float] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    non_reasoning_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    output_truncated: Optional[bool] = None
    estimated_cost_usd: Optional[float] = None

    evaluation_status: str = "EVALUATION_UNAVAILABLE"
    answer_correct: Optional[bool] = None
    answer_complete: Optional[bool] = None
