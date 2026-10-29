"""Data shapes for Lab09: real-time (sequential/concurrent) measurements,
Batch job lifecycle state, and per-request Batch reconciliation records.

Never stores the model's raw response text -- only derived correctness
fields, consistent with this project's data-privacy policy (see
labs.lab01_token_anatomy and labs.lab08_concurrency.models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CONDITION_SEQUENTIAL = "sequential"
CONDITION_CONCURRENT = "concurrent"
CONDITION_BATCH = "batch"
REALTIME_CONDITIONS = (CONDITION_SEQUENTIAL, CONDITION_CONCURRENT)
ALL_CONDITIONS = (CONDITION_SEQUENTIAL, CONDITION_CONCURRENT, CONDITION_BATCH)

STATUS_SUCCESS = "SUCCESS"
STATUS_RATE_LIMITED = "RATE_LIMITED"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_CONNECTION_ERROR = "CONNECTION_ERROR"
STATUS_PROVIDER_ERROR = "PROVIDER_ERROR"
STATUS_UNKNOWN_ERROR = "UNKNOWN_ERROR"
REQUEST_STATUSES = (
    STATUS_SUCCESS, STATUS_RATE_LIMITED, STATUS_TIMEOUT,
    STATUS_CONNECTION_ERROR, STATUS_PROVIDER_ERROR, STATUS_UNKNOWN_ERROR,
)

EVENT_QUEUED = "QUEUED"
EVENT_PERMIT_ACQUIRED = "PERMIT_ACQUIRED"
EVENT_REQUEST_STARTED = "REQUEST_STARTED"
EVENT_REQUEST_COMPLETED = "REQUEST_COMPLETED"

# -- Provider-documented Batch job statuses (console.groq.com/docs/batch, verified 2026-10-29) --
PROVIDER_BATCH_STATUSES = (
    "validating", "in_progress", "finalizing", "completed",
    "failed", "expired", "cancelling", "cancelled",
)

# -- Local lifecycle states -- not all of these are literal provider statuses;
# some (NOT_SUBMITTED, SUBMISSION_IN_PROGRESS, SUBMISSION_UNCERTAIN,
# RESULTS_IMPORTED, EVALUATED) are THIS PROJECT's own bookkeeping on top of
# the provider's state machine.
LOCAL_STATE_NOT_SUBMITTED = "NOT_SUBMITTED"
LOCAL_STATE_SUBMISSION_IN_PROGRESS = "SUBMISSION_IN_PROGRESS"
LOCAL_STATE_SUBMISSION_UNCERTAIN = "SUBMISSION_UNCERTAIN"
LOCAL_STATE_SUBMITTED = "SUBMITTED"
LOCAL_STATE_RESULTS_AVAILABLE = "RESULTS_AVAILABLE"
LOCAL_STATE_RESULTS_IMPORTED = "RESULTS_IMPORTED"
LOCAL_STATE_EVALUATED = "EVALUATED"
LOCAL_LIFECYCLE_STATES = (
    LOCAL_STATE_NOT_SUBMITTED, LOCAL_STATE_SUBMISSION_IN_PROGRESS, LOCAL_STATE_SUBMISSION_UNCERTAIN,
    LOCAL_STATE_SUBMITTED, LOCAL_STATE_RESULTS_AVAILABLE, LOCAL_STATE_RESULTS_IMPORTED, LOCAL_STATE_EVALUATED,
)

# -- Batch reconciliation statuses (by custom_id; spec Part 8) --
RECON_COMPLETED = "COMPLETED"
RECON_FAILED = "FAILED"
RECON_MISSING = "MISSING"
RECON_DUPLICATE = "DUPLICATE"
RECON_UNKNOWN_ID = "UNKNOWN_ID"
RECON_MALFORMED = "MALFORMED"


def execution_mode_for(max_concurrent: int) -> str:
    return CONDITION_SEQUENTIAL if max_concurrent == 1 else CONDITION_CONCURRENT


@dataclass(frozen=True)
class PlannedRealtimeCall:
    execution_index: int
    condition: str
    max_concurrent: int
    task_id: str
    task_index: int


@dataclass
class RequestTimelineEvent:
    execution_index: int
    condition: str
    max_concurrent: int
    task_id: str
    event_type: str
    relative_ms: float


@dataclass
class RealtimeMeasurement:
    experiment_run_id: str
    execution_index: int
    task_id: str
    category: str
    condition: str
    max_concurrent: int
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
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    estimated_standard_cost_usd: Optional[float] = None

    evaluation_status: str = "EVALUATION_UNAVAILABLE"
    answer_correct: Optional[bool] = None


@dataclass
class BatchRequestMeasurement:
    """One reconciled Batch API request, keyed by custom_id (== task_id)."""

    experiment_run_id: str
    task_id: str
    category: str
    custom_id: str
    reconciliation_status: str
    error_message: Optional[str] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_standard_cost_usd: Optional[float] = None
    estimated_batch_cost_usd: Optional[float] = None

    evaluation_status: str = "EVALUATION_UNAVAILABLE"
    answer_correct: Optional[bool] = None


@dataclass
class ReconciliationCounts:
    planned_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    missing_requests: int = 0
    returned_output_rows: int = 0
    duplicate_result_ids: int = 0
    unexpected_result_ids: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0


@dataclass
class BatchJobMetadata:
    experiment_run_id: str
    workload_version: str
    model: str
    local_lifecycle_state: str = LOCAL_STATE_NOT_SUBMITTED
    completion_window: str = "24h"

    input_file_id: Optional[str] = None
    batch_id: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None

    provider_status: Optional[str] = None
    created_at: Optional[str] = None
    in_progress_at: Optional[str] = None
    finalizing_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    expired_at: Optional[str] = None

    request_counts_total: Optional[int] = None
    request_counts_completed: Optional[int] = None
    request_counts_failed: Optional[int] = None

    submitted_at: Optional[str] = None
    last_checked_at: Optional[str] = None
    last_known_error: Optional[str] = None
    pricing_policy_version: str = "groq-batch-v1-50pct-2026-10-29"
