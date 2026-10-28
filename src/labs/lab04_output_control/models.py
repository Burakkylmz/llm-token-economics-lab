"""Data shapes for Lab04's output-control and invocation-optimization
experiments.

Two measurement kinds, kept separate (same rationale as
labs.lab02_context_growth.models): OutputLimitMeasurement answers RQ1-RQ4
(output budget, truncation, reasoning overhead, answer quality);
InvocationMeasurement answers RQ5-RQ7 (invocation count, workflow latency,
quality-constrained cost). Neither stores the model's raw response text --
only derived completeness/correctness counts and a char count, consistent
with this project's data-privacy policy (see
labs.lab02_context_growth.models).

`output_truncated` is derived STRICTLY from the provider's own
`finish_reason` (True iff "length"), never from
`completion_tokens == max_completion_tokens` alone -- that equality is
necessary but not sufficient evidence of truncation (see Part D of this
lab's design). It is None when finish_reason itself is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PlannedOutputLimitCall:
    execution_index: int
    condition_id: str
    repetition_index: int
    max_completion_tokens: int


@dataclass
class OutputLimitMeasurement:
    experiment_run_id: str
    condition_id: str
    execution_index: int
    repetition_index: int
    timestamp: str
    model: str
    max_completion_tokens: int
    reasoning_effort: str
    succeeded: bool
    attempts_used: int
    error: Optional[str] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    non_reasoning_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    finish_reason: Optional[str] = None
    output_truncated: Optional[bool] = None
    system_fingerprint: Optional[str] = None
    final_answer_char_count: Optional[int] = None

    total_latency_ms: Optional[float] = None
    provider_reported_processing_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None

    required_field_count: int = 0
    present_field_count: int = 0
    correct_field_count: int = 0
    incorrect_field_count: int = 0
    missing_field_count: int = 0
    duplicate_field_count: int = 0
    extra_field_count: int = 0
    field_completeness_ratio: Optional[float] = None
    field_correctness_ratio: Optional[float] = None
    # "accepted" | "incomplete" | "missing_response" -- see
    # labs.lab04_output_control.evaluation.
    answer_evaluation_status: str = "missing_response"
    overall_answer_accepted: bool = False


@dataclass(frozen=True)
class PlannedInvocationCall:
    """`question_index` is None for the single combined-call strategy (all
    three questions in one request) and 0..2 for the multiple-calls
    strategy (one question per request)."""

    execution_index: int
    strategy: str
    question_index: Optional[int]


@dataclass
class InvocationMeasurement:
    experiment_run_id: str
    strategy: str
    execution_index: int
    question_index: Optional[int]
    timestamp: str
    model: str
    max_completion_tokens: int
    reasoning_effort: str
    succeeded: bool
    attempts_used: int
    error: Optional[str] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    non_reasoning_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    finish_reason: Optional[str] = None
    output_truncated: Optional[bool] = None
    final_answer_char_count: Optional[int] = None

    # Per-request client latency (Part G.10) -- distinct from the
    # STRATEGY-level workflow_wall_clock_ms, which is not a per-call field
    # (see labs.lab04_output_control.stats.InvocationStrategySummary).
    client_elapsed_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None

    required_field_count: int = 0
    present_field_count: int = 0
    correct_field_count: int = 0
    incorrect_field_count: int = 0
    missing_field_count: int = 0
    duplicate_field_count: int = 0
    extra_field_count: int = 0
    field_completeness_ratio: Optional[float] = None
    field_correctness_ratio: Optional[float] = None
    answer_evaluation_status: str = "missing_response"
    overall_answer_accepted: bool = False
