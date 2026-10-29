"""Data shapes for Lab10's conversation-summarization experiment.

Exactly three conditions exist (Part 3.1) -- never more, never fewer:
  - CONDITION_FULL_HISTORY_BASELINE: the control condition, no summary call.
  - CONDITION_SUMMARIZATION_LOW / CONDITION_SUMMARIZATION_MEDIUM: one
    summary (reasoning_effort="low"/"medium") generated once per dataset
    and reused across every follow-up question in that condition.

SUMMARIZATION_EFFORTS is deliberately a fixed tuple, not a configuration
value: the whole A/B/C experiment design depends on exactly these two
summarization efforts existing, so it is a constant here rather than an
exposed TOML knob that could silently break the experiment's structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CONDITION_FULL_HISTORY_BASELINE = "full_history_baseline"
CONDITION_SUMMARIZATION_LOW = "summarization_low"
CONDITION_SUMMARIZATION_MEDIUM = "summarization_medium"
CONDITIONS = (CONDITION_FULL_HISTORY_BASELINE, CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM)
SUMMARY_CONDITIONS = (CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM)

SUMMARIZATION_EFFORTS = ("low", "medium")
CONDITION_TO_EFFORT = {CONDITION_SUMMARIZATION_LOW: "low", CONDITION_SUMMARIZATION_MEDIUM: "medium"}

CALL_PURPOSE_SUMMARY_GENERATION = "summary_generation"
CALL_PURPOSE_FOLLOWUP_ANSWER = "followup_answer"

SUMMARY_PENDING_GENERATION = "PENDING_GENERATION"


@dataclass(frozen=True)
class PlannedCall:
    execution_index: int
    dataset_id: str
    condition_id: str
    call_purpose: str
    reasoning_effort: str
    question_id: Optional[str] = None


@dataclass
class SummaryMeasurement:
    experiment_run_id: str
    execution_index: int
    dataset_id: str
    condition_id: str
    reasoning_effort: str
    timestamp: str
    model: str
    request_status: str
    error: Optional[str] = None

    summary_generation_prompt_tokens: Optional[int] = None
    summary_generation_completion_tokens: Optional[int] = None
    summary_generation_reasoning_tokens: Optional[int] = None
    summary_generation_non_reasoning_completion_tokens: Optional[int] = None
    summary_generation_cached_tokens: Optional[int] = None
    summary_generation_total_tokens: Optional[int] = None
    summary_generation_latency_ms: Optional[float] = None
    summary_generation_estimated_cost_usd: Optional[float] = None
    finish_reason: Optional[str] = None

    original_message_count: Optional[int] = None
    original_conversation_character_count: Optional[int] = None
    generated_summary_character_count: Optional[int] = None

    json_valid: bool = False
    schema_errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class FollowupMeasurement:
    experiment_run_id: str
    execution_index: int
    dataset_id: str
    condition_id: str
    question_id: str
    timestamp: str
    model: str
    reasoning_effort: str
    request_status: str
    error: Optional[str] = None

    followup_prompt_tokens: Optional[int] = None
    followup_completion_tokens: Optional[int] = None
    followup_reasoning_tokens: Optional[int] = None
    followup_non_reasoning_completion_tokens: Optional[int] = None
    followup_cached_tokens: Optional[int] = None
    followup_total_tokens: Optional[int] = None
    followup_latency_ms: Optional[float] = None
    followup_estimated_cost_usd: Optional[float] = None
    finish_reason: Optional[str] = None

    evaluation_status: str = "EVALUATION_UNAVAILABLE"
    answer_correct: Optional[bool] = None


@dataclass
class TraceRecord:
    """Full input/output inspection record (Part 8.2). `response_visible_content`
    holds only the final answer text for approved synthetic fixtures --
    never model reasoning traces, never secrets/API keys (Part 8.2/8.3)."""

    experiment_run_id: str
    dataset_id: str
    condition_id: str
    call_id: str
    call_purpose: str
    question_id: Optional[str]
    request_model: str
    request_parameters: dict
    request_messages: list
    response_visible_content: Optional[str]
    response_finish_reason: Optional[str]
    response_usage: dict
    response_latency_ms: Optional[float]
    evaluation_status: Optional[str]
