"""Data shapes for Lab02's context-growth/context-bloat experiment.

Two measurement kinds, kept as separate dataclasses (not shoehorned into
one shape) because they answer different research questions and carry
different identity: ContextMeasurement is one repetition of one
(minimal/expanded/irrelevant)_context condition, answering RQ1-RQ3/RQ5;
TurnMeasurement is one turn of the single growing multi-turn conversation,
answering RQ4. Both deliberately never store the model's raw response text
(see labs.lab02_context_growth.evaluation) -- only a derived correctness
status/char count, consistent with this project's data-privacy stance
(see labs.lab01_token_anatomy.effort_models for the same policy there).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PlannedContextCall:
    """One planned API call: a context condition plus which repetition this is."""

    execution_index: int
    condition_id: str
    repetition_index: int


@dataclass
class ContextMeasurement:
    """One completed (successful or failed) attempt at a PlannedContextCall."""

    experiment_run_id: str
    condition_id: str
    execution_index: int
    repetition_index: int
    timestamp: str
    model: str
    max_completion_tokens: int
    succeeded: bool
    attempts_used: int
    error: Optional[str] = None

    context_chars: Optional[int] = None
    expected_answer: Optional[str] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    non_reasoning_completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    finish_reason: Optional[str] = None
    system_fingerprint: Optional[str] = None
    final_answer_char_count: Optional[int] = None

    total_latency_ms: Optional[float] = None
    provider_reported_processing_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None

    # "correct" | "incorrect" | "format_invalid" | "missing_response" |
    # "unavailable" (legacy data with no stored answer text to evaluate).
    # See labs.lab02_context_growth.evaluation for the exact rules.
    answer_evaluation_status: str = "unavailable"
    # True only for "correct"; False for "incorrect"/"format_invalid"; None
    # for "missing_response"/"unavailable" -- API success and answer
    # correctness are deliberately different concepts (see evaluation.py).
    answer_correct: Optional[bool] = None


@dataclass
class TurnMeasurement:
    """One turn of Lab02's single growing multi-turn conversation.

    `messages_count` is the number of chat messages sent TO the API on this
    turn's request (i.e. including this turn's new user message but before
    its assistant reply is appended) -- a direct, model-agnostic proxy for
    how much conversation history is being re-sent.
    """

    experiment_run_id: str
    turn_index: int
    timestamp: str
    model: str
    max_completion_tokens: int
    messages_count: int
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
    total_latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None

    # Derived, cumulative-growth metrics (see
    # labs.lab02_context_growth.runner.execute_multi_turn_growth).
    incremental_prompt_tokens: Optional[int] = None
    cumulative_prompt_tokens: Optional[int] = None
    cumulative_total_tokens: Optional[int] = None
    cumulative_estimated_cost_usd: Optional[float] = None
