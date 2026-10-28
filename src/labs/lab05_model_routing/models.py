"""Data shapes for Lab05's model-routing experiment.

Never stores the model's raw response text (same data-privacy policy as
every other redesigned lab in this project) -- only derived
correctness/completeness status and a char count. `matched_rules` is a
list of short rule-name strings (e.g. "multi_step_reasoning_indicators"),
never raw reasoning trace content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PlannedRoutingCall:
    execution_index: int
    strategy: str  # "always_small" | "always_large" | "rule_based_routing"
    task_id: str


@dataclass
class RoutingMeasurement:
    """One completed (successful or failed) attempt at a PlannedRoutingCall."""

    experiment_run_id: str
    execution_index: int
    strategy: str
    task_id: str
    task_category: str
    complexity_label: str
    timestamp: str

    selected_model: str
    model_tier: str
    routing_reason: str
    matched_rules: list = field(default_factory=list)
    policy_version: str = ""

    reasoning_effort: str = ""
    max_completion_tokens: int = 0
    succeeded: bool = False
    attempts_used: int = 0
    error: Optional[str] = None

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

    estimated_input_cost_usd: Optional[float] = None
    estimated_output_cost_usd: Optional[float] = None
    estimated_cost_usd: Optional[float] = None

    # "CORRECT" | "INCORRECT" | "INCOMPLETE" | "INVALID_FORMAT" | "TRUNCATED" |
    # "EVALUATION_UNAVAILABLE" | "STRUCTURAL_VALIDATION_ONLY". See
    # labs.lab05_model_routing.evaluation.
    evaluation_status: str = "EVALUATION_UNAVAILABLE"
    # True only for a genuinely verified correct answer; None for
    # structural-validation-only (code) tasks and for missing responses --
    # never fabricated as True/False when unknown.
    answer_correct: Optional[bool] = None
    answer_complete: Optional[bool] = None
    evaluation_detail: str = ""
