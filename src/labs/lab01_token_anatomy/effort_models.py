"""Data shapes for Lab01's `effort` execution mode: a reasoning_effort-only
comparison experiment.

Replaces the temperature x top_p x reasoning_effort full factorial engine
(labs.lab01_token_anatomy.factorial_models, retired -- see
labs.lab01_token_anatomy.cli._run_legacy_factorial_stub) with a leaner
design: `temperature` and `top_p` are not independent variables here at
all, not fields set to some default marker -- they are simply never part
of this experiment's condition identity or request, because provider
defaults apply and this project never fabricates a value it didn't
explicitly send (see effort_runner.build_messages_and_params_for_condition).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def make_effort_condition_id(prompt_type: str, reasoning_effort: str) -> str:
    """Deterministic, collision-free by construction: both inputs are a
    small, fixed set of strings (no floats involved), unlike the retired
    factorial engine's condition_id which needed float-rounding safeguards."""
    return f"{prompt_type}__{reasoning_effort}"


@dataclass(frozen=True)
class EffortCondition:
    """One unique (prompt_type, reasoning_effort) point -- before repetitions."""

    condition_id: str
    prompt_type: str
    reasoning_effort: str


@dataclass(frozen=True)
class PlannedEffortCall:
    """One planned API call: a condition plus which repetition of it this is."""

    execution_index: int
    condition: EffortCondition
    repetition_index: int
    api_seed: Optional[int]


@dataclass
class EffortMeasurement:
    """One completed (successful or failed) attempt at a PlannedEffortCall.

    Field set mirrors labs.lab01_token_anatomy.factorial_models.
    FactorialMeasurement minus `temperature`/`top_p` -- every other
    observability guarantee (None vs 0, no raw text persisted, no double
    counting) carries over unchanged.
    """

    experiment_run_id: str
    condition_id: str
    execution_index: int
    repetition_index: int
    timestamp: str
    prompt_type: str
    model: str
    reasoning_effort: str
    max_completion_tokens: int
    api_seed: Optional[int]
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
    system_fingerprint: Optional[str] = None
    final_answer_char_count: Optional[int] = None

    total_latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None

    quality_evaluation_method: str = "not_evaluated"
    quality_heuristic_passed: Optional[bool] = None
