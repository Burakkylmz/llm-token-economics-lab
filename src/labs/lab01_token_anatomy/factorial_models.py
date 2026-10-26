"""Data shapes for Lab01's full-factorial experiment engine.

Kept separate from lab_core.metrics's ExperimentResult/TokenCostMetrics on
purpose: those were designed around a single flat call per basic-mode
repetition, shared by every lab. A factorial condition carries its own
identity (condition_id, the four independent variables, execution/repetition
indices) that is much more useful as first-class, flat columns in
raw_measurements.csv than buried inside a generic `model_parameters` dict --
which is exactly what forcing this into ExperimentResult would do. The actual
per-call work (the Groq SDK call, cost calculation, reasoning-token
derivation) still goes through the existing lab_core.groq_client /
labs.lab01_token_anatomy.cost / .observability helpers; only the *record
shape* is new.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def format_condition_param(value: float) -> str:
    """Canonical 2-decimal string for a condition_id component.

    All factorial parameter values used by this project (0.5/1.0/1.5,
    0.7/0.9/1.0, ...) are exactly representable at 2 decimal places. Rounding
    here -- rather than relying on however a float happens to repr -- is what
    guarantees the same (prompt_type, effort, temperature, top_p) always
    produces the exact same condition_id string, regardless of
    floating-point formatting noise.
    """
    return f"{round(value, 2):.2f}"


def make_condition_id(prompt_type: str, reasoning_effort: str, temperature: float, top_p: float) -> str:
    return (
        f"{prompt_type}__{reasoning_effort}"
        f"__temp_{format_condition_param(temperature)}"
        f"__topp_{format_condition_param(top_p)}"
    )


@dataclass(frozen=True)
class FactorialCondition:
    """One unique point in the factorial matrix (before repetitions)."""

    condition_id: str
    prompt_type: str
    reasoning_effort: str
    temperature: float
    top_p: float


@dataclass(frozen=True)
class PlannedCall:
    """One planned API call: a condition plus which repetition of it this is.

    `execution_index` is this call's 1-based position in the (possibly
    randomized) execution order -- distinct from `repetition_index`, which
    identifies the repetition within its condition independent of ordering.
    """

    execution_index: int
    condition: FactorialCondition
    repetition_index: int
    api_seed: Optional[int]


@dataclass
class FactorialMeasurement:
    """One completed (successful or failed) attempt at a PlannedCall.

    Exactly one FactorialMeasurement is ever recorded per PlannedCall, even
    if the underlying API call needed several retry attempts internally
    (`attempts_used` records that count) -- retries never produce duplicate
    successful rows.
    """

    experiment_run_id: str
    condition_id: str
    execution_index: int
    repetition_index: int
    timestamp: str
    prompt_type: str
    model: str
    reasoning_effort: str
    temperature: float
    top_p: float
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

    # Quality signal: deliberately NOT a general "correctness" claim -- see
    # labs.lab01_token_anatomy.factorial_runner.evaluate_quality_heuristic.
    quality_evaluation_method: str = "not_evaluated"
    quality_heuristic_passed: Optional[bool] = None
