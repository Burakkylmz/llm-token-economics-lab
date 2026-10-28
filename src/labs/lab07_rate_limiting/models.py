"""Data shapes and constants for Lab07's retry-strategy simulation.

Every measurement here is synthetic simulation data, never real Groq
telemetry -- see the module docstring in labs.lab07_rate_limiting.runner
for the explicit "real_api_calls = 0 / real_token_usage = None"
measurement-integrity rule this schema exists to make easy to honor (there
is simply no field here that could be mistaken for a real provider usage
field).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

STRATEGY_NO_BACKOFF = "no_backoff"
STRATEGY_EXPONENTIAL_BACKOFF = "exponential_backoff"
STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER = "exponential_backoff_with_jitter"
STRATEGY_RATE_LIMIT_AWARE = "rate_limit_aware"

STRATEGIES: tuple[str, ...] = (
    STRATEGY_NO_BACKOFF,
    STRATEGY_EXPONENTIAL_BACKOFF,
    STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER,
    STRATEGY_RATE_LIMIT_AWARE,
)

OUTCOME_SUCCEEDED = "SUCCEEDED"
OUTCOME_FAILED_MAX_ATTEMPTS = "FAILED_MAX_ATTEMPTS"
OUTCOME_FAILED_RETRY_BUDGET = "FAILED_RETRY_BUDGET"
OUTCOME_FAILED_DEADLINE = "FAILED_DEADLINE"
OUTCOME_FAILED_NON_RETRYABLE_ERROR = "FAILED_NON_RETRYABLE_ERROR"
OUTCOME_SIMULATION_ERROR = "SIMULATION_ERROR"

DELAY_REASON_NO_BACKOFF = "no_backoff"
DELAY_REASON_EXPONENTIAL_BACKOFF = "exponential_backoff"
DELAY_REASON_EXPONENTIAL_BACKOFF_JITTER = "exponential_backoff_jitter"
DELAY_REASON_RETRY_AFTER_HEADER = "retry_after_header"
DELAY_REASON_FALLBACK_MISSING_HEADER = "fallback_exponential_backoff_missing_header"
DELAY_REASON_FALLBACK_INVALID_HEADER = "fallback_exponential_backoff_invalid_header"
DELAY_REASON_NONE = "none"


@dataclass(frozen=True)
class PlannedSimulationRun:
    """One (scenario, strategy) combination -- the unit the CLI/matrix
    plans and persists per-run results for. `client_ids` is the full list
    of simulated clients this run will execute (length 1 for the
    single-client scenarios)."""

    execution_index: int
    scenario_id: str
    strategy_id: str
    client_ids: tuple[int, ...]


@dataclass
class AttemptRecord:
    """One simulated attempt -- never a real HTTP request. Bounded,
    auditable fields only; `retry_after_header` is the raw synthetic
    string the simulated server returned (or None), never a parsed/
    real-world value."""

    scenario_id: str
    strategy_id: str
    simulation_run_id: str
    client_id: int
    attempt_number: int
    retry_number: int
    virtual_timestamp_seconds: float
    http_status: int
    retry_after_header: Optional[str]
    scheduled_delay_seconds: Optional[float]
    delay_reason: str
    # Set only on a client's LAST attempt record; None on every earlier one.
    final_outcome: Optional[str] = None
    collision: bool = False


@dataclass
class ClientSimulationResult:
    scenario_id: str
    strategy_id: str
    simulation_run_id: str
    client_id: int
    succeeded: bool
    outcome: str
    total_attempts: int
    retry_count: int
    rate_limit_response_count: int
    total_scheduled_wait_seconds: float
    # Virtual time at which this client's successful response arrived --
    # None for a client that never succeeded.
    virtual_completion_time_seconds: Optional[float]
