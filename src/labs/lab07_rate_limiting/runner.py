"""The deterministic, virtual-time retry simulation engine.

Measurement integrity (Part K, non-negotiable): this module never imports
`groq`, never constructs a real client, never performs real networking,
and never sleeps on a real wall clock. Every number it produces is
synthetic simulation data:

    real_api_calls = 0
    real_token_usage = None
    real_api_cost = 0

A single engine (`simulate_scenario`) drives BOTH single-client scenarios
(Transient Failures, Server Recovery Window, Retry-After Variations) and
multi-client scenarios (Shared Contention, Quota Pressure) -- a
single-client "scenario" is simply the N=1 case of the same discrete-event
simulation, so there is no separate code path to keep in sync, and the
`retry_collisions` metric is correctly always 0 for N=1 (a true fact about
a system with only one client, never a fabricated placeholder).

Determinism/tie-breaking (Part F): events are processed in
`(virtual_time, client_id)` order -- Python's tuple comparison in the
heap breaks exact-time ties by ascending client_id, a fixed, documented
policy, not an accident of dict/set iteration order.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Optional

from lab_core.retry import compute_backoff_delay_seconds
from labs.lab07_rate_limiting.models import (
    DELAY_REASON_EXPONENTIAL_BACKOFF,
    DELAY_REASON_EXPONENTIAL_BACKOFF_JITTER,
    DELAY_REASON_FALLBACK_INVALID_HEADER,
    DELAY_REASON_FALLBACK_MISSING_HEADER,
    DELAY_REASON_NO_BACKOFF,
    DELAY_REASON_NONE,
    DELAY_REASON_RETRY_AFTER_HEADER,
    OUTCOME_FAILED_DEADLINE,
    OUTCOME_FAILED_MAX_ATTEMPTS,
    OUTCOME_FAILED_RETRY_BUDGET,
    OUTCOME_SIMULATION_ERROR,
    OUTCOME_SUCCEEDED,
    STRATEGY_EXPONENTIAL_BACKOFF,
    STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER,
    STRATEGY_NO_BACKOFF,
    STRATEGY_RATE_LIMIT_AWARE,
    AttemptRecord,
    ClientSimulationResult,
)
from labs.lab07_rate_limiting.retry_after import parse_retry_after
from labs.lab07_rate_limiting.scenarios import ScenarioSpec, virtual_time_to_utc


def _decide_delay(
    strategy_id: str,
    attempt_index: int,
    retry_after_header: Optional[str],
    virtual_time: float,
    base_delay_seconds: float,
    max_delay_seconds: float,
    rng: random.Random,
) -> tuple[float, str]:
    """Returns (delay_seconds, delay_reason). Never silently shortens a
    valid-but-large Retry-After value -- an "excessive"/unusable header
    always falls back to the documented exponential-backoff policy
    instead (Part I's explicit rule)."""
    if strategy_id == STRATEGY_NO_BACKOFF:
        return 0.0, DELAY_REASON_NO_BACKOFF

    if strategy_id == STRATEGY_RATE_LIMIT_AWARE:
        parsed = parse_retry_after(retry_after_header, now_utc=virtual_time_to_utc(virtual_time))
        if parsed.valid:
            return parsed.delay_seconds, DELAY_REASON_RETRY_AFTER_HEADER
        fallback_delay = compute_backoff_delay_seconds(attempt_index, base_delay_seconds, max_delay_seconds, jitter=False)
        reason = DELAY_REASON_FALLBACK_MISSING_HEADER if retry_after_header is None else DELAY_REASON_FALLBACK_INVALID_HEADER
        return fallback_delay, reason

    jitter = strategy_id == STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER
    delay = compute_backoff_delay_seconds(attempt_index, base_delay_seconds, max_delay_seconds, jitter=jitter, rng=rng if jitter else None)
    return delay, (DELAY_REASON_EXPONENTIAL_BACKOFF_JITTER if jitter else DELAY_REASON_EXPONENTIAL_BACKOFF)


@dataclass
class _ClientState:
    attempt_index: int = 1
    retry_count: int = 0
    rate_limit_response_count: int = 0
    total_scheduled_wait_seconds: float = 0.0
    done: bool = False
    outcome: Optional[str] = None
    virtual_completion_time_seconds: Optional[float] = None


def simulate_scenario(
    spec: ScenarioSpec,
    strategy_id: str,
    simulation_run_id: str,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    max_total_wait_seconds: Optional[float],
    simulation_deadline_seconds: Optional[float],
    rng: random.Random,
    max_simulation_events: int = 10_000,
) -> tuple[list[AttemptRecord], list[ClientSimulationResult]]:
    effective_max_wait = (
        spec.max_total_wait_seconds_override if spec.max_total_wait_seconds_override is not None else max_total_wait_seconds
    )

    state: dict[int, _ClientState] = {cid: _ClientState() for cid in spec.client_ids}
    heap: list[tuple[float, int]] = [(0.0, cid) for cid in spec.client_ids]
    heapq.heapify(heap)
    records: list[AttemptRecord] = []
    events_processed = 0
    event_ceiling_hit = False

    while heap:
        if events_processed >= max_simulation_events:
            event_ceiling_hit = True
            break
        events_processed += 1

        virtual_time, client_id = heapq.heappop(heap)
        client_state = state[client_id]
        if client_state.done:
            continue

        attempt_index = client_state.attempt_index
        response = spec.server.handle(virtual_time, client_id, attempt_index, spec.token_demand)
        retry_number = attempt_index - 1
        succeeded = response.http_status == 200

        if succeeded:
            client_state.done = True
            client_state.outcome = OUTCOME_SUCCEEDED
            client_state.virtual_completion_time_seconds = virtual_time
            records.append(
                AttemptRecord(
                    spec.scenario_id, strategy_id, simulation_run_id, client_id, attempt_index, retry_number,
                    virtual_time, response.http_status, response.retry_after_header, None, DELAY_REASON_NONE,
                    final_outcome=OUTCOME_SUCCEEDED, collision=response.collision,
                )
            )
            continue

        client_state.rate_limit_response_count += 1

        if attempt_index >= max_attempts:
            client_state.done = True
            client_state.outcome = OUTCOME_FAILED_MAX_ATTEMPTS
            records.append(
                AttemptRecord(
                    spec.scenario_id, strategy_id, simulation_run_id, client_id, attempt_index, retry_number,
                    virtual_time, response.http_status, response.retry_after_header, None, DELAY_REASON_NONE,
                    final_outcome=OUTCOME_FAILED_MAX_ATTEMPTS, collision=response.collision,
                )
            )
            continue

        delay, delay_reason = _decide_delay(
            strategy_id, attempt_index, response.retry_after_header, virtual_time, base_delay_seconds, max_delay_seconds, rng
        )

        if effective_max_wait is not None and client_state.total_scheduled_wait_seconds + delay > effective_max_wait:
            client_state.done = True
            client_state.outcome = OUTCOME_FAILED_RETRY_BUDGET
            records.append(
                AttemptRecord(
                    spec.scenario_id, strategy_id, simulation_run_id, client_id, attempt_index, retry_number,
                    virtual_time, response.http_status, response.retry_after_header, delay, delay_reason,
                    final_outcome=OUTCOME_FAILED_RETRY_BUDGET, collision=response.collision,
                )
            )
            continue

        if simulation_deadline_seconds is not None and virtual_time + delay > simulation_deadline_seconds:
            client_state.done = True
            client_state.outcome = OUTCOME_FAILED_DEADLINE
            records.append(
                AttemptRecord(
                    spec.scenario_id, strategy_id, simulation_run_id, client_id, attempt_index, retry_number,
                    virtual_time, response.http_status, response.retry_after_header, delay, delay_reason,
                    final_outcome=OUTCOME_FAILED_DEADLINE, collision=response.collision,
                )
            )
            continue

        records.append(
            AttemptRecord(
                spec.scenario_id, strategy_id, simulation_run_id, client_id, attempt_index, retry_number,
                virtual_time, response.http_status, response.retry_after_header, delay, delay_reason,
                final_outcome=None, collision=response.collision,
            )
        )
        client_state.retry_count += 1
        client_state.total_scheduled_wait_seconds += delay
        client_state.attempt_index += 1
        heapq.heappush(heap, (virtual_time + delay, client_id))

    if event_ceiling_hit:
        for client_id, client_state in state.items():
            if not client_state.done:
                client_state.done = True
                client_state.outcome = OUTCOME_SIMULATION_ERROR

    results = [
        ClientSimulationResult(
            scenario_id=spec.scenario_id,
            strategy_id=strategy_id,
            simulation_run_id=simulation_run_id,
            client_id=client_id,
            succeeded=client_state.outcome == OUTCOME_SUCCEEDED,
            outcome=client_state.outcome or OUTCOME_SIMULATION_ERROR,
            total_attempts=client_state.attempt_index,
            retry_count=client_state.retry_count,
            rate_limit_response_count=client_state.rate_limit_response_count,
            total_scheduled_wait_seconds=client_state.total_scheduled_wait_seconds,
            virtual_completion_time_seconds=client_state.virtual_completion_time_seconds,
        )
        for client_id, client_state in state.items()
    ]
    return records, results
