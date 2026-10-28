from __future__ import annotations

import random

from labs.lab07_rate_limiting.models import (
    OUTCOME_FAILED_DEADLINE,
    OUTCOME_FAILED_MAX_ATTEMPTS,
    OUTCOME_FAILED_RETRY_BUDGET,
    OUTCOME_SIMULATION_ERROR,
    OUTCOME_SUCCEEDED,
    STRATEGY_EXPONENTIAL_BACKOFF,
    STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER,
    STRATEGY_NO_BACKOFF,
    STRATEGY_RATE_LIMIT_AWARE,
)
from labs.lab07_rate_limiting.runner import simulate_scenario
from labs.lab07_rate_limiting.scenarios import ScenarioSpec
from labs.lab07_rate_limiting.server_simulator import SimulatedServer


def _run(spec, strategy, max_attempts=5, base_delay=1.0, max_delay=30.0, max_wait=None, deadline=None, seed=42, max_events=10_000):
    rng = random.Random(seed)
    return simulate_scenario(spec, strategy, "run-1", max_attempts, base_delay, max_delay, max_wait, deadline, rng, max_events)


def test_no_backoff_retries_with_zero_delay_and_succeeds():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=2), (0,))
    records, results = _run(spec, STRATEGY_NO_BACKOFF)
    assert results[0].succeeded is True
    assert results[0].retry_count == 2
    assert results[0].total_scheduled_wait_seconds == 0.0
    retry_records = [r for r in records if r.final_outcome is None]
    assert all(r.scheduled_delay_seconds == 0.0 for r in retry_records)


def test_exponential_backoff_delays_grow_correctly():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=3), (0,))
    records, results = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF, base_delay=1.0, max_delay=30.0)
    delays = [r.scheduled_delay_seconds for r in records if r.final_outcome is None]
    assert delays == [1.0, 2.0, 4.0]
    assert results[0].succeeded is True


def test_max_delay_cap_enforced():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=6), (0,))
    records, results = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF, max_attempts=8, base_delay=1.0, max_delay=5.0)
    delays = [r.scheduled_delay_seconds for r in records if r.final_outcome is None]
    assert max(delays) == 5.0


def test_same_seed_produces_same_jitter_delays():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=4), (0,))
    records_a, _ = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER, seed=7)
    spec2 = ScenarioSpec("transient_failures", SimulatedServer(fail_count=4), (0,))
    records_b, _ = _run(spec2, STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER, seed=7)
    delays_a = [r.scheduled_delay_seconds for r in records_a]
    delays_b = [r.scheduled_delay_seconds for r in records_b]
    assert delays_a == delays_b


def test_different_seeds_can_produce_different_jitter_delays():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=4), (0,))
    records_a, _ = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER, seed=1)
    spec2 = ScenarioSpec("transient_failures", SimulatedServer(fail_count=4), (0,))
    records_b, _ = _run(spec2, STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER, seed=2)
    delays_a = [r.scheduled_delay_seconds for r in records_a]
    delays_b = [r.scheduled_delay_seconds for r in records_b]
    assert delays_a != delays_b


def test_jitter_stays_within_nominal_bounds():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=4), (0,))
    records, _ = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER, base_delay=1.0, max_delay=100.0)
    nominal = [1.0, 2.0, 4.0, 8.0]
    delays = [r.scheduled_delay_seconds for r in records if r.final_outcome is None]
    for d, n in zip(delays, nominal):
        assert 0 <= d <= n


def test_rate_limit_aware_honors_valid_retry_after_header():
    spec = ScenarioSpec("retry_after_variations", SimulatedServer(retry_after_variants={0: "9"}), (0,))
    records, results = _run(spec, STRATEGY_RATE_LIMIT_AWARE)
    first_retry = next(r for r in records if r.final_outcome is None)
    assert first_retry.scheduled_delay_seconds == 9.0
    assert first_retry.delay_reason == "retry_after_header"
    assert results[0].succeeded is True


def test_rate_limit_aware_falls_back_on_missing_header():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=2), (0,))
    records, results = _run(spec, STRATEGY_RATE_LIMIT_AWARE, base_delay=1.0)
    first_retry = next(r for r in records if r.final_outcome is None)
    assert first_retry.delay_reason == "fallback_exponential_backoff_missing_header"
    assert first_retry.scheduled_delay_seconds == 1.0


def test_rate_limit_aware_falls_back_on_invalid_header_never_truncates():
    spec = ScenarioSpec("retry_after_variations", SimulatedServer(retry_after_variants={0: "not-a-number"}), (0,))
    records, results = _run(spec, STRATEGY_RATE_LIMIT_AWARE)
    first_retry = next(r for r in records if r.final_outcome is None)
    assert first_retry.delay_reason == "fallback_exponential_backoff_invalid_header"
    assert first_retry.scheduled_delay_seconds == 1.0  # exponential_backoff fallback, not a truncated header value


def test_max_attempts_exhaustion_sets_failed_max_attempts():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=100), (0,))
    records, results = _run(spec, STRATEGY_NO_BACKOFF, max_attempts=3)
    assert results[0].succeeded is False
    assert results[0].outcome == OUTCOME_FAILED_MAX_ATTEMPTS
    assert results[0].total_attempts == 3
    assert results[0].retry_count == 2  # the final failed attempt is NOT counted as an additional retry


def test_retry_budget_exceeded_sets_failed_retry_budget():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=100), (0,))
    records, results = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF, max_attempts=10, base_delay=10.0, max_delay=10.0, max_wait=15.0)
    assert results[0].outcome == OUTCOME_FAILED_RETRY_BUDGET
    assert results[0].succeeded is False


def test_simulation_deadline_exceeded_sets_failed_deadline():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=100), (0,))
    records, results = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF, max_attempts=10, base_delay=10.0, max_delay=10.0, deadline=15.0)
    assert results[0].outcome == OUTCOME_FAILED_DEADLINE


def test_simulation_event_ceiling_enforced():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=100), (0,))
    records, results = _run(spec, STRATEGY_NO_BACKOFF, max_attempts=100000, max_events=5)
    assert results[0].outcome == OUTCOME_SIMULATION_ERROR


def test_multi_client_shared_capacity_deterministic_tie_break():
    spec = ScenarioSpec("shared_contention", SimulatedServer(window_seconds=1.0, capacity_requests_per_window=1), (0, 1, 2))
    records, results = _run(spec, STRATEGY_NO_BACKOFF, max_attempts=2)
    # all three attempt at virtual_time=0.0 -- lower client_id wins the tie.
    first_attempts = sorted((r for r in records if r.attempt_number == 1), key=lambda r: r.client_id)
    assert first_attempts[0].http_status == 200
    assert first_attempts[1].http_status == 429
    assert first_attempts[2].http_status == 429
    assert sum(1 for r in first_attempts if r.collision) == 2


def test_retry_collisions_are_zero_for_single_client_scenarios():
    spec = ScenarioSpec("transient_failures", SimulatedServer(fail_count=2), (0,))
    records, results = _run(spec, STRATEGY_EXPONENTIAL_BACKOFF)
    assert all(r.collision is False for r in records)


def test_no_client_state_leakage_between_clients():
    spec = ScenarioSpec("shared_contention", SimulatedServer(window_seconds=1.0, capacity_requests_per_window=10), (0, 1))
    records, results = _run(spec, STRATEGY_NO_BACKOFF, max_attempts=3)
    client0_records = [r for r in records if r.client_id == 0]
    client1_records = [r for r in records if r.client_id == 1]
    assert len(client0_records) == 1  # succeeded on first attempt, no leaked state from client 1
    assert len(client1_records) == 1


def test_deterministic_scheduling_reproducible_across_runs():
    spec_a = ScenarioSpec("shared_contention", SimulatedServer(window_seconds=1.0, capacity_requests_per_window=2), tuple(range(5)))
    spec_b = ScenarioSpec("shared_contention", SimulatedServer(window_seconds=1.0, capacity_requests_per_window=2), tuple(range(5)))
    records_a, _ = _run(spec_a, STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER, seed=99)
    records_b, _ = _run(spec_b, STRATEGY_EXPONENTIAL_BACKOFF_WITH_JITTER, seed=99)
    assert [(r.client_id, r.attempt_number, r.http_status, r.scheduled_delay_seconds) for r in records_a] == [
        (r.client_id, r.attempt_number, r.http_status, r.scheduled_delay_seconds) for r in records_b
    ]
