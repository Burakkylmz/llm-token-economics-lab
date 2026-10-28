from __future__ import annotations

from labs.lab07_rate_limiting.models import AttemptRecord, ClientSimulationResult, OUTCOME_SUCCEEDED
from labs.lab07_rate_limiting.stats import group_by_scenario, summarize_strategy


def _client_result(client_id=0, succeeded=True, total_attempts=1, retry_count=0, rate_limit=0, wait=0.0, completion=None):
    return ClientSimulationResult(
        scenario_id="s", strategy_id="strat", simulation_run_id="run", client_id=client_id, succeeded=succeeded,
        outcome=OUTCOME_SUCCEEDED if succeeded else "FAILED_MAX_ATTEMPTS", total_attempts=total_attempts,
        retry_count=retry_count, rate_limit_response_count=rate_limit, total_scheduled_wait_seconds=wait,
        virtual_completion_time_seconds=completion,
    )


def _attempt(client_id=0, collision=False, http_status=429):
    return AttemptRecord(
        scenario_id="s", strategy_id="strat", simulation_run_id="run", client_id=client_id, attempt_number=1,
        retry_number=0, virtual_timestamp_seconds=0.0, http_status=http_status, retry_after_header=None,
        scheduled_delay_seconds=1.0, delay_reason="exponential_backoff", collision=collision,
    )


def test_success_rate_computed_correctly():
    results = [_client_result(0, succeeded=True), _client_result(1, succeeded=False)]
    summary = summarize_strategy("s", "strat", results, [])
    assert summary.success_rate == 0.5
    assert summary.successful_clients == 1
    assert summary.failed_clients == 1


def test_total_attempts_and_retry_count_summed():
    results = [_client_result(0, total_attempts=3, retry_count=2), _client_result(1, total_attempts=1, retry_count=0)]
    summary = summarize_strategy("s", "strat", results, [])
    assert summary.total_attempts == 4
    assert summary.total_retry_count == 2


def test_mean_and_max_wait_computed():
    results = [_client_result(0, wait=2.0), _client_result(1, wait=4.0)]
    summary = summarize_strategy("s", "strat", results, [])
    assert summary.mean_wait_seconds == 3.0
    assert summary.max_wait_seconds == 4.0
    assert summary.total_scheduled_wait_seconds == 6.0


def test_recovery_time_only_averages_successful_clients():
    results = [_client_result(0, succeeded=True, completion=5.0), _client_result(1, succeeded=False, completion=None)]
    summary = summarize_strategy("s", "strat", results, [])
    assert summary.mean_recovery_time_seconds == 5.0
    assert summary.completion_time_seconds == 5.0


def test_retry_collision_count_from_records():
    records = [_attempt(0, collision=True), _attempt(1, collision=False), _attempt(2, collision=True)]
    summary = summarize_strategy("s", "strat", [_client_result(0), _client_result(1), _client_result(2)], records)
    assert summary.retry_collision_count == 2


def test_single_client_scenario_has_genuinely_zero_collisions():
    records = [_attempt(0, collision=False)]
    summary = summarize_strategy("s", "strat", [_client_result(0)], records)
    assert summary.retry_collision_count == 0


def test_empty_client_list_does_not_crash():
    summary = summarize_strategy("s", "strat", [], [])
    assert summary.total_clients == 0
    assert summary.success_rate is None
    assert summary.mean_wait_seconds is None
    assert summary.max_wait_seconds is None


def test_group_by_scenario():
    from labs.lab07_rate_limiting.stats import StrategySummary

    summaries = [
        StrategySummary("s1", "a", 1, 1, 0, 1.0, 1, 0, 0, 1.0, 0.0, 0.0, 0.0, None, None, 0),
        StrategySummary("s2", "b", 1, 1, 0, 1.0, 1, 0, 0, 1.0, 0.0, 0.0, 0.0, None, None, 0),
    ]
    grouped = group_by_scenario(summaries)
    assert set(grouped.keys()) == {"s1", "s2"}
