"""Per-(scenario, strategy) summary statistics for Lab07.

Reliability (success_rate, recovery time) and efficiency (total_attempts,
total_429_responses, total_scheduled_wait_seconds) are reported as
SEPARATE fields, never collapsed into one score -- Part K's explicit
rule that the strategy with the shortest wait time is never, by that fact
alone, the best strategy.

`retry_collision_count` is computed directly from the attempt records'
own `collision` flag (set only by shared-capacity contention in
labs.lab07_rate_limiting.server_simulator) -- for a single-client scenario
this is always, genuinely, 0, never a placeholder.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from labs.lab07_rate_limiting.models import AttemptRecord, ClientSimulationResult


@dataclass
class StrategySummary:
    scenario_id: str
    strategy_id: str
    total_clients: int
    successful_clients: int
    failed_clients: int
    success_rate: Optional[float]
    total_attempts: int
    total_retry_count: int
    total_429_responses: int
    mean_attempts_per_client: Optional[float]
    total_scheduled_wait_seconds: float
    mean_wait_seconds: Optional[float]
    max_wait_seconds: Optional[float]
    mean_recovery_time_seconds: Optional[float]
    completion_time_seconds: Optional[float]
    retry_collision_count: int


def summarize_strategy(
    scenario_id: str, strategy_id: str, client_results: Sequence[ClientSimulationResult], records: Sequence[AttemptRecord]
) -> StrategySummary:
    total = len(client_results)
    successes = [c for c in client_results if c.succeeded]
    waits = [c.total_scheduled_wait_seconds for c in client_results]
    recovery_times = [c.virtual_completion_time_seconds for c in successes if c.virtual_completion_time_seconds is not None]

    return StrategySummary(
        scenario_id=scenario_id,
        strategy_id=strategy_id,
        total_clients=total,
        successful_clients=len(successes),
        failed_clients=total - len(successes),
        success_rate=(len(successes) / total) if total else None,
        total_attempts=sum(c.total_attempts for c in client_results),
        total_retry_count=sum(c.retry_count for c in client_results),
        total_429_responses=sum(c.rate_limit_response_count for c in client_results),
        mean_attempts_per_client=(sum(c.total_attempts for c in client_results) / total) if total else None,
        total_scheduled_wait_seconds=sum(waits),
        mean_wait_seconds=(sum(waits) / total) if total else None,
        max_wait_seconds=max(waits) if waits else None,
        mean_recovery_time_seconds=(sum(recovery_times) / len(recovery_times)) if recovery_times else None,
        completion_time_seconds=max(recovery_times) if recovery_times else None,
        retry_collision_count=sum(1 for r in records if r.collision),
    )


def summarize_all(
    plan_results: Sequence[tuple[str, str, Sequence[ClientSimulationResult], Sequence[AttemptRecord]]],
) -> list[StrategySummary]:
    return [summarize_strategy(scenario_id, strategy_id, client_results, records) for scenario_id, strategy_id, client_results, records in plan_results]


def group_by_scenario(summaries: Sequence[StrategySummary]) -> dict[str, list[StrategySummary]]:
    grouped: dict[str, list[StrategySummary]] = defaultdict(list)
    for s in summaries:
        grouped[s.scenario_id].append(s)
    return dict(grouped)
