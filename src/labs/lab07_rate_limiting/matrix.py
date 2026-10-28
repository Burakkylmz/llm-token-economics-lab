"""Builds Lab07's simulation plan: profiles (pilot/default/extended) x
scenarios x strategies.

Unlike every live-API lab in this project, there is no API-call budget
here (Part L: "LAB07 tamamen offline olduğu için API çağrı bütçesi
gerekmiyor") -- the thing profiles actually bound is simulation workload
(client count, event count), enforced separately via
lab07.max_simulation_events/max_clients (see labs.lab07_rate_limiting.runner).
"""

from __future__ import annotations

from labs.lab07_rate_limiting.models import STRATEGIES, PlannedSimulationRun
from labs.lab07_rate_limiting.scenarios import (
    DEFAULT_SCENARIOS,
    EXTENDED_SCENARIOS,
    PILOT_SCENARIOS,
)

PROFILE_PILOT = "pilot"
PROFILE_DEFAULT = "default"
PROFILE_EXTENDED = "extended"
PROFILES = (PROFILE_PILOT, PROFILE_DEFAULT, PROFILE_EXTENDED)


def scenarios_for_profile(profile: str) -> tuple[str, ...]:
    if profile == PROFILE_PILOT:
        return PILOT_SCENARIOS
    if profile == PROFILE_EXTENDED:
        return EXTENDED_SCENARIOS
    return DEFAULT_SCENARIOS


def build_simulation_plan(scenario_ids, client_ids_by_scenario: dict[str, tuple[int, ...]], start_execution_index: int = 1) -> list[PlannedSimulationRun]:
    plan: list[PlannedSimulationRun] = []
    index = start_execution_index
    for scenario_id in scenario_ids:
        for strategy_id in STRATEGIES:
            plan.append(
                PlannedSimulationRun(
                    execution_index=index, scenario_id=scenario_id, strategy_id=strategy_id,
                    client_ids=client_ids_by_scenario[scenario_id],
                )
            )
            index += 1
    return plan
