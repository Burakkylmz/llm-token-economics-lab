from __future__ import annotations

from labs.lab07_rate_limiting.matrix import build_simulation_plan, scenarios_for_profile
from labs.lab07_rate_limiting.models import STRATEGIES
from labs.lab07_rate_limiting.scenarios import (
    DEFAULT_SCENARIOS,
    EXTENDED_SCENARIOS,
    PILOT_SCENARIOS,
    SCENARIO_TRANSIENT_FAILURES,
)


def test_pilot_is_one_scenario():
    assert scenarios_for_profile("pilot") == PILOT_SCENARIOS
    assert len(PILOT_SCENARIOS) == 1


def test_default_is_three_scenarios():
    assert scenarios_for_profile("default") == DEFAULT_SCENARIOS
    assert len(DEFAULT_SCENARIOS) == 3


def test_extended_is_five_scenarios():
    assert scenarios_for_profile("extended") == EXTENDED_SCENARIOS
    assert len(EXTENDED_SCENARIOS) == 5


def test_pilot_plans_four_simulation_runs():
    scenario_ids = scenarios_for_profile("pilot")
    plan = build_simulation_plan(scenario_ids, {SCENARIO_TRANSIENT_FAILURES: (0,)})
    assert len(plan) == 4  # 1 scenario x 4 strategies


def test_default_plans_twelve_simulation_runs():
    scenario_ids = scenarios_for_profile("default")
    client_ids = {sid: (0,) for sid in scenario_ids}
    plan = build_simulation_plan(scenario_ids, client_ids)
    assert len(plan) == 12  # 3 scenarios x 4 strategies


def test_extended_plans_twenty_simulation_runs():
    scenario_ids = scenarios_for_profile("extended")
    client_ids = {sid: (0,) for sid in scenario_ids}
    plan = build_simulation_plan(scenario_ids, client_ids)
    assert len(plan) == 20  # 5 scenarios x 4 strategies


def test_every_scenario_gets_every_strategy():
    scenario_ids = ["s1", "s2"]
    client_ids = {sid: (0,) for sid in scenario_ids}
    plan = build_simulation_plan(scenario_ids, client_ids)
    for scenario_id in scenario_ids:
        strategies_seen = {p.strategy_id for p in plan if p.scenario_id == scenario_id}
        assert strategies_seen == set(STRATEGIES)


def test_execution_index_is_sequential():
    client_ids = {"s1": (0,)}
    plan = build_simulation_plan(["s1"], client_ids, start_execution_index=5)
    assert [p.execution_index for p in plan] == [5, 6, 7, 8]
