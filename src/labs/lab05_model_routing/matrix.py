"""Builds Lab05's experiment plan: profiles (pilot/default/extended) x
strategies (always_small/always_large/rule_based_routing), paired and
interleaved per task.

Execution order alternates which strategy goes first every other task
(Part G's example: Task1 small-then-router, Task2 router-then-small, ...)
so no single strategy is systematically always-first or always-last across
the whole run -- this reduces, but per this lab's own documented
limitation, does NOT eliminate, time-dependent confounders (e.g. provider
load drifting over the course of a run).
"""

from __future__ import annotations

from typing import Sequence

from labs.lab05_model_routing.golden_dataset import task_ids_for_profile
from labs.lab05_model_routing.models import PlannedRoutingCall

STRATEGY_ALWAYS_SMALL = "always_small"
STRATEGY_ALWAYS_LARGE = "always_large"
STRATEGY_ROUTER = "rule_based_routing"

PROFILE_PILOT = "pilot"
PROFILE_DEFAULT = "default"
PROFILE_EXTENDED = "extended"
PROFILES = (PROFILE_PILOT, PROFILE_DEFAULT, PROFILE_EXTENDED)


def strategies_for_profile(profile: str) -> tuple[str, ...]:
    if profile == PROFILE_EXTENDED:
        return (STRATEGY_ALWAYS_SMALL, STRATEGY_ALWAYS_LARGE, STRATEGY_ROUTER)
    return (STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER)


def task_ids_for(profile: str) -> tuple[str, ...]:
    return task_ids_for_profile(profile)


def build_routing_plan(
    task_ids: Sequence[str], strategies: Sequence[str], start_execution_index: int = 1
) -> list[PlannedRoutingCall]:
    plan: list[PlannedRoutingCall] = []
    index = start_execution_index
    for i, task_id in enumerate(task_ids):
        order = strategies if i % 2 == 0 else tuple(reversed(strategies))
        for strategy in order:
            plan.append(PlannedRoutingCall(execution_index=index, strategy=strategy, task_id=task_id))
            index += 1
    return plan


def build_plan_for_profile(profile: str, start_execution_index: int = 1) -> list[PlannedRoutingCall]:
    return build_routing_plan(task_ids_for(profile), strategies_for_profile(profile), start_execution_index)
