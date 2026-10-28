"""Builds Lab08's concurrency-level execution plan.

Default order alternates which concurrency-level order runs first every
other repetition (Part E's example: repetition 1 runs levels low-to-high,
repetition 2 high-to-low, ...), the same reasoning this project already
applies in labs.lab06_streaming.matrix/labs.lab07_rate_limiting -- reduces,
but does not eliminate, temporal/provider-load confounders.
"""

from __future__ import annotations

import random

from labs.lab08_concurrency.experiments import DEFAULT_TASK_IDS, PILOT_TASK_IDS
from labs.lab08_concurrency.models import PlannedConcurrencyCall

PROFILE_PILOT = "pilot"
PROFILE_DEFAULT = "default"
PROFILE_EXTENDED = "extended"
PROFILES = (PROFILE_PILOT, PROFILE_DEFAULT, PROFILE_EXTENDED)

PILOT_LEVELS: tuple[int, ...] = (1, 3)
EXTENDED_LEVELS: tuple[int, ...] = (1, 2, 3, 5)


def levels_for_profile(profile: str, default_levels: tuple[int, ...]) -> tuple[int, ...]:
    if profile == PROFILE_PILOT:
        return PILOT_LEVELS
    if profile == PROFILE_EXTENDED:
        return EXTENDED_LEVELS
    return default_levels


def task_ids_for_profile(profile: str) -> tuple[str, ...]:
    return PILOT_TASK_IDS if profile == PROFILE_PILOT else DEFAULT_TASK_IDS


def build_plan(
    task_ids: tuple[str, ...],
    levels: tuple[int, ...],
    repetitions: int = 1,
    randomize: bool = False,
    seed: int = 42,
    start_execution_index: int = 1,
) -> list[PlannedConcurrencyCall]:
    if repetitions < 1:
        raise ValueError("repetitions must be a positive integer")

    plan: list[PlannedConcurrencyCall] = []
    index = start_execution_index
    rng = random.Random(seed) if randomize else None

    for repetition_index in range(1, repetitions + 1):
        if randomize:
            order = list(levels)
            rng.shuffle(order)
        else:
            order = list(levels) if repetition_index % 2 == 1 else list(reversed(levels))
        for level in order:
            for task_index, task_id in enumerate(task_ids):
                plan.append(
                    PlannedConcurrencyCall(
                        execution_index=index, repetition_index=repetition_index, max_concurrent=level,
                        task_id=task_id, task_index=task_index,
                    )
                )
                index += 1
    return plan
