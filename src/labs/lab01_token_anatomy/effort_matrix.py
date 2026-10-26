"""Builds the reasoning-effort condition set and execution plan for Lab01's
`effort` mode -- the single-factor analogue of
labs.lab01_token_anatomy.factorial_matrix, minus temperature/top_p.
"""

from __future__ import annotations

import itertools
import random
from typing import Sequence

from lab_core.config import Lab01EffortConfig
from labs.lab01_token_anatomy.experiments import EXPERIMENTS
from labs.lab01_token_anatomy.effort_models import EffortCondition, PlannedEffortCall, make_effort_condition_id


class EffortMatrixError(ValueError):
    """Raised when the effort configuration cannot be turned into a valid matrix."""


def build_conditions(prompts: Sequence[str], reasoning_efforts: Sequence[str]) -> list[EffortCondition]:
    """The exhaustive cross-product of prompt_type x reasoning_effort.

    Every prompt name must be one of Lab01's existing basic-mode experiments
    (short_prompt/long_prompt) -- reused verbatim, never reworded.
    """
    unknown_prompts = [p for p in prompts if p not in EXPERIMENTS]
    if unknown_prompts:
        raise EffortMatrixError(
            f"Unknown prompt type(s) {unknown_prompts}; must be one of {sorted(EXPERIMENTS.keys())}"
        )

    conditions: list[EffortCondition] = []
    seen_ids: set[str] = set()
    for prompt_type, effort in itertools.product(prompts, reasoning_efforts):
        condition_id = make_effort_condition_id(prompt_type, effort)
        if condition_id in seen_ids:
            raise EffortMatrixError(f"Duplicate condition_id '{condition_id}' -- duplicate prompt/effort values?")
        seen_ids.add(condition_id)
        conditions.append(EffortCondition(condition_id=condition_id, prompt_type=prompt_type, reasoning_effort=effort))
    return conditions


def build_conditions_from_config(config: Lab01EffortConfig) -> list[EffortCondition]:
    return build_conditions(config.prompts, config.reasoning_efforts)


def build_execution_plan(
    conditions: Sequence[EffortCondition],
    repetitions: int,
    randomize_execution_order: bool,
    randomization_seed: int,
    use_paired_api_seed: bool = False,
) -> list[PlannedEffortCall]:
    """Same ordering/reproducibility design as
    labs.lab01_token_anatomy.factorial_matrix.build_execution_plan: a pure
    function of its arguments, so resuming an interrupted run can re-derive
    the exact original plan deterministically.
    """
    if repetitions < 1:
        raise EffortMatrixError("repetitions must be >= 1")

    pairs = [(condition, repetition_index) for condition in conditions for repetition_index in range(1, repetitions + 1)]

    if randomize_execution_order:
        random.Random(randomization_seed).shuffle(pairs)

    plan: list[PlannedEffortCall] = []
    for index, (condition, repetition_index) in enumerate(pairs, start=1):
        api_seed = randomization_seed + repetition_index if use_paired_api_seed else None
        plan.append(
            PlannedEffortCall(
                execution_index=index,
                condition=condition,
                repetition_index=repetition_index,
                api_seed=api_seed,
            )
        )
    return plan


def build_plan_from_config(config: Lab01EffortConfig) -> list[PlannedEffortCall]:
    conditions = build_conditions_from_config(config)
    return build_execution_plan(
        conditions,
        repetitions=config.repetitions,
        randomize_execution_order=config.randomize_execution_order,
        randomization_seed=config.randomization_seed,
        use_paired_api_seed=config.use_paired_api_seed,
    )
