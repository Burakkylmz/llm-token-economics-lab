"""Builds the full-factorial condition set and execution plan for Lab01.

Two separate, pure, independently-testable steps:
  1. build_conditions(): the exhaustive set of unique (prompt_type,
     reasoning_effort, temperature, top_p) points -- condition IDENTITY,
     independent of how many times each is repeated or in what order.
  2. build_execution_plan(): repeats each condition `repetitions` times and
     decides the order those calls actually run in (optionally randomized,
     always reproducible from the same seed).
"""

from __future__ import annotations

import itertools
import random
from typing import Sequence

from lab_core.config import Lab01FactorialConfig
from labs.lab01_token_anatomy.experiments import EXPERIMENTS
from labs.lab01_token_anatomy.factorial_models import FactorialCondition, PlannedCall, make_condition_id


class FactorialMatrixError(ValueError):
    """Raised when the factorial configuration cannot be turned into a valid matrix."""


def build_conditions(
    prompts: Sequence[str],
    reasoning_efforts: Sequence[str],
    temperatures: Sequence[float],
    top_p_values: Sequence[float],
) -> list[FactorialCondition]:
    """The exhaustive, deduplicated cross-product of all four factors.

    Every prompt name must be one of Lab01's existing basic-mode experiments
    (short_prompt/long_prompt) -- the factorial engine reuses those exact
    prompt strings (see build_messages_for_condition in factorial_runner.py),
    it does not invent new prompt text.
    """
    unknown_prompts = [p for p in prompts if p not in EXPERIMENTS]
    if unknown_prompts:
        raise FactorialMatrixError(
            f"Unknown prompt type(s) {unknown_prompts}; must be one of {sorted(EXPERIMENTS.keys())}"
        )

    conditions: list[FactorialCondition] = []
    seen_ids: dict[str, tuple] = {}
    for prompt_type, effort, temperature, top_p in itertools.product(
        prompts, reasoning_efforts, temperatures, top_p_values
    ):
        condition_id = make_condition_id(prompt_type, effort, temperature, top_p)
        existing = seen_ids.get(condition_id)
        if existing is not None:
            raise FactorialMatrixError(
                f"condition_id collision: '{condition_id}' was generated for both "
                f"{existing} and {(prompt_type, effort, temperature, top_p)} -- two "
                "distinct parameter combinations rounded to the same 2-decimal id."
            )
        seen_ids[condition_id] = (prompt_type, effort, temperature, top_p)
        conditions.append(
            FactorialCondition(
                condition_id=condition_id,
                prompt_type=prompt_type,
                reasoning_effort=effort,
                temperature=temperature,
                top_p=top_p,
            )
        )
    return conditions


def build_conditions_from_config(config: Lab01FactorialConfig) -> list[FactorialCondition]:
    return build_conditions(config.prompts, config.reasoning_efforts, config.temperatures, config.top_p_values)


def build_execution_plan(
    conditions: Sequence[FactorialCondition],
    repetitions: int,
    randomize_execution_order: bool,
    randomization_seed: int,
    use_paired_api_seed: bool = False,
) -> list[PlannedCall]:
    """Expand each condition into `repetitions` calls and order them.

    The canonical (pre-randomization) order is condition-major: all
    repetitions of condition 1, then all of condition 2, etc. When
    `randomize_execution_order` is True, that list is shuffled with a
    `random.Random(randomization_seed)` instance -- a pure function of the
    seed, so the exact same plan (and therefore the exact same execution
    order) is reproduced by calling this again with the same arguments. This
    is also what makes resuming an interrupted run safe: re-deriving the
    full plan and filtering out already-completed calls reconstructs the
    original order deterministically, with no separate "order" file needed.

    `api_seed` (the Groq `seed` request parameter, NOT `randomization_seed`
    above) is only ever set when `use_paired_api_seed` is True, in which case
    every call sharing the same repetition_index across ALL conditions gets
    `randomization_seed + repetition_index` as its api_seed -- a "paired
    seed" design intended to make cross-condition comparisons slightly less
    confounded by whatever that seed happens to draw. Per Groq's own API
    docs, this is a best-effort determinism hint, never a guarantee.
    """
    if repetitions < 1:
        raise FactorialMatrixError("repetitions must be >= 1")

    pairs = [(condition, repetition_index) for condition in conditions for repetition_index in range(1, repetitions + 1)]

    if randomize_execution_order:
        random.Random(randomization_seed).shuffle(pairs)

    plan: list[PlannedCall] = []
    for index, (condition, repetition_index) in enumerate(pairs, start=1):
        api_seed = randomization_seed + repetition_index if use_paired_api_seed else None
        plan.append(
            PlannedCall(
                execution_index=index,
                condition=condition,
                repetition_index=repetition_index,
                api_seed=api_seed,
            )
        )
    return plan


def build_plan_from_config(config: Lab01FactorialConfig) -> list[PlannedCall]:
    conditions = build_conditions_from_config(config)
    return build_execution_plan(
        conditions,
        repetitions=config.repetitions,
        randomize_execution_order=config.randomize_execution_order,
        randomization_seed=config.randomization_seed,
        use_paired_api_seed=config.use_paired_api_seed,
    )
