"""Builds the context-condition execution plan for Lab02 -- the
context-growth analogue of labs.lab01_token_anatomy.effort_matrix, minus
the prompt_type/reasoning_effort cross-product (Lab02 has exactly one
factor: which of the three context conditions this call uses).

`multi_turn_growth` is intentionally never part of this plan: it is a
single, inherently sequential conversation (each turn depends on the
previous turn's reply), so it is always executed after the (optionally
randomized/interleaved) context-condition plan below, never mixed into it.
"""

from __future__ import annotations

import random
from typing import Sequence

from labs.lab02_context_growth.experiments import CONTEXT_CONDITION_ORDER
from labs.lab02_context_growth.models import PlannedContextCall


class ContextMatrixError(ValueError):
    """Raised when the context plan configuration is invalid."""


def build_context_plan(
    condition_ids: Sequence[str],
    repetitions: int,
    randomize_context_order: bool = False,
    randomization_seed: int = 42,
) -> list[PlannedContextCall]:
    """One PlannedContextCall per (condition, repetition) pair.

    A pure function of its arguments (same pattern as
    labs.lab01_token_anatomy.effort_matrix.build_execution_plan): the same
    inputs always produce the exact same plan, so a dry-run plan and the
    plan actually executed are guaranteed identical.
    """
    unknown = [c for c in condition_ids if c not in CONTEXT_CONDITION_ORDER]
    if unknown:
        raise ContextMatrixError(
            f"Unknown context condition id(s) {unknown}; must be one of {list(CONTEXT_CONDITION_ORDER)}"
        )
    if repetitions < 1:
        raise ContextMatrixError("repetitions must be >= 1")

    pairs = [(condition_id, rep) for condition_id in condition_ids for rep in range(1, repetitions + 1)]

    if randomize_context_order:
        random.Random(randomization_seed).shuffle(pairs)

    return [
        PlannedContextCall(execution_index=index, condition_id=condition_id, repetition_index=rep)
        for index, (condition_id, rep) in enumerate(pairs, start=1)
    ]
