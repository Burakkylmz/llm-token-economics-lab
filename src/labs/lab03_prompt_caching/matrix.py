"""Builds Lab03's paired/interleaved execution plan.

The key structural fix over the old design: the OUTER loop is over pairs
(repetitions), and the INNER loop is over conditions -- so the plan is
naturally "Pair 1: repeated_prefix, cache_unfriendly; Pair 2: ...", never
"all repeated_prefix calls, then all cache_unfriendly calls" (the old
block-sequential order this task's design explicitly asked to replace).

A pure function of its arguments (same pattern as
labs.lab02_context_growth.matrix.build_context_plan): the same inputs
always produce the exact same plan, so a dry-run plan and the plan
actually executed are guaranteed identical. Randomization, when enabled,
only reorders which condition goes FIRST WITHIN each pair -- it never
moves a call across pairs, since multi_turn-style/history-dependent
caching effects mean pair grouping itself should stay fixed and legible.
"""

from __future__ import annotations

import random
from typing import Sequence

from labs.lab03_prompt_caching.experiments import CONDITION_ORDER, PAIRED_QUESTIONS
from labs.lab03_prompt_caching.models import PlannedCacheCall


class CacheMatrixError(ValueError):
    """Raised when the cache plan configuration is invalid."""


def build_cache_plan(
    condition_ids: Sequence[str],
    repetitions: int,
    randomize_condition_order_within_pair: bool = False,
    randomization_seed: int = 42,
) -> list[PlannedCacheCall]:
    """One PlannedCacheCall per (pair, condition) pair, paired/interleaved
    by construction: `condition_ids` is usually both conditions (the `all`
    CLI mode -- true pairing) or a single condition (the single-condition
    CLI modes -- just that one condition repeated, still cycling through
    the same paired questions for cross-run comparability).
    """
    unknown = [c for c in condition_ids if c not in CONDITION_ORDER]
    if unknown:
        raise CacheMatrixError(f"Unknown condition id(s) {unknown}; must be one of {list(CONDITION_ORDER)}")
    if repetitions < 1:
        raise CacheMatrixError("repetitions must be >= 1")

    ordered_conditions = [c for c in CONDITION_ORDER if c in condition_ids]
    rng = random.Random(randomization_seed) if randomize_condition_order_within_pair else None

    plan: list[PlannedCacheCall] = []
    execution_index = 0
    for pair_index in range(1, repetitions + 1):
        question_index = (pair_index - 1) % len(PAIRED_QUESTIONS)
        this_pair_conditions = list(ordered_conditions)
        if rng is not None:
            rng.shuffle(this_pair_conditions)
        for condition_id in this_pair_conditions:
            execution_index += 1
            plan.append(
                PlannedCacheCall(
                    execution_index=execution_index,
                    condition_id=condition_id,
                    pair_index=pair_index,
                    question_index=question_index,
                )
            )
    return plan
