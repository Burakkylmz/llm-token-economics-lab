"""Builds Lab06's paired, interleaved execution plan.

Default order alternates which condition goes first every other pair
(pair 1: non_streaming, streaming; pair 2: streaming, non_streaming; ...)
so neither condition is systematically always-first across the whole run.
`randomize=True` instead shuffles each pair's order with a seeded RNG
(reproducible, not true randomness) -- an explicit alternative, not the
default. Neither approach eliminates every time-dependent confounder
(e.g. provider load drifting over the run) -- see this lab's
documentation for that explicit limitation.
"""

from __future__ import annotations

import random

from labs.lab06_streaming.models import CONDITION_NON_STREAMING, CONDITION_STREAMING, PlannedStreamingCall

CONDITION_ORDER = (CONDITION_NON_STREAMING, CONDITION_STREAMING)

PROFILE_PILOT = "pilot"
PROFILE_DEFAULT = "default"
PROFILE_EXTENDED = "extended"
PROFILES = (PROFILE_PILOT, PROFILE_DEFAULT, PROFILE_EXTENDED)

# Fixed pair counts for pilot/extended (Part D.7); "default"'s pair count
# is the one configurable value (lab06.paired_repetitions).
_FIXED_PROFILE_PAIRS = {PROFILE_PILOT: 2, PROFILE_EXTENDED: 8}


def pairs_for_profile(profile: str, default_pairs: int) -> int:
    if profile == PROFILE_DEFAULT:
        return default_pairs
    return _FIXED_PROFILE_PAIRS[profile]


def build_streaming_plan(
    pairs: int, randomize: bool = False, seed: int = 42, start_execution_index: int = 1
) -> list[PlannedStreamingCall]:
    if pairs < 1:
        raise ValueError("pairs must be a positive integer")

    plan: list[PlannedStreamingCall] = []
    index = start_execution_index
    rng = random.Random(seed) if randomize else None

    for pair_id in range(1, pairs + 1):
        if randomize:
            order = list(CONDITION_ORDER)
            rng.shuffle(order)
        else:
            order = list(CONDITION_ORDER) if pair_id % 2 == 1 else list(reversed(CONDITION_ORDER))
        for condition in order:
            plan.append(PlannedStreamingCall(execution_index=index, pair_id=pair_id, condition=condition))
            index += 1
    return plan


def build_single_condition_plan(condition: str, pairs: int, start_execution_index: int = 1) -> list[PlannedStreamingCall]:
    """For the backward-compatible `lab06 non_streaming`/`lab06 streaming`
    single-condition commands -- one call per "pair" (pair_id still
    assigned so the same persistence/stats code paths work), no pairing
    partner."""
    return [
        PlannedStreamingCall(execution_index=start_execution_index + i, pair_id=i + 1, condition=condition)
        for i in range(pairs)
    ]
