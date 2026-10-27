"""`cache_probe` -- the strongest deterministic exact-prefix caching
experiment this project can construct against the Groq API.

Design rationale (Part B of this lab's redesign task): `lab03 all`'s
`repeated_prefix` condition already demonstrates Behavior B ("stable
prefix, dynamic suffix") -- its tag is byte-identical across every pair,
but the user QUESTION differs pair to pair (question_index cycles with
pair_index), since that condition's job is answering RQ1-RQ5 with
comparable-but-distinct paired questions. `cache_probe` demonstrates
Behavior A ("exact full-prompt reuse") instead: every planned call pins
`question_index=0`, so the ENTIRE request -- system tag, knowledge base,
rules, AND user question -- is byte-identical across all probe calls, not
just the leading prefix. This maximizes the chance Groq's exact-prefix
cache matching has something to match against, without ever fabricating
or forcing a hit (a provider-managed cache is never guaranteed to fire,
even on an exact repeat).

No new message-construction logic is introduced: `cache_probe` reuses
`CONDITION_REPEATED_PREFIX`'s existing tag/knowledge-base/rules verbatim
via `labs.lab03_prompt_caching.experiments.build_cache_messages`, and its
calls execute through the EXACT SAME `execute_planned_cache_call` runner
as every other condition -- only the PLAN differs (question_index pinned,
not cycled).
"""

from __future__ import annotations

from labs.lab03_prompt_caching.diagnostics import compute_request_hash
from labs.lab03_prompt_caching.experiments import CONDITION_REPEATED_PREFIX, build_cache_messages
from labs.lab03_prompt_caching.models import PlannedCacheCall

DEFAULT_PROBE_CALLS = 3
# Always question_index 0 -- pinning this (never cycling it the way the
# paired matrix does) is exactly what makes every probe call identical.
PROBE_QUESTION_INDEX = 0


def build_probe_plan(probe_calls: int = DEFAULT_PROBE_CALLS) -> list[PlannedCacheCall]:
    """`probe_calls` sequential, deterministic PlannedCacheCall entries, all
    condition_id=repeated_prefix, all question_index=0 -- every one of them
    builds to the exact same message payload (see verify_probe_plan_is_identical)."""
    if probe_calls < 1:
        raise ValueError("probe_calls must be >= 1")
    return [
        PlannedCacheCall(
            execution_index=i,
            condition_id=CONDITION_REPEATED_PREFIX,
            pair_index=i,
            question_index=PROBE_QUESTION_INDEX,
        )
        for i in range(1, probe_calls + 1)
    ]


def verify_probe_plan_is_identical(plan: list[PlannedCacheCall], model: str, max_completion_tokens: int) -> dict:
    """Offline (no API call) stability check: hashes the exact payload each
    planned probe call would send, and confirms they are all identical.
    Hashes are a verification/diagnostic tool only -- NOT evidence a real
    cache hit occurred; only a live provider response can show that.
    """
    hashes = []
    stable_prefix_length = None
    full_prompt_length = None
    for call in plan:
        messages = build_cache_messages(call.condition_id, call.pair_index, call.question_index)
        hashes.append(compute_request_hash(model, messages, max_completion_tokens))
        full_prompt_length = sum(len(m["content"]) for m in messages)
        stable_prefix_length = len(messages[0]["content"])  # the system message is the stable/cacheable prefix

    return {
        "stable": len(set(hashes)) <= 1,
        "request_hash": hashes[0] if hashes else None,
        "stable_prefix_length": stable_prefix_length,
        "full_prompt_length": full_prompt_length,
        "probe_calls": len(plan),
    }
