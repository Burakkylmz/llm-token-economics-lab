"""Lab03 experiments: prompt caching behavior on Groq.

Prompt caching on Groq (as of Oct 2026) is:
  - fully automatic/implicit -- there is no cache-control request parameter;
  - supported only on openai/gpt-oss-20b, openai/gpt-oss-120b,
    openai/gpt-oss-safeguard-20b;
  - keyed on an exact-prefix match, so only identical leading text can hit;
  - not guaranteed: provider-side cache eviction, load, and the 2h TTL mean
    a hit is an *observation*, never something this code should assume or
    force. A single miss is not a bug in the experiment.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from groq import Groq

from lab_core.config import AppConfig
from lab_core.groq_client import call_sync
from lab_core.metrics import ExecutionMetadata, ExperimentResult
from labs.lab01_token_anatomy.cost import calculate_cost

# A long, fixed prefix (system-prompt-like) reused identically across calls
# in the cache-friendly condition, so a cache hit is possible on repeat.
STATIC_PREFIX = (
    "You are a customer support assistant for an API platform. Always answer "
    "using the following policy verbatim where relevant: refunds are issued "
    "within 5 business days, support tickets are answered within 24 hours, "
    "and escalations go to a human agent after 2 failed automated attempts. "
    "Be concise and professional in every response you give to the user. "
) * 10  # repeated to exceed the model's minimum cacheable prefix length

_DYNAMIC_SUFFIXES = [
    "What is the refund policy?",
    "How long until a human answers my ticket?",
    "When does my issue get escalated to a person?",
    "Can you restate the support ticket response time?",
    "What happens after two failed automated attempts?",
]


def _build_messages(prefix: str, suffix: str) -> list[dict]:
    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": suffix},
    ]


def run_repeated_prefix(
    client: Groq,
    config: AppConfig,
    model: str,
    repetitions: int,
) -> list[ExperimentResult]:
    """Cache-friendly condition: identical static prefix, varying suffix."""
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    for i in range(repetitions):
        suffix = _DYNAMIC_SUFFIXES[i % len(_DYNAMIC_SUFFIXES)]
        call = call_sync(client, model=model, messages=_build_messages(STATIC_PREFIX, suffix))
        if pricing is not None:
            call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)

        results.append(
            ExperimentResult(
                metadata=ExecutionMetadata(
                    experiment_id="lab03_repeated_prefix",
                    run_id=str(uuid.uuid4()),
                    model_id=model,
                    execution_mode="sync",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_parameters={"repetition": i},
                ),
                tokens_and_cost=call.tokens_and_cost,
                performance=call.performance,
            )
        )

    return results


def run_cache_unfriendly(
    client: Groq,
    config: AppConfig,
    model: str,
    repetitions: int,
) -> list[ExperimentResult]:
    """Cache-unfriendly condition: a different prefix on every call (no exact-prefix match is possible)."""
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    for i in range(repetitions):
        unique_prefix = f"[call #{i} nonce={uuid.uuid4()}] " + STATIC_PREFIX
        suffix = _DYNAMIC_SUFFIXES[i % len(_DYNAMIC_SUFFIXES)]
        call = call_sync(client, model=model, messages=_build_messages(unique_prefix, suffix))
        if pricing is not None:
            call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)

        results.append(
            ExperimentResult(
                metadata=ExecutionMetadata(
                    experiment_id="lab03_cache_unfriendly",
                    run_id=str(uuid.uuid4()),
                    model_id=model,
                    execution_mode="sync",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_parameters={"repetition": i},
                ),
                tokens_and_cost=call.tokens_and_cost,
                performance=call.performance,
            )
        )

    return results


def observed_cache_hit_rate(results: list[ExperimentResult]) -> float | None:
    """Fraction of results with a non-zero, non-None cached_tokens.

    Returns None if no result reports cached_tokens at all (so an empty
    experiment or an unsupported model doesn't get silently reported as 0%).
    """
    reported = [r.tokens_and_cost.cached_tokens for r in results if r.tokens_and_cost.cached_tokens is not None]
    if not reported:
        return None
    hits = sum(1 for c in reported if c > 0)
    return hits / len(reported)
