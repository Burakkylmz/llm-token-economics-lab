"""Lab02 experiments: how growing/irrelevant context affects input tokens, cost and latency.

All conditions ask essentially the same question, so completion content
should be similar; what varies is the size and relevance of the context
placed before that question. Keep this the only controlled variable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from groq import Groq

from lab_core.config import AppConfig
from lab_core.groq_client import call_sync
from lab_core.metrics import ExecutionMetadata, ExperimentResult
from labs.lab01_token_anatomy.cost import calculate_cost

QUESTION = "Based on the context above, what is the maximum number of retries described?"

RELEVANT_CONTEXT = (
    "Our API client retry policy: on a transient failure (HTTP 429 or 5xx), "
    "the client retries up to 5 times total, using exponential backoff with "
    "full jitter, starting at a 1 second base delay and capping at 30 seconds."
)

# Repeated filler clearly unrelated to the retry question, to simulate
# "context bloat" (irrelevant information inflating the prompt).
_IRRELEVANT_PARAGRAPH = (
    "The cafeteria on the third floor serves lunch from 11:30 to 14:00 and "
    "the vegetarian option changes weekly according to a rotating seasonal "
    "menu planned by the facilities team. "
)

EXPANDED_RELEVANT_CONTEXT = (RELEVANT_CONTEXT + " ") * 20

IRRELEVANT_CONTEXT = (_IRRELEVANT_PARAGRAPH * 30) + RELEVANT_CONTEXT

CONDITIONS: dict[str, str] = {
    "minimal_context": RELEVANT_CONTEXT,
    "expanded_context": EXPANDED_RELEVANT_CONTEXT,
    "irrelevant_context": IRRELEVANT_CONTEXT,
}


def _build_messages(context: str) -> list[dict]:
    return [
        {"role": "system", "content": "Answer using only the information in the user's context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {QUESTION}"},
    ]


def run_context_experiment(
    client: Groq,
    config: AppConfig,
    condition: str,
    model: str,
    repetitions: int,
) -> list[ExperimentResult]:
    context = CONDITIONS[condition]
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    for _ in range(repetitions):
        call = call_sync(client, model=model, messages=_build_messages(context))
        if pricing is not None:
            call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)

        results.append(
            ExperimentResult(
                metadata=ExecutionMetadata(
                    experiment_id=f"lab02_{condition}",
                    run_id=str(uuid.uuid4()),
                    model_id=model,
                    execution_mode="sync",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_parameters={"context_chars": len(context)},
                ),
                tokens_and_cost=call.tokens_and_cost,
                performance=call.performance,
            )
        )

    return results


def run_multi_turn_growth(
    client: Groq,
    config: AppConfig,
    model: str,
    turns: int,
) -> list[ExperimentResult]:
    """Grow a single conversation turn by turn and measure prompt_tokens at each turn.

    Each turn's prompt_tokens should reflect the accumulated history, letting
    participants observe token growth independent of any single message's size.
    """
    pricing = config.pricing.get(model)
    messages: list[dict] = [
        {"role": "system", "content": "You are a concise assistant. Answer in one short sentence."}
    ]
    results: list[ExperimentResult] = []

    for turn in range(1, turns + 1):
        messages.append({"role": "user", "content": f"Tell me a short, distinct fact about topic #{turn}."})
        call = call_sync(client, model=model, messages=messages)
        if pricing is not None:
            call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)

        results.append(
            ExperimentResult(
                metadata=ExecutionMetadata(
                    experiment_id="lab02_multi_turn_growth",
                    run_id=str(uuid.uuid4()),
                    model_id=model,
                    execution_mode="sync",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_parameters={"turn": turn},
                ),
                tokens_and_cost=call.tokens_and_cost,
                performance=call.performance,
            )
        )

        assistant_content = call.raw_response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_content})

    return results
