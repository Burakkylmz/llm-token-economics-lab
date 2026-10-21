"""Lab01 experiments: token breakdown and cost calculation for short vs long prompts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from groq import Groq

from lab_core.config import AppConfig
from lab_core.groq_client import call_sync
from lab_core.metrics import ExecutionMetadata, ExperimentResult
from labs.lab01_token_anatomy.cost import calculate_cost

SHORT_PROMPT = "What is the capital of France?"

LONG_PROMPT = (
    "You are a technical writer. Write a detailed, multi-paragraph explanation "
    "of how token-based pricing works for large language model APIs. Cover: "
    "what input tokens and output tokens are, how a tokenizer splits text "
    "into subword units, why longer prompts cost more, and how providers "
    "such as Groq report token usage in their API responses. Write at least "
    "four paragraphs."
)

EXPERIMENTS: dict[str, str] = {
    "short_prompt": SHORT_PROMPT,
    "long_prompt": LONG_PROMPT,
}


def run_experiment(
    client: Groq,
    config: AppConfig,
    experiment: str,
    model: str,
    repetitions: int,
) -> list[ExperimentResult]:
    prompt = EXPERIMENTS[experiment]
    pricing = config.pricing.get(model)
    results: list[ExperimentResult] = []

    for _ in range(repetitions):
        call = call_sync(client, model=model, messages=[{"role": "user", "content": prompt}])
        if pricing is not None:
            call.tokens_and_cost.estimated_cost_usd = calculate_cost(call.tokens_and_cost, pricing)

        results.append(
            ExperimentResult(
                metadata=ExecutionMetadata(
                    experiment_id=f"lab01_{experiment}",
                    run_id=str(uuid.uuid4()),
                    model_id=model,
                    execution_mode="sync",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_parameters={"prompt_chars": len(prompt)},
                ),
                tokens_and_cost=call.tokens_and_cost,
                performance=call.performance,
            )
        )

    return results
