"""Orchestrates one Lab10 experiment plan (Part 7.2/8): executes each
`PlannedCall` in order, generating each dataset's two summaries exactly
once and reusing them for every follow-up question in that condition.

A summary that has not yet been generated (e.g. the plan was filtered to
skip summary-generation calls, as `lab10 evaluate` does) is represented as
`PENDING_GENERATION`, never fabricated (Part 8.4) -- any follow-up call
for that condition is refused rather than silently falling back to
something else.
"""

from __future__ import annotations

from typing import Optional

from groq import Groq

from lab_core.config import ModelPricing
from labs.lab10_conversation_summarization.answering import answer_question
from labs.lab10_conversation_summarization.context import render_summary_as_context
from labs.lab10_conversation_summarization.datasets import Conversation, GoldenQuestion, render_transcript
from labs.lab10_conversation_summarization.models import (
    CALL_PURPOSE_FOLLOWUP_ANSWER,
    CALL_PURPOSE_SUMMARY_GENERATION,
    CONDITION_FULL_HISTORY_BASELINE,
    CONDITION_TO_EFFORT,
    FollowupMeasurement,
    PlannedCall,
    SummaryMeasurement,
    TraceRecord,
)
from labs.lab10_conversation_summarization.summarization import generate_summary


class SummaryNotYetGenerated(RuntimeError):
    """Raised when a follow-up call for a summarization condition is
    scheduled before that condition's summary call has run."""


def execute_plan(
    client: Groq,
    summary_model: str,
    summary_max_completion_tokens: int,
    answer_model: str,
    answer_reasoning_effort: str,
    answer_max_completion_tokens: int,
    pricing: Optional[ModelPricing],
    experiment_run_id: str,
    conversations: dict[str, Conversation],
    questions: dict[str, dict[str, GoldenQuestion]],
    plan: list[PlannedCall],
    preloaded_summaries: Optional[dict[tuple[str, str], dict]] = None,
) -> tuple[list[SummaryMeasurement], list[FollowupMeasurement], list[TraceRecord], dict[tuple[str, str], dict]]:
    """`preloaded_summaries` (keyed by (dataset_id, condition_id)) lets
    `lab10 evaluate` reuse an already-generated summary artifact without
    regenerating it -- those (dataset_id, condition_id) pairs are then
    never dispatched through `generate_summary` even if the plan still
    contains a summary_generation call for them (`lab10 evaluate` builds a
    plan containing only followup_answer calls, so this is purely defensive)."""
    summary_measurements: list[SummaryMeasurement] = []
    followup_measurements: list[FollowupMeasurement] = []
    traces: list[TraceRecord] = []
    generated_summaries: dict[tuple[str, str], dict] = dict(preloaded_summaries or {})

    for call in plan:
        conversation = conversations[call.dataset_id]

        if call.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION:
            measurement, summary_dict, trace = generate_summary(
                client, summary_model, call.reasoning_effort, summary_max_completion_tokens,
                conversation, pricing, experiment_run_id, call.execution_index, call.condition_id,
            )
            summary_measurements.append(measurement)
            traces.append(trace)
            if summary_dict is not None:
                generated_summaries[(call.dataset_id, call.condition_id)] = summary_dict
            continue

        assert call.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER
        question = questions[call.dataset_id][call.question_id]

        if call.condition_id == CONDITION_FULL_HISTORY_BASELINE:
            context_payload = render_transcript(conversation)
        else:
            summary_dict = generated_summaries.get((call.dataset_id, call.condition_id))
            if summary_dict is None:
                # Fail clearly rather than silently sending a PENDING_GENERATION
                # placeholder to the model as if it were real context (Part 8.4
                # forbids fabricating a missing summary; Part 17.1 requires
                # `lab10 evaluate` to fail clearly on an incompatible artifact).
                raise SummaryNotYetGenerated(
                    f"No generated summary available for dataset '{call.dataset_id}' / "
                    f"condition '{call.condition_id}' -- cannot answer question "
                    f"'{call.question_id}'. Nothing was fabricated; generate the summary first."
                )
            context_payload = render_summary_as_context(summary_dict)

        measurement, trace = answer_question(
            client, answer_model, answer_reasoning_effort, answer_max_completion_tokens, context_payload,
            question, pricing, experiment_run_id, call.execution_index, call.dataset_id, call.condition_id,
        )
        followup_measurements.append(measurement)
        traces.append(trace)

    return summary_measurements, followup_measurements, traces, generated_summaries
