"""Experiment matrix for Lab10 (Part 7).

Calls per dataset = 3Q + 2 (2 summary-generation calls + Q baseline
follow-ups + Q low-summary follow-ups + Q medium-summary follow-ups).
Profiles select which datasets and how many questions per dataset:

  pilot:    1 dataset,  2 questions -> 1 x (3*2+2)  =  8 calls
  default:  1 dataset,  4 questions -> 1 x (3*4+2)  = 14 calls
  extended: 2 datasets, 4 questions -> 2 x (3*4+2)  = 28 calls

Execution order per dataset is fixed and deterministic (never randomized):
summary generation (low, then medium) always completes before any
downstream follow-up call that would need it -- Part 7.2's explicit
ordering requirement.
"""

from __future__ import annotations

from labs.lab10_conversation_summarization.datasets import DATASET_IDS, GoldenQuestion, load_golden_questions
from labs.lab10_conversation_summarization.models import (
    CALL_PURPOSE_FOLLOWUP_ANSWER,
    CALL_PURPOSE_SUMMARY_GENERATION,
    CONDITION_FULL_HISTORY_BASELINE,
    CONDITION_SUMMARIZATION_LOW,
    CONDITION_SUMMARIZATION_MEDIUM,
    CONDITION_TO_EFFORT,
    PlannedCall,
)

PROFILE_PILOT = "pilot"
PROFILE_DEFAULT = "default"
PROFILE_EXTENDED = "extended"
PROFILES = (PROFILE_PILOT, PROFILE_DEFAULT, PROFILE_EXTENDED)

PILOT_QUESTION_COUNT = 2
DEFAULT_QUESTION_COUNT = 4
EXTENDED_QUESTION_COUNT = 4

PILOT_DATASET_IDS = (DATASET_IDS[0],)
EXTENDED_DATASET_IDS = DATASET_IDS


def datasets_for_profile(profile: str, default_dataset_ids: tuple[str, ...] = (DATASET_IDS[0],)) -> tuple[str, ...]:
    if profile == PROFILE_PILOT:
        return PILOT_DATASET_IDS
    if profile == PROFILE_EXTENDED:
        return EXTENDED_DATASET_IDS
    if profile == PROFILE_DEFAULT:
        return default_dataset_ids
    raise ValueError(f"Unsupported profile: {profile}")


def question_count_for_profile(profile: str, default_question_count: int = DEFAULT_QUESTION_COUNT) -> int:
    if profile == PROFILE_PILOT:
        return PILOT_QUESTION_COUNT
    if profile == PROFILE_EXTENDED:
        return EXTENDED_QUESTION_COUNT
    if profile == PROFILE_DEFAULT:
        return default_question_count
    raise ValueError(f"Unsupported profile: {profile}")


def select_questions(dataset_id: str, count: int, version: str = "v1") -> list[GoldenQuestion]:
    """The first `count` golden questions, in their stable, versioned
    dataset-file order -- the same subset every time for a given count."""
    all_questions = load_golden_questions(dataset_id, version)
    if count > len(all_questions):
        raise ValueError(f"Dataset '{dataset_id}' only has {len(all_questions)} golden questions, requested {count}")
    return all_questions[:count]


def calls_per_dataset(question_count: int) -> int:
    return 3 * question_count + 2


def total_planned_calls(profile: str, default_dataset_ids: tuple[str, ...] = (DATASET_IDS[0],), default_question_count: int = DEFAULT_QUESTION_COUNT) -> int:
    datasets = datasets_for_profile(profile, default_dataset_ids)
    question_count = question_count_for_profile(profile, default_question_count)
    return len(datasets) * calls_per_dataset(question_count)


def build_plan(
    profile: str,
    default_dataset_ids: tuple[str, ...] = (DATASET_IDS[0],),
    default_question_count: int = DEFAULT_QUESTION_COUNT,
    dataset_version: str = "v1",
) -> list[PlannedCall]:
    datasets = datasets_for_profile(profile, default_dataset_ids)
    question_count = question_count_for_profile(profile, default_question_count)

    plan: list[PlannedCall] = []
    execution_index = 1
    for dataset_id in datasets:
        questions = select_questions(dataset_id, question_count, dataset_version)

        for condition_id in (CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM):
            plan.append(
                PlannedCall(
                    execution_index=execution_index, dataset_id=dataset_id, condition_id=condition_id,
                    call_purpose=CALL_PURPOSE_SUMMARY_GENERATION, reasoning_effort=CONDITION_TO_EFFORT[condition_id],
                )
            )
            execution_index += 1

        for condition_id in (CONDITION_FULL_HISTORY_BASELINE, CONDITION_SUMMARIZATION_LOW, CONDITION_SUMMARIZATION_MEDIUM):
            for question in questions:
                plan.append(
                    PlannedCall(
                        execution_index=execution_index, dataset_id=dataset_id, condition_id=condition_id,
                        call_purpose=CALL_PURPOSE_FOLLOWUP_ANSWER, reasoning_effort="",
                        question_id=question.question_id,
                    )
                )
                execution_index += 1

    return plan
