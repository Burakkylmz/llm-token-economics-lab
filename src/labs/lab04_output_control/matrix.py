"""Builds Lab04's two execution plans.

Deliberately NOT paired/interleaved like Lab03's matrix: output_limit_sweep's
conditions are independent (no shared "pair" concept across them), so a
plain condition-major order (all repetitions of one limit, then the next)
is used -- this also matches the terminal output example in this lab's
design (Part I), which prints a condition's full summary before moving on.

`start_execution_index` lets `all` mode continue execution numbering across
both experiments without the two plans needing to know about each other.
"""

from __future__ import annotations

from typing import Sequence

from labs.lab04_output_control.experiments import QUESTIONS, condition_label_for
from labs.lab04_output_control.models import PlannedInvocationCall, PlannedOutputLimitCall


def build_output_limit_plan(
    output_limit_values: Sequence[int], repetitions: int, start_execution_index: int = 1
) -> list[PlannedOutputLimitCall]:
    plan: list[PlannedOutputLimitCall] = []
    index = start_execution_index
    for max_completion_tokens in output_limit_values:
        condition_id = condition_label_for(max_completion_tokens)
        for repetition_index in range(1, repetitions + 1):
            plan.append(
                PlannedOutputLimitCall(
                    execution_index=index,
                    condition_id=condition_id,
                    repetition_index=repetition_index,
                    max_completion_tokens=max_completion_tokens,
                )
            )
            index += 1
    return plan


def build_invocation_plan(start_execution_index: int = 1) -> list[PlannedInvocationCall]:
    """Always exactly 1 (single) + len(QUESTIONS) (multiple) calls -- this
    lab's invocation_comparison shape is fixed, not configurable, since
    varying it would no longer be the same controlled comparison."""
    plan = [PlannedInvocationCall(execution_index=start_execution_index, strategy="single", question_index=None)]
    plan += [
        PlannedInvocationCall(execution_index=start_execution_index + 1 + i, strategy="multiple", question_index=i)
        for i in range(len(QUESTIONS))
    ]
    return plan
