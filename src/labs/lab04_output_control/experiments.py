"""Lab04 experiments: output-token budgeting/truncation and invocation-count
optimization on Groq.

Two independent experiments, both built around deterministic,
field-extraction tasks (never an open-ended "explain this" prompt -- see
Part D of this lab's redesign) so completeness and correctness can be
judged exactly, not guessed at:

  - output_limit_sweep: the SAME retry-policy-extraction task, sent with a
    varying `max_completion_tokens` cap, to show how an output budget
    interacts with reasoning-token consumption, truncation
    (`finish_reason`), and answer completeness/correctness.
  - invocation_comparison: the same three trivia questions this project has
    always used, answered either in one combined call or in three separate
    calls, to show the fixed per-call overhead of unnecessary extra
    invocations without changing what is actually being asked.

Message structure follows this project's established convention: static
task content first (the policy document / the question), the output-format
contract last -- see labs.lab03_prompt_caching.experiments for the same
layout rule.
"""

from __future__ import annotations

from labs.lab04_output_control.evaluation import (
    RETRY_POLICY_SYSTEM_MESSAGE,
    RETRY_POLICY_USER_MESSAGE,
)

# --- Output-limit sweep ----------------------------------------------------

# Recommended starting profile (Part C of this lab's design) -- these are
# the config defaults (lab04.output_limit_values), not a hard requirement:
# any positive, duplicate-free list of token limits is accepted. A value
# outside this mapping is labeled by its raw token count instead (see
# condition_label_for below), so an override never produces an unlabeled
# condition.
OUTPUT_LIMIT_PROFILE_NAMES = {
    64: "restrictive",
    256: "moderate",
    1024: "generous",
}


def condition_label_for(max_completion_tokens: int) -> str:
    return OUTPUT_LIMIT_PROFILE_NAMES.get(max_completion_tokens, f"max_{max_completion_tokens}")


def build_output_limit_messages() -> list[dict]:
    """The output-limit sweep's ONLY independent variable is
    `max_completion_tokens` (Part D's controlled-prompt rule): this message
    pair is identical on every call, across every condition and repetition."""
    return [
        {"role": "system", "content": RETRY_POLICY_SYSTEM_MESSAGE},
        {"role": "user", "content": RETRY_POLICY_USER_MESSAGE},
    ]


# --- Invocation comparison -------------------------------------------------

QUESTIONS = [
    "What is the capital of Japan?",
    "What is the boiling point of water in Celsius at sea level?",
    "Who wrote the play 'Hamlet'?",
]

REQUIRED_INVOCATION_FIELDS = ("capital_of_japan", "boiling_point_celsius", "hamlet_author")

EXPECTED_INVOCATION_FIELD_VALUES = {
    "capital_of_japan": "Tokyo",
    "boiling_point_celsius": "100",
    "hamlet_author": "William Shakespeare",
}


def build_invocation_single_messages() -> list[dict]:
    """All three questions, answered in one call, using the same
    "field_name: value" output contract as the retry-policy task -- so the
    exact same deterministic multi-field evaluator applies to both."""
    instruction = (
        "Answer all three questions below. Respond with EXACTLY three lines, "
        "each in the form \"field_name: value\", in this exact order, and "
        "nothing else (no explanation, no restating the questions):\n\n"
        "capital_of_japan: <answer>\n"
        "boiling_point_celsius: <answer>\n"
        "hamlet_author: <answer>\n\n"
        "Questions:\n"
        + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(QUESTIONS))
    )
    return [{"role": "user", "content": instruction}]


def build_invocation_multiple_messages(question_index: int) -> list[dict]:
    """One question per call, asked plainly -- deliberately NOT the same
    strict output contract as the single-call strategy, since forcing an
    artificial "field_name: value" format onto a single bare question would
    not reflect how multiple independent invocations are actually used.
    Equivalent correctness evaluation is still guaranteed by
    labs.lab04_output_control.evaluation.evaluate_single_answer (see its
    docstring for why this is still a fair, equivalent comparison)."""
    return [{"role": "user", "content": QUESTIONS[question_index]}]


def expected_invocation_answer_for(question_index: int) -> str:
    field = REQUIRED_INVOCATION_FIELDS[question_index]
    return EXPECTED_INVOCATION_FIELD_VALUES[field]
