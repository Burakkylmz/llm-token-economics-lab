from __future__ import annotations

from labs.lab02_context_growth.experiments import (
    CONDITIONS,
    CONTEXT_CONDITION_ORDER,
    EXPANDED_RELEVANT_CONTEXT,
    EXPECTED_ANSWER,
    IRRELEVANT_CONTEXT,
    QUESTION,
    RELEVANT_CONTEXT,
    build_context_messages,
)


def test_three_context_conditions_in_fixed_order():
    assert CONTEXT_CONDITION_ORDER == ("minimal_context", "expanded_context", "irrelevant_context")
    assert set(CONDITIONS.keys()) == set(CONTEXT_CONDITION_ORDER)


def test_expanded_context_is_larger_than_minimal():
    assert len(EXPANDED_RELEVANT_CONTEXT) > len(RELEVANT_CONTEXT)


def test_irrelevant_context_contains_the_relevant_fact_too():
    # irrelevant_context must still contain the canonical answer, so answer
    # quality is held constant across conditions -- only size/relevance varies.
    assert RELEVANT_CONTEXT in IRRELEVANT_CONTEXT


def test_expanded_and_irrelevant_context_are_comparable_in_size():
    # Not required to be byte-identical (see the lab's documented confounders),
    # but must be in the same ballpark -- otherwise "bloat" vs "growth" would
    # really just be testing "longer vs shorter", which isn't this experiment's
    # question.
    ratio = len(EXPANDED_RELEVANT_CONTEXT) / len(IRRELEVANT_CONTEXT)
    assert 0.8 <= ratio <= 1.25


def test_canonical_answer_appears_exactly_once_in_expanded_and_irrelevant():
    assert EXPANDED_RELEVANT_CONTEXT.count(RELEVANT_CONTEXT) == 1
    assert IRRELEVANT_CONTEXT.count(RELEVANT_CONTEXT) == 1


def test_retry_policy_is_unambiguous_about_total_vs_retry_attempts():
    # The original fixture's "retries up to 5 times total" was ambiguous
    # about whether 5 meant total attempts or retry attempts. The new
    # fixture must state both numbers explicitly and distinctly.
    assert "Maximum total attempts: 5" in RELEVANT_CONTEXT
    assert "Maximum retry attempts: 4" in RELEVANT_CONTEXT
    assert "retries up to 5 times total" not in RELEVANT_CONTEXT


def test_expected_answer_matches_the_unambiguous_policy():
    assert EXPECTED_ANSWER == "4"


def test_question_asks_for_retry_attempts_excluding_initial():
    assert "excluding the initial attempt" in QUESTION


def test_build_context_messages_includes_context_and_question():
    messages = build_context_messages(RELEVANT_CONTEXT)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert RELEVANT_CONTEXT in messages[1]["content"]
    assert QUESTION in messages[1]["content"]


def test_all_three_conditions_use_the_same_system_instruction_and_question():
    # Part 5.4's control requirement: same question, same system instruction,
    # same output-format requirement across all three conditions.
    system_messages = {build_context_messages(ctx)[0]["content"] for ctx in CONDITIONS.values()}
    assert len(system_messages) == 1
    for ctx in CONDITIONS.values():
        assert QUESTION in build_context_messages(ctx)[1]["content"]
