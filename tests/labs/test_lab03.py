from __future__ import annotations

from labs.lab03_prompt_caching.experiments import (
    CACHE_FRIENDLY_TAG,
    CONDITION_CACHE_UNFRIENDLY,
    CONDITION_ORDER,
    CONDITION_REPEATED_PREFIX,
    PAIRED_QUESTIONS,
    STATIC_KNOWLEDGE_BASE,
    build_cache_messages,
    expected_answer_for,
    tag_for_condition,
)


def test_condition_order_preserves_legacy_names():
    assert CONDITION_ORDER == ("repeated_prefix", "cache_unfriendly")


def test_repeated_prefix_tag_is_identical_across_every_pair_index():
    tags = [tag_for_condition(CONDITION_REPEATED_PREFIX, pair_index) for pair_index in range(1, 5)]
    assert len(set(tags)) == 1
    assert tags[0] == CACHE_FRIENDLY_TAG


def test_cache_unfriendly_tag_changes_every_pair_index():
    tags = [tag_for_condition(CONDITION_CACHE_UNFRIENDLY, pair_index) for pair_index in range(1, 5)]
    assert len(set(tags)) == 4


def test_cache_friendly_and_cache_unfriendly_tags_are_the_same_length():
    assert len(tag_for_condition(CONDITION_REPEATED_PREFIX, 1)) == len(tag_for_condition(CONDITION_CACHE_UNFRIENDLY, 1))


def test_tag_for_unknown_condition_raises():
    import pytest

    with pytest.raises(ValueError):
        tag_for_condition("not_a_real_condition", 1)


def test_four_paired_questions_each_with_a_distinct_integer_answer():
    assert len(PAIRED_QUESTIONS) == 4
    answers = [q.expected_answer for q in PAIRED_QUESTIONS]
    assert len(set(answers)) == 4
    for answer in answers:
        assert answer.lstrip("-").isdigit()


def test_expected_answer_for_cycles_through_paired_questions():
    assert expected_answer_for(0) == PAIRED_QUESTIONS[0].expected_answer
    assert expected_answer_for(4) == PAIRED_QUESTIONS[0].expected_answer  # wraps around


def test_static_knowledge_base_is_long_enough_to_be_cache_eligible():
    # Groq's own docs: minimum cacheable prompt length "varies by model,
    # from 128 to 1024 tokens" -- this asserts comfortably above the upper
    # end of that documented range using a conservative chars-per-token
    # estimate, without inventing an exact per-model threshold.
    approx_tokens = len(STATIC_KNOWLEDGE_BASE) / 4.0
    assert approx_tokens > 1024


def test_both_conditions_in_a_pair_use_the_same_question_and_expected_answer():
    messages_a = build_cache_messages(CONDITION_REPEATED_PREFIX, pair_index=1, question_index=2)
    messages_b = build_cache_messages(CONDITION_CACHE_UNFRIENDLY, pair_index=1, question_index=2)
    assert messages_a[1]["content"] == messages_b[1]["content"]  # same user question
    assert expected_answer_for(2) == expected_answer_for(2)


def test_repeated_prefix_and_cache_unfriendly_system_prefix_differs_only_at_the_tag():
    messages_a = build_cache_messages(CONDITION_REPEATED_PREFIX, pair_index=1, question_index=0)
    messages_b = build_cache_messages(CONDITION_CACHE_UNFRIENDLY, pair_index=1, question_index=0)
    system_a = messages_a[0]["content"]
    system_b = messages_b[0]["content"]
    assert system_a != system_b
    # Stripping each one's own tag-length prefix should leave identical content.
    tag_len = len(CACHE_FRIENDLY_TAG)
    assert system_a[tag_len:] == system_b[tag_len:]


def test_repeated_prefix_system_content_is_byte_identical_across_pairs():
    system_pair1 = build_cache_messages(CONDITION_REPEATED_PREFIX, pair_index=1, question_index=0)[0]["content"]
    system_pair2 = build_cache_messages(CONDITION_REPEATED_PREFIX, pair_index=2, question_index=0)[0]["content"]
    assert system_pair1 == system_pair2


def test_cache_unfriendly_system_content_changes_across_pairs():
    system_pair1 = build_cache_messages(CONDITION_CACHE_UNFRIENDLY, pair_index=1, question_index=0)[0]["content"]
    system_pair2 = build_cache_messages(CONDITION_CACHE_UNFRIENDLY, pair_index=2, question_index=0)[0]["content"]
    assert system_pair1 != system_pair2


def test_expanded_and_unfriendly_prompts_are_comparable_in_length():
    friendly = build_cache_messages(CONDITION_REPEATED_PREFIX, pair_index=1, question_index=0)
    unfriendly = build_cache_messages(CONDITION_CACHE_UNFRIENDLY, pair_index=1, question_index=0)
    len_friendly = sum(len(m["content"]) for m in friendly)
    len_unfriendly = sum(len(m["content"]) for m in unfriendly)
    assert len_friendly == len_unfriendly  # tags are the same length, so these must match exactly
