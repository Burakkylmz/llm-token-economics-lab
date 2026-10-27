from __future__ import annotations

from lab_core.config import RetryConfig
from labs.lab03_prompt_caching.cache_status import CACHE_HIT, CACHE_MISS, CACHE_USAGE_UNAVAILABLE, INVALID_USAGE
from labs.lab03_prompt_caching.models import PlannedCacheCall
from labs.lab03_prompt_caching.runner import Budget, execute_planned_cache_call

RETRY_CONFIG = RetryConfig(max_attempts=2, base_delay_seconds=0.001, max_delay_seconds=0.002, jitter=False)


def _planned_call(condition_id="repeated_prefix", execution_index=1, pair_index=1, question_index=0):
    return PlannedCacheCall(execution_index=execution_index, condition_id=condition_id, pair_index=pair_index, question_index=question_index)


def test_successful_call_records_full_cache_observability(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=1500)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )

    assert m.succeeded is True
    assert m.prompt_tokens == 2000
    assert m.cached_tokens == 1500
    assert m.uncached_prompt_tokens == 500
    assert m.request_cache_hit is True
    assert m.request_cache_ratio == 0.75
    assert m.estimated_cache_savings_usd is not None and m.estimated_cache_savings_usd > 0
    assert m.answer_correct is True
    assert m.answer_evaluation_status == "correct"
    assert m.cache_status == CACHE_HIT
    assert m.prompt_tokens_details_present is True
    assert m.cached_tokens_field_present is True
    assert m.request_hash is not None
    assert len(m.request_hash) == 64
    assert m.sdk_version is not None


def test_cached_tokens_none_stays_none_not_zero(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=None)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )

    assert m.cached_tokens is None
    assert m.uncached_prompt_tokens is None
    assert m.request_cache_hit is None
    assert m.request_cache_ratio is None
    assert m.estimated_cache_savings_usd is None
    assert m.estimated_cost_usd is not None  # total cost is still computable
    assert m.cache_status == CACHE_USAGE_UNAVAILABLE
    assert m.prompt_tokens_details_present is False
    assert m.cached_tokens_field_present is False


def test_cached_tokens_reported_zero_is_a_real_miss_not_missing(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )

    assert m.cached_tokens == 0
    assert m.request_cache_hit is False
    assert m.estimated_cache_savings_usd == 0.0
    assert m.cache_status == CACHE_MISS
    assert m.prompt_tokens_details_present is True  # the field WAS reported, just with cached_tokens=0


def test_inconsistent_cache_usage_is_flagged_not_silently_fixed(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", prompt_tokens=100, completion_tokens=5, cached_tokens=400)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )

    assert m.cached_tokens == 400  # raw value preserved exactly as reported
    assert m.cache_status == INVALID_USAGE
    assert m.cache_usage_inconsistent is True
    assert m.uncached_prompt_tokens is None
    assert m.estimated_cache_savings_usd is None


def test_reasoning_tokens_never_double_counted(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", prompt_tokens=2000, completion_tokens=20, reasoning_tokens=15, cached_tokens=0)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )

    assert m.reasoning_tokens == 15
    assert m.non_reasoning_completion_tokens == 5


def test_incorrect_answer_is_api_success_but_not_correct(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="99", prompt_tokens=2000, completion_tokens=2, cached_tokens=0)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )

    assert m.succeeded is True
    assert m.answer_correct is False
    assert m.answer_evaluation_status == "incorrect"


def test_format_invalid_answer_is_not_correct_even_with_the_right_digit(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="The answer is not 5.", prompt_tokens=2000, completion_tokens=8, cached_tokens=0)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )

    assert m.answer_evaluation_status == "format_invalid"
    assert m.answer_correct is False


def test_finish_reason_length_preserved(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", finish_reason="length", prompt_tokens=2000, completion_tokens=2048, cached_tokens=0)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", config_with_pricing.pricing["openai/gpt-oss-20b"], 2048, "run-1", _planned_call()
    )
    assert m.finish_reason == "length"


def test_no_pricing_leaves_cost_and_savings_none(fake_sync_client_factory, make_response):
    response = make_response(content="5", prompt_tokens=2000, completion_tokens=2, cached_tokens=100)
    client = fake_sync_client_factory([response])

    m = execute_planned_cache_call(client, RETRY_CONFIG, "some-model", None, 2048, "run-1", _planned_call())
    assert m.estimated_cost_usd is None
    assert m.estimated_cache_savings_usd is None


def test_paired_calls_send_the_same_question_but_different_prefix(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="5", prompt_tokens=2000, completion_tokens=2, cached_tokens=0) for _ in range(2)]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    execute_planned_cache_call(client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "run-1", _planned_call("repeated_prefix", 1, 1, 0))
    execute_planned_cache_call(client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "run-1", _planned_call("cache_unfriendly", 2, 1, 0))

    sent = client.completions.calls
    assert sent[0]["messages"][1]["content"] == sent[1]["messages"][1]["content"]  # same user question
    assert sent[0]["messages"][0]["content"] != sent[1]["messages"][0]["content"]  # different prefix tag


def test_budget_blocks_a_call_before_it_is_made(fake_sync_client_factory, make_response, config_with_pricing):
    budget = Budget(max_api_calls=0, max_estimated_cost_usd=None, per_call_cost_ceiling=0.0)
    assert budget.can_make_another_call() is not None
