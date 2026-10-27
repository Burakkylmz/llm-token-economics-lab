from __future__ import annotations

from lab_core.config import RetryConfig
from labs.lab02_context_growth.models import PlannedContextCall
from labs.lab02_context_growth.runner import Budget, execute_multi_turn_growth, execute_planned_context_call

RETRY_CONFIG = RetryConfig(max_attempts=2, base_delay_seconds=0.001, max_delay_seconds=0.002, jitter=False)


def _planned_call(condition_id="minimal_context", execution_index=1, repetition_index=1):
    return PlannedContextCall(execution_index=execution_index, condition_id=condition_id, repetition_index=repetition_index)


# --- Context condition calls -----------------------------------------------


def test_successful_call_records_full_observability(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="4", prompt_tokens=50, completion_tokens=5, reasoning_tokens=None)
    client = fake_sync_client_factory([response])

    measurement = execute_planned_context_call(
        client, RETRY_CONFIG, "llama-3.1-8b-instant", config_with_pricing.pricing["llama-3.1-8b-instant"], 2048, "run-1", _planned_call()
    )

    assert measurement.succeeded is True
    assert measurement.prompt_tokens == 50
    assert measurement.completion_tokens == 5
    assert measurement.reasoning_tokens is None
    assert measurement.non_reasoning_completion_tokens is None  # None in, None out -- never fabricated
    assert measurement.answer_correct is True
    assert measurement.answer_evaluation_status == "correct"
    assert measurement.estimated_cost_usd is not None
    assert measurement.total_latency_ms is not None
    assert measurement.context_chars is not None


def test_reasoning_tokens_never_double_counted_in_non_reasoning(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="4", prompt_tokens=50, completion_tokens=20, reasoning_tokens=15)
    client = fake_sync_client_factory([response])

    measurement = execute_planned_context_call(
        client, RETRY_CONFIG, "llama-3.1-8b-instant", config_with_pricing.pricing["llama-3.1-8b-instant"], 2048, "run-1", _planned_call()
    )

    assert measurement.reasoning_tokens == 15
    assert measurement.non_reasoning_completion_tokens == 5  # 20 - 15, not 20 + 15


def test_incorrect_answer_is_scored_incorrect_not_failure(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", prompt_tokens=50, completion_tokens=2)
    client = fake_sync_client_factory([response])

    measurement = execute_planned_context_call(
        client, RETRY_CONFIG, "llama-3.1-8b-instant", config_with_pricing.pricing["llama-3.1-8b-instant"], 2048, "run-1", _planned_call()
    )

    assert measurement.succeeded is True  # API-level success
    assert measurement.answer_correct is False
    assert measurement.answer_evaluation_status == "incorrect"


def test_format_invalid_answer_is_not_correct(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="The answer is 4, probably.", prompt_tokens=50, completion_tokens=8)
    client = fake_sync_client_factory([response])

    measurement = execute_planned_context_call(
        client, RETRY_CONFIG, "llama-3.1-8b-instant", config_with_pricing.pricing["llama-3.1-8b-instant"], 2048, "run-1", _planned_call()
    )

    assert measurement.answer_evaluation_status == "format_invalid"
    assert measurement.answer_correct is False


def test_finish_reason_length_is_preserved_as_truncation_signal(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="4", finish_reason="length", prompt_tokens=50, completion_tokens=2048)
    client = fake_sync_client_factory([response])

    measurement = execute_planned_context_call(
        client, RETRY_CONFIG, "llama-3.1-8b-instant", config_with_pricing.pricing["llama-3.1-8b-instant"], 2048, "run-1", _planned_call()
    )

    assert measurement.finish_reason == "length"


def test_no_pricing_leaves_cost_none_rather_than_fabricated(fake_sync_client_factory, make_response):
    response = make_response(content="4", prompt_tokens=50, completion_tokens=2)
    client = fake_sync_client_factory([response])

    measurement = execute_planned_context_call(client, RETRY_CONFIG, "some-model", None, 2048, "run-1", _planned_call())

    assert measurement.estimated_cost_usd is None


def test_different_condition_ids_send_different_context(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="4", prompt_tokens=50, completion_tokens=2) for _ in range(3)]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["llama-3.1-8b-instant"]

    for i, condition_id in enumerate(["minimal_context", "expanded_context", "irrelevant_context"], start=1):
        execute_planned_context_call(
            client, RETRY_CONFIG, "llama-3.1-8b-instant", pricing, 2048, "run-1", _planned_call(condition_id, i, 1)
        )

    sent_lengths = [len(call["messages"][1]["content"]) for call in client.completions.calls]
    assert len(set(sent_lengths)) == 3  # all three conditions send genuinely different content
    assert sent_lengths[0] == min(sent_lengths)  # minimal_context is always the smallest


# --- Multi-turn growth -------------------------------------------------


def test_multi_turn_grows_message_history(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content=f"fact-{i}", prompt_tokens=10 * i, completion_tokens=5) for i in range(1, 4)]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["llama-3.1-8b-instant"]

    measurements = execute_multi_turn_growth(client, RETRY_CONFIG, "llama-3.1-8b-instant", pricing, 2048, "run-1", turns=3)

    assert len(measurements) == 3
    message_counts = [len(call["messages"]) for call in client.completions.calls]
    assert message_counts == sorted(message_counts)
    assert message_counts[0] < message_counts[-1]
    assert client.completions.calls[1]["messages"][-2]["content"] == "fact-1"


def test_incremental_and_cumulative_prompt_tokens_are_computed_correctly(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [
        make_response(content="a", prompt_tokens=98, completion_tokens=5),
        make_response(content="b", prompt_tokens=133, completion_tokens=5),
        make_response(content="c", prompt_tokens=163, completion_tokens=5),
    ]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["llama-3.1-8b-instant"]

    measurements = execute_multi_turn_growth(client, RETRY_CONFIG, "llama-3.1-8b-instant", pricing, 2048, "run-1", turns=3)

    assert measurements[0].incremental_prompt_tokens is None  # no prior turn to diff against
    assert measurements[1].incremental_prompt_tokens == 133 - 98
    assert measurements[2].incremental_prompt_tokens == 163 - 133
    assert measurements[0].cumulative_prompt_tokens == 98
    assert measurements[1].cumulative_prompt_tokens == 98 + 133
    assert measurements[2].cumulative_prompt_tokens == 98 + 133 + 163


def test_cumulative_cost_accumulates_across_turns(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="a", prompt_tokens=100, completion_tokens=5) for _ in range(2)]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["llama-3.1-8b-instant"]

    measurements = execute_multi_turn_growth(client, RETRY_CONFIG, "llama-3.1-8b-instant", pricing, 2048, "run-1", turns=2)

    assert measurements[0].cumulative_estimated_cost_usd == measurements[0].estimated_cost_usd
    assert measurements[1].cumulative_estimated_cost_usd > measurements[0].cumulative_estimated_cost_usd


def test_missing_usage_fields_do_not_break_cumulative_calculation(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [
        make_response(content="a", prompt_tokens=None, completion_tokens=None, total_tokens=None),
        make_response(content="b", prompt_tokens=50, completion_tokens=5),
    ]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["llama-3.1-8b-instant"]

    measurements = execute_multi_turn_growth(client, RETRY_CONFIG, "llama-3.1-8b-instant", pricing, 2048, "run-1", turns=2)

    assert len(measurements) == 2
    assert measurements[0].cumulative_prompt_tokens is None
    assert measurements[1].cumulative_prompt_tokens == 50


def test_budget_stops_multi_turn_growth_early(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="a", prompt_tokens=50, completion_tokens=5) for _ in range(4)]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["llama-3.1-8b-instant"]
    budget = Budget(max_api_calls=2, max_estimated_cost_usd=None, per_call_cost_ceiling=0.0)

    measurements = execute_multi_turn_growth(
        client, RETRY_CONFIG, "llama-3.1-8b-instant", pricing, 2048, "run-1", turns=4, budget=budget
    )

    assert len(measurements) == 2
