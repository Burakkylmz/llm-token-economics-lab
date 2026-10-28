from __future__ import annotations

from lab_core.config import RetryConfig
from labs.lab04_output_control.models import PlannedInvocationCall, PlannedOutputLimitCall
from labs.lab04_output_control.runner import Budget, execute_planned_output_limit_call, run_invocation_strategy

RETRY_CONFIG = RetryConfig(max_attempts=2, base_delay_seconds=0.001, max_delay_seconds=0.002, jitter=False)

RETRY_POLICY_ALL_CORRECT = """\
max_retry_attempts: 4
max_total_attempts: 5
base_delay_seconds: 1
last_nominal_delay_seconds: 8
retryable_http_status: 429
"""


def _planned_output_limit_call(max_completion_tokens=256, execution_index=1, repetition_index=1, condition_id="moderate"):
    return PlannedOutputLimitCall(
        execution_index=execution_index, condition_id=condition_id, repetition_index=repetition_index,
        max_completion_tokens=max_completion_tokens,
    )


def test_fully_correct_answer_is_accepted(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content=RETRY_POLICY_ALL_CORRECT, prompt_tokens=200, completion_tokens=40, reasoning_tokens=10)
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    m = execute_planned_output_limit_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call()
    )

    assert m.succeeded is True
    assert m.overall_answer_accepted is True
    assert m.correct_field_count == 5
    assert m.reasoning_tokens == 10
    assert m.non_reasoning_completion_tokens == 30
    assert m.estimated_cost_usd is not None


def test_truncated_response_is_detected_from_finish_reason(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="max_retry_attempts: 4", finish_reason="length", prompt_tokens=200, completion_tokens=16, reasoning_tokens=16)
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    m = execute_planned_output_limit_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call(max_completion_tokens=16)
    )

    assert m.output_truncated is True
    assert m.overall_answer_accepted is False
    assert m.missing_field_count == 4


def test_finish_reason_stop_but_all_budget_went_to_reasoning_is_not_truncated_but_still_fails_quality(
    fake_sync_client_factory, make_response, config_with_pricing
):
    # completion_tokens == max_completion_tokens is NOT proof of truncation
    # without finish_reason confirming it (Part D's non-negotiable rule).
    response = make_response(content="", finish_reason="stop", prompt_tokens=200, completion_tokens=64, reasoning_tokens=64)
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    m = execute_planned_output_limit_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call(max_completion_tokens=64)
    )

    assert m.output_truncated is False
    assert m.answer_evaluation_status == "missing_response"
    assert m.overall_answer_accepted is False


def test_missing_finish_reason_leaves_output_truncated_none(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content=RETRY_POLICY_ALL_CORRECT, finish_reason=None, prompt_tokens=200, completion_tokens=40)
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    m = execute_planned_output_limit_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call()
    )
    assert m.output_truncated is None


def test_missing_reasoning_token_data_leaves_non_reasoning_none(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content=RETRY_POLICY_ALL_CORRECT, prompt_tokens=200, completion_tokens=40, reasoning_tokens=None)
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    m = execute_planned_output_limit_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call()
    )
    assert m.reasoning_tokens is None
    assert m.non_reasoning_completion_tokens is None


def test_failed_call_keeps_every_optional_field_none(fake_sync_client_factory, config_with_pricing):
    import httpx
    from groq import APIConnectionError

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client = fake_sync_client_factory(
        [APIConnectionError(request=request) for _ in range(RETRY_CONFIG.max_attempts)]
    )
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    m = execute_planned_output_limit_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call()
    )
    assert m.succeeded is False
    assert m.prompt_tokens is None
    assert m.overall_answer_accepted is False
    assert m.answer_evaluation_status == "missing_response"


def test_sends_correct_max_completion_tokens_and_reasoning_effort(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content=RETRY_POLICY_ALL_CORRECT, prompt_tokens=200, completion_tokens=40)
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    execute_planned_output_limit_call(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call(max_completion_tokens=256)
    )
    sent = client.completions.calls[0]
    assert sent["max_completion_tokens"] == 256
    assert sent["reasoning_effort"] == "low"


def test_prompt_is_identical_across_every_output_limit_condition(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content=RETRY_POLICY_ALL_CORRECT) for _ in range(3)]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]

    for tokens in (64, 256, 1024):
        execute_planned_output_limit_call(
            client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, "low", "run-1", _planned_output_limit_call(max_completion_tokens=tokens)
        )
    sent_messages = [c["messages"] for c in client.completions.calls]
    assert sent_messages[0] == sent_messages[1] == sent_messages[2]


# --- Invocation strategy ---------------------------------------------------


def _budget(max_calls=10):
    return Budget(max_api_calls=max_calls, max_estimated_cost_usd=None, per_call_cost_ceiling=0.0)


def test_single_strategy_makes_exactly_one_call(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="capital_of_japan: Tokyo\nboiling_point_celsius: 100\nhamlet_author: William Shakespeare")
    client = fake_sync_client_factory([response])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    plan = [PlannedInvocationCall(execution_index=1, strategy="single", question_index=None)]

    result = run_invocation_strategy(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "low", "run-1", plan, _budget()
    )
    assert len(client.completions.calls) == 1
    assert len(result.measurements) == 1
    assert result.measurements[0].overall_answer_accepted is True
    assert result.workflow_wall_clock_ms is not None


def test_multiple_strategy_makes_exactly_three_calls(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content=a) for a in ("Tokyo", "100", "William Shakespeare")]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    plan = [PlannedInvocationCall(execution_index=i + 1, strategy="multiple", question_index=i) for i in range(3)]

    result = run_invocation_strategy(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "low", "run-1", plan, _budget()
    )
    assert len(client.completions.calls) == 3
    assert all(m.overall_answer_accepted for m in result.measurements)


def test_missing_answer_in_multiple_strategy_is_detected(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content=a) for a in ("Tokyo", "", "William Shakespeare")]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    plan = [PlannedInvocationCall(execution_index=i + 1, strategy="multiple", question_index=i) for i in range(3)]

    result = run_invocation_strategy(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "low", "run-1", plan, _budget()
    )
    assert result.measurements[1].answer_evaluation_status == "missing_response"
    assert result.measurements[1].overall_answer_accepted is False


def test_reasoning_tokens_not_double_counted_in_multiple_strategy(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="Tokyo", completion_tokens=20, reasoning_tokens=15) for _ in range(3)]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    plan = [PlannedInvocationCall(execution_index=i + 1, strategy="multiple", question_index=0) for i in range(3)]

    result = run_invocation_strategy(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "low", "run-1", plan, _budget()
    )
    for m in result.measurements:
        assert m.non_reasoning_completion_tokens == 5


def test_per_request_latency_kept_separately_from_measurements(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content=a) for a in ("Tokyo", "100", "William Shakespeare")]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    plan = [PlannedInvocationCall(execution_index=i + 1, strategy="multiple", question_index=i) for i in range(3)]

    result = run_invocation_strategy(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "low", "run-1", plan, _budget()
    )
    for m in result.measurements:
        assert m.client_elapsed_ms is not None
    # Workflow wall-clock is its own number, not a re-derivation of the sum
    # of per-request latencies.
    assert result.workflow_wall_clock_ms is not None


def test_budget_exhaustion_stops_the_strategy_early(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="Tokyo")]
    client = fake_sync_client_factory(responses)
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    plan = [PlannedInvocationCall(execution_index=i + 1, strategy="multiple", question_index=i) for i in range(3)]

    result = run_invocation_strategy(
        client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "low", "run-1", plan, _budget(max_calls=1)
    )
    assert len(result.measurements) == 1


def test_empty_plan_returns_no_workflow_wall_clock(fake_sync_client_factory, config_with_pricing):
    client = fake_sync_client_factory([])
    pricing = config_with_pricing.pricing["openai/gpt-oss-20b"]
    result = run_invocation_strategy(client, RETRY_CONFIG, "openai/gpt-oss-20b", pricing, 2048, "low", "run-1", [], _budget())
    assert result.measurements == []
    assert result.workflow_wall_clock_ms is None
