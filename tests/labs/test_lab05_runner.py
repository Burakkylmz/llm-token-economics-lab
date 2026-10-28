from __future__ import annotations

from labs.lab05_model_routing.golden_dataset import TASK_BY_ID
from labs.lab05_model_routing.matrix import STRATEGY_ALWAYS_LARGE, STRATEGY_ALWAYS_SMALL, STRATEGY_ROUTER
from labs.lab05_model_routing.models import PlannedRoutingCall
from labs.lab05_model_routing.runner import decide_for_strategy, execute_planned_routing_call


def test_always_small_strategy_always_selects_small_model():
    task = TASK_BY_ID["task_009"]  # a multi-step task that the router would send large
    decision = decide_for_strategy(STRATEGY_ALWAYS_SMALL, task, "small-m", "large-m")
    assert decision.selected_model == "small-m"
    assert decision.model_tier == "small"
    assert decision.matched_rules == ()


def test_always_large_strategy_always_selects_large_model():
    task = TASK_BY_ID["task_001"]  # a simple task the router would send small
    decision = decide_for_strategy(STRATEGY_ALWAYS_LARGE, task, "small-m", "large-m")
    assert decision.selected_model == "large-m"
    assert decision.model_tier == "large"


def test_router_strategy_uses_the_configured_policy():
    task = TASK_BY_ID["task_009"]
    decision = decide_for_strategy(STRATEGY_ROUTER, task, "small-m", "large-m")
    assert decision.selected_model == "large-m"
    assert decision.matched_rules


def test_unknown_strategy_raises():
    import pytest

    with pytest.raises(ValueError):
        decide_for_strategy("not_a_real_strategy", TASK_BY_ID["task_001"], "small-m", "large-m")


def _planned_call(strategy="always_small", task_id="task_001", execution_index=1):
    return PlannedRoutingCall(execution_index=execution_index, strategy=strategy, task_id=task_id)


def test_successful_call_is_evaluated_and_costed(fake_sync_client_factory, make_response, config_with_pricing):
    task = TASK_BY_ID["task_001"]  # factual_retrieval, exact_match, expected "Rome"
    response = make_response(content="Rome", prompt_tokens=50, completion_tokens=10, reasoning_tokens=2)
    client = fake_sync_client_factory([response])

    m = execute_planned_routing_call(client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(), task)

    assert m.succeeded is True
    assert m.answer_correct is True
    assert m.evaluation_status == "CORRECT"
    assert m.estimated_cost_usd is not None
    assert m.estimated_input_cost_usd is not None
    assert m.estimated_output_cost_usd is not None
    # input + output cost must reconcile with the single total figure.
    assert abs((m.estimated_input_cost_usd + m.estimated_output_cost_usd) - m.estimated_cost_usd) < 1e-12
    assert m.selected_model == config_with_pricing.lab05.small_model
    assert m.model_tier == "small"


def test_router_strategy_selects_large_model_for_multi_step_task(fake_sync_client_factory, make_response, config_with_pricing):
    task = TASK_BY_ID["task_009"]
    response = make_response(content="83", prompt_tokens=80, completion_tokens=30)
    client = fake_sync_client_factory([response])

    m = execute_planned_routing_call(
        client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(strategy="rule_based_routing", task_id="task_009"), task
    )
    assert m.selected_model == config_with_pricing.lab05.large_model
    assert m.model_tier == "large"
    assert "multi_step_reasoning_indicators" in m.matched_rules
    assert m.answer_correct is True


def test_wrong_answer_is_not_correct(fake_sync_client_factory, make_response, config_with_pricing):
    task = TASK_BY_ID["task_001"]
    response = make_response(content="Paris")
    client = fake_sync_client_factory([response])
    m = execute_planned_routing_call(client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(), task)
    assert m.answer_correct is False
    assert m.evaluation_status == "INCORRECT"


def test_reasoning_tokens_not_double_counted(fake_sync_client_factory, make_response, config_with_pricing):
    task = TASK_BY_ID["task_001"]
    response = make_response(content="Rome", completion_tokens=20, reasoning_tokens=15)
    client = fake_sync_client_factory([response])
    m = execute_planned_routing_call(client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(), task)
    assert m.non_reasoning_completion_tokens == 5


def test_missing_cached_tokens_stays_none_not_zero(fake_sync_client_factory, make_response, config_with_pricing):
    task = TASK_BY_ID["task_001"]
    response = make_response(content="Rome", cached_tokens=None)
    client = fake_sync_client_factory([response])
    m = execute_planned_routing_call(client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(), task)
    assert m.cached_tokens is None


def test_failed_call_keeps_every_optional_field_none(fake_sync_client_factory, config_with_pricing):
    import httpx
    from groq import APIConnectionError

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    client = fake_sync_client_factory([APIConnectionError(request=request) for _ in range(config_with_pricing.retry.max_attempts)])
    task = TASK_BY_ID["task_001"]

    m = execute_planned_routing_call(client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(), task)
    assert m.succeeded is False
    assert m.prompt_tokens is None
    assert m.answer_correct is None
    assert m.evaluation_status == "EVALUATION_UNAVAILABLE"


def test_routing_reason_and_matched_rules_are_persisted_on_the_measurement(fake_sync_client_factory, make_response, config_with_pricing):
    task = TASK_BY_ID["task_011"]
    response = make_response(content="5.99")
    client = fake_sync_client_factory([response])
    m = execute_planned_routing_call(
        client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(strategy="rule_based_routing", task_id="task_011"), task
    )
    assert m.routing_reason
    assert "conditional_constraints" in m.matched_rules


def test_correct_model_specific_pricing_is_used(fake_sync_client_factory, make_response, config_with_pricing):
    task = TASK_BY_ID["task_001"]
    response = make_response(content="Rome", prompt_tokens=1000, completion_tokens=1000)
    client = fake_sync_client_factory([response])
    m_small = execute_planned_routing_call(client, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(strategy="always_small"), task)

    response2 = make_response(content="Rome", prompt_tokens=1000, completion_tokens=1000)
    client2 = fake_sync_client_factory([response2])
    m_large = execute_planned_routing_call(client2, config_with_pricing.retry, config_with_pricing, "run-1", _planned_call(strategy="always_large"), task)

    # Different models, same usage -> different cost (unless prices happen to collide).
    assert m_small.estimated_cost_usd != m_large.estimated_cost_usd
