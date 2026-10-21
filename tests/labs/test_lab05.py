from __future__ import annotations

from labs.lab05_model_routing.experiments import (
    run_fixed_model_baseline,
    run_rule_based_routing,
    summarize_quality_cost_latency,
)
from labs.lab05_model_routing.golden_dataset import GOLDEN_DATASET, score
from labs.lab05_model_routing.routing import choose_model


def test_golden_dataset_check_passes_on_correct_answer():
    example = next(e for e in GOLDEN_DATASET if e.task_id == "simple_capital")
    assert score(example, "Rome") is True
    assert score(example, "I don't know") is False


def test_router_sends_short_simple_prompts_to_small_model():
    assert choose_model("What is 2 + 2?", "small", "large") == "small"


def test_router_sends_code_prompts_to_large_model():
    assert choose_model("Write a Python function that sorts a list.", "small", "large") == "large"


def test_fixed_model_baseline_calls_once_per_dataset_item(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="Rome"), make_response(content="19"), make_response(content="9"), make_response(content="def is_palindrome(s): pass")]
    client = fake_sync_client_factory(responses)

    results = run_fixed_model_baseline(client, config_with_pricing, "llama-3.1-8b-instant")

    assert len(results) == len(GOLDEN_DATASET)
    assert all(r.metadata.model_id == "llama-3.1-8b-instant" for r in results)
    assert [r.metadata.model_parameters["passed"] for r in results] == [True, True, True, True]


def test_rule_based_routing_uses_both_models(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="Rome"), make_response(content="19"), make_response(content="9"), make_response(content="def is_palindrome(s): pass")]
    client = fake_sync_client_factory(responses)

    results = run_rule_based_routing(client, config_with_pricing)

    used_models = {r.metadata.model_id for r in results}
    assert used_models == {config_with_pricing.lab05.small_model, config_with_pricing.lab05.large_model}


def test_summarize_quality_cost_latency_computes_accuracy(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content="Rome"), make_response(content="wrong"), make_response(content="9"), make_response(content="nope")]
    client = fake_sync_client_factory(responses)

    results = run_fixed_model_baseline(client, config_with_pricing, "llama-3.1-8b-instant")
    summary = summarize_quality_cost_latency(results)
    assert summary["accuracy"] == 0.5
