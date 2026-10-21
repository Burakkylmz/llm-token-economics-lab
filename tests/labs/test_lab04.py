from __future__ import annotations

from labs.lab04_output_control.experiments import (
    QUESTIONS,
    run_multiple_invocations,
    run_output_limit_sweep,
    run_single_invocation,
    summarize_invocation_comparison,
)


def test_output_limit_sweep_passes_each_max_tokens_value(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response(completion_tokens=n) for n in (16, 64, 256)])
    results = run_output_limit_sweep(client, config_with_pricing, "llama-3.1-8b-instant", [16, 64, 256])

    assert [call["max_tokens"] for call in client.completions.calls] == [16, 64, 256]
    assert [r.metadata.model_parameters["max_tokens"] for r in results] == [16, 64, 256]


def test_single_invocation_makes_exactly_one_call_for_all_questions(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response(content="1. Tokyo\n2. 100\n3. Shakespeare")])
    results = run_single_invocation(client, config_with_pricing, "llama-3.1-8b-instant")

    assert len(client.completions.calls) == 1
    sent = client.completions.calls[0]["messages"][0]["content"]
    assert all(q in sent for q in QUESTIONS)
    assert results[0].metadata.model_parameters["question_count"] == len(QUESTIONS)


def test_multiple_invocations_makes_one_call_per_question(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response() for _ in QUESTIONS])
    results = run_multiple_invocations(client, config_with_pricing, "llama-3.1-8b-instant")

    assert len(client.completions.calls) == len(QUESTIONS)
    assert len(results) == len(QUESTIONS)


def test_summarize_invocation_comparison_reflects_call_count_difference(fake_sync_client_factory, make_response, config_with_pricing):
    single_client = fake_sync_client_factory([make_response(prompt_tokens=50, completion_tokens=30)])
    single_results = run_single_invocation(single_client, config_with_pricing, "llama-3.1-8b-instant")

    multi_client = fake_sync_client_factory([make_response(prompt_tokens=20, completion_tokens=10) for _ in QUESTIONS])
    multi_results = run_multiple_invocations(multi_client, config_with_pricing, "llama-3.1-8b-instant")

    summary = summarize_invocation_comparison(single_results, multi_results)
    assert summary["single_invocation"]["api_call_count"] == 1
    assert summary["multiple_invocations"]["api_call_count"] == len(QUESTIONS)
    assert summary["multiple_invocations"]["total_prompt_tokens"] == 20 * len(QUESTIONS)
