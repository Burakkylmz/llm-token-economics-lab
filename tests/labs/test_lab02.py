from __future__ import annotations

from labs.lab02_context_growth.experiments import (
    CONDITIONS,
    EXPANDED_RELEVANT_CONTEXT,
    IRRELEVANT_CONTEXT,
    RELEVANT_CONTEXT,
    run_context_experiment,
    run_multi_turn_growth,
)


def test_expanded_context_is_larger_than_minimal():
    assert len(EXPANDED_RELEVANT_CONTEXT) > len(RELEVANT_CONTEXT)


def test_irrelevant_context_contains_the_relevant_fact_too():
    # irrelevant_context must still contain the answer, so quality is held
    # constant across conditions -- only prompt size/relevance varies.
    assert RELEVANT_CONTEXT in IRRELEVANT_CONTEXT


def test_run_context_experiment_sends_context_in_prompt(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="5", prompt_tokens=50, completion_tokens=2)
    client = fake_sync_client_factory([response])

    results = run_context_experiment(
        client=client,
        config=config_with_pricing,
        condition="minimal_context",
        model="llama-3.1-8b-instant",
        repetitions=1,
    )

    assert len(results) == 1
    sent_content = client.completions.calls[0]["messages"][1]["content"]
    assert RELEVANT_CONTEXT in sent_content
    assert results[0].metadata.model_parameters["context_chars"] == len(CONDITIONS["minimal_context"])


def test_run_multi_turn_growth_grows_message_history(fake_sync_client_factory, make_response, config_with_pricing):
    responses = [make_response(content=f"fact-{i}", prompt_tokens=10 * i, completion_tokens=5) for i in range(1, 4)]
    client = fake_sync_client_factory(responses)

    results = run_multi_turn_growth(client=client, config=config_with_pricing, model="llama-3.1-8b-instant", turns=3)

    assert len(results) == 3
    # Each successive call should carry more messages than the last (growing history).
    message_counts = [len(call["messages"]) for call in client.completions.calls]
    assert message_counts == sorted(message_counts)
    assert message_counts[0] < message_counts[-1]
    # Prior assistant replies must be threaded back into the conversation.
    assert client.completions.calls[1]["messages"][-2]["content"] == "fact-1"
