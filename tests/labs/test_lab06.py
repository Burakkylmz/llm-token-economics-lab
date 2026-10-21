from __future__ import annotations

from labs.lab06_streaming.experiments import run_non_streaming, run_streaming


def test_non_streaming_computes_output_tokens_per_second(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response(completion_tokens=100)])
    result = run_non_streaming(client, config_with_pricing, "llama-3.1-8b-instant")

    assert result.tokens_and_cost.completion_tokens == 100
    assert result.performance.output_tokens_per_second is not None
    assert result.performance.output_tokens_per_second > 0


def test_streaming_measures_ttft_before_first_visible_content(fake_sync_client_factory, make_chunk, config_with_pricing):
    # First chunk carries no visible text (e.g. a role-only delta), second chunk does.
    chunks = [make_chunk(content=None), make_chunk(content="Once"), make_chunk(content=" upon a time")]
    client = fake_sync_client_factory([chunks])

    result = run_streaming(client, config_with_pricing, "llama-3.1-8b-instant")

    assert result.performance.ttft_ms is not None
    assert result.performance.first_visible_content_ms is not None
    assert result.performance.ttft_ms <= result.performance.first_visible_content_ms
    assert result.metadata.model_parameters["chunk_count"] == 3


def test_streaming_never_fabricates_token_usage(fake_sync_client_factory, make_chunk, config_with_pricing):
    client = fake_sync_client_factory([[make_chunk(content="hi")]])
    result = run_streaming(client, config_with_pricing, "llama-3.1-8b-instant")
    assert result.tokens_and_cost.prompt_tokens is None
    assert result.tokens_and_cost.completion_tokens is None
