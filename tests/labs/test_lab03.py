from __future__ import annotations

from labs.lab03_prompt_caching.experiments import (
    STATIC_PREFIX,
    observed_cache_hit_rate,
    run_cache_unfriendly,
    run_repeated_prefix,
)


def test_repeated_prefix_sends_identical_system_prompt_every_time(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response(cached_tokens=100) for _ in range(3)])
    run_repeated_prefix(client=client, config=config_with_pricing, model="openai/gpt-oss-20b", repetitions=3)

    prefixes = [call["messages"][0]["content"] for call in client.completions.calls]
    assert all(p == STATIC_PREFIX for p in prefixes)


def test_cache_unfriendly_sends_a_different_prefix_every_time(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response(cached_tokens=0) for _ in range(3)])
    run_cache_unfriendly(client=client, config=config_with_pricing, model="openai/gpt-oss-20b", repetitions=3)

    prefixes = [call["messages"][0]["content"] for call in client.completions.calls]
    assert len(set(prefixes)) == 3


def test_observed_cache_hit_rate_counts_nonzero_cached_tokens(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory(
        [make_response(cached_tokens=100), make_response(cached_tokens=0), make_response(cached_tokens=50)]
    )
    results = run_repeated_prefix(client=client, config=config_with_pricing, model="openai/gpt-oss-20b", repetitions=3)
    assert observed_cache_hit_rate(results) == 2 / 3


def test_observed_cache_hit_rate_is_none_when_unreported(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response(cached_tokens=None) for _ in range(2)])
    results = run_repeated_prefix(client=client, config=config_with_pricing, model="llama-3.1-8b-instant", repetitions=2)
    assert observed_cache_hit_rate(results) is None
