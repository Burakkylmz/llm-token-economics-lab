from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def config_with_pricing(tmp_path):
    """A fully valid AppConfig with plausible (test-only, not verified) pricing
    for every model used across the lab test suites."""
    from lab_core.config import load_config

    toml_content = """
[groq]
default_model = "llama-3.1-8b-instant"
request_timeout_seconds = 30.0

[execution]
live_api_enabled = false
cost_warning_message = "warning"

[experiment]
default_repetitions = 3
results_dir = "results"

[retry]
max_attempts = 3
base_delay_seconds = 1.0
max_delay_seconds = 10.0
jitter = true

[concurrency]
max_concurrent_requests = 4

[pricing.models."llama-3.1-8b-instant"]
input_per_1m_usd = 0.05
output_per_1m_usd = 0.08
cached_input_per_1m_usd = 0.025
price_verified_date = "2026-10-21"

[pricing.models."llama-3.3-70b-versatile"]
input_per_1m_usd = 0.59
output_per_1m_usd = 0.79
cached_input_per_1m_usd = 0.295
price_verified_date = "2026-10-21"

[pricing.models."openai/gpt-oss-20b"]
input_per_1m_usd = 0.075
output_per_1m_usd = 0.3
cached_input_per_1m_usd = 0.0375
price_verified_date = "2026-10-21"

[pricing.models."openai/gpt-oss-120b"]
input_per_1m_usd = 0.15
output_per_1m_usd = 0.6
cached_input_per_1m_usd = 0.075
price_verified_date = "2026-10-21"

[lab03]
caching_capable_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

[lab05]
small_model = "llama-3.1-8b-instant"
large_model = "llama-3.3-70b-versatile"

[lab09]
batch_completion_window = "24h"
"""
    path = tmp_path / "test_config.toml"
    path.write_text(toml_content, encoding="utf-8")
    return load_config(path)


@pytest.fixture
def make_usage():
    """Builds a fake Groq SDK `usage` object with only the attributes given.

    Mirrors the real SDK's behavior of exposing attributes via attribute
    access (not dict access), and lets a test simulate a response where
    some fields (e.g. cached_tokens, reasoning_tokens) are simply absent.
    """

    def _make(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        total_time=0.5,
        cached_tokens=None,
        reasoning_tokens=None,
    ):
        prompt_tokens_details = (
            SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None
        )
        completion_tokens_details = (
            SimpleNamespace(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens is not None
            else None
        )
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_time=total_time,
            prompt_tokens_details=prompt_tokens_details,
            completion_tokens_details=completion_tokens_details,
        )

    return _make


@pytest.fixture
def make_response(make_usage):
    """Builds a fake non-streaming chat-completion response object."""

    def _make(content: str = "hello", **usage_kwargs):
        usage = make_usage(**usage_kwargs)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage)

    return _make


@pytest.fixture
def make_chunk():
    """Builds a fake streaming chunk with the shape lab_core.groq_client.call_stream consumes."""

    def _make(content: str | None = None):
        delta = SimpleNamespace(content=content)
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice])

    return _make


@pytest.fixture
def fake_sync_client_factory():
    """Builds a minimal fake Groq client that returns canned responses in order.

    Mirrors only the shape lab_core.groq_client.call_sync actually uses
    (client.chat.completions.create(**kwargs)), and records every call's
    kwargs in `.calls` so tests can assert on what was sent.
    """

    class _FakeCompletions:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls: list[dict] = []

        def create(self, **kwargs):
            # Snapshot mutable arguments (e.g. a `messages` list a caller may
            # keep appending to across turns) so recorded calls reflect what
            # was sent at call time, not whatever the list looks like later.
            snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in kwargs.items()}
            self.calls.append(snapshot)
            if not self._responses:
                raise AssertionError("fake client ran out of canned responses")
            next_response = self._responses.pop(0)
            if isinstance(next_response, Exception):
                raise next_response
            return next_response

    class _FakeChat:
        def __init__(self, completions):
            self.completions = completions

    class _FakeClient:
        def __init__(self, responses):
            self.completions = _FakeCompletions(responses)
            self.chat = _FakeChat(self.completions)

    def _make(responses):
        return _FakeClient(responses)

    return _make


@pytest.fixture
def fake_async_client_factory():
    """Async counterpart of fake_sync_client_factory."""

    class _FakeAsyncCompletions:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls: list[dict] = []

        async def create(self, **kwargs):
            snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in kwargs.items()}
            self.calls.append(snapshot)
            if not self._responses:
                raise AssertionError("fake async client ran out of canned responses")
            next_response = self._responses.pop(0)
            if isinstance(next_response, Exception):
                raise next_response
            return next_response

    class _FakeAsyncChat:
        def __init__(self, completions):
            self.completions = completions

    class _FakeAsyncClient:
        def __init__(self, responses):
            self.completions = _FakeAsyncCompletions(responses)
            self.chat = _FakeAsyncChat(self.completions)

    def _make(responses):
        return _FakeAsyncClient(responses)

    return _make
