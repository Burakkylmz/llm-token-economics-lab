"""Regression tests for the shared live API safety/measurement setup."""

from types import SimpleNamespace

import pytest

from lab_core import lab_cli_support as support


def _config(*, live_enabled=True):
    return SimpleNamespace(
        execution=SimpleNamespace(
            live_api_enabled=live_enabled,
            cost_warning_message="Test requests may incur cost",
        ),
        groq=SimpleNamespace(request_timeout_seconds=17.5),
        experiment=SimpleNamespace(results_dir="results"),
    )


def test_shared_clients_receive_explicit_timeout_and_zero_sdk_retries(monkeypatch):
    constructed = []

    def fake_client(**kwargs):
        constructed.append(kwargs)
        return object()

    monkeypatch.setattr(support, "load_config", lambda path: _config())
    monkeypatch.setattr(support, "Groq", fake_client)
    monkeypatch.setattr(support, "AsyncGroq", fake_client)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    context = support.build_context(requested_live=True)
    assert context.live is True
    assert context.client is not None
    assert context.async_client is not None
    assert constructed == [
        {"api_key": "test-key-not-real", "timeout": 17.5, "max_retries": 0},
        {"api_key": "test-key-not-real", "timeout": 17.5, "max_retries": 0},
    ]


def test_dry_run_never_constructs_clients(monkeypatch):
    monkeypatch.setattr(support, "load_config", lambda path: _config())
    monkeypatch.setattr(support, "Groq", lambda **kwargs: pytest.fail("unexpected sync client"))
    monkeypatch.setattr(support, "AsyncGroq", lambda **kwargs: pytest.fail("unexpected async client"))

    context = support.build_context(requested_live=False)
    assert context.live is False
    assert context.client is None
    assert context.async_client is None


def test_config_guard_precedes_client_creation(monkeypatch):
    monkeypatch.setattr(support, "load_config", lambda path: _config(live_enabled=False))
    monkeypatch.setattr(support, "Groq", lambda **kwargs: pytest.fail("unexpected client"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    with pytest.raises(support.LiveExecutionNotAllowed):
        support.build_context(requested_live=True)
