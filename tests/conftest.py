from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


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
