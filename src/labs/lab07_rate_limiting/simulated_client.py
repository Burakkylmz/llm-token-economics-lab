"""A controlled rate-limit failure injector.

Wraps any client exposing `.chat.completions.create` and raises a real
`groq.RateLimitError` (with a genuine httpx.Response carrying a 429 status
and an optional retry-after header) for the first `fail_count` calls, then
delegates to the wrapped client normally.

This is deliberate, documented simulation -- never real excessive traffic
against the Groq API -- exercising the exact exception type and
header-parsing path a production retry layer would have to handle.
"""

from __future__ import annotations

from typing import Optional

import httpx
from groq import RateLimitError


def make_rate_limit_error(retry_after_seconds: Optional[str] = None) -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after_seconds} if retry_after_seconds is not None else {}
    response = httpx.Response(status_code=429, headers=headers, request=request)
    return RateLimitError("Rate limit exceeded (simulated)", response=response, body=None)


class _FlakyCompletions:
    def __init__(self, real_completions, fail_count: int, retry_after_seconds: Optional[str]):
        self._real = real_completions
        self._fail_count = fail_count
        self._retry_after_seconds = retry_after_seconds
        self.attempt_count = 0

    def create(self, **kwargs):
        self.attempt_count += 1
        if self.attempt_count <= self._fail_count:
            raise make_rate_limit_error(self._retry_after_seconds)
        return self._real.create(**kwargs)


class _ChatProxy:
    def __init__(self, completions):
        self.completions = completions


class FlakyClient:
    def __init__(self, real_client, fail_count: int, retry_after_seconds: Optional[str] = None):
        self.chat = _ChatProxy(_FlakyCompletions(real_client.chat.completions, fail_count, retry_after_seconds))
