from __future__ import annotations

import pytest

from lab_core.retry import compute_backoff_delay_seconds


def test_delay_grows_exponentially_without_jitter():
    delays = [
        compute_backoff_delay_seconds(attempt=a, base_delay_seconds=1.0, max_delay_seconds=100.0, jitter=False)
        for a in (1, 2, 3)
    ]
    assert delays == [1.0, 2.0, 4.0]


def test_delay_is_capped_at_max_delay():
    delay = compute_backoff_delay_seconds(attempt=10, base_delay_seconds=1.0, max_delay_seconds=5.0, jitter=False)
    assert delay == 5.0


def test_jitter_stays_within_bounds():
    for _ in range(50):
        delay = compute_backoff_delay_seconds(attempt=3, base_delay_seconds=1.0, max_delay_seconds=100.0, jitter=True)
        assert 0 <= delay <= 4.0


def test_attempt_must_be_at_least_one():
    with pytest.raises(ValueError):
        compute_backoff_delay_seconds(attempt=0, base_delay_seconds=1.0, max_delay_seconds=10.0, jitter=False)
