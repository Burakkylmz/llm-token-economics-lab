from __future__ import annotations

from labs.lab03_prompt_caching.cache_metrics import (
    compute_observed_request_cache_hit_rate,
    compute_request_cache_hit,
    compute_request_cache_ratio,
    compute_token_cache_ratio,
)
from labs.lab03_prompt_caching.models import CacheMeasurement


def _measurement(cached_tokens, prompt_tokens, succeeded=True) -> CacheMeasurement:
    return CacheMeasurement(
        experiment_run_id="r",
        condition_id="repeated_prefix",
        execution_index=1,
        pair_index=1,
        question_index=0,
        timestamp="t",
        model="m",
        max_completion_tokens=100,
        succeeded=succeeded,
        attempts_used=1,
        cached_tokens=cached_tokens,
        prompt_tokens=prompt_tokens,
    )


def test_request_cache_hit_none_when_unreported():
    assert compute_request_cache_hit(None) is None


def test_request_cache_hit_true_on_positive_cached_tokens():
    assert compute_request_cache_hit(100) is True


def test_request_cache_hit_false_on_reported_zero():
    assert compute_request_cache_hit(0) is False


def test_request_cache_ratio_none_when_missing():
    assert compute_request_cache_ratio(None, 1000) is None
    assert compute_request_cache_ratio(100, None) is None
    assert compute_request_cache_ratio(100, 0) is None


def test_request_cache_ratio_computed_correctly():
    assert compute_request_cache_ratio(400, 1000) == 0.4


def test_metric_a_observed_hit_rate_example_from_spec():
    # 3 calls, 2 of them cache hits -> 66.7%
    measurements = [_measurement(100, 1000), _measurement(0, 1000), _measurement(50, 1000)]
    rate = compute_observed_request_cache_hit_rate(measurements)
    assert round(rate, 3) == round(2 / 3, 3)


def test_metric_a_none_when_nothing_reported():
    measurements = [_measurement(None, 1000), _measurement(None, 1000)]
    assert compute_observed_request_cache_hit_rate(measurements) is None


def test_metric_b_token_cache_ratio_example_from_spec():
    # total 6000 input tokens, 4800 cached -> 80%
    measurements = [_measurement(4800, 6000)]
    ratio = compute_token_cache_ratio(measurements)
    assert ratio == 0.8


def test_metric_a_and_metric_b_are_not_the_same_number():
    # 2 calls: one with a tiny cache hit on a huge prompt, one with a full
    # miss on a small prompt. Request hit rate = 50%, but token ratio is
    # weighted toward the big prompt's small hit -- a different number.
    measurements = [_measurement(10, 10000), _measurement(0, 100)]
    hit_rate = compute_observed_request_cache_hit_rate(measurements)
    token_ratio = compute_token_cache_ratio(measurements)
    assert hit_rate == 0.5
    assert token_ratio == 10 / 10100
    assert hit_rate != token_ratio


def test_failed_calls_excluded_from_both_metrics():
    measurements = [_measurement(100, 1000, succeeded=False)]
    assert compute_observed_request_cache_hit_rate(measurements) is None
    assert compute_token_cache_ratio(measurements) is None
