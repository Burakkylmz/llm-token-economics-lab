from __future__ import annotations

from datetime import datetime, timedelta, timezone

from labs.lab07_rate_limiting.retry_after import MAX_SANE_RETRY_AFTER_SECONDS, parse_retry_after
from labs.lab07_rate_limiting.server_simulator import SimulatedServer

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --- Retry-After parsing ----------------------------------------------------


def test_missing_header_is_invalid():
    result = parse_retry_after(None)
    assert result.valid is False
    assert result.reason == "missing"


def test_empty_string_header_is_invalid():
    result = parse_retry_after("   ")
    assert result.valid is False
    assert result.reason == "missing"


def test_valid_integer_seconds():
    result = parse_retry_after("7")
    assert result.valid is True
    assert result.delay_seconds == 7.0
    assert result.reason == "numeric_seconds"


def test_valid_decimal_seconds():
    result = parse_retry_after("2.5")
    assert result.valid is True
    assert result.delay_seconds == 2.5


def test_zero_is_valid_immediate_retry():
    result = parse_retry_after("0")
    assert result.valid is True
    assert result.delay_seconds == 0.0


def test_negative_is_invalid():
    result = parse_retry_after("-5")
    assert result.valid is False
    assert result.reason == "negative"


def test_nan_is_invalid():
    result = parse_retry_after("nan")
    assert result.valid is False
    assert result.reason == "nan"


def test_infinity_is_invalid():
    result = parse_retry_after("inf")
    assert result.valid is False
    assert result.reason == "infinite"


def test_malformed_string_is_invalid():
    result = parse_retry_after("not-a-number")
    assert result.valid is False
    assert result.reason == "malformed"


def test_excessively_large_value_is_invalid_never_truncated():
    result = parse_retry_after(str(MAX_SANE_RETRY_AFTER_SECONDS + 1))
    assert result.valid is False
    assert result.reason == "excessive"
    assert result.delay_seconds is None  # never a clamped/truncated value


def test_http_date_valid_with_injected_clock():
    future = _NOW + timedelta(seconds=10)
    header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = parse_retry_after(header, now_utc=_NOW)
    assert result.valid is True
    assert abs(result.delay_seconds - 10.0) < 1e-6
    assert result.reason == "http_date"


def test_http_date_without_injected_clock_is_unusable_not_real_wall_clock():
    future = _NOW + timedelta(seconds=10)
    header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = parse_retry_after(header, now_utc=None)
    assert result.valid is False
    assert result.reason == "http_date_without_injected_clock"


def test_http_date_in_the_past_is_treated_as_immediate():
    past = _NOW - timedelta(seconds=10)
    header = past.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = parse_retry_after(header, now_utc=_NOW)
    assert result.valid is True
    assert result.delay_seconds == 0.0


def test_excessive_http_date_is_invalid():
    future = _NOW + timedelta(seconds=MAX_SANE_RETRY_AFTER_SECONDS + 100)
    header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    result = parse_retry_after(header, now_utc=_NOW)
    assert result.valid is False
    assert result.reason == "excessive"


def test_never_raises_on_any_input():
    for value in (None, "", "   ", "abc", "-1", "nan", "inf", "99999999999", "garbage date string"):
        parse_retry_after(value, now_utc=_NOW)  # must not raise


# --- Simulated server --------------------------------------------------------


def test_fixed_transient_failures_rejects_exactly_fail_count_times():
    server = SimulatedServer(fail_count=2)
    statuses = [server.handle(0.0, 0, attempt).http_status for attempt in (1, 2, 3)]
    assert statuses == [429, 429, 200]


def test_recovery_window_rejects_before_recovery_time():
    server = SimulatedServer(recovery_time=3.0)
    assert server.handle(2.9, 0, 1).http_status == 429
    assert server.handle(3.0, 0, 1).http_status == 200


def test_retry_after_variant_only_applies_to_first_attempt():
    server = SimulatedServer(retry_after_variants={0: "5"})
    first = server.handle(0.0, 0, 1)
    assert first.http_status == 429
    assert first.retry_after_header == "5"
    second = server.handle(0.0, 0, 2)
    assert second.http_status == 200  # no second rejection -- one-shot


def test_shared_capacity_rejects_once_window_is_full():
    server = SimulatedServer(window_seconds=1.0, capacity_requests_per_window=2)
    responses = [server.handle(0.1, cid, 1) for cid in (0, 1, 2)]
    statuses = [r.http_status for r in responses]
    assert statuses == [200, 200, 429]
    assert responses[2].collision is True


def test_shared_capacity_resets_in_the_next_window():
    server = SimulatedServer(window_seconds=1.0, capacity_requests_per_window=1)
    assert server.handle(0.1, 0, 1).http_status == 200
    assert server.handle(0.2, 1, 1).http_status == 429
    assert server.handle(1.1, 2, 1).http_status == 200  # new window, capacity reset


def test_token_quota_rejects_when_token_demand_exceeds_cap():
    server = SimulatedServer(window_seconds=10.0, capacity_tokens_per_window=150)
    first = server.handle(0.0, 0, 1, token_demand=100)
    second = server.handle(0.0, 1, 1, token_demand=100)
    assert first.http_status == 200
    assert second.http_status == 429
    assert second.collision is True


def test_legacy_fail_count_retry_after_injection():
    server = SimulatedServer(fail_count=1, fail_count_retry_after="9")
    response = server.handle(0.0, 0, 1)
    assert response.http_status == 429
    assert response.retry_after_header == "9"


def test_mechanisms_apply_in_documented_order_fail_count_before_recovery():
    # recovery_time is in the future, AND fail_count is also set -- the
    # fixed-fail-count mechanism (checked first) determines the outcome
    # for attempts within its count, regardless of recovery_time.
    server = SimulatedServer(fail_count=1, recovery_time=100.0)
    assert server.handle(0.0, 0, 1).http_status == 429  # fail_count rule
    # attempt 2 is beyond fail_count, so now recovery_time's rule applies.
    assert server.handle(0.0, 0, 2).http_status == 429  # recovery_time rule (0.0 < 100.0)
