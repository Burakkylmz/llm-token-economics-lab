"""Robust, offline Retry-After header parsing.

Replaces the previous `delay = float(retry_after_header)`, which raised an
uncaught exception on any malformed value. Distinguishes HTTP's standard
`delta-seconds`/`HTTP-date` forms from provider-specific decimal-second
extensions, and always returns a safe, explicit (valid, reason) result --
never raises.

HTTP-date parsing never reads the real wall clock: `now_utc` must be
injected (Lab07's virtual-clock discipline) or HTTP-date headers are
reported as unusable ("http_date_without_injected_clock") rather than
silently falling back to `datetime.now()`.

An "excessively large" delay (above `MAX_SANE_RETRY_AFTER_SECONDS`) is
treated as INVALID, not truncated -- Part I's explicit rule: "Retry-After
süresini sessizce kısaltarak header'a uyulduğunu iddia etme." A caller
that sees `valid=False` falls back to its own documented policy (see
labs.lab07_rate_limiting.runner) instead of waiting a clamped amount while
still claiming to have honored the header.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

# A provider could never reasonably ask a client to wait more than this --
# a value above it is treated as a malformed/untrustworthy header, not
# honored (even partially). One hour, documented, not derived from any
# real Groq-published ceiling (none was found).
MAX_SANE_RETRY_AFTER_SECONDS = 3600.0


@dataclass
class ParsedRetryAfter:
    delay_seconds: Optional[float]
    valid: bool
    reason: str


def parse_retry_after(header_value: Optional[str], now_utc: Optional[datetime] = None) -> ParsedRetryAfter:
    if header_value is None:
        return ParsedRetryAfter(None, False, "missing")

    stripped = header_value.strip()
    if not stripped:
        return ParsedRetryAfter(None, False, "missing")

    try:
        seconds = float(stripped)
    except ValueError:
        seconds = None

    if seconds is not None:
        if math.isnan(seconds):
            return ParsedRetryAfter(None, False, "nan")
        if math.isinf(seconds):
            return ParsedRetryAfter(None, False, "infinite")
        if seconds < 0:
            return ParsedRetryAfter(None, False, "negative")
        if seconds > MAX_SANE_RETRY_AFTER_SECONDS:
            return ParsedRetryAfter(None, False, "excessive")
        return ParsedRetryAfter(seconds, True, "numeric_seconds")

    try:
        parsed_date = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, IndexError):
        return ParsedRetryAfter(None, False, "malformed")
    if parsed_date is None:
        return ParsedRetryAfter(None, False, "malformed")

    if now_utc is None:
        return ParsedRetryAfter(None, False, "http_date_without_injected_clock")

    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=now_utc.tzinfo)

    delta_seconds = (parsed_date - now_utc).total_seconds()
    if delta_seconds < 0:
        return ParsedRetryAfter(0.0, True, "http_date_in_past")
    if delta_seconds > MAX_SANE_RETRY_AFTER_SECONDS:
        return ParsedRetryAfter(None, False, "excessive")
    return ParsedRetryAfter(delta_seconds, True, "http_date")
