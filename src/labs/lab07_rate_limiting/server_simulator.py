"""A single, configurable fake server used to build every Lab07 scenario.

Deliberately NOT five separate scenario classes: each of the five
scenarios (Part F/G) is just a different parameterization of the same
three independent mechanisms a server might apply, checked in a fixed,
documented order:

  1. a per-client fixed leading-failure count (Scenario 1: Fixed Transient
     Failures),
  2. a global "not available before this virtual time" recovery window
     (Scenario 2: Server Recovery Window),
  3. a one-shot, per-client Retry-After header injected on that client's
     first attempt (Scenario 4: Retry-After Header Behavior),
  4. a shared, per-time-window request/token capacity (Scenario 3: Shared
     Contention, and Scenario 5: Request/Token Quota Pressure -- RPM uses
     only the request cap, TPM adds the token cap).

Never issues a real HTTP request, never imports anything from `groq` --
this is a pure, offline, in-memory state machine over virtual time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerResponse:
    http_status: int  # 200 or 429 -- this simulator never returns any other status
    retry_after_header: Optional[str] = None
    # True only when THIS 429 was caused by shared-capacity contention
    # (mechanism 4 above) -- the metric labs.lab07_rate_limiting.stats
    # calls retry_collisions counts exactly these, never fixed-fail-count
    # or recovery-window rejections (which are not "collisions" between
    # clients).
    collision: bool = False


@dataclass
class SimulatedServer:
    # Mechanism 1 -- Scenario 1 (Fixed Transient Failures): reject every
    # client's first `fail_count` attempts unconditionally. The legacy
    # `--retry-after-seconds` CLI flag sets `fail_count_retry_after` to
    # inject a fixed header on those same rejections (None = no header,
    # the historical default).
    fail_count: int = 0
    fail_count_retry_after: Optional[str] = None

    # Mechanism 2 -- Scenario 2 (Server Recovery Window): reject EVERY
    # attempt (any client) before this virtual time.
    recovery_time: float = 0.0

    # Mechanism 3 -- Scenario 4 (Retry-After Header Behavior): on a
    # client's first attempt only, if that client_id is a key here,
    # return 429 with exactly this raw header string (never parsed here --
    # labs.lab07_rate_limiting.retry_after does that on the consuming
    # side).
    retry_after_variants: dict[int, Optional[str]] = field(default_factory=dict)

    # Mechanism 4 -- Scenario 3 (Shared Contention) / Scenario 5 (Quota
    # Pressure): a shared capacity budget per fixed-width virtual-time
    # window, covering ALL clients together. `capacity_requests_per_window`
    # alone models an RPM-style cap; adding `capacity_tokens_per_window`
    # (with each request's own `token_demand`) additionally models a
    # TPM-style cap -- whichever is hit first rejects the request.
    window_seconds: Optional[float] = None
    capacity_requests_per_window: Optional[int] = None
    capacity_tokens_per_window: Optional[int] = None

    _window_usage: dict[int, tuple[int, int]] = field(default_factory=dict, repr=False, compare=False)

    def handle(self, virtual_time: float, client_id: int, attempt_index: int, token_demand: int = 1) -> ServerResponse:
        if attempt_index <= self.fail_count:
            return ServerResponse(429, self.fail_count_retry_after)

        if virtual_time < self.recovery_time:
            return ServerResponse(429, None)

        if attempt_index == 1 and client_id in self.retry_after_variants:
            return ServerResponse(429, self.retry_after_variants[client_id])

        if self.window_seconds is not None:
            window_index = int(virtual_time // self.window_seconds)
            used_requests, used_tokens = self._window_usage.get(window_index, (0, 0))
            over_requests = (
                self.capacity_requests_per_window is not None and used_requests >= self.capacity_requests_per_window
            )
            over_tokens = (
                self.capacity_tokens_per_window is not None
                and used_tokens + token_demand > self.capacity_tokens_per_window
            )
            if over_requests or over_tokens:
                remaining_in_window = self.window_seconds - (virtual_time % self.window_seconds)
                return ServerResponse(429, f"{remaining_in_window:.2f}", collision=True)
            self._window_usage[window_index] = (used_requests + 1, used_tokens + token_demand)

        return ServerResponse(200, None)
