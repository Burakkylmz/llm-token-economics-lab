"""The five Lab07 scenarios, each a specific parameterization of
labs.lab07_rate_limiting.server_simulator.SimulatedServer (see that
module's docstring for why one configurable server, not five classes).

Scenario 1-3 are this project's documented Default Grand Test; 4-5 are
Extended-profile-only (Part G, "optional advanced simulations").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Optional

from labs.lab07_rate_limiting.server_simulator import SimulatedServer

SCENARIO_TRANSIENT_FAILURES = "transient_failures"
SCENARIO_SERVER_RECOVERY_WINDOW = "server_recovery_window"
SCENARIO_SHARED_CONTENTION = "shared_contention"
SCENARIO_RETRY_AFTER_VARIATIONS = "retry_after_variations"
SCENARIO_QUOTA_PRESSURE = "quota_pressure"

PILOT_SCENARIOS: tuple[str, ...] = (SCENARIO_TRANSIENT_FAILURES,)
DEFAULT_SCENARIOS: tuple[str, ...] = (SCENARIO_TRANSIENT_FAILURES, SCENARIO_SERVER_RECOVERY_WINDOW, SCENARIO_SHARED_CONTENTION)
EXTENDED_ONLY_SCENARIOS: tuple[str, ...] = (SCENARIO_RETRY_AFTER_VARIATIONS, SCENARIO_QUOTA_PRESSURE)
EXTENDED_SCENARIOS: tuple[str, ...] = DEFAULT_SCENARIOS + EXTENDED_ONLY_SCENARIOS

# A fixed, arbitrary epoch used ONLY to anchor virtual time for HTTP-date
# Retry-After parsing (Part I) -- never the real wall clock.
SIMULATION_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def virtual_time_to_utc(virtual_time_seconds: float) -> datetime:
    return SIMULATION_EPOCH + timedelta(seconds=virtual_time_seconds)


# Scenario 4's eight header variants (Part G/I) -- client_id -> raw header
# value the simulated server returns on that client's first attempt.
RETRY_AFTER_VARIANT_LABELS: dict[int, str] = {
    0: "valid_integer",
    1: "valid_decimal",
    2: "missing",
    3: "malformed",
    4: "zero",
    5: "negative",
    6: "exceeds_wait_budget",
    7: "http_date",
}
RETRY_AFTER_VARIANT_HEADERS: dict[int, Optional[str]] = {
    0: "5",
    1: "2.5",
    2: None,
    3: "not-a-number",
    4: "0",
    5: "-5",
    6: "80",
    7: format_datetime(virtual_time_to_utc(3.0), usegmt=True),
}


@dataclass
class ScenarioSpec:
    scenario_id: str
    server: SimulatedServer
    client_ids: tuple[int, ...]
    token_demand: int = 1
    # Overrides this run's lab07.max_total_wait_seconds, only when the
    # scenario needs a specific (documented) budget to make its point --
    # e.g. Scenario 4's "exceeds_wait_budget" variant.
    max_total_wait_seconds_override: Optional[float] = None


def build_scenario(
    scenario_id: str,
    *,
    transient_fail_count: int,
    recovery_time_seconds: float,
    contention_client_count: int,
    contention_capacity_per_window: int,
    contention_window_seconds: float,
    quota_client_count: int,
    quota_rpm: int,
    quota_tpm: int,
    quota_token_demand_per_request: int,
    quota_window_seconds: float,
) -> ScenarioSpec:
    if scenario_id == SCENARIO_TRANSIENT_FAILURES:
        return ScenarioSpec(scenario_id, SimulatedServer(fail_count=transient_fail_count), (0,))

    if scenario_id == SCENARIO_SERVER_RECOVERY_WINDOW:
        return ScenarioSpec(scenario_id, SimulatedServer(recovery_time=recovery_time_seconds), (0,))

    if scenario_id == SCENARIO_SHARED_CONTENTION:
        server = SimulatedServer(
            window_seconds=contention_window_seconds, capacity_requests_per_window=contention_capacity_per_window
        )
        return ScenarioSpec(scenario_id, server, tuple(range(contention_client_count)))

    if scenario_id == SCENARIO_RETRY_AFTER_VARIATIONS:
        server = SimulatedServer(retry_after_variants=dict(RETRY_AFTER_VARIANT_HEADERS))
        return ScenarioSpec(
            scenario_id, server, tuple(RETRY_AFTER_VARIANT_HEADERS.keys()), max_total_wait_seconds_override=10.0
        )

    if scenario_id == SCENARIO_QUOTA_PRESSURE:
        server = SimulatedServer(
            window_seconds=quota_window_seconds,
            capacity_requests_per_window=quota_rpm,
            capacity_tokens_per_window=quota_tpm,
        )
        return ScenarioSpec(
            scenario_id, server, tuple(range(quota_client_count)), token_demand=quota_token_demand_per_request
        )

    raise ValueError(f"Unknown scenario_id '{scenario_id}'")
