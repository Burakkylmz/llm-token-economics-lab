"""Loading and validation for the TOML-based lab configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


class ConfigError(ValueError):
    """Raised when a configuration file is missing required fields or has invalid values."""


@dataclass(frozen=True)
class GroqConfig:
    default_model: str
    request_timeout_seconds: float


@dataclass(frozen=True)
class ExecutionConfig:
    live_api_enabled: bool
    cost_warning_message: str


@dataclass(frozen=True)
class ExperimentConfig:
    default_repetitions: int
    results_dir: str


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float
    jitter: bool


@dataclass(frozen=True)
class ConcurrencyConfig:
    max_concurrent_requests: int


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m_usd: float
    output_per_1m_usd: float
    cached_input_per_1m_usd: float
    price_verified_date: str


@dataclass(frozen=True)
class Lab05Config:
    """Lab05's model-routing experiment.

    `small_model`/`large_model` remain required (unchanged from before this
    redesign). The new fields below are optional with documented defaults
    matching this project's default 12-call experiment (6 tasks x 2
    strategies): a config that predates this redesign and only ever set
    small_model/large_model keeps working identically.
    """

    small_model: str
    large_model: str
    default_profile: str
    reasoning_effort: str
    max_completion_tokens: int
    randomize_task_order: bool
    randomization_seed: int
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]


@dataclass(frozen=True)
class Lab03Config:
    """Lab03's prompt-caching experiment.

    `caching_capable_models` was already required before this redesign;
    every other field is new and optional with safe, documented defaults
    (see _build_lab03 below), so a config file that only ever set
    `caching_capable_models` keeps loading and running `lab03 all --live`
    unchanged -- exactly this project's documented 6-call default (3
    paired repetitions x 2 conditions).
    """

    caching_capable_models: tuple[str, ...]
    repetitions: int
    max_completion_tokens: int
    randomize_condition_order_within_pair: bool
    randomization_seed: int
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]
    # Number of sequential, byte-identical requests `lab03 cache_probe`
    # sends by default -- a separate, smaller budget from `repetitions`
    # above (which governs `lab03 all`/single-condition modes).
    cache_probe_calls: int


@dataclass(frozen=True)
class Lab04Config:
    """Lab04's output-control/invocation-optimization experiment.

    Entirely optional at the AppConfig level (same pattern as Lab02Config):
    `lab04 all --live` must keep working with its documented 10-call default
    (3 output-limit values x 2 repetitions + 1 single-invocation call + 3
    multiple-invocation calls) even for a config that predates this section.
    """

    enabled: bool
    model: str
    output_limit_values: tuple[int, ...]
    output_limit_repetitions: int
    reasoning_effort: str
    invocation_max_completion_tokens: int
    invocation_execution_mode: str
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]


@dataclass(frozen=True)
class Lab06Config:
    """Lab06's streaming/non-streaming perceived-latency experiment.

    Entirely optional at the AppConfig level (same pattern as
    Lab02Config/Lab04Config): `lab06 all --live` must keep working with
    its documented 10-call default (5 paired repetitions x 2 conditions)
    even for a config that predates this section.
    """

    enabled: bool
    model: str
    default_profile: str
    paired_repetitions: int
    reasoning_effort: str
    max_completion_tokens: int
    randomize_condition_order: bool
    randomization_seed: int
    request_usage_when_supported: bool
    record_event_timeline: bool
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]


@dataclass(frozen=True)
class Lab07Config:
    """Lab07's fully offline retry-strategy/rate-limiting simulation.

    Entirely optional at the AppConfig level (same pattern as
    Lab02Config/Lab04Config/Lab06Config): `lab07 all --config
    config/default.toml` must keep working with its documented 12-run
    default (3 scenarios x 4 strategies) even for a config that predates
    this section. Deliberately decoupled from the global `[retry]`
    section (used by every live-API lab for REAL operational resilience)
    -- Lab07's own retry-shaped fields here are a SIMULATION parameter
    (this lab's explicit experiment variable), not an operational concern,
    so changing one can never silently change the other.
    """

    enabled: bool
    default_profile: str
    randomization_seed: int
    max_simulation_events: int
    max_clients: int
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float
    max_total_wait_seconds: Optional[float]
    simulation_deadline_seconds: Optional[float]
    transient_fail_count: int
    recovery_time_seconds: float
    contention_client_count: int
    contention_capacity_per_window: int
    contention_window_seconds: float
    quota_client_count: int
    quota_rpm: int
    quota_tpm: int
    quota_token_demand_per_request: int
    quota_window_seconds: float


@dataclass(frozen=True)
class Lab08Config:
    """Lab08's sequential-vs-concurrent async execution experiment.

    Entirely optional at the AppConfig level (same pattern as
    Lab02Config/Lab04Config/Lab06Config/Lab07Config): `lab08 all --live`
    must keep working with its documented 16-call default (8 tasks x 2
    concurrency levels) even for a config that predates this section.
    """

    enabled: bool
    model: str
    default_profile: str
    # Configures only the "default" profile's concurrency levels (pilot=
    # (1, 3), extended=(1, 2, 3, 5) are fixed, documented scopes -- same
    # precedent as Lab06Config.paired_repetitions/Lab07Config.default_profile).
    concurrency_levels: tuple[int, ...]
    repetitions: int
    reasoning_effort: str
    max_completion_tokens: int
    randomize_condition_order: bool
    randomization_seed: int
    # A hard ceiling on any single concurrency level (default-profile
    # config value or --max-concurrent CLI override) -- enforced before
    # any live call, independent of the API-call budget below.
    max_allowed_concurrency: int
    # This lab's OWN AsyncGroq client is constructed with this
    # max_retries value (default 0) so the SDK's hidden automatic retries
    # never inflate measured latency/attempt counts -- never applied to
    # any other lab's shared client.
    sdk_max_retries: int
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]


@dataclass(frozen=True)
class Lab09Config:
    batch_completion_window: str


# openai/gpt-oss-20b and openai/gpt-oss-120b (this project's default/pricing
# models) support exactly these reasoning_effort levels, per the installed
# `groq` SDK's own type stub (groq/types/chat/completion_create_params.py,
# groq==1.7.0): "openai/gpt-oss-20b and openai/gpt-oss-120b support 'low',
# 'medium', or 'high'." The SDK's broader Literal type also allows
# "none"/"default", but those are qwen3-model-specific and not valid for
# this lab's models -- rejected explicitly rather than silently accepted.
SUPPORTED_REASONING_EFFORTS = ("low", "medium", "high")
# Per Groq's API docs (same SDK docstring): temperature is "between 0 and
# 2"; top_p is a nucleus-sampling probability mass, so must be in (0, 1].
MIN_TEMPERATURE_INCLUSIVE = 0.0
MAX_TEMPERATURE_INCLUSIVE = 2.0
MAX_TOP_P_INCLUSIVE = 1.0


@dataclass(frozen=True)
class Lab01FactorialConfig:
    """Full-factorial experiment matrix for Lab01's `factorial` execution mode.

    Deliberately optional at the AppConfig level (see `_build_lab01_factorial`):
    a `config/live.toml` created before this feature existed, and missing the
    whole `[lab01.factorial]` section, must keep loading and running Lab01's
    basic mode without any change -- only the `factorial` CLI mode requires
    this section to be present.
    """

    enabled: bool
    prompts: tuple[str, ...]
    reasoning_efforts: tuple[str, ...]
    temperatures: tuple[float, ...]
    top_p_values: tuple[float, ...]
    repetitions: int
    max_completion_tokens: int
    randomize_execution_order: bool
    randomization_seed: int
    use_paired_api_seed: bool
    max_api_calls: Optional[int] = None
    max_estimated_cost_usd: Optional[float] = None


@dataclass(frozen=True)
class Lab01EffortConfig:
    """Lab01's `effort` execution mode: compares reasoning_effort levels only.

    Replaces [lab01.factorial] (temperature/top_p/reasoning_effort full
    factorial, retired for Groq rate-limit/quota reasons -- see
    labs.lab01_token_anatomy.cli._run_legacy_factorial_stub) as the active
    multi-call Lab01 experiment. Like Lab01FactorialConfig, this whole
    section is optional at the AppConfig level: a config file that omits
    [lab01.effort] still runs Lab01's basic mode unchanged.
    """

    enabled: bool
    prompts: tuple[str, ...]
    reasoning_efforts: tuple[str, ...]
    repetitions: int
    max_completion_tokens: int
    randomize_execution_order: bool
    randomization_seed: int
    use_paired_api_seed: bool
    max_api_calls: Optional[int] = None
    max_estimated_cost_usd: Optional[float] = None


@dataclass(frozen=True)
class Lab02Config:
    """Lab02's context-growth/context-bloat experiment.

    Unlike Lab01FactorialConfig/Lab01EffortConfig, this section is NOT
    optional at the AppConfig level: `lab02 all --live` must keep working
    with its documented 10-call default even for a config/live.toml that
    predates this feature and has no [lab02] section at all (see
    _build_lab02 below, which fills in these exact defaults when the
    section is absent).
    """

    enabled: bool
    repetitions: int
    multi_turn_turns: int
    max_completion_tokens: int
    randomize_context_order: bool
    randomization_seed: int
    max_api_calls: Optional[int]
    max_estimated_cost_usd: Optional[float]


@dataclass(frozen=True)
class AppConfig:
    groq: GroqConfig
    execution: ExecutionConfig
    experiment: ExperimentConfig
    retry: RetryConfig
    concurrency: ConcurrencyConfig
    pricing: dict[str, ModelPricing]
    lab02: Lab02Config
    lab03: Lab03Config
    lab04: Lab04Config
    lab05: Lab05Config
    lab06: Lab06Config
    lab07: Lab07Config
    lab08: Lab08Config
    lab09: Lab09Config
    lab01_factorial: Optional[Lab01FactorialConfig]
    lab01_effort: Optional[Lab01EffortConfig]


def _require(section: dict, key: str, section_name: str) -> object:
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in [{section_name}]")
    return section[key]


def _build_groq(raw: dict) -> GroqConfig:
    section = raw.get("groq", {})
    return GroqConfig(
        default_model=str(_require(section, "default_model", "groq")),
        request_timeout_seconds=float(_require(section, "request_timeout_seconds", "groq")),
    )


def _build_execution(raw: dict) -> ExecutionConfig:
    section = raw.get("execution", {})
    live_api_enabled = _require(section, "live_api_enabled", "execution")
    if not isinstance(live_api_enabled, bool):
        raise ConfigError("execution.live_api_enabled must be a boolean")
    return ExecutionConfig(
        live_api_enabled=live_api_enabled,
        cost_warning_message=str(_require(section, "cost_warning_message", "execution")),
    )


def _build_experiment(raw: dict) -> ExperimentConfig:
    section = raw.get("experiment", {})
    repetitions = int(_require(section, "default_repetitions", "experiment"))
    if repetitions < 1:
        raise ConfigError("experiment.default_repetitions must be >= 1")
    return ExperimentConfig(
        default_repetitions=repetitions,
        results_dir=str(_require(section, "results_dir", "experiment")),
    )


def _build_retry(raw: dict) -> RetryConfig:
    section = raw.get("retry", {})
    max_attempts = int(_require(section, "max_attempts", "retry"))
    if max_attempts < 1:
        raise ConfigError("retry.max_attempts must be >= 1")
    base_delay = float(_require(section, "base_delay_seconds", "retry"))
    max_delay = float(_require(section, "max_delay_seconds", "retry"))
    if base_delay <= 0 or max_delay <= 0:
        raise ConfigError("retry delay values must be > 0")
    if max_delay < base_delay:
        raise ConfigError("retry.max_delay_seconds must be >= retry.base_delay_seconds")
    jitter = _require(section, "jitter", "retry")
    if not isinstance(jitter, bool):
        raise ConfigError("retry.jitter must be a boolean")
    return RetryConfig(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay,
        max_delay_seconds=max_delay,
        jitter=jitter,
    )


def _build_concurrency(raw: dict) -> ConcurrencyConfig:
    section = raw.get("concurrency", {})
    max_concurrent = int(_require(section, "max_concurrent_requests", "concurrency"))
    if max_concurrent < 1:
        raise ConfigError("concurrency.max_concurrent_requests must be >= 1")
    return ConcurrencyConfig(max_concurrent_requests=max_concurrent)


def _build_pricing(raw: dict) -> dict[str, ModelPricing]:
    pricing_section = raw.get("pricing", {}).get("models", {})
    pricing: dict[str, ModelPricing] = {}
    for model_id, values in pricing_section.items():
        for key in ("input_per_1m_usd", "output_per_1m_usd", "cached_input_per_1m_usd"):
            if key not in values:
                raise ConfigError(f"Missing required key '{key}' for pricing.models.\"{model_id}\"")
            if float(values[key]) < 0:
                raise ConfigError(f"pricing.models.\"{model_id}\".{key} must be >= 0")
        pricing[model_id] = ModelPricing(
            input_per_1m_usd=float(values["input_per_1m_usd"]),
            output_per_1m_usd=float(values["output_per_1m_usd"]),
            cached_input_per_1m_usd=float(values["cached_input_per_1m_usd"]),
            price_verified_date=str(values.get("price_verified_date", "")),
        )
    return pricing


def _build_lab02(raw: dict) -> Lab02Config:
    """Optional [lab02] section with safe, documented defaults: 3 context
    conditions x 2 repetitions (6 calls) + 4 multi-turn calls = 10 planned
    calls by default, matching this project's default experiment matrix.
    A config file that omits [lab02] entirely gets exactly these defaults,
    so `lab02 all --live` keeps working unchanged on pre-existing configs.
    """
    section = raw.get("lab02", {})

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("lab02.enabled must be a boolean")

    repetitions = section.get("repetitions", 2)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ConfigError("lab02.repetitions must be a positive integer")

    multi_turn_turns = section.get("multi_turn_turns", 4)
    if isinstance(multi_turn_turns, bool) or not isinstance(multi_turn_turns, int) or multi_turn_turns < 1:
        raise ConfigError("lab02.multi_turn_turns must be a positive integer")

    # Large enough that a short, strictly-formatted integer answer is never
    # truncated by its own reasoning overhead (finish_reason="length" is
    # still recorded on every call so truncation stays visible if it
    # happens), but far below this project's factorial-scale 4096 default
    # since Lab02's answers are intentionally one short token.
    max_completion_tokens = section.get("max_completion_tokens", 2048)
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ConfigError("lab02.max_completion_tokens must be a positive integer")

    randomize_context_order = section.get("randomize_context_order", False)
    if not isinstance(randomize_context_order, bool):
        raise ConfigError("lab02.randomize_context_order must be a boolean")

    randomization_seed = section.get("randomization_seed", 42)
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab02.randomization_seed must be an integer")

    # Exactly this project's documented default experiment matrix (3
    # conditions x 2 repetitions + 4 multi-turn calls). A CLI override
    # (--repetitions/--turns) that would push the plan above this is
    # refused before any API call -- see labs.lab02_context_growth.cli.
    max_api_calls = section.get("max_api_calls", 10)
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab02.max_api_calls must be a positive integer or omitted")

    # Worst case at the 10-call default: 10 x 2048 max_completion_tokens x
    # $0.30/1e6 (gpt-oss-20b output price) = $0.00614, plus a negligible
    # prompt-token share. $0.05 leaves generous headroom while still being
    # a real, enforced cap rather than "unlimited".
    max_estimated_cost_usd = section.get("max_estimated_cost_usd", 0.05)
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab02.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    known_keys = {
        "enabled",
        "repetitions",
        "multi_turn_turns",
        "max_completion_tokens",
        "randomize_context_order",
        "randomization_seed",
        "max_api_calls",
        "max_estimated_cost_usd",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab02 has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab02Config(
        enabled=enabled,
        repetitions=repetitions,
        multi_turn_turns=multi_turn_turns,
        max_completion_tokens=max_completion_tokens,
        randomize_context_order=randomize_context_order,
        randomization_seed=randomization_seed,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )


def _build_lab03(raw: dict) -> Lab03Config:
    """`caching_capable_models` remains required (unchanged from before this
    redesign). The new fields below are optional with documented defaults
    matching this project's default 6-call paired experiment (3 repetitions
    x 2 conditions): a config that predates this redesign and only ever
    set caching_capable_models keeps working identically.
    """
    section = raw.get("lab03", {})
    models = _require(section, "caching_capable_models", "lab03")
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise ConfigError("lab03.caching_capable_models must be a list of strings")

    repetitions = section.get("repetitions", 3)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ConfigError("lab03.repetitions must be a positive integer")

    # Large enough that a short, strictly-formatted integer answer is never
    # truncated by its own reasoning overhead (finish_reason="length" is
    # still recorded on every call so truncation stays visible), matching
    # Lab02's same documented rationale -- deliberately NOT the task
    # spec's own illustrative 512, which this project's live-verified
    # reasoning-token observations (see docs/instructor/README.md) show is
    # too tight a margin for a gpt-oss reasoning model on some calls.
    max_completion_tokens = section.get("max_completion_tokens", 2048)
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ConfigError("lab03.max_completion_tokens must be a positive integer")

    randomize_condition_order_within_pair = section.get("randomize_condition_order_within_pair", False)
    if not isinstance(randomize_condition_order_within_pair, bool):
        raise ConfigError("lab03.randomize_condition_order_within_pair must be a boolean")

    randomization_seed = section.get("randomization_seed", 42)
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab03.randomization_seed must be an integer")

    # Exactly this project's documented default experiment matrix (3
    # repetitions x 2 conditions = 6 calls). A CLI override (--repetitions)
    # that would push the plan above this is refused before any API call.
    max_api_calls = section.get("max_api_calls", 8)
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab03.max_api_calls must be a positive integer or omitted")

    # Worst case at the 8-call extended-pilot ceiling: 8 x 2048
    # max_completion_tokens x $0.30/1e6 (gpt-oss-20b output price) =
    # $0.0049, plus a negligible prompt-token share. $0.05 leaves generous
    # headroom while still being a real, enforced cap.
    max_estimated_cost_usd = section.get("max_estimated_cost_usd", 0.05)
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab03.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    # `lab03 cache_probe`'s own, smaller, dedicated budget: 3 sequential,
    # byte-identical requests by default -- the strongest exact-prefix-match
    # condition this project can construct, independent of the 6-call
    # paired-matrix `repetitions` default above.
    cache_probe_calls = section.get("cache_probe_calls", 3)
    if isinstance(cache_probe_calls, bool) or not isinstance(cache_probe_calls, int) or cache_probe_calls < 1:
        raise ConfigError("lab03.cache_probe_calls must be a positive integer")

    known_keys = {
        "caching_capable_models",
        "repetitions",
        "max_completion_tokens",
        "randomize_condition_order_within_pair",
        "randomization_seed",
        "max_api_calls",
        "max_estimated_cost_usd",
        "cache_probe_calls",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab03 has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab03Config(
        caching_capable_models=tuple(models),
        repetitions=repetitions,
        max_completion_tokens=max_completion_tokens,
        randomize_condition_order_within_pair=randomize_condition_order_within_pair,
        randomization_seed=randomization_seed,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
        cache_probe_calls=cache_probe_calls,
    )


def _validate_positive_int_list(values: object, label: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{label} must be a non-empty list of positive integers")
    result: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 1:
            raise ConfigError(f"{label} must contain only positive integers")
        result.append(raw_value)
    if len(set(result)) != len(result):
        raise ConfigError(f"{label} contains duplicate values")
    return result


def _build_lab04(raw: dict) -> Lab04Config:
    """Optional [lab04] section with safe, documented defaults: 3
    output-limit values x 2 repetitions (6 calls) + 1 single-invocation
    call + 3 multiple-invocation calls = 10 planned calls by default,
    matching this project's documented default experiment matrix. A
    config file that omits [lab04] entirely gets exactly these defaults.
    """
    section = raw.get("lab04", {})

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("lab04.enabled must be a boolean")

    model = str(section.get("model", "openai/gpt-oss-20b"))

    output_limit_values = _validate_positive_int_list(
        section.get("output_limit_values", [64, 256, 1024]), "lab04.output_limit_values"
    )

    output_limit_repetitions = section.get("output_limit_repetitions", 2)
    if (
        isinstance(output_limit_repetitions, bool)
        or not isinstance(output_limit_repetitions, int)
        or output_limit_repetitions < 1
    ):
        raise ConfigError("lab04.output_limit_repetitions must be a positive integer")

    # openai/gpt-oss-20b/120b only support these reasoning_effort levels --
    # see SUPPORTED_REASONING_EFFORTS above. The SAME value is used for both
    # the output-limit sweep and the invocation comparison, so "same
    # reasoning effort across conditions" holds by construction rather than
    # needing two separately-validated fields that could drift apart.
    reasoning_effort = str(section.get("reasoning_effort", "low"))
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ConfigError(
            f"lab04.reasoning_effort: unsupported value '{reasoning_effort}' -- "
            f"this project's models only support {SUPPORTED_REASONING_EFFORTS}"
        )

    # Deliberately generous (same rationale as lab02/lab03's own
    # max_completion_tokens default): the invocation-comparison experiment
    # studies invocation COUNT, not output budgeting, so its own budget
    # must not introduce the very truncation failure mode output_limit_sweep
    # exists to study. Real cost is still driven by actual completion_tokens
    # used, never by this ceiling.
    invocation_max_completion_tokens = section.get("invocation_max_completion_tokens", 2048)
    if (
        isinstance(invocation_max_completion_tokens, bool)
        or not isinstance(invocation_max_completion_tokens, int)
        or invocation_max_completion_tokens < 1
    ):
        raise ConfigError("lab04.invocation_max_completion_tokens must be a positive integer")

    # Only "sequential" is implemented/validated here by design -- concurrent
    # multiple-invocation execution is LAB08's subject, not LAB04's; see this
    # lab's module docstrings.
    invocation_execution_mode = str(section.get("invocation_execution_mode", "sequential"))
    if invocation_execution_mode != "sequential":
        raise ConfigError(
            "lab04.invocation_execution_mode: only 'sequential' is supported -- concurrent "
            "execution is LAB08's subject, not LAB04's"
        )

    # Exactly this project's documented default experiment matrix (3 x 2 + 1
    # + 3 = 10 calls). A CLI override that would push the plan above this is
    # refused before any API call.
    max_api_calls = section.get("max_api_calls", 10)
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab04.max_api_calls must be a positive integer or omitted")

    # Worst case at the 10-call default: 10 x 2048 (the largest of
    # output_limit_values/invocation_max_completion_tokens' defaults) x
    # $0.30/1e6 (gpt-oss-20b output price) = $0.00614, plus a negligible
    # prompt-token share. $0.05 leaves generous headroom while still being a
    # real, enforced cap.
    max_estimated_cost_usd = section.get("max_estimated_cost_usd", 0.05)
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab04.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    known_keys = {
        "enabled",
        "model",
        "output_limit_values",
        "output_limit_repetitions",
        "reasoning_effort",
        "invocation_max_completion_tokens",
        "invocation_execution_mode",
        "max_api_calls",
        "max_estimated_cost_usd",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab04 has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab04Config(
        enabled=enabled,
        model=model,
        output_limit_values=tuple(output_limit_values),
        output_limit_repetitions=output_limit_repetitions,
        reasoning_effort=reasoning_effort,
        invocation_max_completion_tokens=invocation_max_completion_tokens,
        invocation_execution_mode=invocation_execution_mode,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )


LAB05_PROFILES = ("pilot", "default", "extended")


def _build_lab05(raw: dict) -> Lab05Config:
    section = raw.get("lab05", {})
    small_model = str(_require(section, "small_model", "lab05"))
    large_model = str(_require(section, "large_model", "lab05"))

    default_profile = section.get("default_profile", "default")
    if default_profile not in LAB05_PROFILES:
        raise ConfigError(f"lab05.default_profile must be one of {LAB05_PROFILES}")

    reasoning_effort = section.get("reasoning_effort", "low")
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ConfigError(
            f"lab05.reasoning_effort: unsupported value '{reasoning_effort}' -- "
            f"this project's models only support {SUPPORTED_REASONING_EFFORTS}"
        )

    max_completion_tokens = section.get("max_completion_tokens", 1024)
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ConfigError("lab05.max_completion_tokens must be a positive integer")

    randomize_task_order = section.get("randomize_task_order", False)
    if not isinstance(randomize_task_order, bool):
        raise ConfigError("lab05.randomize_task_order must be a boolean")

    randomization_seed = section.get("randomization_seed", 42)
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab05.randomization_seed must be an integer")

    # Exactly this project's documented Extended Experiment ceiling (6
    # tasks x 3 strategies = 18 calls) -- the largest planned scope
    # documented for this lab. A --profile override that would push the
    # plan above this is refused before any API call.
    max_api_calls = section.get("max_api_calls", 18)
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab05.max_api_calls must be a positive integer or omitted")

    # Worst case at the 18-call extended ceiling: 18 x 1024
    # max_completion_tokens x $0.60/1e6 (gpt-oss-120b output price) =
    # $0.011, plus a negligible prompt-token share. $0.05 leaves generous
    # headroom while still being a real, enforced cap.
    max_estimated_cost_usd = section.get("max_estimated_cost_usd", 0.05)
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab05.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    known_keys = {
        "small_model",
        "large_model",
        "default_profile",
        "reasoning_effort",
        "max_completion_tokens",
        "randomize_task_order",
        "randomization_seed",
        "max_api_calls",
        "max_estimated_cost_usd",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab05 has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab05Config(
        small_model=small_model,
        large_model=large_model,
        default_profile=default_profile,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        randomize_task_order=randomize_task_order,
        randomization_seed=randomization_seed,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )


LAB06_PROFILES = ("pilot", "default", "extended")


def _build_lab06(raw: dict) -> Lab06Config:
    """Optional [lab06] section with safe, documented defaults: 5 paired
    repetitions x 2 conditions (non_streaming, streaming) = 10 planned
    calls by default, matching this project's documented default
    experiment matrix. A config file that omits [lab06] entirely gets
    exactly these defaults.
    """
    section = raw.get("lab06", {})

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("lab06.enabled must be a boolean")

    model = str(section.get("model", "openai/gpt-oss-20b"))

    default_profile = section.get("default_profile", "default")
    if default_profile not in LAB06_PROFILES:
        raise ConfigError(f"lab06.default_profile must be one of {LAB06_PROFILES}")

    # Only the "default" profile's pair count is configurable -- "pilot"
    # (2 pairs) and "extended" (8 pairs) are fixed, documented scopes (see
    # labs.lab06_streaming.matrix.PROFILE_PAIRS).
    paired_repetitions = section.get("paired_repetitions", 5)
    if isinstance(paired_repetitions, bool) or not isinstance(paired_repetitions, int) or paired_repetitions < 1:
        raise ConfigError("lab06.paired_repetitions must be a positive integer")

    reasoning_effort = section.get("reasoning_effort", "low")
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ConfigError(
            f"lab06.reasoning_effort: unsupported value '{reasoning_effort}' -- "
            f"this project's models only support {SUPPORTED_REASONING_EFFORTS}"
        )

    max_completion_tokens = section.get("max_completion_tokens", 768)
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ConfigError("lab06.max_completion_tokens must be a positive integer")

    randomize_condition_order = section.get("randomize_condition_order", False)
    if not isinstance(randomize_condition_order, bool):
        raise ConfigError("lab06.randomize_condition_order must be a boolean")

    randomization_seed = section.get("randomization_seed", 42)
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab06.randomization_seed must be an integer")

    request_usage_when_supported = section.get("request_usage_when_supported", True)
    if not isinstance(request_usage_when_supported, bool):
        raise ConfigError("lab06.request_usage_when_supported must be a boolean")

    record_event_timeline = section.get("record_event_timeline", True)
    if not isinstance(record_event_timeline, bool):
        raise ConfigError("lab06.record_event_timeline must be a boolean")

    # Exactly this project's documented Extended Experiment ceiling (8
    # pairs x 2 conditions = 16 calls) -- the largest planned scope
    # documented for this lab. A --profile override that would push the
    # plan above this is refused before any API call.
    max_api_calls = section.get("max_api_calls", 16)
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab06.max_api_calls must be a positive integer or omitted")

    # Worst case at the 16-call extended ceiling: 16 x 768
    # max_completion_tokens x $0.30/1e6 (gpt-oss-20b output price) =
    # $0.00369, plus a negligible prompt-token share. $0.05 leaves
    # generous headroom while still being a real, enforced cap.
    max_estimated_cost_usd = section.get("max_estimated_cost_usd", 0.05)
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab06.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    known_keys = {
        "enabled",
        "model",
        "default_profile",
        "paired_repetitions",
        "reasoning_effort",
        "max_completion_tokens",
        "randomize_condition_order",
        "randomization_seed",
        "request_usage_when_supported",
        "record_event_timeline",
        "max_api_calls",
        "max_estimated_cost_usd",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab06 has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab06Config(
        enabled=enabled,
        model=model,
        default_profile=default_profile,
        paired_repetitions=paired_repetitions,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        randomize_condition_order=randomize_condition_order,
        randomization_seed=randomization_seed,
        request_usage_when_supported=request_usage_when_supported,
        record_event_timeline=record_event_timeline,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )


LAB07_PROFILES = ("pilot", "default", "extended")


def _build_lab07(raw: dict) -> Lab07Config:
    """Optional [lab07] section with safe, documented defaults: 3
    scenarios x 4 strategies = 12 planned simulation runs by default
    (this project's documented default matrix). A config file that omits
    [lab07] entirely gets exactly these defaults.
    """
    section = raw.get("lab07", {})

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("lab07.enabled must be a boolean")

    default_profile = section.get("default_profile", "default")
    if default_profile not in LAB07_PROFILES:
        raise ConfigError(f"lab07.default_profile must be one of {LAB07_PROFILES}")

    def _positive_int(key: str, default: int) -> int:
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"lab07.{key} must be a positive integer")
        return value

    def _non_negative_int(key: str, default: int) -> int:
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"lab07.{key} must be a non-negative integer")
        return value

    def _positive_float(key: str, default: float) -> float:
        value = section.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"lab07.{key} must be a positive number")
        return float(value)

    def _optional_positive_float(key: str, default: Optional[float]) -> Optional[float]:
        value = section.get(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"lab07.{key} must be a positive number or omitted")
        return float(value)

    randomization_seed = section.get("randomization_seed", 42)
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab07.randomization_seed must be an integer")

    max_simulation_events = _positive_int("max_simulation_events", 10_000)
    max_clients = _positive_int("max_clients", 50)
    max_attempts = _positive_int("max_attempts", 5)
    base_delay_seconds = _positive_float("base_delay_seconds", 1.0)
    max_delay_seconds = _positive_float("max_delay_seconds", 30.0)
    if max_delay_seconds < base_delay_seconds:
        raise ConfigError("lab07.max_delay_seconds must be >= lab07.base_delay_seconds")
    max_total_wait_seconds = _optional_positive_float("max_total_wait_seconds", 120.0)
    simulation_deadline_seconds = _optional_positive_float("simulation_deadline_seconds", None)

    transient_fail_count = _non_negative_int("transient_fail_count", 2)
    recovery_time_seconds = _positive_float("recovery_time_seconds", 3.0)

    contention_client_count = _positive_int("contention_client_count", 8)
    contention_capacity_per_window = _positive_int("contention_capacity_per_window", 2)
    contention_window_seconds = _positive_float("contention_window_seconds", 1.0)

    quota_client_count = _positive_int("quota_client_count", 3)
    quota_rpm = _positive_int("quota_rpm", 5)
    quota_tpm = _positive_int("quota_tpm", 250)
    quota_token_demand_per_request = _positive_int("quota_token_demand_per_request", 100)
    quota_window_seconds = _positive_float("quota_window_seconds", 60.0)

    if contention_client_count > max_clients:
        raise ConfigError("lab07.contention_client_count must not exceed lab07.max_clients")
    if quota_client_count > max_clients:
        raise ConfigError("lab07.quota_client_count must not exceed lab07.max_clients")

    known_keys = {
        "enabled", "default_profile", "randomization_seed", "max_simulation_events", "max_clients",
        "max_attempts", "base_delay_seconds", "max_delay_seconds", "max_total_wait_seconds",
        "simulation_deadline_seconds", "transient_fail_count", "recovery_time_seconds",
        "contention_client_count", "contention_capacity_per_window", "contention_window_seconds",
        "quota_client_count", "quota_rpm", "quota_tpm", "quota_token_demand_per_request", "quota_window_seconds",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab07 has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab07Config(
        enabled=enabled,
        default_profile=default_profile,
        randomization_seed=randomization_seed,
        max_simulation_events=max_simulation_events,
        max_clients=max_clients,
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        max_total_wait_seconds=max_total_wait_seconds,
        simulation_deadline_seconds=simulation_deadline_seconds,
        transient_fail_count=transient_fail_count,
        recovery_time_seconds=recovery_time_seconds,
        contention_client_count=contention_client_count,
        contention_capacity_per_window=contention_capacity_per_window,
        contention_window_seconds=contention_window_seconds,
        quota_client_count=quota_client_count,
        quota_rpm=quota_rpm,
        quota_tpm=quota_tpm,
        quota_token_demand_per_request=quota_token_demand_per_request,
        quota_window_seconds=quota_window_seconds,
    )


LAB08_PROFILES = ("pilot", "default", "extended")


def _build_lab08(raw: dict) -> Lab08Config:
    """Optional [lab08] section with safe, documented defaults: 8 tasks x
    2 concurrency levels (1, 5) = 16 planned calls by default, matching
    this project's documented default experiment matrix. A config file
    that omits [lab08] entirely gets exactly these defaults.
    """
    section = raw.get("lab08", {})

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("lab08.enabled must be a boolean")

    model = str(section.get("model", "openai/gpt-oss-20b"))

    default_profile = section.get("default_profile", "default")
    if default_profile not in LAB08_PROFILES:
        raise ConfigError(f"lab08.default_profile must be one of {LAB08_PROFILES}")

    concurrency_levels = _validate_positive_int_list(
        section.get("concurrency_levels", [1, 5]), "lab08.concurrency_levels"
    )

    repetitions = section.get("repetitions", 1)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ConfigError("lab08.repetitions must be a positive integer")

    reasoning_effort = section.get("reasoning_effort", "low")
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ConfigError(
            f"lab08.reasoning_effort: unsupported value '{reasoning_effort}' -- "
            f"this project's models only support {SUPPORTED_REASONING_EFFORTS}"
        )

    max_completion_tokens = section.get("max_completion_tokens", 256)
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ConfigError("lab08.max_completion_tokens must be a positive integer")

    randomize_condition_order = section.get("randomize_condition_order", False)
    if not isinstance(randomize_condition_order, bool):
        raise ConfigError("lab08.randomize_condition_order must be a boolean")

    randomization_seed = section.get("randomization_seed", 42)
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab08.randomization_seed must be an integer")

    max_allowed_concurrency = section.get("max_allowed_concurrency", 10)
    if (
        isinstance(max_allowed_concurrency, bool)
        or not isinstance(max_allowed_concurrency, int)
        or max_allowed_concurrency < 1
    ):
        raise ConfigError("lab08.max_allowed_concurrency must be a positive integer")
    if max(concurrency_levels) > max_allowed_concurrency:
        raise ConfigError("lab08.concurrency_levels must not exceed lab08.max_allowed_concurrency")

    sdk_max_retries = section.get("sdk_max_retries", 0)
    if isinstance(sdk_max_retries, bool) or not isinstance(sdk_max_retries, int) or sdk_max_retries < 0:
        raise ConfigError("lab08.sdk_max_retries must be a non-negative integer")

    # Exactly this project's documented Extended Experiment ceiling (8
    # tasks x 4 concurrency levels = 32 calls) -- the largest planned
    # scope documented for this lab. A --profile override that would push
    # the plan above this is refused before any API call.
    max_api_calls = section.get("max_api_calls", 32)
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab08.max_api_calls must be a positive integer or omitted")

    # Worst case at the 32-call extended ceiling: 32 x 256
    # max_completion_tokens x $0.30/1e6 (gpt-oss-20b output price) =
    # $0.00246, plus a negligible prompt-token share. $0.05 leaves
    # generous headroom while still being a real, enforced cap.
    max_estimated_cost_usd = section.get("max_estimated_cost_usd", 0.05)
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab08.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    known_keys = {
        "enabled", "model", "default_profile", "concurrency_levels", "repetitions", "reasoning_effort",
        "max_completion_tokens", "randomize_condition_order", "randomization_seed", "max_allowed_concurrency",
        "sdk_max_retries", "max_api_calls", "max_estimated_cost_usd",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab08 has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab08Config(
        enabled=enabled,
        model=model,
        default_profile=default_profile,
        concurrency_levels=tuple(concurrency_levels),
        repetitions=repetitions,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        randomize_condition_order=randomize_condition_order,
        randomization_seed=randomization_seed,
        max_allowed_concurrency=max_allowed_concurrency,
        sdk_max_retries=sdk_max_retries,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )


def _build_lab09(raw: dict) -> Lab09Config:
    section = raw.get("lab09", {})
    return Lab09Config(
        batch_completion_window=str(_require(section, "batch_completion_window", "lab09")),
    )


def _validate_nonempty_string_list(values: object, label: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{label} must be a non-empty list of strings")
    if not all(isinstance(v, str) for v in values):
        raise ConfigError(f"{label} must contain only strings")
    if len(set(values)) != len(values):
        raise ConfigError(f"{label} contains duplicate values")
    return values


def _validate_float_list(
    values: object,
    label: str,
    minimum: float,
    maximum: float,
    exclusive_min: bool = False,
) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"{label} must be a non-empty list of numbers")

    result: list[float] = []
    for raw_value in values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ConfigError(f"{label} must contain only numbers")
        value = float(raw_value)
        lower_ok = value > minimum if exclusive_min else value >= minimum
        if not (lower_ok and value <= maximum):
            bound = f"({minimum}, {maximum}]" if exclusive_min else f"[{minimum}, {maximum}]"
            raise ConfigError(f"{label} value {value} is outside the supported range {bound}")
        result.append(value)

    # Round before the duplicate check so e.g. 0.7 and 0.7000001 (distinct
    # floats that would format identically) can never silently produce two
    # "different" factorial conditions with colliding ids later on.
    if len({round(v, 6) for v in result}) != len(result):
        raise ConfigError(f"{label} contains duplicate values")
    return result


def _build_lab01_factorial(raw: dict) -> Optional[Lab01FactorialConfig]:
    lab01_section = raw.get("lab01", {})
    if "factorial" not in lab01_section:
        # Whole section absent: Lab01 basic mode is unaffected. `factorial`
        # execution mode will refuse to run with an explicit, actionable
        # error (see labs.lab01_token_anatomy.factorial_cli) rather than a
        # KeyError/ConfigError here.
        return None

    section = lab01_section["factorial"]

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("lab01.factorial.enabled must be a boolean")

    prompts = _validate_nonempty_string_list(
        _require(section, "prompts", "lab01.factorial"), "lab01.factorial.prompts"
    )
    reasoning_efforts = _validate_nonempty_string_list(
        _require(section, "reasoning_efforts", "lab01.factorial"), "lab01.factorial.reasoning_efforts"
    )
    for effort in reasoning_efforts:
        if effort not in SUPPORTED_REASONING_EFFORTS:
            raise ConfigError(
                f"lab01.factorial.reasoning_efforts: unsupported value '{effort}' -- "
                f"this project's models only support {SUPPORTED_REASONING_EFFORTS} "
                "(see the SUPPORTED_REASONING_EFFORTS comment in this module)"
            )

    temperatures = _validate_float_list(
        _require(section, "temperatures", "lab01.factorial"),
        "lab01.factorial.temperatures",
        MIN_TEMPERATURE_INCLUSIVE,
        MAX_TEMPERATURE_INCLUSIVE,
    )
    top_p_values = _validate_float_list(
        _require(section, "top_p_values", "lab01.factorial"),
        "lab01.factorial.top_p_values",
        0.0,
        MAX_TOP_P_INCLUSIVE,
        exclusive_min=True,
    )

    repetitions = _require(section, "repetitions", "lab01.factorial")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ConfigError("lab01.factorial.repetitions must be a positive integer")

    max_completion_tokens = _require(section, "max_completion_tokens", "lab01.factorial")
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ConfigError("lab01.factorial.max_completion_tokens must be a positive integer")

    randomize_execution_order = _require(section, "randomize_execution_order", "lab01.factorial")
    if not isinstance(randomize_execution_order, bool):
        raise ConfigError("lab01.factorial.randomize_execution_order must be a boolean")

    randomization_seed = _require(section, "randomization_seed", "lab01.factorial")
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab01.factorial.randomization_seed must be an integer")

    use_paired_api_seed = section.get("use_paired_api_seed", False)
    if not isinstance(use_paired_api_seed, bool):
        raise ConfigError("lab01.factorial.use_paired_api_seed must be a boolean")

    max_api_calls = section.get("max_api_calls")
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab01.factorial.max_api_calls must be a positive integer or omitted")

    max_estimated_cost_usd = section.get("max_estimated_cost_usd")
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab01.factorial.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    known_keys = {
        "enabled",
        "prompts",
        "reasoning_efforts",
        "temperatures",
        "top_p_values",
        "repetitions",
        "max_completion_tokens",
        "randomize_execution_order",
        "randomization_seed",
        "use_paired_api_seed",
        "max_api_calls",
        "max_estimated_cost_usd",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab01.factorial has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab01FactorialConfig(
        enabled=enabled,
        prompts=tuple(prompts),
        reasoning_efforts=tuple(reasoning_efforts),
        temperatures=tuple(temperatures),
        top_p_values=tuple(top_p_values),
        repetitions=repetitions,
        max_completion_tokens=max_completion_tokens,
        randomize_execution_order=randomize_execution_order,
        randomization_seed=randomization_seed,
        use_paired_api_seed=use_paired_api_seed,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )


def _build_lab01_effort(raw: dict) -> Optional[Lab01EffortConfig]:
    """Same optional-section, same validation style as _build_lab01_factorial
    above, just without temperature/top_p -- see Lab01EffortConfig."""
    lab01_section = raw.get("lab01", {})
    if "effort" not in lab01_section:
        return None

    section = lab01_section["effort"]

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("lab01.effort.enabled must be a boolean")

    prompts = _validate_nonempty_string_list(
        _require(section, "prompts", "lab01.effort"), "lab01.effort.prompts"
    )
    reasoning_efforts = _validate_nonempty_string_list(
        _require(section, "reasoning_efforts", "lab01.effort"), "lab01.effort.reasoning_efforts"
    )
    for effort in reasoning_efforts:
        if effort not in SUPPORTED_REASONING_EFFORTS:
            raise ConfigError(
                f"lab01.effort.reasoning_efforts: unsupported value '{effort}' -- "
                f"this project's models only support {SUPPORTED_REASONING_EFFORTS} "
                "(see the SUPPORTED_REASONING_EFFORTS comment in this module)"
            )

    repetitions = _require(section, "repetitions", "lab01.effort")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ConfigError("lab01.effort.repetitions must be a positive integer")

    max_completion_tokens = _require(section, "max_completion_tokens", "lab01.effort")
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ConfigError("lab01.effort.max_completion_tokens must be a positive integer")

    randomize_execution_order = _require(section, "randomize_execution_order", "lab01.effort")
    if not isinstance(randomize_execution_order, bool):
        raise ConfigError("lab01.effort.randomize_execution_order must be a boolean")

    randomization_seed = _require(section, "randomization_seed", "lab01.effort")
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise ConfigError("lab01.effort.randomization_seed must be an integer")

    use_paired_api_seed = section.get("use_paired_api_seed", False)
    if not isinstance(use_paired_api_seed, bool):
        raise ConfigError("lab01.effort.use_paired_api_seed must be a boolean")

    max_api_calls = section.get("max_api_calls")
    if max_api_calls is not None:
        if isinstance(max_api_calls, bool) or not isinstance(max_api_calls, int) or max_api_calls < 1:
            raise ConfigError("lab01.effort.max_api_calls must be a positive integer or omitted")

    max_estimated_cost_usd = section.get("max_estimated_cost_usd")
    if max_estimated_cost_usd is not None:
        if (
            isinstance(max_estimated_cost_usd, bool)
            or not isinstance(max_estimated_cost_usd, (int, float))
            or max_estimated_cost_usd <= 0
        ):
            raise ConfigError("lab01.effort.max_estimated_cost_usd must be a positive number or omitted")
        max_estimated_cost_usd = float(max_estimated_cost_usd)

    known_keys = {
        "enabled",
        "prompts",
        "reasoning_efforts",
        "repetitions",
        "max_completion_tokens",
        "randomize_execution_order",
        "randomization_seed",
        "use_paired_api_seed",
        "max_api_calls",
        "max_estimated_cost_usd",
    }
    unknown_keys = set(section.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"lab01.effort has unknown configuration key(s): {sorted(unknown_keys)}")

    return Lab01EffortConfig(
        enabled=enabled,
        prompts=tuple(prompts),
        reasoning_efforts=tuple(reasoning_efforts),
        repetitions=repetitions,
        max_completion_tokens=max_completion_tokens,
        randomize_execution_order=randomize_execution_order,
        randomization_seed=randomization_seed,
        use_paired_api_seed=use_paired_api_seed,
        max_api_calls=max_api_calls,
        max_estimated_cost_usd=max_estimated_cost_usd,
    )


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate the TOML configuration at `path` (default: config/default.toml)."""
    config_path = path if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    return AppConfig(
        groq=_build_groq(raw),
        execution=_build_execution(raw),
        experiment=_build_experiment(raw),
        retry=_build_retry(raw),
        concurrency=_build_concurrency(raw),
        pricing=_build_pricing(raw),
        lab02=_build_lab02(raw),
        lab03=_build_lab03(raw),
        lab04=_build_lab04(raw),
        lab05=_build_lab05(raw),
        lab06=_build_lab06(raw),
        lab07=_build_lab07(raw),
        lab08=_build_lab08(raw),
        lab09=_build_lab09(raw),
        lab01_factorial=_build_lab01_factorial(raw),
        lab01_effort=_build_lab01_effort(raw),
    )
