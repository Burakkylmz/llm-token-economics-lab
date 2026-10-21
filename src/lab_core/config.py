"""Loading and validation for the TOML-based lab configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

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
class AppConfig:
    groq: GroqConfig
    execution: ExecutionConfig
    experiment: ExperimentConfig
    retry: RetryConfig
    concurrency: ConcurrencyConfig
    pricing: dict[str, ModelPricing]


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
    )
