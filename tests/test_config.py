from __future__ import annotations

import pytest

from lab_core.config import ConfigError, load_config

VALID_TOML = """
[groq]
default_model = "llama-3.1-8b-instant"
request_timeout_seconds = 30.0

[execution]
live_api_enabled = false
cost_warning_message = "warning"

[experiment]
default_repetitions = 5
results_dir = "results"

[retry]
max_attempts = 5
base_delay_seconds = 1.0
max_delay_seconds = 30.0
jitter = true

[concurrency]
max_concurrent_requests = 5

[pricing.models."llama-3.1-8b-instant"]
input_per_1m_usd = 0.05
output_per_1m_usd = 0.08
cached_input_per_1m_usd = 0.025
price_verified_date = "2026-10-21"
"""


def _write(tmp_path, content: str):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_default_config_loads_and_is_safe_by_default():
    from lab_core.config import DEFAULT_CONFIG_PATH

    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.execution.live_api_enabled is False


def test_valid_config_loads(tmp_path):
    config = load_config(_write(tmp_path, VALID_TOML))
    assert config.groq.default_model == "llama-3.1-8b-instant"
    assert config.retry.max_attempts == 5
    assert config.pricing["llama-3.1-8b-instant"].input_per_1m_usd == 0.05


def test_missing_section_raises(tmp_path):
    broken = VALID_TOML.replace("[groq]", "[not_groq]")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_live_api_enabled_must_be_bool(tmp_path):
    broken = VALID_TOML.replace("live_api_enabled = false", 'live_api_enabled = "false"')
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_retry_max_delay_must_be_gte_base_delay(tmp_path):
    broken = VALID_TOML.replace("max_delay_seconds = 30.0", "max_delay_seconds = 0.1")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_repetitions_must_be_positive(tmp_path):
    broken = VALID_TOML.replace("default_repetitions = 5", "default_repetitions = 0")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_negative_pricing_rejected(tmp_path):
    broken = VALID_TOML.replace("input_per_1m_usd = 0.05", "input_per_1m_usd = -1.0")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))
