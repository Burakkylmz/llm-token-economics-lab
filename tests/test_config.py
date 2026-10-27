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

[lab03]
caching_capable_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

[lab05]
small_model = "llama-3.1-8b-instant"
large_model = "llama-3.3-70b-versatile"

[lab09]
batch_completion_window = "24h"
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


# --- lab01.factorial: optional section, backward compatibility ------------


def test_lab01_factorial_section_absent_is_none(tmp_path):
    # A config file written before this feature existed (no [lab01.factorial]
    # at all) must keep loading and running Lab01 basic mode unchanged.
    config = load_config(_write(tmp_path, VALID_TOML))
    assert config.lab01_factorial is None


FACTORIAL_TOML = (
    VALID_TOML
    + """
[lab01.factorial]
enabled = true
prompts = ["short_prompt", "long_prompt"]
reasoning_efforts = ["low", "medium", "high"]
temperatures = [0.5, 1.0, 1.5]
top_p_values = [0.7, 0.9, 1.0]
repetitions = 10
max_completion_tokens = 4096
randomize_execution_order = true
randomization_seed = 42
"""
)


def test_lab01_factorial_section_present_and_valid(tmp_path):
    config = load_config(_write(tmp_path, FACTORIAL_TOML))
    f = config.lab01_factorial
    assert f is not None
    assert f.enabled is True
    assert f.prompts == ("short_prompt", "long_prompt")
    assert f.reasoning_efforts == ("low", "medium", "high")
    assert f.temperatures == (0.5, 1.0, 1.5)
    assert f.top_p_values == (0.7, 0.9, 1.0)
    assert f.repetitions == 10
    assert f.max_completion_tokens == 4096
    assert f.use_paired_api_seed is False
    assert f.max_api_calls is None
    assert f.max_estimated_cost_usd is None


def test_lab01_factorial_optional_budget_fields(tmp_path):
    toml = FACTORIAL_TOML + "\nmax_api_calls = 100\nmax_estimated_cost_usd = 0.5\n"
    config = load_config(_write(tmp_path, toml))
    assert config.lab01_factorial.max_api_calls == 100
    assert config.lab01_factorial.max_estimated_cost_usd == 0.5


def test_lab01_factorial_empty_prompts_rejected(tmp_path):
    broken = FACTORIAL_TOML.replace('prompts = ["short_prompt", "long_prompt"]', "prompts = []")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_invalid_temperature_rejected(tmp_path):
    broken = FACTORIAL_TOML.replace("temperatures = [0.5, 1.0, 1.5]", "temperatures = [0.5, 1.0, 2.5]")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_invalid_top_p_rejected(tmp_path):
    broken = FACTORIAL_TOML.replace("top_p_values = [0.7, 0.9, 1.0]", "top_p_values = [0.7, 0.9, 1.1]")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_top_p_zero_rejected(tmp_path):
    # top_p is a nucleus-sampling probability mass; 0 is out of range (exclusive lower bound).
    broken = FACTORIAL_TOML.replace("top_p_values = [0.7, 0.9, 1.0]", "top_p_values = [0.0, 0.9, 1.0]")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_unsupported_reasoning_effort_rejected(tmp_path):
    broken = FACTORIAL_TOML.replace(
        'reasoning_efforts = ["low", "medium", "high"]', 'reasoning_efforts = ["low", "none"]'
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_duplicate_levels_rejected(tmp_path):
    broken = FACTORIAL_TOML.replace("temperatures = [0.5, 1.0, 1.5]", "temperatures = [0.5, 0.5, 1.5]")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_near_duplicate_floats_rejected(tmp_path):
    # Distinct floats that would format identically at the precision used for
    # condition_id must be rejected here, before they could ever produce a
    # silent condition_id collision downstream.
    broken = FACTORIAL_TOML.replace("temperatures = [0.5, 1.0, 1.5]", "temperatures = [0.5, 0.5000001, 1.5]")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_non_positive_repetitions_rejected(tmp_path):
    broken = FACTORIAL_TOML.replace("repetitions = 10", "repetitions = 0")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_non_positive_max_completion_tokens_rejected(tmp_path):
    broken = FACTORIAL_TOML.replace("max_completion_tokens = 4096", "max_completion_tokens = 0")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_unknown_key_rejected(tmp_path):
    broken = FACTORIAL_TOML + "\nthis_is_not_a_real_key = 1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_negative_max_api_calls_rejected(tmp_path):
    broken = FACTORIAL_TOML + "\nmax_api_calls = -5\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab01_factorial_negative_max_estimated_cost_rejected(tmp_path):
    broken = FACTORIAL_TOML + "\nmax_estimated_cost_usd = -1.0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


# --- lab02: NOT optional, always present with safe defaults ---------------


def test_lab02_section_absent_still_gets_documented_defaults(tmp_path):
    # VALID_TOML has no [lab02] section at all -- `lab02 all --live` must
    # still plan exactly this project's documented 10-call default.
    config = load_config(_write(tmp_path, VALID_TOML))
    lab02 = config.lab02
    assert lab02.enabled is True
    assert lab02.repetitions == 2
    assert lab02.multi_turn_turns == 4
    assert lab02.max_api_calls == 10


def test_lab02_section_present_overrides_defaults(tmp_path):
    toml = (
        VALID_TOML
        + """
[lab02]
enabled = true
repetitions = 3
multi_turn_turns = 5
max_completion_tokens = 1024
randomize_context_order = true
randomization_seed = 7
max_api_calls = 50
max_estimated_cost_usd = 0.25
"""
    )
    config = load_config(_write(tmp_path, toml))
    lab02 = config.lab02
    assert lab02.repetitions == 3
    assert lab02.multi_turn_turns == 5
    assert lab02.max_completion_tokens == 1024
    assert lab02.randomize_context_order is True
    assert lab02.randomization_seed == 7
    assert lab02.max_api_calls == 50
    assert lab02.max_estimated_cost_usd == 0.25


def test_lab02_non_positive_repetitions_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab02]\nrepetitions = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab02_unknown_key_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab02]\nnot_a_real_key = 1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab02_negative_max_estimated_cost_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab02]\nmax_estimated_cost_usd = -1.0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


# --- lab03: caching_capable_models stays required; everything else is new,
#     optional, and backward compatible -------------------------------------


def _with_lab03_overrides(extra_lines: str) -> str:
    # VALID_TOML already has one [lab03] table -- TOML forbids a second
    # [lab03] header, so new keys are injected into the EXISTING table
    # rather than appended as a duplicate section.
    return VALID_TOML.replace(
        '[lab03]\ncaching_capable_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]\n',
        '[lab03]\ncaching_capable_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]\n' + extra_lines,
    )


def test_lab03_section_with_only_legacy_key_still_gets_documented_defaults(tmp_path):
    # VALID_TOML's [lab03] section only ever set caching_capable_models --
    # `lab03 all --live` must still plan exactly this project's documented
    # 6-call default.
    config = load_config(_write(tmp_path, VALID_TOML))
    lab03 = config.lab03
    assert lab03.caching_capable_models == ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
    assert lab03.repetitions == 3
    assert lab03.max_completion_tokens == 2048
    assert lab03.randomize_condition_order_within_pair is False
    assert lab03.max_api_calls == 8
    assert lab03.max_estimated_cost_usd == 0.05


def test_lab03_section_present_overrides_defaults(tmp_path):
    toml = _with_lab03_overrides(
        "repetitions = 4\n"
        "max_completion_tokens = 1024\n"
        "randomize_condition_order_within_pair = true\n"
        "randomization_seed = 7\n"
        "max_api_calls = 50\n"
        "max_estimated_cost_usd = 0.25\n"
    )
    config = load_config(_write(tmp_path, toml))
    lab03 = config.lab03
    assert lab03.repetitions == 4
    assert lab03.max_completion_tokens == 1024
    assert lab03.randomize_condition_order_within_pair is True
    assert lab03.randomization_seed == 7
    assert lab03.max_api_calls == 50
    assert lab03.max_estimated_cost_usd == 0.25


def test_lab03_non_positive_repetitions_rejected(tmp_path):
    broken = _with_lab03_overrides("repetitions = 0\n")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab03_unknown_key_rejected(tmp_path):
    broken = _with_lab03_overrides("not_a_real_key = 1\n")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab03_negative_max_estimated_cost_rejected(tmp_path):
    broken = _with_lab03_overrides("max_estimated_cost_usd = -1.0\n")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))
