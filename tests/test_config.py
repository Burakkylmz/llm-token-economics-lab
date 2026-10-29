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


# --- lab04: entirely optional, documented defaults -------------------------


def test_lab04_section_absent_still_gets_documented_defaults(tmp_path):
    config = load_config(_write(tmp_path, VALID_TOML))
    lab04 = config.lab04
    assert lab04.enabled is True
    assert lab04.model == "openai/gpt-oss-20b"
    assert lab04.output_limit_values == (64, 256, 1024)
    assert lab04.output_limit_repetitions == 2
    assert lab04.reasoning_effort == "low"
    assert lab04.invocation_execution_mode == "sequential"
    assert lab04.max_api_calls == 10
    assert lab04.max_estimated_cost_usd == 0.05


def test_lab04_section_present_overrides_defaults(tmp_path):
    toml = VALID_TOML + """
[lab04]
enabled = true
model = "openai/gpt-oss-120b"
output_limit_values = [32, 128]
output_limit_repetitions = 3
reasoning_effort = "high"
invocation_max_completion_tokens = 512
invocation_execution_mode = "sequential"
max_api_calls = 20
max_estimated_cost_usd = 0.25
"""
    config = load_config(_write(tmp_path, toml))
    lab04 = config.lab04
    assert lab04.model == "openai/gpt-oss-120b"
    assert lab04.output_limit_values == (32, 128)
    assert lab04.output_limit_repetitions == 3
    assert lab04.reasoning_effort == "high"
    assert lab04.invocation_max_completion_tokens == 512
    assert lab04.max_api_calls == 20
    assert lab04.max_estimated_cost_usd == 0.25


def test_lab04_non_positive_output_limit_repetitions_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab04]\noutput_limit_repetitions = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab04_duplicate_output_limit_values_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab04]\noutput_limit_values = [64, 64, 256]\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab04_unsupported_reasoning_effort_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab04]\nreasoning_effort = "extreme"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab04_non_sequential_invocation_execution_mode_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab04]\ninvocation_execution_mode = "parallel"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab04_unknown_key_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab04]\nnot_a_real_key = 1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab04_negative_max_estimated_cost_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab04]\nmax_estimated_cost_usd = -1.0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


# --- lab05: small_model/large_model stay required; everything else is new,
#     optional, documented defaults -------------------------------------


def _with_lab05_overrides(extra_lines: str) -> str:
    # VALID_TOML already has one [lab05] table -- TOML forbids a second
    # [lab05] header, so new keys are injected into the EXISTING table
    # instead of appending a conflicting second one.
    return VALID_TOML.replace(
        '[lab05]\nsmall_model = "llama-3.1-8b-instant"\nlarge_model = "llama-3.3-70b-versatile"\n',
        '[lab05]\nsmall_model = "llama-3.1-8b-instant"\nlarge_model = "llama-3.3-70b-versatile"\n' + extra_lines,
    )


def test_lab05_section_with_only_legacy_keys_still_gets_documented_defaults(tmp_path):
    config = load_config(_write(tmp_path, VALID_TOML))
    lab05 = config.lab05
    assert lab05.small_model == "llama-3.1-8b-instant"
    assert lab05.large_model == "llama-3.3-70b-versatile"
    assert lab05.default_profile == "default"
    assert lab05.reasoning_effort == "low"
    assert lab05.max_completion_tokens == 1024
    assert lab05.max_api_calls == 18
    assert lab05.max_estimated_cost_usd == 0.05


def test_lab05_section_present_overrides_defaults(tmp_path):
    toml = _with_lab05_overrides(
        'default_profile = "pilot"\n'
        'reasoning_effort = "high"\n'
        "max_completion_tokens = 512\n"
        "randomize_task_order = true\n"
        "randomization_seed = 7\n"
        "max_api_calls = 8\n"
        "max_estimated_cost_usd = 0.25\n"
    )
    config = load_config(_write(tmp_path, toml))
    lab05 = config.lab05
    assert lab05.default_profile == "pilot"
    assert lab05.reasoning_effort == "high"
    assert lab05.max_completion_tokens == 512
    assert lab05.randomize_task_order is True
    assert lab05.randomization_seed == 7
    assert lab05.max_api_calls == 8
    assert lab05.max_estimated_cost_usd == 0.25


def test_lab05_invalid_profile_rejected(tmp_path):
    broken = _with_lab05_overrides('default_profile = "nonexistent"\n')
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab05_unsupported_reasoning_effort_rejected(tmp_path):
    broken = _with_lab05_overrides('reasoning_effort = "extreme"\n')
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab05_unknown_key_rejected(tmp_path):
    broken = _with_lab05_overrides("not_a_real_key = 1\n")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab05_negative_max_estimated_cost_rejected(tmp_path):
    broken = _with_lab05_overrides("max_estimated_cost_usd = -1.0\n")
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


# --- lab06: entirely optional, documented defaults -------------------------


def test_lab06_section_absent_still_gets_documented_defaults(tmp_path):
    config = load_config(_write(tmp_path, VALID_TOML))
    lab06 = config.lab06
    assert lab06.enabled is True
    assert lab06.model == "openai/gpt-oss-20b"
    assert lab06.default_profile == "default"
    assert lab06.paired_repetitions == 5
    assert lab06.reasoning_effort == "low"
    assert lab06.max_completion_tokens == 768
    assert lab06.request_usage_when_supported is True
    assert lab06.record_event_timeline is True
    assert lab06.max_api_calls == 16
    assert lab06.max_estimated_cost_usd == 0.05


def test_lab06_section_present_overrides_defaults(tmp_path):
    toml = VALID_TOML + """
[lab06]
enabled = true
model = "openai/gpt-oss-120b"
default_profile = "pilot"
paired_repetitions = 3
reasoning_effort = "high"
max_completion_tokens = 512
randomize_condition_order = true
randomization_seed = 7
request_usage_when_supported = false
record_event_timeline = false
max_api_calls = 8
max_estimated_cost_usd = 0.25
"""
    config = load_config(_write(tmp_path, toml))
    lab06 = config.lab06
    assert lab06.model == "openai/gpt-oss-120b"
    assert lab06.default_profile == "pilot"
    assert lab06.paired_repetitions == 3
    assert lab06.reasoning_effort == "high"
    assert lab06.max_completion_tokens == 512
    assert lab06.randomize_condition_order is True
    assert lab06.randomization_seed == 7
    assert lab06.request_usage_when_supported is False
    assert lab06.record_event_timeline is False
    assert lab06.max_api_calls == 8
    assert lab06.max_estimated_cost_usd == 0.25


def test_lab06_non_positive_paired_repetitions_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab06]\npaired_repetitions = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab06_invalid_profile_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab06]\ndefault_profile = "nonexistent"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab06_unsupported_reasoning_effort_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab06]\nreasoning_effort = "extreme"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab06_unknown_key_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab06]\nnot_a_real_key = 1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab06_negative_max_estimated_cost_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab06]\nmax_estimated_cost_usd = -1.0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


# --- lab07: entirely optional, documented defaults -------------------------


def test_lab07_section_absent_still_gets_documented_defaults(tmp_path):
    config = load_config(_write(tmp_path, VALID_TOML))
    lab07 = config.lab07
    assert lab07.enabled is True
    assert lab07.default_profile == "default"
    assert lab07.randomization_seed == 42
    assert lab07.max_simulation_events == 10000
    assert lab07.max_clients == 50
    assert lab07.max_attempts == 5
    assert lab07.base_delay_seconds == 1.0
    assert lab07.max_delay_seconds == 30.0
    assert lab07.max_total_wait_seconds == 120.0
    assert lab07.simulation_deadline_seconds is None
    assert lab07.transient_fail_count == 2
    assert lab07.recovery_time_seconds == 3.0
    assert lab07.contention_client_count == 8
    assert lab07.contention_capacity_per_window == 2
    assert lab07.quota_client_count == 3
    assert lab07.quota_rpm == 5
    assert lab07.quota_tpm == 250


def test_lab07_section_present_overrides_defaults(tmp_path):
    toml = VALID_TOML + """
[lab07]
enabled = true
default_profile = "pilot"
randomization_seed = 7
max_simulation_events = 500
max_clients = 10
max_attempts = 3
base_delay_seconds = 0.5
max_delay_seconds = 10.0
max_total_wait_seconds = 60.0
transient_fail_count = 1
recovery_time_seconds = 2.0
contention_client_count = 4
contention_capacity_per_window = 1
contention_window_seconds = 0.5
quota_client_count = 2
quota_rpm = 3
quota_tpm = 100
quota_token_demand_per_request = 50
quota_window_seconds = 30.0
"""
    config = load_config(_write(tmp_path, toml))
    lab07 = config.lab07
    assert lab07.default_profile == "pilot"
    assert lab07.randomization_seed == 7
    assert lab07.max_simulation_events == 500
    assert lab07.max_clients == 10
    assert lab07.max_attempts == 3
    assert lab07.transient_fail_count == 1
    assert lab07.recovery_time_seconds == 2.0
    assert lab07.contention_client_count == 4


def test_lab07_invalid_profile_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab07]\ndefault_profile = "nonexistent"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab07_zero_max_attempts_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab07]\nmax_attempts = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab07_negative_transient_fail_count_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab07]\ntransient_fail_count = -1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab07_zero_transient_fail_count_is_valid(tmp_path):
    # Zero simulated failures (server succeeds on the very first attempt)
    # is a legitimate configuration, not an error.
    toml = VALID_TOML + "\n[lab07]\ntransient_fail_count = 0\n"
    config = load_config(_write(tmp_path, toml))
    assert config.lab07.transient_fail_count == 0


def test_lab07_max_delay_less_than_base_delay_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab07]\nbase_delay_seconds = 10.0\nmax_delay_seconds = 1.0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab07_contention_client_count_exceeding_max_clients_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab07]\nmax_clients = 2\ncontention_client_count = 8\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab07_quota_client_count_exceeding_max_clients_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab07]\nmax_clients = 2\nquota_client_count = 3\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab07_unknown_key_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab07]\nnot_a_real_key = 1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


# --- lab08: optional section, backward compatibility -----------------------


def test_lab08_section_absent_still_gets_documented_defaults(tmp_path):
    config = load_config(_write(tmp_path, VALID_TOML))
    lab08 = config.lab08
    assert lab08.enabled is True
    assert lab08.model == "openai/gpt-oss-20b"
    assert lab08.default_profile == "default"
    assert lab08.concurrency_levels == (1, 5)
    assert lab08.repetitions == 1
    assert lab08.reasoning_effort == "low"
    assert lab08.max_allowed_concurrency == 10
    assert lab08.sdk_max_retries == 0
    assert lab08.max_api_calls == 32
    assert lab08.max_estimated_cost_usd == 0.05


def test_lab08_section_present_overrides_defaults(tmp_path):
    toml = VALID_TOML + """
[lab08]
enabled = true
model = "openai/gpt-oss-120b"
default_profile = "pilot"
concurrency_levels = [1, 2, 4]
repetitions = 2
reasoning_effort = "medium"
max_completion_tokens = 128
randomize_condition_order = true
randomization_seed = 7
max_allowed_concurrency = 8
sdk_max_retries = 1
max_api_calls = 16
max_estimated_cost_usd = 0.02
"""
    config = load_config(_write(tmp_path, toml))
    lab08 = config.lab08
    assert lab08.default_profile == "pilot"
    assert lab08.concurrency_levels == (1, 2, 4)
    assert lab08.repetitions == 2
    assert lab08.reasoning_effort == "medium"
    assert lab08.sdk_max_retries == 1


def test_lab08_invalid_profile_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab08]\ndefault_profile = "nonexistent"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab08_zero_concurrency_level_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab08]\nconcurrency_levels = [0, 5]\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab08_duplicate_concurrency_levels_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab08]\nconcurrency_levels = [1, 1, 5]\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab08_concurrency_exceeding_max_allowed_concurrency_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab08]\nconcurrency_levels = [1, 20]\nmax_allowed_concurrency = 10\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab08_unsupported_reasoning_effort_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab08]\nreasoning_effort = "extreme"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab08_negative_sdk_max_retries_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab08]\nsdk_max_retries = -1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab08_unknown_key_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab08]\nnot_a_real_key = 1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab08_negative_max_estimated_cost_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab08]\nmax_estimated_cost_usd = -0.5\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


# --- lab10: optional section, backward compatibility -----------------------


def test_lab10_section_absent_still_gets_documented_defaults(tmp_path):
    config = load_config(_write(tmp_path, VALID_TOML))
    lab10 = config.lab10
    assert lab10.enabled is True
    assert lab10.model == "openai/gpt-oss-20b"
    assert lab10.default_profile == "default"
    assert lab10.dataset_version == "v1"
    assert lab10.summarization_max_completion_tokens == 1536
    assert lab10.answering_reasoning_effort == "low"
    assert lab10.answering_max_completion_tokens == 256
    assert lab10.persist_synthetic_io is True
    assert lab10.show_all_io_in_terminal is False
    assert lab10.max_api_calls == 28
    assert lab10.max_estimated_cost_usd == 0.10


def test_lab10_section_present_overrides_defaults(tmp_path):
    toml = VALID_TOML + """
[lab10]
enabled = true
model = "openai/gpt-oss-120b"
default_profile = "pilot"
dataset_version = "v1"
summarization_max_completion_tokens = 2048
answering_reasoning_effort = "medium"
answering_max_completion_tokens = 512
persist_synthetic_io = false
show_all_io_in_terminal = true
max_api_calls = 8
max_estimated_cost_usd = 0.02
"""
    config = load_config(_write(tmp_path, toml))
    lab10 = config.lab10
    assert lab10.default_profile == "pilot"
    assert lab10.summarization_max_completion_tokens == 2048
    assert lab10.answering_reasoning_effort == "medium"
    assert lab10.persist_synthetic_io is False
    assert lab10.max_api_calls == 8


def test_lab10_invalid_profile_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab10]\ndefault_profile = "nonexistent"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_unsupported_reasoning_effort_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab10]\nanswering_reasoning_effort = "extreme"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_non_positive_summarization_max_completion_tokens_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab10]\nsummarization_max_completion_tokens = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_non_positive_answering_max_completion_tokens_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab10]\nanswering_max_completion_tokens = 0\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_empty_dataset_version_rejected(tmp_path):
    broken = VALID_TOML + '\n[lab10]\ndataset_version = ""\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_negative_max_api_calls_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab10]\nmax_api_calls = -1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_negative_max_estimated_cost_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab10]\nmax_estimated_cost_usd = -0.5\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_unknown_key_rejected(tmp_path):
    broken = VALID_TOML + "\n[lab10]\nnot_a_real_key = 1\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))


def test_lab10_persist_synthetic_io_must_be_bool(tmp_path):
    broken = VALID_TOML + '\n[lab10]\npersist_synthetic_io = "yes"\n'
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, broken))
