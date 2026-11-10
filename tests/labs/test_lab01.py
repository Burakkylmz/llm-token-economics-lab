from __future__ import annotations

import json
import warnings

import pytest

from lab_core.config import ModelPricing
from lab_core.metrics import TokenCostMetrics
from lab_core.results_writer import append_csv, write_json
from labs.lab01_token_anatomy.cost import calculate_cost
from labs.lab01_token_anatomy.experiments import run_experiment
from labs.lab01_token_anatomy.observability import (
    compute_non_reasoning_completion_tokens,
    summarize_values,
    tally_finish_reasons,
)


def _pricing(**overrides) -> ModelPricing:
    defaults = dict(
        input_per_1m_usd=1.0,
        output_per_1m_usd=2.0,
        cached_input_per_1m_usd=0.5,
        price_verified_date="2026-10-21",
    )
    defaults.update(overrides)
    return ModelPricing(**defaults)


def test_calculate_cost_basic():
    tokens = TokenCostMetrics(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost = calculate_cost(tokens, _pricing())
    assert cost == 1.0 + 2.0


def test_calculate_cost_applies_cached_discount():
    tokens = TokenCostMetrics(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=1_000_000)
    cost = calculate_cost(tokens, _pricing())
    assert cost == 0.5  # fully cached: uncached portion is 0, cached portion billed at cached rate


def test_calculate_cost_does_not_double_count_reasoning_tokens():
    # reasoning_tokens is a subset of completion_tokens and must not add extra cost on its own.
    with_reasoning = TokenCostMetrics(prompt_tokens=100, completion_tokens=500, reasoning_tokens=300)
    without_reasoning = TokenCostMetrics(prompt_tokens=100, completion_tokens=500, reasoning_tokens=None)
    pricing = _pricing()
    assert calculate_cost(with_reasoning, pricing) == calculate_cost(without_reasoning, pricing)


def test_calculate_cost_returns_none_when_usage_unavailable():
    tokens = TokenCostMetrics(prompt_tokens=None, completion_tokens=None)
    assert calculate_cost(tokens, _pricing()) is None


def test_run_experiment_sends_correct_prompt_and_computes_cost(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="Paris", prompt_tokens=8, completion_tokens=4, total_tokens=12)
    client = fake_sync_client_factory([response])

    results = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="llama-3.1-8b-instant",
        repetitions=1,
    )

    assert len(results) == 1
    result = results[0]
    assert result.tokens_and_cost.prompt_tokens == 8
    assert result.tokens_and_cost.completion_tokens == 4
    assert result.tokens_and_cost.estimated_cost_usd is not None
    assert client.completions.calls[0]["model"] == "llama-3.1-8b-instant"
    assert "capital of France" in client.completions.calls[0]["messages"][0]["content"]


def test_run_experiment_repeats_the_requested_number_of_times(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response() for _ in range(3)])
    results = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="long_prompt",
        model="llama-3.1-8b-instant",
        repetitions=3,
    )
    assert len(results) == 3


# --- Observability: reasoning tokens, finish_reason, derived metrics -------


def test_compute_non_reasoning_completion_tokens_when_reported():
    assert compute_non_reasoning_completion_tokens(completion_tokens=41, reasoning_tokens=24) == 17


def test_compute_non_reasoning_completion_tokens_none_when_reasoning_missing():
    assert compute_non_reasoning_completion_tokens(completion_tokens=41, reasoning_tokens=None) is None


def test_compute_non_reasoning_completion_tokens_handles_explicit_zero():
    # 0 is a real reported value, distinct from "unavailable" (None).
    assert compute_non_reasoning_completion_tokens(completion_tokens=41, reasoning_tokens=0) == 41


def test_compute_non_reasoning_completion_tokens_none_when_completion_missing():
    assert compute_non_reasoning_completion_tokens(completion_tokens=None, reasoning_tokens=10) is None


def test_compute_non_reasoning_completion_tokens_warns_and_returns_none_when_reasoning_exceeds_completion():
    with pytest.warns(UserWarning, match="Inconsistent token accounting"):
        result = compute_non_reasoning_completion_tokens(completion_tokens=10, reasoning_tokens=30)
    assert result is None


def test_compute_non_reasoning_completion_tokens_warns_and_returns_none_when_reasoning_negative():
    with pytest.warns(UserWarning, match="Inconsistent token accounting"):
        result = compute_non_reasoning_completion_tokens(completion_tokens=10, reasoning_tokens=-5)
    assert result is None


def test_summarize_values_computes_min_max_mean_stdev_range():
    stats = summarize_values([32, 38, 39, 39, 41])
    assert stats.count_available == 5
    assert stats.count_total == 5
    assert stats.minimum == 32
    assert stats.maximum == 41
    assert stats.mean == pytest.approx(37.8)
    assert stats.range == 9
    assert stats.standard_deviation is not None


def test_summarize_values_reports_partial_availability_without_treating_none_as_zero():
    stats = summarize_values([24, None, 30, None, 18])
    assert stats.count_available == 3
    assert stats.count_total == 5
    assert stats.mean == pytest.approx((24 + 30 + 18) / 3)


def test_summarize_values_stdev_is_none_below_two_measurements():
    stats = summarize_values([42])
    assert stats.count_available == 1
    assert stats.standard_deviation is None
    assert stats.mean == 42


def test_summarize_values_all_none_leaves_everything_unavailable():
    stats = summarize_values([None, None])
    assert stats.count_available == 0
    assert stats.count_total == 2
    assert stats.mean is None
    assert stats.standard_deviation is None


def test_tally_finish_reasons_groups_missing_as_unavailable():
    tally = tally_finish_reasons(["stop", "stop", "length", None])
    assert tally == {"stop": 2, "length": 1, "unavailable": 1}


def test_run_experiment_reports_reasoning_and_non_reasoning_tokens(
    fake_sync_client_factory, make_response, config_with_pricing
):
    response = make_response(content="answer", prompt_tokens=78, completion_tokens=41, reasoning_tokens=24)
    client = fake_sync_client_factory([response])

    [result] = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=1,
    )

    assert result.tokens_and_cost.reasoning_tokens == 24
    assert result.derived.non_reasoning_completion_tokens == 17
    assert result.derived.final_answer_char_count == len("answer")


def test_run_experiment_leaves_derived_fields_none_when_reasoning_unreported(
    fake_sync_client_factory, make_response, config_with_pricing
):
    response = make_response(content="Paris", completion_tokens=4, reasoning_tokens=None)
    client = fake_sync_client_factory([response])

    [result] = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=1,
    )

    assert result.tokens_and_cost.reasoning_tokens is None
    assert result.derived.non_reasoning_completion_tokens is None


def test_run_experiment_captures_finish_reason(fake_sync_client_factory, make_response, config_with_pricing):
    response = make_response(content="truncated...", finish_reason="length")
    client = fake_sync_client_factory([response])

    [result] = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=1,
    )

    assert result.tokens_and_cost.finish_reason == "length"


def test_run_experiment_does_not_double_count_reasoning_tokens_in_cost(
    fake_sync_client_factory, make_response, config_with_pricing
):
    # completion_tokens already includes reasoning_tokens per the OpenAI-compatible
    # usage convention -- cost must be based on completion_tokens alone.
    with_reasoning = make_response(completion_tokens=500, reasoning_tokens=300)
    without_reasoning = make_response(completion_tokens=500, reasoning_tokens=None)
    client = fake_sync_client_factory([with_reasoning, without_reasoning])

    results = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=2,
    )

    assert results[0].tokens_and_cost.estimated_cost_usd == results[1].tokens_and_cost.estimated_cost_usd


def test_run_experiment_captures_system_fingerprint_when_present(
    fake_sync_client_factory, make_response, config_with_pricing
):
    response = make_response(system_fingerprint="fp_abc123")
    client = fake_sync_client_factory([response])

    [result] = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=1,
    )

    assert result.metadata.system_fingerprint == "fp_abc123"


def test_run_experiment_assigns_sequential_run_index(fake_sync_client_factory, make_response, config_with_pricing):
    client = fake_sync_client_factory([make_response() for _ in range(3)])

    results = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=3,
    )

    assert [r.metadata.run_index for r in results] == [1, 2, 3]


# --- Serialization: new fields round-trip through JSON/CSV, raw text excluded ---


def test_json_serialization_includes_new_observability_fields(
    fake_sync_client_factory, make_response, config_with_pricing, tmp_path
):
    response = make_response(content="The capital of France is Paris.", completion_tokens=41, reasoning_tokens=24, finish_reason="stop")
    client = fake_sync_client_factory([response])
    [result] = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=1,
    )

    output_path = tmp_path / "lab01_token_anatomy.json"
    write_json([result], output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))[0]

    assert payload["tokens_and_cost"]["finish_reason"] == "stop"
    assert payload["derived"]["non_reasoning_completion_tokens"] == 17
    assert payload["derived"]["final_answer_char_count"] == len("The capital of France is Paris.")
    assert payload["metadata"]["run_index"] == 1


def test_json_serialization_does_not_persist_raw_answer_text(
    fake_sync_client_factory, make_response, config_with_pricing, tmp_path
):
    response = make_response(content="SECRET_ANSWER_TEXT_MARKER")
    client = fake_sync_client_factory([response])
    [result] = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=1,
    )

    output_path = tmp_path / "lab01_token_anatomy.json"
    write_json([result], output_path)
    raw_json_text = output_path.read_text(encoding="utf-8")

    assert "SECRET_ANSWER_TEXT_MARKER" not in raw_json_text


def test_csv_serialization_includes_new_observability_fields(
    fake_sync_client_factory, make_response, config_with_pricing, tmp_path
):
    response = make_response(completion_tokens=41, reasoning_tokens=24, finish_reason="stop")
    client = fake_sync_client_factory([response])
    [result] = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=1,
    )

    output_path = tmp_path / "lab01_token_anatomy.csv"
    append_csv([result], output_path)

    import csv

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["tokens_and_cost.finish_reason"] == "stop"
    assert row["derived.non_reasoning_completion_tokens"] == "17"
    assert row["metadata.run_index"] == "1"


def test_csv_serialization_keeps_one_row_per_call_across_experiments(
    fake_sync_client_factory, make_response, config_with_pricing, tmp_path
):
    responses = [make_response() for _ in range(3)]
    client = fake_sync_client_factory(responses)
    results = run_experiment(
        client=client,
        config=config_with_pricing,
        experiment="short_prompt",
        model="openai/gpt-oss-20b",
        repetitions=3,
    )

    output_path = tmp_path / "lab01_token_anatomy.csv"
    append_csv(results, output_path)

    import csv

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3


def test_calculate_cost_rejects_invalid_usage():
    pricing = _pricing()
    for tokens in (
        TokenCostMetrics(prompt_tokens=100, completion_tokens=0, cached_tokens=400),
        TokenCostMetrics(prompt_tokens=100, completion_tokens=0, cached_tokens=-1),
        TokenCostMetrics(prompt_tokens=-1, completion_tokens=1),
        TokenCostMetrics(prompt_tokens=1, completion_tokens=-1),
    ):
        assert calculate_cost(tokens, pricing) is None


def test_calculate_cost_accepts_valid_cache_boundaries():
    pricing = _pricing()
    for cached in (0, 100):
        assert calculate_cost(
            TokenCostMetrics(prompt_tokens=100, completion_tokens=1, cached_tokens=cached),
            pricing,
        ) is not None
