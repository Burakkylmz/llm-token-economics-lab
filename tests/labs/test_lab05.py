from __future__ import annotations

from labs.lab05_model_routing.golden_dataset import (
    DEFAULT_TASK_IDS,
    GOLDEN_DATASET,
    PILOT_TASK_IDS,
    TASK_BY_ID,
    get_task,
    task_ids_for_profile,
)
from labs.lab05_model_routing.routing import choose_model, extract_task_features, fixed_decision


def test_dataset_has_at_least_twelve_tasks():
    assert len(GOLDEN_DATASET) >= 12


def test_every_task_id_is_unique():
    ids = [t.task_id for t in GOLDEN_DATASET]
    assert len(ids) == len(set(ids))


def test_dataset_covers_every_required_category():
    required = {
        "factual_retrieval", "simple_arithmetic", "structured_extraction", "classification",
        "multi_step_reasoning", "multi_constraint_decision", "conditional_business_rule",
        "structured_output_generation",
    }
    present = {t.category for t in GOLDEN_DATASET}
    assert required <= present


def test_every_complexity_label_is_simple_or_complex():
    assert {t.complexity_label for t in GOLDEN_DATASET} <= {"simple", "complex"}


def test_every_evaluation_type_is_a_recognized_evaluator():
    from labs.lab05_model_routing.evaluation import EVALUATION_TYPES

    assert {t.evaluation_type for t in GOLDEN_DATASET} <= set(EVALUATION_TYPES)


def test_pilot_and_default_task_ids_reference_real_tasks():
    for task_id in (*PILOT_TASK_IDS, *DEFAULT_TASK_IDS):
        assert task_id in TASK_BY_ID


def test_pilot_is_exactly_four_tasks():
    assert len(PILOT_TASK_IDS) == 4


def test_default_is_exactly_six_tasks():
    assert len(DEFAULT_TASK_IDS) == 6


def test_pilot_and_default_span_more_than_one_category():
    pilot_categories = {TASK_BY_ID[t].category for t in PILOT_TASK_IDS}
    default_categories = {TASK_BY_ID[t].category for t in DEFAULT_TASK_IDS}
    assert len(pilot_categories) > 1
    assert len(default_categories) > 1


def test_task_ids_for_profile_is_deterministic():
    assert task_ids_for_profile("pilot") == task_ids_for_profile("pilot")
    assert task_ids_for_profile("default") == task_ids_for_profile("extended")


def test_get_task_returns_none_for_unknown_id():
    assert get_task("not_a_real_task") is None


def test_dataset_loading_is_deterministic_across_calls():
    from labs.lab05_model_routing.golden_dataset import _load_dataset

    first = _load_dataset()
    second = _load_dataset()
    assert first == second


# --- Routing policy ---------------------------------------------------------


def test_router_sends_short_simple_prompt_to_small_model():
    decision = choose_model("What is 2 + 2? Answer with only the number.", "small", "large")
    assert decision.selected_model == "small"
    assert decision.model_tier == "small"


def test_router_sends_multi_step_reasoning_prompt_to_large_model():
    task = TASK_BY_ID["task_009"]
    decision = choose_model(task.prompt, "small", "large")
    assert decision.selected_model == "large"
    assert "multi_step_reasoning_indicators" in decision.matched_rules


def test_router_sends_conditional_constraint_prompt_to_large_model():
    task = TASK_BY_ID["task_011"]
    decision = choose_model(task.prompt, "small", "large")
    assert decision.selected_model == "large"
    assert "conditional_constraints" in decision.matched_rules


def test_router_sends_code_generation_prompt_to_large_model():
    task = TASK_BY_ID["task_013"]
    decision = choose_model(task.prompt, "small", "large")
    assert decision.selected_model == "large"
    assert "code_generation_indicator" in decision.matched_rules


def test_router_sends_structured_output_generation_prompt_to_large_model():
    task = TASK_BY_ID["task_012"]
    decision = choose_model(task.prompt, "small", "large")
    assert decision.selected_model == "large"
    assert "structured_output_indicator" in decision.matched_rules


def test_same_prompt_always_produces_the_same_decision():
    task = TASK_BY_ID["task_009"]
    a = choose_model(task.prompt, "small", "large")
    b = choose_model(task.prompt, "small", "large")
    assert a.selected_model == b.selected_model
    assert a.matched_rules == b.matched_rules


def test_routing_reason_is_always_recorded():
    decision = choose_model("What is 2 + 2?", "small", "large")
    assert decision.routing_reason
    assert isinstance(decision.routing_reason, str)


def test_router_functions_only_accept_prompt_text_never_a_task_object():
    # choose_model/extract_task_features take only `prompt: str` -- there is
    # no parameter through which a GoldenTask (and therefore its
    # ground-truth complexity_label/expected_answer) could be passed in.
    import inspect

    choose_model_params = inspect.signature(choose_model).parameters
    assert list(choose_model_params.keys()) == ["prompt", "small_model", "large_model"]
    extract_features_params = inspect.signature(extract_task_features).parameters
    assert list(extract_features_params.keys()) == ["prompt"]


def test_prompt_length_alone_is_not_the_only_decision_rule():
    # A long prompt with none of the strong lexical signals must still
    # route small -- length alone never decides.
    long_but_simple = "What is the capital of Italy? " * 10 + "Answer with only the city name."
    decision = choose_model(long_but_simple, "small", "large")
    assert decision.selected_model == "small"
    features = extract_task_features(long_but_simple)
    assert features["is_long_prompt"] is True


def test_extract_task_features_never_includes_ground_truth_fields():
    features = extract_task_features("What is 2 + 2?")
    assert "complexity_label" not in features
    assert "expected_answer" not in features


def test_fixed_decision_never_matches_any_rule():
    decision = fixed_decision("small-model-id", "small", "always_small")
    assert decision.matched_rules == ()
    assert "Fixed baseline" in decision.routing_reason
