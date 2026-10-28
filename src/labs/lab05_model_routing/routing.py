"""A deterministic, rule-based, explainable routing policy -- zero
additional LLM calls (no embedding model, no ML classifier, no LLM judge;
see this lab's design notes for why those are explicitly out of scope for
now).

The router reads ONLY the task's prompt TEXT. It never reads
`GoldenTask.complexity_label` (ground-truth metadata that would never be
available at real deployment time), never reads `expected_answer`, and
never reads any evaluation result -- see `extract_task_features` below,
which is the router's only window onto a task.

Prompt character length is used only as a documented, non-decisive
feature (`is_long_prompt`): it is recorded on every decision for
transparency, but it never by itself causes a large-model selection --
only an explicit lexical signal (multi-step reasoning language,
conditional+constraint language, code-generation language, or
structured-output-generation language) does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

POLICY_VERSION = "v2.0-rule-based"

# Informational only (see module docstring) -- never a sole decision rule.
LONG_PROMPT_CHAR_THRESHOLD = 180

MULTI_STEP_KEYWORDS = ("then", "after that", "in total", "both days", "over the two days", "step by step")
CONDITIONAL_KEYWORDS = ("if ", "unless", "only if", "otherwise", "qualifies", "qualify", "eligible")
CONSTRAINT_KEYWORDS = (" and ", "all of the following", "at least", "below", "at most", "no more than")
CODE_KEYWORDS = ("function", "python", "def ", "write a function", "write a python")
STRUCTURED_OUTPUT_KEYWORDS = ("json object", "respond with only the json", "json schema")


def extract_task_features(prompt: str) -> dict:
    """The complete set of signals the router is allowed to use -- built
    from `prompt` text alone. Every field here is safe to log/display:
    none of it is the expected answer or a ground-truth label."""
    lowered = prompt.lower()
    return {
        "char_length": len(prompt),
        "word_count": len(prompt.split()),
        "numeric_value_count": len(re.findall(r"\d+(?:\.\d+)?", prompt)),
        "has_multi_step_indicator": any(k in lowered for k in MULTI_STEP_KEYWORDS),
        "has_conditional_indicator": any(k in lowered for k in CONDITIONAL_KEYWORDS),
        "has_constraint_language": any(k in lowered for k in CONSTRAINT_KEYWORDS),
        "has_code_indicator": any(k in lowered for k in CODE_KEYWORDS),
        "has_structured_output_indicator": any(k in lowered for k in STRUCTURED_OUTPUT_KEYWORDS),
        "is_long_prompt": len(prompt) > LONG_PROMPT_CHAR_THRESHOLD,
    }


@dataclass(frozen=True)
class RoutingDecision:
    selected_model: str
    model_tier: str  # "small" | "large"
    routing_reason: str
    matched_rules: tuple[str, ...]
    task_features: dict
    policy_version: str


def choose_model(prompt: str, small_model: str, large_model: str) -> RoutingDecision:
    """Deterministic: the same prompt text always produces the same
    decision (no randomness, no external state)."""
    features = extract_task_features(prompt)
    matched: list[str] = []

    if features["has_multi_step_indicator"]:
        matched.append("multi_step_reasoning_indicators")
    if features["has_conditional_indicator"] and features["has_constraint_language"]:
        matched.append("conditional_constraints")
    if features["has_code_indicator"]:
        matched.append("code_generation_indicator")
    if features["has_structured_output_indicator"]:
        matched.append("structured_output_indicator")

    if matched:
        selected_model, tier = large_model, "large"
        reason = "Matched rule(s): " + ", ".join(matched)
    else:
        selected_model, tier = small_model, "small"
        reason = "No large-model routing rule matched"
        if features["is_long_prompt"]:
            reason += " (prompt length alone is not a routing signal)"

    return RoutingDecision(
        selected_model=selected_model,
        model_tier=tier,
        routing_reason=reason,
        matched_rules=tuple(matched),
        task_features=features,
        policy_version=POLICY_VERSION,
    )


def fixed_decision(model: str, tier: str, strategy_label: str) -> RoutingDecision:
    """A non-routing "decision" for the two fixed baselines (always_small/
    always_large) -- expressed as the same RoutingDecision shape so every
    strategy's measurements share one schema, with an honest reason that
    makes clear no routing logic ran."""
    return RoutingDecision(
        selected_model=model,
        model_tier=tier,
        routing_reason=f"Fixed baseline: {strategy_label} strategy (no routing policy evaluated)",
        matched_rules=(),
        task_features={},
        policy_version="baseline-fixed",
    )
