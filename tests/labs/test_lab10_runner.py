from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from labs.lab10_conversation_summarization.datasets import load_conversation, load_golden_questions
from labs.lab10_conversation_summarization.matrix import build_plan
from labs.lab10_conversation_summarization.models import CALL_PURPOSE_FOLLOWUP_ANSWER, CALL_PURPOSE_SUMMARY_GENERATION
from labs.lab10_conversation_summarization.runner import SummaryNotYetGenerated, execute_plan

DATASET_ID = "customer_support_001"

VALID_SUMMARY = json.dumps({
    "overview": "o", "facts": [{"name": "order_id", "value": "ORD-74291", "status": "confirmed"}],
    "decisions": [{"topic": "delivery time preference", "current_value": "afternoon", "superseded_value": "morning"}],
    "open_issues": [], "timeline": [],
})


def _usage(pt=100, ct=20, rt=5):
    return SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct, total_time=0.1,
                            prompt_tokens_details=None, completion_tokens_details=SimpleNamespace(reasoning_tokens=rt))


def _resp(content, finish_reason="stop", **kwargs):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=_usage(**kwargs), system_fingerprint=None)


class _FakeCompletions:
    def __init__(self, answer_for_question=None, fail_on_calls=frozenset()):
        self.calls: list[dict] = []
        self._n = 0
        self._answer_for_question = answer_for_question or {}
        self._fail_on_calls = fail_on_calls

    def create(self, **kwargs):
        self._n += 1
        self.calls.append(dict(kwargs))
        if self._n in self._fail_on_calls:
            raise ConnectionError("simulated failure")
        user_content = kwargs["messages"][-1]["content"]
        if "Question:\n" in user_content:
            question_text = user_content.split("Question:\n")[1].split("\n")[0]
            answer = self._answer_for_question.get(question_text, "ORD-74291")
            return _resp(json.dumps({"answer": answer}))
        return _resp(VALID_SUMMARY)


def _fake_client(**kwargs):
    completions = _FakeCompletions(**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _questions_dict():
    return {DATASET_ID: {q.question_id: q for q in load_golden_questions(DATASET_ID)}}


def test_execute_plan_generates_exactly_one_summary_per_effort():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    plan = build_plan("pilot")
    summaries, followups, traces, generated = execute_plan(
        client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan,
    )
    assert len(summaries) == 2
    assert (DATASET_ID, "summarization_low") in generated
    assert (DATASET_ID, "summarization_medium") in generated


def test_execute_plan_reuses_the_same_summary_across_all_its_followups():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    plan = build_plan("default")
    execute_plan(client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan)
    # 2 summary-generation calls + 4 baseline + 4 low + 4 medium = 14 total calls,
    # never more than one summary call per effort despite 4 follow-up questions each.
    summary_gen_calls = sum(1 for c in completions.calls if "Question:\n" not in c["messages"][-1]["content"])
    assert summary_gen_calls == 2


def test_execute_plan_baseline_uses_full_transcript_as_context():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    plan = [c for c in build_plan("pilot") if c.condition_id == "full_history_baseline" and c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER][:1]
    execute_plan(client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan)
    sent = completions.calls[0]["messages"][-1]["content"]
    assert "ORD-74291" in sent  # the raw transcript, not a summary, is embedded verbatim


def test_execute_plan_summarization_condition_uses_rendered_summary_as_context():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    plan = build_plan("pilot")
    execute_plan(client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan)
    low_followup_calls = [
        c for c in completions.calls
        if "Question:\n" in c["messages"][-1]["content"] and "Overview:" in c["messages"][-1]["content"]
    ]
    assert len(low_followup_calls) >= 1


def test_execute_plan_never_leaks_golden_expected_answer_into_any_request():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    plan = build_plan("pilot")
    execute_plan(client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan)
    for call in completions.calls:
        for message in call["messages"]:
            assert "NOT_STATED\n" not in message["content"] or "Expected Answer Format" in message["content"]
            # The literal golden identifier answers never appear tagged as "expected".
            assert "expected_answer" not in message["content"].lower()


def test_execute_plan_evaluates_followups_against_expected_answers():
    client, completions = _fake_client(answer_for_question={"What is the customer's order ID?": "ORD-74291"})
    conversation = load_conversation(DATASET_ID)
    plan = [c for c in build_plan("pilot") if c.call_purpose != CALL_PURPOSE_SUMMARY_GENERATION or True]
    summaries, followups, traces, generated = execute_plan(
        client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), build_plan("pilot"),
    )
    order_id_followups = [f for f in followups if f.question_id == "q1_order_id"]
    assert all(f.evaluation_status == "CORRECT" for f in order_id_followups)


def test_execute_plan_preloaded_summary_skips_regeneration():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    preloaded = {(DATASET_ID, "summarization_low"): json.loads(VALID_SUMMARY), (DATASET_ID, "summarization_medium"): json.loads(VALID_SUMMARY)}
    plan = [c for c in build_plan("pilot") if c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER]
    summaries, followups, traces, generated = execute_plan(
        client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan, preloaded_summaries=preloaded,
    )
    assert summaries == []  # no summary_generation calls were ever dispatched
    summary_gen_calls = sum(1 for c in completions.calls if "Question:\n" not in c["messages"][-1]["content"])
    assert summary_gen_calls == 0
    assert len(followups) == 6  # pilot: baseline + low + medium, 2 questions each


def test_execute_plan_raises_clearly_when_summary_missing_for_summarization_followup():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    plan = [c for c in build_plan("pilot") if c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER]
    with pytest.raises(SummaryNotYetGenerated):
        execute_plan(client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan)


def test_execute_plan_one_followup_failure_does_not_lose_other_results():
    client, completions = _fake_client(fail_on_calls=frozenset({3}))  # 3rd call overall (1st followup, baseline q1)
    conversation = load_conversation(DATASET_ID)
    plan = build_plan("pilot")
    summaries, followups, traces, generated = execute_plan(
        client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan,
    )
    assert len(followups) == 6
    statuses = [f.request_status for f in followups]
    assert statuses.count("UNKNOWN_ERROR") == 1
    assert statuses.count("SUCCESS") == 5


def test_execute_plan_traces_never_include_golden_expected_answer_as_a_labeled_field():
    client, completions = _fake_client()
    conversation = load_conversation(DATASET_ID)
    plan = build_plan("pilot")
    summaries, followups, traces, generated = execute_plan(
        client, "m", 1536, "m", "low", 256, None, "run-1", {DATASET_ID: conversation}, _questions_dict(), plan,
    )
    for t in traces:
        assert "expected_answer" not in str(t.request_parameters).lower()
        for m in t.request_messages:
            assert "expected_answer" not in m["content"].lower()
