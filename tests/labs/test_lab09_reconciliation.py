from __future__ import annotations

from lab_core.config import ModelPricing
from labs.lab09_batch_processing.datasets import GoldenTask
from labs.lab09_batch_processing.models import (
    RECON_COMPLETED,
    RECON_FAILED,
    RECON_MALFORMED,
    RECON_MISSING,
    RECON_UNKNOWN_ID,
)
from labs.lab09_batch_processing.reconciliation import reconcile_batch_results

PRICING = ModelPricing(input_per_1m_usd=0.075, output_per_1m_usd=0.30, cached_input_per_1m_usd=0.0375, price_verified_date="2026-10-21")


def _task(task_id: str, expected: str = "POSITIVE") -> GoldenTask:
    return GoldenTask(task_id=task_id, category="classification_sentiment", prompt=f"classify {task_id}", expected_answer=expected, evaluation_type="exact_match", metadata={})


def _output_row(custom_id: str, content: str, prompt_tokens=10, completion_tokens=2) -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
                "choices": [{"message": {"content": content}}],
            },
        },
        "error": None,
    }


def test_reconcile_completed_and_correct():
    tasks = [_task("task_001")]
    rows = [_output_row("task_001", "POSITIVE")]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    assert counts.completed_requests == 1
    assert counts.correct_answers == 1
    assert measurements[0].reconciliation_status == RECON_COMPLETED
    assert measurements[0].answer_correct is True
    assert measurements[0].estimated_standard_cost_usd is not None
    assert measurements[0].estimated_batch_cost_usd is not None


def test_reconcile_missing_request_never_fabricated():
    tasks = [_task("task_001"), _task("task_002")]
    rows = [_output_row("task_001", "POSITIVE")]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    assert counts.missing_requests == 1
    missing = next(m for m in measurements if m.task_id == "task_002")
    assert missing.reconciliation_status == RECON_MISSING
    assert missing.prompt_tokens is None
    assert missing.answer_correct is None


def test_reconcile_failed_request_from_error_field():
    tasks = [_task("task_001")]
    rows = [{"custom_id": "task_001", "response": None, "error": {"message": "rate limited"}}]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    assert counts.failed_requests == 1
    assert measurements[0].reconciliation_status == RECON_FAILED
    assert measurements[0].error_message == "rate limited"


def test_reconcile_duplicate_custom_id_counted_not_double_applied():
    tasks = [_task("task_001")]
    rows = [_output_row("task_001", "POSITIVE"), _output_row("task_001", "POSITIVE")]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    assert counts.duplicate_result_ids == 1
    assert counts.completed_requests == 1  # only the first occurrence is applied


def test_reconcile_unknown_custom_id_reported_not_dropped():
    tasks = [_task("task_001")]
    rows = [_output_row("task_001", "POSITIVE"), _output_row("task_999", "NEGATIVE")]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    assert counts.unexpected_result_ids == 1
    unknown = next(m for m in measurements if m.reconciliation_status == RECON_UNKNOWN_ID)
    assert unknown.custom_id == "task_999"


def test_reconcile_malformed_line_reported_not_dropped():
    tasks = [_task("task_001")]
    rows = [_output_row("task_001", "POSITIVE"), {"_malformed_line": "garbage"}]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    malformed = [m for m in measurements if m.reconciliation_status == RECON_MALFORMED]
    assert len(malformed) == 1


def test_reconcile_row_missing_custom_id_treated_as_malformed():
    tasks = [_task("task_001")]
    rows = [{"response": {"body": {}}, "error": None}]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    assert counts.missing_requests == 1  # task_001 itself never arrived
    assert any(m.reconciliation_status == RECON_MALFORMED for m in measurements)


def test_reconcile_missing_usage_does_not_crash_and_leaves_cost_none():
    tasks = [_task("task_001")]
    rows = [{"custom_id": "task_001", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "POSITIVE"}}]}}, "error": None}]
    measurements, counts = reconcile_batch_results("run1", tasks, rows, [], "model-x", PRICING)
    assert measurements[0].prompt_tokens is None
    assert measurements[0].estimated_standard_cost_usd is None
    assert measurements[0].estimated_batch_cost_usd is None


def test_reconcile_error_rows_are_included_alongside_output_rows():
    tasks = [_task("task_001"), _task("task_002")]
    output_rows = [_output_row("task_001", "POSITIVE")]
    error_rows = [{"custom_id": "task_002", "response": None, "error": {"message": "expired"}}]
    measurements, counts = reconcile_batch_results("run1", tasks, output_rows, error_rows, "model-x", PRICING)
    assert counts.completed_requests == 1
    assert counts.failed_requests == 1
