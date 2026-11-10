"""Stable shared-policy imports and backward-compatible Lab01 aliases."""

from lab_core.budget import Budget, conservative_per_call_cost_ceiling
from lab_core.operational_retry import call_with_operational_retry
from labs.lab01_token_anatomy import factorial_runner


def test_lab01_legacy_imports_are_shared_core_symbols():
    assert factorial_runner.Budget is Budget
    assert factorial_runner.conservative_per_call_cost_ceiling is conservative_per_call_cost_ceiling
    assert factorial_runner.call_with_operational_retry is call_with_operational_retry


def test_shared_budget_blocks_requests_without_mutating_accounting():
    budget = Budget(max_api_calls=1, max_estimated_cost_usd=0.05, per_call_cost_ceiling=0.01)
    assert budget.can_make_another_call() is None
    budget.record_call(0.002)
    assert budget.can_make_another_call() is not None
    assert budget.calls_made == 1
    assert budget.accumulated_estimated_cost_usd == 0.002
