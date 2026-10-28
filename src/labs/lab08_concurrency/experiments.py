"""Lab08's shared workload: eight distinct, deterministic arithmetic
tasks, used identically (same prompts, same expected answers, same task
count and order) across every concurrency level -- the only independent
variable this lab's controlled comparison varies is `max_concurrent`
itself (Part D).
"""

from __future__ import annotations

from dataclasses import dataclass

# Deliberately distinct operand pairs (not the old "i+i" repeated
# pattern) so no two tasks share an expected answer -- purely a clarity
# improvement, not a correctness requirement.
_OPERAND_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (10, 11), (13, 14), (20, 21),
)


@dataclass(frozen=True)
class ConcurrencyTask:
    task_id: str
    prompt: str
    expected_answer: str


def _build_workload() -> tuple[ConcurrencyTask, ...]:
    tasks = []
    for i, (a, b) in enumerate(_OPERAND_PAIRS, start=1):
        tasks.append(
            ConcurrencyTask(
                task_id=f"task_{i:03d}",
                prompt=f"What is {a} plus {b}? Answer with only the number.",
                expected_answer=str(a + b),
            )
        )
    return tuple(tasks)


WORKLOAD: tuple[ConcurrencyTask, ...] = _build_workload()
WORKLOAD_BY_ID: dict[str, ConcurrencyTask] = {t.task_id: t for t in WORKLOAD}

# Pilot uses a stable 4-task subset; default/extended use the full 8.
PILOT_TASK_IDS: tuple[str, ...] = tuple(t.task_id for t in WORKLOAD[:4])
DEFAULT_TASK_IDS: tuple[str, ...] = tuple(t.task_id for t in WORKLOAD)
