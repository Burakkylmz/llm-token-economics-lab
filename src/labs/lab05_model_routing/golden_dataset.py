"""A tiny, fixed golden dataset with deterministic pass/fail checks.

Quality is evaluated with simple, auditable string checks rather than an
LLM judge, so results are reproducible and don't depend on another model
call. This dataset is intentionally small and illustrative, not a
statistically rigorous benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GoldenExample:
    task_id: str
    complexity: str  # "simple" | "complex" -- ground truth label, not used by the router
    prompt: str
    check: Callable[[str], bool]


GOLDEN_DATASET: tuple[GoldenExample, ...] = (
    GoldenExample(
        task_id="simple_capital",
        complexity="simple",
        prompt="What is the capital of Italy? Answer with only the city name.",
        check=lambda text: "rome" in text.lower(),
    ),
    GoldenExample(
        task_id="simple_arithmetic",
        complexity="simple",
        prompt="What is 12 + 7? Answer with only the number.",
        check=lambda text: "19" in text,
    ),
    GoldenExample(
        task_id="complex_word_problem",
        complexity="complex",
        prompt=(
            "A farmer has 17 sheep. All but 9 of them run away. "
            "How many sheep does the farmer have left? Answer with only the number."
        ),
        check=lambda text: "9" in text,
    ),
    GoldenExample(
        task_id="complex_code_generation",
        complexity="complex",
        prompt=(
            "Write a Python function named is_palindrome(s) that returns True if "
            "s is a palindrome (ignoring case), False otherwise. Output only the code."
        ),
        check=lambda text: "def is_palindrome" in text,
    ),
)


def score(example: GoldenExample, completion_text: str) -> bool:
    return bool(example.check(completion_text))
