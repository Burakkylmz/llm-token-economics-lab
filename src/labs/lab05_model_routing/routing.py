"""A deliberately simple, rule-based routing policy.

This is a heuristic on the *prompt text itself* (length + a small keyword
list), not a cheat that reads the golden dataset's ground-truth
`complexity` label -- a real router never has that label available.
"""

from __future__ import annotations

_LONG_PROMPT_CHAR_THRESHOLD = 110
_COMPLEXITY_KEYWORDS = ("function", "python", "code", "write a", "explain")


def choose_model(prompt: str, small_model: str, large_model: str) -> str:
    is_long = len(prompt) > _LONG_PROMPT_CHAR_THRESHOLD
    has_complexity_keyword = any(keyword in prompt.lower() for keyword in _COMPLEXITY_KEYWORDS)
    return large_model if (is_long or has_complexity_keyword) else small_model
