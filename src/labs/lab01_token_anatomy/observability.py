"""Analysis helpers for Lab01's completion-token-variability observability.

These functions turn raw, per-call measurements (already normalized by
lab_core.groq_client) into the derived metrics and summary statistics Lab01
reports. Nothing here calls the Groq API or knows about cost -- that split
(raw extraction / cost / variability analysis) mirrors the project's existing
layering, just extended for this lab's new questions.
"""

from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence


def compute_non_reasoning_completion_tokens(
    completion_tokens: Optional[int], reasoning_tokens: Optional[int]
) -> Optional[int]:
    """completion_tokens minus reasoning_tokens, only when both are known.

    Not named `visible_output_tokens`: completion tokens outside the
    reasoning span are not guaranteed to equal the token count of the final
    answer text a user actually sees (see `final_answer_char_count` for a
    separate, character-based diagnostic of that).

    Returns None -- rather than a fabricated or negative number -- when
    either figure is missing, or when the provider's own figures are
    inconsistent (reasoning_tokens negative, or larger than completion_tokens).
    An inconsistency also raises a warning so it stays visible instead of
    being silently absorbed into the data.
    """
    if completion_tokens is None or reasoning_tokens is None:
        return None

    non_reasoning = completion_tokens - reasoning_tokens
    if reasoning_tokens < 0 or non_reasoning < 0:
        warnings.warn(
            "Inconsistent token accounting from provider: "
            f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens}. "
            "non_reasoning_completion_tokens reported as None rather than a negative number.",
            stacklevel=2,
        )
        return None
    return non_reasoning


@dataclass
class VariabilityStats:
    """Summary statistics for one metric across a set of repeated calls.

    `standard_deviation` is the SAMPLE standard deviation (statistics.stdev;
    Bessel's correction, n-1 denominator): each repetition is treated as one
    sample drawn from the (unobserved) full distribution of completions the
    model could produce for that prompt, not as the entire population of
    interest. It is None when fewer than 2 measurements are available, since
    sample standard deviation is undefined below n=2.

    A handful of repetitions (Lab01's default is 5) is NOT a statistically
    robust sample -- these numbers are descriptive of this one run, not proof
    of a general distribution.
    """

    count_available: int
    count_total: int
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    standard_deviation: Optional[float] = None
    range: Optional[float] = None


def summarize_values(values: Sequence[Optional[float]]) -> VariabilityStats:
    """Build a VariabilityStats from `values`, skipping (not zeroing) Nones.

    A None in `values` means that repetition did not report the metric at all
    (e.g. no reasoning_tokens on that call) -- it is excluded from the
    arithmetic rather than treated as 0, and `count_available` records how
    many usable measurements actually went into the statistics.
    """
    available = [float(v) for v in values if v is not None]
    count_total = len(values)
    count_available = len(available)

    if count_available == 0:
        return VariabilityStats(count_available=0, count_total=count_total)

    minimum = min(available)
    maximum = max(available)
    stats = VariabilityStats(
        count_available=count_available,
        count_total=count_total,
        minimum=minimum,
        maximum=maximum,
        mean=statistics.mean(available),
        range=maximum - minimum,
    )
    if count_available >= 2:
        stats.standard_deviation = statistics.stdev(available)
    return stats


def tally_finish_reasons(finish_reasons: Sequence[Optional[str]]) -> dict[str, int]:
    """Count occurrences of each finish_reason, grouping missing values under 'unavailable'."""
    tally: dict[str, int] = {}
    for reason in finish_reasons:
        key = reason if reason is not None else "unavailable"
        tally[key] = tally.get(key, 0) + 1
    return tally
