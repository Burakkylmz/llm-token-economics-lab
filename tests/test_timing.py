from __future__ import annotations

import time

from lab_core.timing import StopwatchTimer


def test_stopwatch_measures_positive_elapsed_time():
    with StopwatchTimer() as timer:
        time.sleep(0.01)
    assert timer.result is not None
    assert timer.result.elapsed_ms >= 10


def test_stopwatch_result_is_none_before_exit():
    timer = StopwatchTimer()
    assert timer.result is None
