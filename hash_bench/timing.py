# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The measurement method, recorded with every number it produces.

One rep dispatches `iters` independent calls on the same device input and blocks
once, at the end. Blocking per call would measure the host round trip instead of
the kernel on any backend with an asynchronous queue; blocking once amortizes it
over `iters`. The calls are independent rather than chained (`x = fn(x)`)
because a chain measures latency under a serial dependency, and what a consumer
batches over is throughput.

`iters` is calibrated per call so a rep spans `target_rep_ns`, which is what
keeps a nanosecond-scale CPU permutation and a millisecond-scale GPU sweep on
the same clock resolution. The reported number is the MEDIAN over reps of the
mean over iters: the mean inside a rep because the reps are what carry the
run-to-run noise, and the median across them because a preempted rep is an
outlier rather than a distribution.

Every field below rides in the row, so a number can be re-derived rather than
trusted.
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from collections.abc import Callable
from typing import Any

DEFAULT_WARMUP = 3
DEFAULT_REPS = 7
DEFAULT_TARGET_REP_NS = 20_000_000  # 20 ms
MAX_ITERS = 100_000

ASYNC_DISPATCH = "iters independent calls per rep, blocked once at the end"
SYNC_DISPATCH = "iters independent synchronous calls per rep; nothing to block on"


@dataclasses.dataclass(frozen=True)
class Method:
    """What was done to produce a number."""

    warmup: int = DEFAULT_WARMUP
    reps: int = DEFAULT_REPS
    target_rep_ns: int = DEFAULT_TARGET_REP_NS
    iters: int = 0  # filled in by calibration
    statistic: str = "median over reps of the mean over iters"
    dispatch: str = ASYNC_DISPATCH
    timer: str = "time.perf_counter_ns"

    def with_iters(self, iters: int) -> "Method":
        return dataclasses.replace(self, iters=iters)

    def synchronous(self) -> "Method":
        """The same method against a call that returns complete. Every field a
        comparison rests on — warmup, reps, the calibration target, the
        statistic, the timer — is unchanged; only the account of what the block
        at the end of a rep waited for differs, because for a native reference
        it waited for nothing."""
        return dataclasses.replace(self, dispatch=SYNC_DISPATCH)


@dataclasses.dataclass(frozen=True)
class Measurement:
    ns_per_call: float
    ns_per_call_min: float
    # (max - min) / median across reps. A row whose spread is large is a row
    # whose median is not worth quoting, and the reader can see that from the
    # row instead of from the raw reps.
    spread: float
    method: Method


Blocker = Callable[[Any], None]


def block_frx(out: Any) -> None:
    """Wait for an frx call's result. The default, and what every backend leg
    with an asynchronous queue needs."""
    import frx

    frx.block_until_ready(out)


def block_none(_out: Any) -> None:
    """Wait for nothing. A native reference call has already returned its
    result, so there is no queue to drain; saying so explicitly is what keeps
    `frx` out of a reference worker's timed region."""


def _one_rep(fn: Callable[[Any], Any], x: Any, iters: int, block: Blocker) -> int:
    start = time.perf_counter_ns()
    out = None
    for _ in range(iters):
        out = fn(x)
    block(out)
    return time.perf_counter_ns() - start


def calibrate(
    fn: Callable[[Any], Any],
    x: Any,
    target_rep_ns: int,
    block: Blocker = block_frx,
) -> int:
    """How many calls make one rep span `target_rep_ns`, doubling from one until
    the measured span is long enough to extrapolate from."""
    iters = 1
    while iters < MAX_ITERS:
        elapsed = _one_rep(fn, x, iters, block)
        if elapsed >= target_rep_ns // 4:
            return max(1, min(MAX_ITERS, round(iters * target_rep_ns / elapsed)))
        iters *= 2
    return MAX_ITERS


def measure(
    fn: Callable[[Any], Any],
    x: Any,
    method: Method | None = None,
    block: Blocker = block_frx,
) -> Measurement:
    """Time `fn(x)` under `method`, returning nanoseconds per call."""
    method = method or Method()
    for _ in range(method.warmup):
        block(fn(x))
    iters = method.iters or calibrate(fn, x, method.target_rep_ns, block)
    per_call = [_one_rep(fn, x, iters, block) / iters for _ in range(method.reps)]
    median = statistics.median(per_call)
    return Measurement(
        ns_per_call=median,
        ns_per_call_min=min(per_call),
        spread=(max(per_call) - min(per_call)) / median if median else 0.0,
        method=method.with_iters(iters),
    )
