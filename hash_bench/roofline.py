# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The two ceilings a hash row is reported against, both measured on the machine
that runs the sweep.

A spec-sheet peak would make the fraction a property of the datasheet rather
than of this machine and this compiler, and the fraction is the whole point: it
says how much of what the hardware can do the arm reached. So both ceilings come
from probes compiled through the same frx and the same plugin as the rows.

- **Memory.** An elementwise scale over a large array — one read and one write
  per element, no reuse — is the streaming ceiling. Measured once per process.
- **Arithmetic.** A long chain of multiplies over many independent lanes, in the
  ROW'S dtype. A hash's arithmetic ceiling is a multiply rate in its own field
  or word type; comparing a field multiply count against a `uint32` rate would
  be comparing two different operations. Measured once per dtype.

A row is reported against whichever ceiling it is nearer, and both fractions
appear so the choice is visible rather than asserted. A row with no arithmetic
model (`registry.OpModel`) gets the memory fraction only.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from hash_bench import timing

# Big enough that the arrays cannot sit in any cache the backend has, so the
# scale probe measures the path a batched hash sweep actually streams over.
_MEMORY_PROBE_ELEMENTS = 1 << 24  # 64 MiB of uint32, 128 MiB of traffic

# Lanes wide enough to fill the machine, chain long enough that the 8 bytes of
# traffic per lane are negligible beside the multiplies over it.
_ARITH_PROBE_LANES = 1 << 20
_ARITH_PROBE_CHAIN = 64


@dataclasses.dataclass(frozen=True)
class MemoryPeak:
    bytes_per_s: float
    probe: str


@dataclasses.dataclass(frozen=True)
class ArithPeak:
    ops_per_s: float
    unit: str
    probe: str


_memory_peak: MemoryPeak | None = None
_arith_peaks: dict[str, ArithPeak] = {}


def memory_peak(method: timing.Method | None = None) -> MemoryPeak:
    """The streaming read+write ceiling, measured once per process."""
    global _memory_peak
    if _memory_peak is not None:
        return _memory_peak
    import frx
    import frx.numpy as fnp

    x = fnp.full((_MEMORY_PROBE_ELEMENTS,), 3, dtype=fnp.uint32)
    fn = frx.jit(lambda a: a * fnp.uint32(3))
    m = timing.measure(fn, x, method)
    traffic = 2 * _MEMORY_PROBE_ELEMENTS * 4
    _memory_peak = MemoryPeak(
        bytes_per_s=traffic / (m.ns_per_call * 1e-9),
        probe=(
            f"elementwise uint32 scale over {_MEMORY_PROBE_ELEMENTS} elements, "
            "traffic counted as one read plus one write"
        ),
    )
    return _memory_peak


def arith_peak(dtype: Any, unit: str, method: timing.Method | None = None) -> ArithPeak:
    """The multiply-rate ceiling in `dtype`, measured once per dtype.

    The chain is dependent within a lane and independent across lanes, so the
    rate is throughput-limited rather than latency-limited as long as the lane
    count exceeds the machine's parallelism — which `_ARITH_PROBE_LANES` is
    sized for.
    """
    key = str(dtype)
    cached = _arith_peaks.get(key)
    if cached is not None:
        return cached
    import frx
    import frx.numpy as fnp

    x = fnp.full((_ARITH_PROBE_LANES,), 3, dtype=dtype)

    def chain(a: Any) -> Any:
        for _ in range(_ARITH_PROBE_CHAIN):
            a = a * a
        return a

    m = timing.measure(frx.jit(chain), x, method)
    ops = _ARITH_PROBE_LANES * _ARITH_PROBE_CHAIN
    peak = ArithPeak(
        ops_per_s=ops / (m.ns_per_call * 1e-9),
        unit=unit,
        probe=(
            f"{_ARITH_PROBE_CHAIN} chained multiplies over {_ARITH_PROBE_LANES} "
            f"independent {dtype} lanes"
        ),
    )
    _arith_peaks[key] = peak
    return peak


def reset_peaks() -> None:
    """Drop the per-process caches. For tests that measure under a changed
    backend or arm inside one process."""
    global _memory_peak
    _memory_peak = None
    _arith_peaks.clear()


def fractions(
    bytes_per_s: float,
    ops_per_s: float | None,
    memory: MemoryPeak,
    arith: ArithPeak | None,
) -> dict[str, Any]:
    """The roofline block of a row: both fractions, and which one binds.

    `bound` names the ceiling the row is nearer, which is the one a reader
    should hold it to. With no arithmetic model there is only one candidate, and
    saying so is the point: it means nobody has written down what this hash
    costs, not that it is memory-bound.
    """
    memory_fraction = bytes_per_s / memory.bytes_per_s
    block: dict[str, Any] = {
        "memory_fraction": memory_fraction,
        "peak_bytes_per_s": memory.bytes_per_s,
        "memory_probe": memory.probe,
        "arith_fraction": None,
        "peak_ops_per_s": None,
        "arith_unit": None,
        "arith_probe": None,
        "bound": "memory",
    }
    if ops_per_s is None or arith is None:
        block["bound"] = "memory (no arithmetic model declared for this hash)"
        return block
    arith_fraction = ops_per_s / arith.ops_per_s
    block.update(
        arith_fraction=arith_fraction,
        peak_ops_per_s=arith.ops_per_s,
        arith_unit=arith.unit,
        arith_probe=arith.probe,
        bound="arithmetic" if arith_fraction >= memory_fraction else "memory",
    )
    return block
