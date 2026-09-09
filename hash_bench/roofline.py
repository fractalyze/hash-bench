# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The two ceilings a hash row is reported against, both measured on the machine
that runs the sweep.

A spec-sheet peak would make the fraction a property of the datasheet rather
than of this machine and this compiler, and the fraction is the whole point: it
says how much of what the hardware can do the arm reached. So both ceilings come
from probes compiled through the same frx and the same plugin as the rows.

- **Memory.** An elementwise scale over a large array — one read and one write
  per element, no reuse — is the streaming ceiling.
- **Arithmetic.** A long chain of multiplies over many independent lanes, in the
  ROW'S dtype. A hash's arithmetic ceiling is a multiply rate in its own field
  or word type; comparing a field multiply count against a `uint32` rate would
  be comparing two different operations.

A `Peaks` is measured ONCE for a backend leg and handed to every arm that runs
on it. Re-measuring per arm would give the arms of one leg different
denominators, so a share and the total it is a share of would come from
different measurements and the three arms would stop being comparable — which
is the only thing a reader wants from them.

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


def dtype_name(dtype: Any) -> str:
    """A dtype's readable spelling (`koalabear_mont`), which is both the key a
    peak is filed under and what its probe text names. `str()` on the class
    gives `<class 'zk_dtypes.koalabear_mont'>`, and a row may hand over either
    the class or an array's dtype, so both are normalized here."""
    import numpy as np

    return str(np.dtype(dtype))


@dataclasses.dataclass(frozen=True)
class MemoryPeak:
    bytes_per_s: float
    probe: str


@dataclasses.dataclass(frozen=True)
class ArithPeak:
    ops_per_s: float
    unit: str
    probe: str


@dataclasses.dataclass(frozen=True)
class Peaks:
    """One leg's ceilings, measured together and reused across its arms.

    `arith` is keyed by `(dtype, unit)` rather than by dtype alone: the unit
    names what a count means, and two units over one dtype are two different
    operations whose rates must not be shared.
    """

    memory: MemoryPeak
    arith: dict[tuple[str, str], ArithPeak] = dataclasses.field(default_factory=dict)

    def arith_for(self, dtype: Any, unit: str) -> ArithPeak | None:
        return self.arith.get((dtype_name(dtype), unit))

    def to_json(self) -> dict[str, Any]:
        """A form that survives the pipe from the probe process to the arm
        workers. The tuple key is flattened, JSON having no tuple keys."""
        return {
            "memory": dataclasses.asdict(self.memory),
            "arith": [
                {"dtype": dtype, **dataclasses.asdict(peak)}
                for (dtype, _unit), peak in self.arith.items()
            ],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Peaks":
        arith = {}
        for entry in payload["arith"]:
            entry = dict(entry)
            dtype = entry.pop("dtype")
            peak = ArithPeak(**entry)
            arith[(dtype, peak.unit)] = peak
        return cls(memory=MemoryPeak(**payload["memory"]), arith=arith)


def memory_peak(method: timing.Method | None = None) -> MemoryPeak:
    """The streaming read+write ceiling."""
    import frx
    import frx.numpy as fnp

    x = fnp.full((_MEMORY_PROBE_ELEMENTS,), 3, dtype=fnp.uint32)
    fn = frx.jit(lambda a: a * fnp.uint32(3))
    m = timing.measure(fn, x, method)
    traffic = 2 * _MEMORY_PROBE_ELEMENTS * 4
    return MemoryPeak(
        bytes_per_s=traffic / (m.ns_per_call * 1e-9),
        probe=(
            f"elementwise uint32 scale over {_MEMORY_PROBE_ELEMENTS} elements, "
            "traffic counted as one read plus one write"
        ),
    )


def arith_peak(dtype: Any, unit: str, method: timing.Method | None = None) -> ArithPeak:
    """The multiply-rate ceiling in `dtype`, reported in `unit`.

    The chain is dependent within a lane and independent across lanes, so the
    rate is throughput-limited rather than latency-limited as long as the lane
    count exceeds the machine's parallelism — which `_ARITH_PROBE_LANES` is
    sized for.
    """
    import frx
    import frx.numpy as fnp

    x = fnp.full((_ARITH_PROBE_LANES,), 3, dtype=dtype)

    def chain(a: Any) -> Any:
        for _ in range(_ARITH_PROBE_CHAIN):
            a = a * a
        return a

    m = timing.measure(frx.jit(chain), x, method)
    ops = _ARITH_PROBE_LANES * _ARITH_PROBE_CHAIN
    return ArithPeak(
        ops_per_s=ops / (m.ns_per_call * 1e-9),
        unit=unit,
        probe=(
            f"{_ARITH_PROBE_CHAIN} chained multiplies over {_ARITH_PROBE_LANES} "
            f"independent {dtype_name(dtype)} lanes"
        ),
    )


def measure_peaks(
    units: list[tuple[Any, str]], method: timing.Method | None = None
) -> Peaks:
    """Every ceiling one leg needs: the streaming one, and an arithmetic one per
    `(dtype, unit)` the leg's rows declare a model in."""
    arith = {}
    for dtype, unit in units:
        key = (dtype_name(dtype), unit)
        if key not in arith:
            arith[key] = arith_peak(dtype, unit, method)
    return Peaks(memory=memory_peak(method), arith=arith)


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
