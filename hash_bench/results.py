# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The row: one measured (hash, batch, backend, arm), and how it is written.

JSON Lines, one row per line, because rows arrive from several worker processes
and a line-appended file needs no merge step. Every row is self-contained —
machine, revisions and method included — so a single line can be quoted without
its file.

`results/README.md` documents each field for a reader; this module is where the
field set is defined, and the two are meant to be read together.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, TextIO

SCHEMA_VERSION = 2


@dataclasses.dataclass(frozen=True)
class Row:
    hash: str
    batch: int
    backend: str
    arm_requested: str
    # What the compiled module showed. Differs from the request whenever the pin
    # or the backend does not offer the requested arm, and the timing then
    # belongs to THIS arm, not to the one asked for.
    arm_observed: str
    kernels: tuple[str, ...]
    # Wall time to lower and compile this call once. It belongs in the row
    # because compile cost is an axis the arms differ on — a declined region
    # inlines its whole round schedule and hands the backend the module the
    # emitter existed to avoid — so an arm that wins at run time while costing
    # more to compile is a different trade, not a free one.
    #
    # Null on a reference arm, which has no run-time compile: the flags that
    # built it are in `reference.copts` and were paid at build time. Null rather
    # than zero, because zero would read as a compile that cost nothing.
    compile_ns: int | None
    ns_per_hash: float
    ns_per_hash_min: float
    spread: float
    hashes_per_s: float
    bytes_moved: int
    bytes_per_s: float
    ops_per_hash: int | None
    ops_per_s: float | None
    ops_unit: str | None
    ops_note: str | None
    roofline: dict[str, Any]
    method: dict[str, Any]
    machine: dict[str, Any]
    # The pinned external implementation this row measured — its revision, the
    # upstream implementation selected, and the flags it was compiled with —
    # or None on a hash-frx arm, whose revisions ride in `machine` instead. A
    # reference number is not quotable without them, and the row is the unit
    # that gets quoted.
    reference: dict[str, Any] | None = None

    @property
    def as_requested(self) -> bool:
        return self.arm_observed == self.arm_requested

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["kernels"] = list(self.kernels)
        d["schema"] = SCHEMA_VERSION
        d["as_requested"] = self.as_requested
        return d


def write(row: Row, stream: TextIO) -> None:
    stream.write(json.dumps(row.to_json()) + "\n")
    stream.flush()


def read(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


_COLUMNS = (
    ("hash", 24),
    ("batch", 8),
    ("backend", 10),
    ("arm", 20),
    ("ns/hash", 12),
    ("GB/s", 9),
    ("compile s", 10),
    ("roofline", 22),
)


def _arm_cell(row: dict[str, Any]) -> str:
    if row["as_requested"]:
        return row["arm_observed"]
    return f"{row['arm_observed']} (<-{row['arm_requested']})"


def _compile_cell(row: dict[str, Any]) -> str:
    """Seconds, or a dash where the arm has nothing to compile at run time."""
    compile_ns = row["compile_ns"]
    return "-" if compile_ns is None else f"{compile_ns / 1e9:.2f}"


def _roofline_cell(row: dict[str, Any]) -> str:
    rl = row["roofline"]
    bound = rl["bound"].split(" ")[0]
    fraction = rl["arith_fraction"] if bound == "arithmetic" else rl["memory_fraction"]
    return f"{fraction * 100:.1f}% of {bound}"


def table(rows: list[dict[str, Any]]) -> str:
    """A fixed-width rendering of the fields a reader scans first. The file
    remains the record; this is for looking at one."""
    header = "  ".join(name.ljust(w) for name, w in _COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = (
            row["hash"],
            str(row["batch"]),
            row["backend"],
            _arm_cell(row),
            f"{row['ns_per_hash']:.3f}",
            f"{row['bytes_per_s'] / 1e9:.2f}",
            _compile_cell(row),
            _roofline_cell(row),
        )
        lines.append(
            "  ".join(c.ljust(w) for c, (_, w) in zip(cells, _COLUMNS, strict=True))
        )
    return "\n".join(lines)
