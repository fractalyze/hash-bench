# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The three arms a hash's one-call unit can lower through, and how to tell
which one actually ran.

An arm is REQUESTED by disabling compiler passes before the plugin loads, and
OBSERVED by reading the compiled module. Both go in the row, because the request
is not evidence: a marker no pass claims inlines silently and still computes the
right bytes, which is the exact failure this harness exists to make visible.

Declining happens in the plugin, not in the frontend. hash-frx has a routing
gate (`hash_frx.fusion.routing`) that decides whether a family puts its own
marker name on the wire, but forcing it makes some families emit a DIFFERENT
marker form rather than no marker — SHA-256 falls back from its whole-message
region to its blocks region, which is a second dedicated kernel and a different
computation. Disabling the recognizer passes leaves the emitted module alone and
takes away only the emitter, which is what a pin without that emitter does.

Both knobs are `--xla_disable_hlo_passes` entries; the constants below say what
each takes away.

`RECOGNIZER_PASSES` is a list of names and so can go stale: a recognizer added
to the plugin and not added here would keep claiming its marker under the
`generic` and `declined` arms. That failure is loud rather than silent — the row
observes `routed` against a `generic` request and says so — and
`sweep_test.DeclineTest` holds every registry row to actually declining, which
is where a new recognizer gets caught.

Which arms a (hash, backend) pair can reach is a property of the pin, not of
this file: a hash-frx family names the backends each of its emitters was written
for, and a request for an arm the pair cannot reach observes whatever did run.
The row records that rather than relabelling the number.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Sequence

# The passes that turn a hash marker into a kernel. Enumerated by compiling each
# registry row under `--xla_dump_hlo_pass_re=.*` and reading which passes
# rewrote the module; `zorch-fused-region-rewriter` covers the markers routed
# through the shared region recognizer (Poseidon2, Keccak-f, the Keccak sponge,
# SHA-256), and BLAKE3 carries its own.
RECOGNIZER_PASSES: tuple[str, ...] = (
    "zorch-fused-region-rewriter",
    "blake3-rewriter",
)

# Converts a small-state while loop with a known trip count into one custom
# fusion. It is what earns a declined region its single kernel on a backend that
# runs the pass, and so is the only thing separating `generic` from `declined`.
LOOP_CONVERSION_PASS = "while-to-for-converter"


def _disable(*passes: str) -> tuple[str, ...]:
    return (f"--xla_disable_hlo_passes={','.join(passes)}",)


class Arm(enum.Enum):
    """The lowering a run asks for."""

    ROUTED = "routed"  # the marker reaches the plugin's dedicated emitter
    GENERIC = "generic"  # no emitter; the loop conversion makes one kernel
    DECLINED = "declined"  # no emitter and no loop conversion; the decomposition

    @property
    def xla_flags(self) -> tuple[str, ...]:
        """The `XLA_FLAGS` tokens this arm adds. They must be in the environment
        before the plugin loads: the flags are parsed once, at load."""
        if self is Arm.ROUTED:
            return ()
        if self is Arm.GENERIC:
            return _disable(*RECOGNIZER_PASSES)
        return _disable(*RECOGNIZER_PASSES, LOOP_CONVERSION_PASS)


class Lowering(enum.Enum):
    """What the compiled module shows actually lowered.

    Separate from `Arm` because a row carries both, and confusing a request with
    an observation is the mistake this harness is built to avoid.
    """

    DEDICATED = "routed"
    STATIC_WHILE = "generic"
    DECOMPOSED = "declined"
    MIXED = "mixed"  # a dedicated kernel AND a static_while in one module

    @property
    def arm_name(self) -> str:
        return self.value


# `%fusion = f32[..] fusion(%a, %b), kind=kCustom, calls=%some_computation, ...`
# The called computation names what claimed the region, which is the only place
# the difference between a dedicated kernel and the generic loop conversion
# survives into the compiled module.
_CUSTOM_FUSION = re.compile(r"kind=kCustom.*?calls=%([A-Za-z0-9_.-]+)")

# The loop conversion's computation, with the usual `.1`, `.2` suffixes a module
# holding several of them carries.
_STATIC_WHILE = "static_while_fusion"


def classify(callees: Sequence[str]) -> Lowering:
    """The arm a module's kCustom callees show.

    No kCustom at all is the decomposition: every region inlined and lowered as
    ordinary fusions. Otherwise the callee names say whether an emitter claimed
    it or the loop conversion did.
    """
    if not callees:
        return Lowering.DECOMPOSED
    converted = [c for c in callees if c.split(".")[0] == _STATIC_WHILE]
    if not converted:
        return Lowering.DEDICATED
    if len(converted) == len(callees):
        return Lowering.STATIC_WHILE
    return Lowering.MIXED


def observe(module_text: str) -> tuple[Lowering, tuple[str, ...]]:
    """The observed arm and the kCustom callee names it was read from, in
    module order."""
    callees = tuple(m.group(1) for m in _CUSTOM_FUSION.finditer(module_text))
    return classify(callees), callees
