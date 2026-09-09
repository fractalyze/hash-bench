# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The table of hashes the sweep measures — one row per hash.

A row names the hash and how to build it; everything else the harness needs is
derived from the object the row builds. Bytes moved come from the arrays a call
actually takes and returns, so no row states a traffic constant. An arithmetic
model is optional and, where a row supplies one, is computed from the
primitive's own parameters rather than written down: `poseidon2_field_mults`
reads the round counts and `alpha` off the params, so a re-parameterized
Poseidon2 re-derives its own op count instead of inheriting a stale literal.

Adding a hash is one `_ROWS` entry.

A row builds its primitive when the sweep asks for a call, not at import: this
module is imported by the orchestrator, which never starts a backend, and
building a hash-frx primitive reads the default backend.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from typing import Any


@dataclasses.dataclass(frozen=True)
class OpModel:
    """A declared count of the dominant arithmetic operation in one hash.

    `unit` names what is counted, because the arithmetic roofline compares it
    against a peak measured in the SAME unit — field multiplies for an algebraic
    permutation, machine-word operations for a bit-oriented one. A row with no
    model reports no arithmetic roofline; it never gets one in another row's
    unit.
    """

    count: int
    unit: str
    note: str


@dataclasses.dataclass(frozen=True)
class Call:
    """One measurable call: the function, its device input, and what one
    invocation costs in traffic and (where modelled) arithmetic."""

    fn: Callable[[Any], Any]
    x: Any
    hashes: int
    bytes_moved: int
    ops: OpModel | None


def _nbytes(arr: Any) -> int:
    return int(arr.size) * int(arr.dtype.itemsize)


def _binary_chain_mults(exponent: int) -> int:
    """Multiplies in the square-and-multiply chain for `x**exponent`.

    Exact for every alpha a Poseidon2 set uses (3, 5, 7): the binary chain and
    the shortest addition chain agree below 15. Stated as the binary chain
    rather than as "the" cost because it is an upper bound in general.
    """
    if exponent < 1:
        raise ValueError(f"exponent must be positive, got {exponent}")
    return (exponent.bit_length() - 1) + (bin(exponent).count("1") - 1)


def poseidon2_field_mults(perm: Any) -> OpModel:
    """Field multiplications in one Poseidon2 permutation, off its params.

    Counts the S-boxes and the internal diagonal. The external layer and the
    internal round's `J`-sum are additions and doublings over the field's
    additive group, so they do not enter a multiply count — which is why the
    unit is named rather than left as "ops".
    """
    p = perm._p
    sbox = _binary_chain_mults(int(p.alpha))
    # `external_rounds` counts one HALF: the schedule runs that many with the
    # initial constants and that many with the terminal ones.
    external = 2 * int(p.external_rounds) * int(p.width) * sbox
    internal = int(p.internal_rounds) * (sbox + int(p.width))
    return OpModel(
        count=external + internal,
        unit="field_mul",
        note=(
            f"{2 * int(p.external_rounds)} external rounds x {p.width} S-boxes "
            f"+ {p.internal_rounds} internal rounds x (1 S-box + {p.width} "
            f"diagonal), S-box = x^{p.alpha} as {sbox} multiplies"
        ),
    )


@dataclasses.dataclass(frozen=True)
class HashSpec:
    """One row of the table."""

    name: str
    kind: str  # "permutation" | "byte_hash"
    build: Callable[[], Any]
    # Byte hashes only: the message length one digest consumes. A hash's cost
    # per byte is not flat (padding, block count), so the length is part of the
    # row's identity and appears in the output.
    message_bytes: int = 0
    op_model: Callable[[Any], OpModel] | None = None

    def call(self, batch: int) -> Call:
        """Build the primitive and shape one batched call over it."""
        import frx
        import frx.numpy as fnp
        import numpy as np

        unit = self.build()
        # A counting pattern rather than zeros, on both kinds: a hash that
        # special-cased an all-zero input would measure a path no consumer runs.
        if self.kind == "permutation":
            x = fnp.arange(batch * unit.width, dtype=unit.dtype).reshape(
                batch, unit.width
            )
            fn = frx.jit(frx.vmap(unit.permute))
        elif self.kind == "byte_hash":
            raw = np.arange(batch * self.message_bytes, dtype=np.uint8)
            x = frx.device_put(raw.reshape(batch, self.message_bytes))
            fn = frx.jit(unit.digest)
        else:
            raise ValueError(f"unknown row kind {self.kind!r}")

        out = frx.eval_shape(fn, x)
        ops = self.op_model(unit) if self.op_model is not None else None
        return Call(
            fn=fn,
            x=x,
            hashes=batch,
            bytes_moved=_nbytes(x) + _nbytes(out),
            ops=ops,
        )


def _poseidon2_koalabear16() -> Any:
    import hash_frx

    # Built from the params rather than taken as the `hash_frx.KoalaBear16`
    # convenience, so the row names the parameter set it measures and a set with
    # no shipped singleton is added the same way.
    return hash_frx.Poseidon2(hash_frx.KOALABEAR16_PARAMS)


def _keccak_f1600() -> Any:
    from hash_frx.keccak.permutation import KeccakF1600

    return KeccakF1600()


def _sha3_256() -> Any:
    import hash_frx

    return hash_frx.Sha3_256()


def _sha256() -> Any:
    import hash_frx

    return hash_frx.Sha256()


def _blake3() -> Any:
    import hash_frx

    return hash_frx.Blake3()


# One message length for every byte hash, so a row-to-row comparison is not
# reading a length difference. 1 KiB spans several blocks of each family
# (64 B for SHA-256, 136 B for SHA3-256, 1024 B = one BLAKE3 chunk).
_MESSAGE_BYTES = 1024

_ROWS: tuple[HashSpec, ...] = (
    HashSpec(
        name="poseidon2-koalabear16",
        kind="permutation",
        build=_poseidon2_koalabear16,
        op_model=poseidon2_field_mults,
    ),
    HashSpec(name="keccak-f1600", kind="permutation", build=_keccak_f1600),
    HashSpec(
        name="sha3-256",
        kind="byte_hash",
        build=_sha3_256,
        message_bytes=_MESSAGE_BYTES,
    ),
    HashSpec(
        name="sha256", kind="byte_hash", build=_sha256, message_bytes=_MESSAGE_BYTES
    ),
    HashSpec(
        name="blake3", kind="byte_hash", build=_blake3, message_bytes=_MESSAGE_BYTES
    ),
)

_BY_NAME = {row.name: row for row in _ROWS}


def names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def rows(selected: tuple[str, ...] = ()) -> Iterator[HashSpec]:
    """The selected rows in table order, or every row when nothing is selected."""
    if not selected:
        yield from _ROWS
        return
    unknown = [n for n in selected if n not in _BY_NAME]
    if unknown:
        raise KeyError(
            f"no such hash: {', '.join(unknown)}; have {', '.join(_BY_NAME)}"
        )
    for name in selected:
        yield _BY_NAME[name]
