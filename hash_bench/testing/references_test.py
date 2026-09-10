# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The references, held to hash-frx's own output.

This is the suite that makes a reference row a reference. A pinned
implementation that is fast and computes something else is not a comparison,
and nothing else in the harness would notice: the row shape, the traffic and
the roofline are all computed from the arrays, none of which knows what the
bytes mean.

So every (reference, hash) pair runs both sides over the same input and asserts
the outputs are equal. The input is the one `registry.HashSpec.call` builds,
which is also the one the sweep measures — a cross-check over a different input
than the sweep runs would leave the measured path unchecked.

Small batches on purpose: this asserts agreement, not throughput, and the
sweep is where the sizes that matter get run.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from hash_bench import backends, references, registry

_BATCH = 4


def _require_device(test: absltest.TestCase, ref: references.Reference) -> None:
    """Skip, visibly, a CUDA reference on a host with no device to run it."""
    if ref.gpu and ref.cuda.device_count() == 0:
        test.skipTest(f"{ref.name} is a CUDA reference and this host has no device")


def _pairs() -> list[tuple[str, str, str]]:
    """Every (reference, hash) the table declares, as named test cases."""
    return [
        (f"{ref}_{hash_name}", ref, hash_name)
        for ref in references.names()
        for hash_name in references.get(ref).hashes()
    ]


class ProvenanceTest(parameterized.TestCase):
    @parameterized.named_parameters((name, name) for name in references.names())
    def test_provenance_names_a_revision_and_the_flags(self, name: str) -> None:
        """The build's account of itself reached the runfiles tree.

        A missing provenance file is not a cosmetic failure: the row would carry
        no revision, and a reference number with no revision is not comparable
        to anything.
        """
        provenance = references.get(name).provenance
        self.assertEqual(provenance.reference, name)
        self.assertNotEmpty(provenance.revision)
        self.assertNotEmpty(provenance.source)
        self.assertNotEmpty(provenance.implementation)
        self.assertNotEmpty(provenance.flags)
        # The upstream is compared with an optimised hash-frx wheel, so a
        # reference whose build does not name an optimised mode is measuring
        # a debug-grade program under the upstream's name.
        self.assertEqual(provenance.compilation_mode, "opt")

    @parameterized.named_parameters(
        (name, name) for name in references.names() if references.get(name).gpu
    )
    def test_a_cuda_reference_names_its_architectures_and_compiler(
        self, name: str
    ) -> None:
        """A kernel built for another architecture runs as JIT-compiled PTX, so
        the setting it was built under is part of what the row measured; the
        nvcc is reported by the kernel's own translation unit."""
        provenance = references.get(name).provenance
        self.assertTrue(
            any(f.startswith("--@rules_cuda//cuda:archs=") for f in provenance.flags),
            provenance.flags,
        )
        self.assertStartsWith(provenance.runtime_dispatch, "nvcc ")

    @parameterized.named_parameters(*_pairs())
    def test_covered_hash_is_in_the_registry(self, name: str, hash_name: str) -> None:
        """A reference covering a hash the registry does not carry would produce
        rows nothing can be compared against."""
        self.assertIn(hash_name, registry.names())
        self.assertNotEmpty(references.get(name).symbol(hash_name))


class AgreementTest(parameterized.TestCase):
    @parameterized.named_parameters(*_pairs())
    def test_reference_reproduces_hash_frx(self, name: str, hash_name: str) -> None:
        ref = references.get(name)
        _require_device(self, ref)
        spec = next(registry.rows((hash_name,)))

        frx_call = spec.call(_BATCH)
        expected = np.asarray(frx_call.fn(frx_call.x))

        ref_call = ref.call(spec, _BATCH)
        result = ref_call.fn(ref_call.x)
        ref.block(result)
        actual = np.asarray(result)

        # The two sides have to be fed the same bytes for the comparison to say
        # anything; that they are is a property of `references` building its
        # input to the pattern `HashSpec.call` builds, so it is asserted rather
        # than assumed.
        np.testing.assert_array_equal(np.asarray(frx_call.x), np.asarray(ref_call.x))
        np.testing.assert_array_equal(actual, expected)

    @parameterized.named_parameters(*_pairs())
    def test_traffic_matches_the_hash_frx_row(self, name: str, hash_name: str) -> None:
        """Both arms move the same bytes, so both are divided by the same memory
        ceiling to mean the same thing. A reference that wrote its digests into
        a differently shaped buffer would report a different roofline fraction
        for the same work."""
        _require_device(self, references.get(name))
        spec = next(registry.rows((hash_name,)))
        self.assertEqual(
            references.get(name).call(spec, _BATCH).bytes_moved,
            spec.call(_BATCH).bytes_moved,
        )


class SelectionTest(absltest.TestCase):
    def test_rows_yields_only_what_the_reference_implements(self) -> None:
        ref = references.get(references.names()[0])
        selected = tuple(spec.name for spec in references.rows(ref))
        self.assertEqual(set(selected), set(ref.hashes()))

    def test_asking_for_an_uncovered_hash_is_not_an_error(self) -> None:
        """One sweep runs several arms over one hash list, so a reference is
        routinely handed a hash it does not implement."""
        ref = references.get(references.names()[0])
        uncovered = [n for n in registry.names() if not ref.covers(n)]
        self.assertNotEmpty(uncovered)
        self.assertEmpty(list(references.rows(ref, tuple(uncovered))))

    def test_every_reference_is_offered_on_legs_that_exist(self) -> None:
        """A leg name no backend defines would drop the reference from every
        sweep without a word."""
        for name in references.names():
            for leg in references.get(name).legs:
                self.assertIn(leg, backends.names(), name)

    def test_unknown_reference_names_the_ones_that_exist(self) -> None:
        with self.assertRaisesRegex(KeyError, "no such reference"):
            references.get("not-a-reference")


if __name__ == "__main__":
    absltest.main()
