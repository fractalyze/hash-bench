# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The table of hashes: that every row builds and runs, that the traffic a row
reports is the traffic its call makes, and that a declared op model is derived
from the primitive rather than written down."""

import numpy as np
from absl.testing import absltest, parameterized

from hash_bench import registry

# The hashes the harness exists to compare. A row may be added freely; one of
# these going missing changes what the harness claims to cover.
_REQUIRED = ("poseidon2-koalabear16", "keccak-f1600", "sha3-256", "sha256", "blake3")


class TableTest(absltest.TestCase):
    def test_the_required_hashes_are_covered(self) -> None:
        self.assertContainsSubset(_REQUIRED, registry.names())

    def test_row_names_are_unique(self) -> None:
        names = [row.name for row in registry.rows()]
        self.assertCountEqual(names, set(names))

    def test_selecting_rows_keeps_the_order_asked_for(self) -> None:
        picked = [row.name for row in registry.rows(("sha256", "blake3"))]
        self.assertEqual(picked, ["sha256", "blake3"])

    def test_an_unknown_name_names_what_is_available(self) -> None:
        with self.assertRaisesRegex(KeyError, "sha512"):
            list(registry.rows(("sha512",)))


class OpModelTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("alpha3", 3, 2),  # x^2 * x
        ("alpha5", 5, 3),  # x^2, x^4, x^4 * x
        ("alpha7", 7, 4),  # x^2, x^3, x^6, x^7
    )
    def test_the_sbox_chain_length(self, alpha: int, expected: int) -> None:
        self.assertEqual(registry._binary_chain_mults(alpha), expected)

    def test_a_nonpositive_exponent_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            registry._binary_chain_mults(0)

    def test_the_poseidon2_count_is_read_off_the_params(self) -> None:
        import hash_frx

        # KoalaBear16: alpha 3, 4+4 external rounds, 20 internal, width 16.
        # 8*16*2 S-box multiplies + 20*(2 + 16) diagonal and S-box.
        perm = hash_frx.Poseidon2(hash_frx.KOALABEAR16_PARAMS)
        self.assertEqual(registry.poseidon2_field_mults(perm).count, 256 + 360)

    def test_a_reparameterized_poseidon2_recounts_itself(self) -> None:
        import hash_frx

        # BabyBear16 differs in both axes the model reads: alpha 7 (4
        # multiplies) and 13 internal rounds. A literal count would not move.
        perm = hash_frx.Poseidon2(hash_frx.BABYBEAR16_PARAMS)
        model = registry.poseidon2_field_mults(perm)
        self.assertEqual(model.count, 8 * 16 * 4 + 13 * (4 + 16))
        self.assertEqual(model.unit, "field_mul")
        self.assertIn("x^7", model.note)


class CallTest(parameterized.TestCase):
    @parameterized.named_parameters(*[(name, name) for name in _REQUIRED])
    def test_a_row_runs_and_reports_the_traffic_it_moved(self, name: str) -> None:
        import frx

        batch = 8
        (spec,) = registry.rows((name,))
        call = spec.call(batch)
        out = frx.block_until_ready(call.fn(call.x))
        self.assertEqual(call.hashes, batch)
        # The row's traffic claim is the arrays its call actually reads and
        # writes, so it is checkable against them rather than against a model.
        expected = registry._nbytes(call.x) + registry._nbytes(out)
        self.assertEqual(call.bytes_moved, expected)

    def test_a_byte_hash_digests_the_message_length_its_row_declares(self) -> None:
        import hashlib

        (spec,) = registry.rows(("sha256",))
        call = spec.call(4)
        self.assertEqual(call.x.shape, (4, spec.message_bytes))
        digests = np.asarray(call.fn(call.x))
        messages = np.asarray(call.x)
        for i in range(4):
            self.assertEqual(
                bytes(digests[i]), hashlib.sha256(bytes(messages[i])).digest()
            )

    def test_a_permutation_row_declares_no_message_length(self) -> None:
        for spec in registry.rows():
            if spec.kind == "permutation":
                self.assertEqual(spec.message_bytes, 0, spec.name)

    def test_only_a_row_with_a_model_reports_ops(self) -> None:
        for spec in registry.rows():
            call = spec.call(2)
            self.assertEqual(call.ops is not None, spec.op_model is not None, spec.name)


if __name__ == "__main__":
    absltest.main()
