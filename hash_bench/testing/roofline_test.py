# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The ceilings: that the probes measure something, and that a row is reported
against the one it is nearer."""

from absl.testing import absltest

from hash_bench import roofline, timing

# The suite is checking that a probe runs and reports a rate, not what the rate
# is, so it runs the shortest reps that still time anything.
_FAST = timing.Method(warmup=1, reps=2, target_rep_ns=2_000_000)


class PeakTest(absltest.TestCase):
    def test_the_memory_probe_reports_a_positive_rate(self) -> None:
        peak = roofline.memory_peak(_FAST)
        self.assertGreater(peak.bytes_per_s, 0.0)
        self.assertIn("read", peak.probe)

    def test_the_arithmetic_probe_reports_a_rate_in_the_asked_unit(self) -> None:
        import frx.numpy as fnp

        peak = roofline.arith_peak(fnp.uint32, "u32_mul", _FAST)
        self.assertGreater(peak.ops_per_s, 0.0)
        self.assertEqual(peak.unit, "u32_mul")
        self.assertIn("uint32", peak.probe)

    def test_a_leg_measures_every_unit_its_rows_declare(self) -> None:
        import frx.numpy as fnp

        peaks = roofline.measure_peaks(
            [(fnp.uint32, "u32_mul"), (fnp.float32, "f32_mul")], _FAST
        )
        self.assertGreater(peaks.memory.bytes_per_s, 0.0)
        self.assertLen(peaks.arith, 2)

    def test_two_units_over_one_dtype_are_two_ceilings(self) -> None:
        # Keying by dtype alone would hand the second unit the first unit's
        # rate, and a unit names what a count MEANS — two units over one dtype
        # are two different operations.
        import frx.numpy as fnp

        peaks = roofline.measure_peaks(
            [(fnp.uint32, "u32_mul"), (fnp.uint32, "u32_bitop")], _FAST
        )
        self.assertLen(peaks.arith, 2)
        self.assertEqual(peaks.arith_for(fnp.uint32, "u32_mul").unit, "u32_mul")
        self.assertEqual(peaks.arith_for(fnp.uint32, "u32_bitop").unit, "u32_bitop")

    def test_a_unit_nobody_measured_has_no_ceiling(self) -> None:
        import frx.numpy as fnp

        peaks = roofline.measure_peaks([(fnp.uint32, "u32_mul")], _FAST)
        self.assertIsNone(peaks.arith_for(fnp.uint32, "field_mul"))


class PeaksTransportTest(absltest.TestCase):
    """Peaks cross a pipe from the probe worker to each arm worker, so the
    round trip is what makes one leg's arms share a denominator."""

    peaks = roofline.Peaks(
        memory=roofline.MemoryPeak(bytes_per_s=100.0, probe="a probe"),
        arith={
            ("koalabear_mont", "field_mul"): roofline.ArithPeak(
                ops_per_s=1000.0, unit="field_mul", probe="a probe"
            )
        },
    )

    def test_a_round_trip_preserves_every_ceiling(self) -> None:
        import json

        restored = roofline.Peaks.from_json(
            json.loads(json.dumps(self.peaks.to_json()))
        )
        self.assertEqual(restored, self.peaks)

    def test_a_dtype_is_keyed_by_its_readable_name(self) -> None:
        # The class spells itself `<class 'zk_dtypes.koalabear_mont'>`; an array
        # spells the same dtype `koalabear_mont`. A row may hand over either.
        import hash_frx

        dtype = hash_frx.Poseidon2(hash_frx.KOALABEAR16_PARAMS).dtype
        self.assertEqual(roofline.dtype_name(dtype), "koalabear_mont")
        self.assertIsNotNone(self.peaks.arith_for(dtype, "field_mul"))


class FractionTest(absltest.TestCase):
    memory = roofline.MemoryPeak(bytes_per_s=100.0, probe="a probe")
    arith = roofline.ArithPeak(ops_per_s=1000.0, unit="field_mul", probe="a probe")

    def test_a_row_with_no_model_is_reported_against_memory_alone(self) -> None:
        block = roofline.fractions(50.0, None, self.memory, None)
        self.assertEqual(block["memory_fraction"], 0.5)
        self.assertIsNone(block["arith_fraction"])
        # Saying WHY there is one candidate matters: it means nobody has written
        # down what this hash costs, not that the hash is memory-bound.
        self.assertIn("no arithmetic model", block["bound"])

    def test_the_larger_fraction_is_the_bound_one(self) -> None:
        block = roofline.fractions(10.0, 900.0, self.memory, self.arith)
        self.assertEqual(block["bound"], "arithmetic")
        self.assertAlmostEqual(block["arith_fraction"], 0.9)

    def test_a_streaming_row_binds_on_memory(self) -> None:
        block = roofline.fractions(90.0, 100.0, self.memory, self.arith)
        self.assertEqual(block["bound"], "memory")

    def test_both_fractions_are_kept_whichever_binds(self) -> None:
        block = roofline.fractions(10.0, 900.0, self.memory, self.arith)
        self.assertAlmostEqual(block["memory_fraction"], 0.1)
        self.assertEqual(block["arith_unit"], "field_mul")


if __name__ == "__main__":
    absltest.main()
