# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The ceilings: that the probes measure something, and that a row is reported
against the one it is nearer."""

from absl.testing import absltest

from hash_bench import roofline, timing

# The suite is checking that a probe runs and reports a rate, not what the rate
# is, so it runs the shortest reps that still time anything.
_FAST = timing.Method(warmup=1, reps=2, target_rep_ns=2_000_000)


class PeakTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        roofline.reset_peaks()

    def test_the_memory_probe_reports_a_positive_rate(self) -> None:
        peak = roofline.memory_peak(_FAST)
        self.assertGreater(peak.bytes_per_s, 0.0)
        self.assertIn("read", peak.probe)

    def test_the_memory_peak_is_measured_once_per_process(self) -> None:
        # Re-probing per row would make two rows in one worker incomparable,
        # since the second would carry a differently-noisy ceiling.
        self.assertIs(roofline.memory_peak(_FAST), roofline.memory_peak(_FAST))

    def test_the_arithmetic_probe_reports_a_rate_in_the_asked_unit(self) -> None:
        import frx.numpy as fnp

        peak = roofline.arith_peak(fnp.uint32, "u32_mul", _FAST)
        self.assertGreater(peak.ops_per_s, 0.0)
        self.assertEqual(peak.unit, "u32_mul")

    def test_arithmetic_peaks_are_cached_per_dtype(self) -> None:
        import frx.numpy as fnp

        a = roofline.arith_peak(fnp.uint32, "u32_mul", _FAST)
        b = roofline.arith_peak(fnp.float32, "f32_mul", _FAST)
        self.assertIs(a, roofline.arith_peak(fnp.uint32, "u32_mul", _FAST))
        self.assertIsNot(a, b)


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
