# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The measurement method: the dispatch shape a row's `method` block claims,
and the statistic it reports."""

import dataclasses
from unittest import mock

from absl.testing import absltest

from hash_bench import timing


class DispatchTest(absltest.TestCase):
    """The claim `Method.dispatch` makes, held to what `measure` does."""

    def setUp(self) -> None:
        super().setUp()
        self.calls = 0
        self.blocks = 0

        def fn(_x: object) -> object:
            self.calls += 1
            return object()

        self.fn = fn

    def _block(self, _out: object) -> None:
        self.blocks += 1

    def test_a_preset_iters_skips_calibration(self) -> None:
        method = timing.Method(warmup=2, reps=3, iters=5)
        timing.measure(self.fn, None, method, block=self._block)
        self.assertEqual(self.calls, 2 + 3 * 5)

    def test_each_rep_blocks_once_not_once_per_call(self) -> None:
        # Blocking per call would measure the host round trip rather than the
        # kernel on any backend with an asynchronous queue.
        timing.measure(
            self.fn, None, timing.Method(warmup=2, reps=3, iters=5), block=self._block
        )
        self.assertEqual(self.blocks, 2 + 3)

    def test_the_chosen_iters_rides_in_the_method(self) -> None:
        # A row's method block has to say how many calls a rep held, or the
        # number cannot be re-derived from it.
        with mock.patch.object(timing, "calibrate", return_value=11):
            measurement = timing.measure(
                self.fn, None, timing.Method(reps=1), block=self._block
            )
        self.assertEqual(measurement.method.iters, 11)

    def test_a_synchronous_call_waits_for_nothing(self) -> None:
        # A native reference has returned its result by the time the call
        # returns, so the reference arms hand `measure` a blocker that does
        # nothing rather than one that reaches for frx.
        timing.measure(
            self.fn,
            None,
            timing.Method(warmup=1, reps=1, iters=2),
            block=timing.block_none,
        )
        self.assertEqual(self.calls, 1 + 2)
        self.assertEqual(self.blocks, 0)

    def test_the_synchronous_method_differs_only_in_the_dispatch_it_claims(
        self,
    ) -> None:
        # Everything a comparison between a reference row and a hash-frx row
        # rests on has to survive the switch; only the account of the block at
        # the end of a rep changes.
        method = timing.Method(warmup=2, reps=3, iters=5)
        synchronous = method.synchronous()
        self.assertEqual(synchronous.dispatch, timing.SYNC_DISPATCH)
        self.assertNotEqual(method.dispatch, synchronous.dispatch)
        self.assertEqual(
            dataclasses.replace(synchronous, dispatch=method.dispatch), method
        )


class StatisticTest(absltest.TestCase):
    def _measure(self, rep_ns: list[int]) -> timing.Measurement:
        method = timing.Method(warmup=0, reps=len(rep_ns), iters=1)
        with mock.patch.object(timing, "_one_rep", side_effect=rep_ns):
            return timing.measure(lambda _x: None, None, method)

    def test_the_reported_number_is_the_median_rep(self) -> None:
        # Median, not mean: a preempted rep is an outlier, not a tail of the
        # distribution.
        self.assertEqual(self._measure([100, 110, 900]).ns_per_call, 110)

    def test_the_fastest_rep_is_kept_as_the_noise_floor(self) -> None:
        self.assertEqual(self._measure([100, 110, 900]).ns_per_call_min, 100)

    def test_spread_is_the_range_over_the_median(self) -> None:
        # A large spread is what tells a reader the median is not worth quoting.
        self.assertAlmostEqual(self._measure([100, 110, 900]).spread, 800 / 110)

    def test_steady_reps_have_no_spread(self) -> None:
        self.assertEqual(self._measure([200, 200, 200]).spread, 0.0)


class CalibrateTest(absltest.TestCase):
    """Calibration against a fake whose rep cost scales with the call count, the
    one property the doubling search relies on."""

    def _calibrate(self, ns_per_call: int, target_rep_ns: int) -> int:
        with mock.patch.object(
            timing,
            "_one_rep",
            side_effect=lambda _fn, _x, iters, _block: iters * ns_per_call,
        ):
            return timing.calibrate(lambda _x: None, None, target_rep_ns)

    def test_iters_scale_to_the_target_rep(self) -> None:
        # 1 ms per call against a 20 ms target is 20 calls per rep.
        self.assertEqual(self._calibrate(1_000_000, 20_000_000), 20)

    def test_a_call_too_fast_to_time_doubles_until_it_is_not(self) -> None:
        # A microsecond call needs thousands of doublings' worth of calls before
        # a rep is long enough to extrapolate from; the answer is still the
        # target divided by the per-call cost.
        self.assertEqual(self._calibrate(1_000, 20_000_000), 20_000)

    def test_a_call_far_too_fast_stops_at_the_ceiling(self) -> None:
        # Without the ceiling a nanosecond call would ask for twenty million
        # dispatches per rep.
        self.assertEqual(self._calibrate(1, 20_000_000), timing.MAX_ITERS)

    def test_calibration_never_asks_for_less_than_one_call(self) -> None:
        # A call far longer than the target: one is as few as a rep can hold.
        self.assertEqual(self._calibrate(10**12, 1_000), 1)


if __name__ == "__main__":
    absltest.main()
