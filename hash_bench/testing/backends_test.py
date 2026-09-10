# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The backend legs: what they name, and what this machine can run."""

from absl.testing import absltest

from hash_bench import backends


class LegTest(absltest.TestCase):
    def test_the_three_legs_the_table_reports(self) -> None:
        self.assertEqual(backends.names(), ("cpu-1core", "cpu", "gpu"))

    def test_an_unknown_leg_names_what_is_available(self) -> None:
        with self.assertRaisesRegex(KeyError, "cpu-1core"):
            backends.get("metal")

    def test_the_one_core_leg_bounds_the_work_and_the_thread_pool(self) -> None:
        # Either alone is wrong: the mask without the flag leaves a full-width
        # Eigen pool thrashing inside one core, and the flag without the mask
        # leaves the runtime free to schedule anywhere.
        leg = backends.get("cpu-1core")
        self.assertTrue(leg.one_core)
        self.assertIn("--xla_cpu_multi_thread_eigen=false", leg.xla_flags)
        self.assertIn(("OMP_NUM_THREADS", "1"), leg.env)

    def test_the_one_core_leg_bounds_every_runtime_a_reference_may_bring(
        self,
    ) -> None:
        # A reference is not lowered by XLA and never sees its flags, so its own
        # pool has to be told separately or the leg pins one core and then lets
        # a reference put a thread per core inside it. One knob per runtime,
        # because they read different ones.
        env = dict(backends.get("cpu-1core").env)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["RAYON_NUM_THREADS"], "1")

    def test_only_the_gpu_leg_needs_a_device(self) -> None:
        needing = [leg.name for leg in backends.LEGS if leg.needs_device]
        self.assertEqual(needing, ["gpu"])


class AvailableTest(absltest.TestCase):
    def test_every_device_free_leg_is_always_available(self) -> None:
        available = backends.available()
        for leg in backends.LEGS:
            if not leg.needs_device:
                self.assertIn(leg.name, available)

    def test_availability_is_a_subset_of_the_legs(self) -> None:
        self.assertContainsSubset(backends.available(), backends.names())


class PinTest(absltest.TestCase):
    def test_pinning_leaves_one_core(self) -> None:
        import os

        before = os.sched_getaffinity(0)
        try:
            backends.pin_to_one_core()
            self.assertEqual(os.sched_getaffinity(0), {min(before)})
        finally:
            os.sched_setaffinity(0, before)


if __name__ == "__main__":
    absltest.main()
