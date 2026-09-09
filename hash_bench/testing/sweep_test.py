# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The driver end to end, and the guard that keeps the arm list honest.

`DeclineTest` is the reason this suite compiles real modules: `arms.RECOGNIZER_PASSES`
is a written-down list of plugin pass names, and a recognizer added to the pin
and not added there would keep claiming its marker under the arms that exist to
take it away. Nothing about the frontend changes when that happens, so only a
compiled module catches it.
"""

import json
import os
import subprocess
import sys

from absl.testing import absltest, parameterized

from hash_bench import arms, registry, results

# The shortest settings that still produce a row: one warmup, one rep, and a rep
# target low enough that calibration stops at a handful of calls.
_FAST = ["--warmup", "1", "--reps", "1", "--target-rep-ns", "1000000"]
_LEG = "cpu"


def _sweep(*extra: str) -> list[dict]:
    """Run the driver as a user would and return the rows it emitted."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "hash_bench.sweep",
            "--quiet",
            "--backend",
            _LEG,
            *_FAST,
            *extra,
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p)),
    )
    if out.returncode != 0:
        raise AssertionError(f"sweep failed:\n{out.stdout}\n{out.stderr}")
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


class DriverTest(absltest.TestCase):
    def test_a_run_emits_one_row_per_hash_batch_and_arm(self) -> None:
        rows = _sweep("--hash", "sha256", "--batch", "4", "--batch", "8")
        self.assertLen(rows, 2 * len(arms.Arm))
        self.assertCountEqual(
            {(r["batch"], r["arm_requested"]) for r in rows},
            {(b, a.value) for b in (4, 8) for a in arms.Arm},
        )

    def test_a_row_carries_the_revisions_that_produced_it(self) -> None:
        (row,) = _sweep("--hash", "sha256", "--batch", "4", "--arm", "routed")
        revisions = row["machine"]["revisions"]
        # The plugin version IS the xla pin, and a row measured against another
        # one is not comparable to this one however alike the machines are.
        self.assertIsNotNone(revisions["frx"])
        self.assertIsNotNone(revisions["hash-frx-version"])
        self.assertEqual(row["schema"], results.SCHEMA_VERSION)
        self.assertGreater(row["ns_per_hash"], 0.0)
        self.assertGreater(row["roofline"]["memory_fraction"], 0.0)

    def test_rows_are_appended_to_the_output_file(self) -> None:
        path = os.path.join(self.create_tempdir().full_path, "rows.jsonl")
        emitted = _sweep("--hash", "sha256", "--batch", "4", "--out", path)
        self.assertEqual(results.read(path), emitted)

    def test_the_worker_runs_under_the_arm_it_was_given(self) -> None:
        (row,) = _sweep("--hash", "sha256", "--batch", "4", "--arm", "declined")
        flags = row["machine"]["env"]["XLA_FLAGS"]
        for pass_name in (*arms.RECOGNIZER_PASSES, arms.LOOP_CONVERSION_PASS):
            self.assertIn(pass_name, flags)


class DeclineTest(parameterized.TestCase):
    @parameterized.named_parameters(*[(n, n) for n in registry.names()])
    def test_no_row_still_routes_once_the_recognizers_are_off(self, name: str) -> None:
        (row,) = _sweep("--hash", name, "--batch", "4", "--arm", "declined")
        self.assertNotEqual(
            row["arm_observed"],
            arms.Arm.ROUTED.value,
            f"{name} kept a dedicated kernel ({row['kernels']}) with every pass "
            f"in arms.RECOGNIZER_PASSES disabled — the pin has a recognizer the "
            f"list does not name",
        )


class RoutedTest(parameterized.TestCase):
    @parameterized.named_parameters(*[(n, n) for n in registry.names()])
    def test_every_row_routes_on_the_cpu_leg_of_this_pin(self, name: str) -> None:
        # The CPU leg carries a dedicated emitter for all of them today. A row
        # that stops routing here has lost its kernel silently — right bytes, no
        # kernel — which is the failure the whole harness is built around.
        (row,) = _sweep("--hash", name, "--batch", "4", "--arm", "routed")
        self.assertEqual(row["arm_observed"], arms.Arm.ROUTED.value, row["kernels"])


if __name__ == "__main__":
    absltest.main()
