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

from hash_bench import arms, references, registry, results, sweep

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


def _arms_over(hash_name: str) -> list[str]:
    """Every arm a default sweep runs over one hash: all three hash-frx
    lowerings, plus each reference that implements it. Read off the tables
    rather than written down, because a reference covers what its codebase
    covers."""
    return [a.value for a in arms.Arm] + [
        name for name in references.names() if references.get(name).covers(hash_name)
    ]


class DriverTest(absltest.TestCase):
    def test_a_run_emits_one_row_per_hash_batch_and_arm(self) -> None:
        expected_arms = _arms_over("sha256")
        rows = _sweep("--hash", "sha256", "--batch", "4", "--batch", "8")
        self.assertLen(rows, 2 * len(expected_arms))
        self.assertCountEqual(
            {(r["batch"], r["arm_requested"]) for r in rows},
            {(b, a) for b in (4, 8) for a in expected_arms},
        )

    def test_a_reference_produces_no_row_for_a_hash_it_does_not_implement(
        self,
    ) -> None:
        # One sweep runs every arm over one hash list, so a reference is
        # routinely handed a hash it does not implement; the table simply has no
        # row there rather than the run failing.
        rows = _sweep("--hash", "sha256", "--batch", "4", "--arm", "xkcp")
        self.assertEmpty(rows)

    def test_a_reference_row_carries_the_revision_and_flags_that_built_it(
        self,
    ) -> None:
        # A reference number names no implementation without them, and the row
        # is the unit that gets quoted.
        (row,) = _sweep("--hash", "sha256", "--batch", "4", "--arm", "openssl")
        reference = row["reference"]
        self.assertEqual(reference["reference"], "openssl")
        self.assertNotEmpty(reference["revision"])
        self.assertNotEmpty(reference["flags"])
        # No run-time compile, and null rather than zero so it does not read as
        # a compile that cost nothing.
        self.assertIsNone(row["compile_ns"])
        # The kernel is the evidence the arm was read from, as on any row.
        self.assertEqual(row["kernels"], ["hash_bench_openssl_sha256"])

    def test_a_hash_frx_row_carries_no_reference_block(self) -> None:
        (row,) = _sweep("--hash", "sha256", "--batch", "4", "--arm", "routed")
        self.assertIsNone(row["reference"])

    def test_an_unknown_arm_names_the_ones_that_exist(self) -> None:
        with self.assertRaises(AssertionError):
            _sweep("--hash", "sha256", "--batch", "4", "--arm", "not-an-arm")

    def test_a_row_carries_the_revisions_that_produced_it(self) -> None:
        (row,) = _sweep("--hash", "sha256", "--batch", "4", "--arm", "routed")
        revisions = row["machine"]["revisions"]
        # The plugin version IS the xla pin, and a row measured against another
        # one is not comparable to this one however alike the machines are.
        self.assertIsNotNone(revisions["frx"])
        self.assertIsNotNone(revisions["hash-frx-version"])
        # Which hash-frx ran. A `git_override`-fetched module has no `.git`, so
        # without the pin a row would name no hash-frx commit at all.
        self.assertIsNotNone(revisions["hash-frx-pin"])
        self.assertEqual(row["schema"], results.SCHEMA_VERSION)
        self.assertGreater(row["ns_per_hash"], 0.0)
        self.assertGreater(row["roofline"]["memory_fraction"], 0.0)

    def test_rows_are_appended_to_the_output_file(self) -> None:
        path = os.path.join(self.create_tempdir().full_path, "rows.jsonl")
        emitted = _sweep("--hash", "sha256", "--batch", "4", "--out", path)
        self.assertEqual(results.read(path), emitted)

    def test_one_leg_reports_one_set_of_ceilings(self) -> None:
        # A share and the total it is a share of have to come from one
        # measurement, or the leg's arms stop being comparable to each other.
        rows = _sweep("--hash", "poseidon2-koalabear16", "--batch", "4")
        self.assertLen(rows, len(_arms_over("poseidon2-koalabear16")))
        self.assertLen({r["roofline"]["peak_bytes_per_s"] for r in rows}, 1)
        self.assertLen({r["roofline"]["peak_ops_per_s"] for r in rows}, 1)

    def test_a_reference_is_held_to_the_same_ceilings_as_the_hash_frx_arms(
        self,
    ) -> None:
        # The reference rows share the leg's Peaks rather than measuring their
        # own. A denominator per arm would make a reference's roofline fraction
        # and a hash-frx row's fractions of two different numbers, which is the
        # one thing a reader wants them not to be.
        rows = _sweep("--hash", "poseidon2-koalabear16", "--batch", "4")
        by_arm = {r["arm_requested"]: r for r in rows}
        self.assertIn("plonky3", by_arm)
        self.assertEqual(
            by_arm["plonky3"]["roofline"]["peak_ops_per_s"],
            by_arm["routed"]["roofline"]["peak_ops_per_s"],
        )

    def test_a_reference_row_declares_the_same_op_model_as_its_hash(self) -> None:
        # A permutation's multiply count is a property of the permutation, not
        # of who implements it, so both arms are held to the same arithmetic
        # ceiling in the same unit — which is what makes the two fractions
        # comparable at all.
        rows = _sweep("--hash", "poseidon2-koalabear16", "--batch", "4")
        by_arm = {r["arm_requested"]: r for r in rows}
        self.assertEqual(
            by_arm["plonky3"]["ops_per_hash"], by_arm["routed"]["ops_per_hash"]
        )
        self.assertEqual(by_arm["plonky3"]["ops_unit"], "field_mul")

    def test_the_worker_runs_under_the_arm_it_was_given(self) -> None:
        (row,) = _sweep("--hash", "sha256", "--batch", "4", "--arm", "declined")
        flags = row["machine"]["env"]["XLA_FLAGS"]
        for pass_name in (*arms.RECOGNIZER_PASSES, arms.LOOP_CONVERSION_PASS):
            self.assertIn(pass_name, flags)


class CeilingTest(absltest.TestCase):
    def test_a_declared_model_never_reports_as_undeclared(self) -> None:
        # A row whose ceiling the leg did not probe must fail, not fall through
        # to the "no arithmetic model declared" wording over a row that declares
        # one.
        from hash_bench import registry, roofline, timing

        (spec,) = registry.rows(("poseidon2-koalabear16",))
        call = spec.call(2)
        empty = roofline.Peaks(
            memory=roofline.MemoryPeak(bytes_per_s=1.0, probe="a probe")
        )
        with self.assertRaisesRegex(SystemExit, "field_mul"):
            sweep._row(
                spec=spec,
                batch=2,
                leg_name="cpu",
                arm=arms.Arm.ROUTED.value,
                call=call,
                compile_ns=1,
                measurement=timing.Measurement(1.0, 1.0, 0.0, timing.Method()),
                observed=arms.Lowering.DEDICATED.arm_name,
                callees=("poseidon2",),
                peaks=empty,
                machine_record={},
                observation="a probe",
            )


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
        # Every row is expected to reach a dedicated emitter on the CPU leg, so
        # one that stops routing has lost its kernel silently — right bytes, no
        # kernel — which is the failure the whole harness is built around. A pin
        # that deliberately drops an emitter changes this expectation, and does
        # so here rather than in a table nobody re-reads.
        (row,) = _sweep("--hash", name, "--batch", "4", "--arm", "routed")
        self.assertEqual(row["arm_observed"], arms.Arm.ROUTED.value, row["kernels"])


if __name__ == "__main__":
    absltest.main()
