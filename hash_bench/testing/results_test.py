# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The row: what it serializes, and what the table shows a reader."""

import json

from absl.testing import absltest

from hash_bench import results


def _row(**overrides: object) -> results.Row:
    fields = dict(
        hash="poseidon2-koalabear16",
        batch=4096,
        backend="cpu",
        arm_requested="routed",
        arm_observed="routed",
        kernels=("poseidon2",),
        compile_ns=1_500_000,
        ns_per_hash=69.3,
        ns_per_hash_min=68.9,
        spread=0.01,
        hashes_per_s=1.44e7,
        bytes_moved=524288,
        bytes_per_s=1.85e9,
        ops_per_hash=616,
        ops_per_s=8.9e9,
        ops_unit="field_mul",
        ops_note="an op model",
        roofline={
            "memory_fraction": 0.06,
            "arith_fraction": 0.28,
            "bound": "arithmetic",
        },
        method={"reps": 7},
        machine={"host": "somewhere"},
    )
    fields.update(overrides)
    return results.Row(**fields)  # type: ignore[arg-type]


class RowTest(absltest.TestCase):
    def test_a_row_round_trips_through_json(self) -> None:
        payload = json.loads(json.dumps(_row().to_json()))
        self.assertEqual(payload["hash"], "poseidon2-koalabear16")
        self.assertEqual(payload["kernels"], ["poseidon2"])
        self.assertEqual(payload["schema"], results.SCHEMA_VERSION)

    def test_an_observed_arm_matching_the_request_is_as_requested(self) -> None:
        self.assertTrue(_row().to_json()["as_requested"])

    def test_an_observed_arm_differing_from_the_request_is_not(self) -> None:
        # A pin need not offer every arm on every backend. Where it does not,
        # the row must not read as if the requested arm had run.
        row = _row(arm_requested="routed", arm_observed="generic")
        self.assertFalse(row.to_json()["as_requested"])


class TableTest(absltest.TestCase):
    def test_a_substituted_arm_is_shown_with_what_was_asked_for(self) -> None:
        rendered = results.table(
            [_row(arm_requested="routed", arm_observed="generic").to_json()]
        )
        self.assertIn("generic (<-routed)", rendered)

    def test_the_bound_ceiling_is_the_one_reported(self) -> None:
        rendered = results.table([_row().to_json()])
        self.assertIn("28.0% of arithmetic", rendered)

    def test_compile_time_is_shown_beside_the_run_time(self) -> None:
        # Compile cost is an axis the arms differ on, so a table without it
        # invites reading a slow-to-compile row as merely slower to run.
        self.assertIn("0.00", results.table([_row(compile_ns=1_500_000).to_json()]))
        self.assertIn(
            "42.00", results.table([_row(compile_ns=42_000_000_000).to_json()])
        )

    def test_a_row_with_no_arithmetic_model_reports_the_memory_fraction(self) -> None:
        row = _row(
            ops_per_hash=None,
            ops_per_s=None,
            ops_unit=None,
            ops_note=None,
            roofline={
                "memory_fraction": 0.18,
                "arith_fraction": None,
                "bound": "memory (no arithmetic model declared for this hash)",
            },
        )
        self.assertIn("18.0% of memory", results.table([row.to_json()]))


if __name__ == "__main__":
    absltest.main()
