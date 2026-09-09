# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The machine record: that a revision it reports belongs to the thing it names."""

import subprocess
from pathlib import Path

from absl.testing import absltest

from hash_bench import machine


class GitShaTest(absltest.TestCase):
    def test_a_symlink_into_another_checkout_reports_that_checkout(self) -> None:
        # The shape a Bazel runfiles tree has: a link under this workspace
        # pointing at a dependency elsewhere. Reading the LINK path would report
        # this repo's commit as the dependency's.
        elsewhere = self.create_tempdir("elsewhere").full_path
        subprocess.run(["git", "init", "-q", elsewhere], check=True)
        (Path(elsewhere) / "a.py").write_text("")
        for args in (
            ["add", "a.py"],
            ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one"],
        ):
            subprocess.run(["git", "-C", elsewhere, *args], check=True)
        expected = subprocess.run(
            ["git", "-C", elsewhere, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        link = Path(self.create_tempdir("links").full_path) / "dep"
        link.symlink_to(elsewhere)
        self.assertEqual(machine._git_sha(link), expected)

    def test_a_path_in_no_checkout_reports_no_sha(self) -> None:
        self.assertIsNone(machine._git_sha(Path(self.create_tempdir().full_path)))


if __name__ == "__main__":
    absltest.main()
