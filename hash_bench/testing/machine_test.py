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

    def test_a_repo_that_only_encloses_the_path_reports_no_sha(self) -> None:
        # The shape a Bazel runfiles tree has: the dependency's files sit under
        # the CONSUMER's gitignored output directory. Asking git about the path
        # returns the consumer's commit, which would be recorded as the
        # dependency's — a row naming the wrong revision, worse than naming none.
        repo = Path(self.create_tempdir("consumer").full_path)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / ".gitignore").write_text("bazel-out/\n")
        (repo / "own.py").write_text("")
        for args in (
            ["add", ".gitignore", "own.py"],
            ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one"],
        ):
            subprocess.run(["git", "-C", str(repo), *args], check=True)
        enclosed = repo / "bazel-out" / "dependency"
        enclosed.mkdir(parents=True)
        (enclosed / "mod.py").write_text("")

        self.assertIsNotNone(machine._git_sha(repo))
        self.assertIsNone(machine._git_sha(enclosed))

    def test_a_path_in_no_checkout_reports_no_sha(self) -> None:
        self.assertIsNone(machine._git_sha(Path(self.create_tempdir().full_path)))


class PinTest(absltest.TestCase):
    def test_the_pinned_hash_frx_commit_is_readable(self) -> None:
        # A `git_override`-fetched module has no `.git`, so this file is the
        # only place the pinned commit survives into a run. It is attached as a
        # runfile for exactly this read; losing that attachment makes every row
        # stop identifying which hash-frx it measured.
        commit = machine._pinned_hash_frx_commit()
        self.assertIsNotNone(commit)
        self.assertRegex(commit, r"^[0-9a-f]{7,40}$")

    def test_revisions_carry_the_pin(self) -> None:
        self.assertIsNotNone(machine.revisions()["hash-frx-pin"])


if __name__ == "__main__":
    absltest.main()
