# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""What ran the sweep, and what it was built from.

A ns/hash number is meaningless without the machine and the toolchain, and this
harness exists to gate emitter deletions across pins, so the row carries both
rather than a table caption carrying them. Everything here is read at runtime
from the process that produced the row: nothing is passed in, so a row cannot
claim a revision it did not run.

The plugin revision is the point of the `revisions` block. `frx-cuda12-plugin`
IS the Fractalyze XLA build, so its version is the xla pin, and a row measured
against a different one is not comparable to this one no matter how similar the
machine.

hash-frx needs two entries rather than one, because it arrives two ways and only
one of them carries a commit at runtime. As a bzlmod module it is fetched with
`git_override`, which deletes `.git` after cloning, so the tree that runs has no
commit to read and only `MODULE.bazel` knows which one it is; as a checkout on
`sys.path` it has a commit and `MODULE.bazel` describes something else. So the
record carries the pin `MODULE.bazel` names AND the checkout sha when there is
one, and `results/README.md` says how to read the pair.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

# The distributions whose versions decide what a row measured. `frx-cuda12-*`
# are absent on a CPU-only host, which is recorded as absent rather than
# skipped: "no GPU plugin installed" is a fact about the row.
_DISTRIBUTIONS = (
    "frx",
    "frxlib",
    "frx-cuda12-pjrt",
    "frx-cuda12-plugin",
    "zk-dtypes",
    "hash-frx",
)


def _git_sha(start: Path) -> str | None:
    """The commit of the checkout that TRACKS `start`, or None when no checkout
    does.

    The tracking check is the whole function. `git -C` answers about whichever
    repository encloses the path, and a Bazel runfiles tree for hash-frx sits
    under the CONSUMER's `bazel-out` — so asking there returns hash-bench's own
    commit and reports it as the dependency's. A repo that encloses the path
    without tracking any file in it is not the path's repo, which is exactly the
    runfiles case (`bazel-out` is gitignored) and exactly the wheel case.
    """
    start = start.resolve()

    def git(*args: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", "-C", str(start), *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None

    tracked = git("ls-files", "--", ".")
    if tracked is None or tracked.returncode != 0 or not tracked.stdout.strip():
        return None
    head = git("rev-parse", "HEAD")
    if head is None or head.returncode != 0:
        return None
    return head.stdout.strip() or None


# The `commit = "<sha>"` of the `git_override` block naming `hash_frx`. Matched
# over the whole file rather than line by line: the two fields are on separate
# lines of one call, and either order occurs.
_HASH_FRX_PIN = re.compile(
    r"git_override\((?=[^)]*module_name\s*=\s*[\"']hash_frx[\"'])"
    r"[^)]*commit\s*=\s*[\"']([0-9a-f]{7,40})[\"']",
    re.DOTALL,
)


def _pinned_hash_frx_commit() -> str | None:
    """The hash-frx commit `MODULE.bazel` pins, or None when this process cannot
    see one (a wheel install, where the workspace is not on disk).

    Two candidate roots, because a runfiles entry may be a symlink back into the
    source tree or a real copy, and the two put the file in different places:
    the unresolved parent is the runfiles root, where `//:MODULE.bazel` is
    attached as data for exactly this read, and the resolved one is the source
    checkout's root.
    """
    here = Path(__file__)
    for root in (here.parent.parent, here.resolve().parent.parent):
        try:
            text = (root / "MODULE.bazel").read_text()
        except OSError:
            continue
        match = _HASH_FRX_PIN.search(text)
        if match:
            return match.group(1)
    return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def revisions() -> dict[str, Any]:
    """Every version that decides what the numbers mean."""
    out: dict[str, Any] = {}
    for dist in _DISTRIBUTIONS:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    # The commit the workspace pins — the one a Bazel run actually executed,
    # and the only place that commit survives into the process.
    out["hash-frx-pin"] = _pinned_hash_frx_commit()
    # Non-null only when hash-frx came from a checkout on `sys.path`, which is
    # then the commit that ran and the pin above is not.
    try:
        import hash_frx

        out["hash-frx-version"] = hash_frx.__version__
        out["hash-frx-sha"] = _git_sha(Path(hash_frx.__file__).parent)
    except ImportError:
        out["hash-frx-version"] = None
        out["hash-frx-sha"] = None
    out["hash-bench-sha"] = _git_sha(Path(__file__).resolve().parent)
    return out


def devices() -> list[dict[str, Any]]:
    import frx

    return [
        {
            "id": d.id,
            "platform": d.platform,
            "kind": getattr(d, "device_kind", None),
        }
        for d in frx.devices()
    ]


def describe() -> dict[str, Any]:
    """The machine record that rides in every row of this run."""
    import frx

    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    else:
        affinity = []
    return {
        "host": socket.gethostname(),
        "os": platform.platform(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "backend": frx.default_backend(),
        "devices": devices(),
        "env": {
            name: os.environ.get(name)
            for name in ("FRX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS")
        },
        "revisions": revisions(),
    }
