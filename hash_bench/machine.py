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
"""

from __future__ import annotations

import os
import platform
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
    """The commit of the checkout `start` physically lives in, or None when it
    is not one (a wheel).

    Resolved first, because a Bazel runfiles entry is a SYMLINK sitting under
    the consumer's workspace: asking git about the link path walks up into
    hash-bench's own checkout and reports hash-bench's commit as the
    dependency's. Resolving points at where the source actually is.
    """
    start = start.resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


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
    # A source checkout on `sys.path` beats the installed distribution, and a
    # dev run routinely has one. Recording the sha next to the version is what
    # lets a reader tell the two cases apart.
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
