# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The backend legs a sweep runs on.

Three, not two: a CPU hash pinned to one core and the same hash across every
core answer different questions — the first is the per-core cost an emitter is
written against, the second is what a consumer sees — and a table that reported
only one of them could not tell a threading win from a codegen one.

The one-core leg pins the process rather than asking XLA for a smaller thread
pool, and does both: the affinity mask is what actually bounds the work, and
the single-threaded Eigen flag is what stops a 24-thread pool from thrashing
inside a one-core mask. Affinity is set in the worker process, so it covers the
backend's own threads, which are spawned when the plugin loads.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys


@dataclasses.dataclass(frozen=True)
class Leg:
    name: str
    frx_platforms: str
    xla_flags: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    one_core: bool = False
    # Whether the leg needs a device to exist. `available()` reads it rather
    # than carrying its own list of which legs those are.
    needs_device: bool = False


LEGS: tuple[Leg, ...] = (
    Leg(
        name="cpu-1core",
        frx_platforms="cpu",
        xla_flags=("--xla_cpu_multi_thread_eigen=false",),
        env=(("OMP_NUM_THREADS", "1"),),
        one_core=True,
    ),
    Leg(name="cpu", frx_platforms="cpu"),
    Leg(name="gpu", frx_platforms="cuda", needs_device=True),
)

_BY_NAME = {leg.name: leg for leg in LEGS}


def names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def get(name: str) -> Leg:
    if name not in _BY_NAME:
        raise KeyError(f"no such backend leg: {name}; have {', '.join(_BY_NAME)}")
    return _BY_NAME[name]


# Asking the plugin is the only way to know whether the GPU leg can run: an
# installed `frx-cuda12-plugin` on a host with no device answers yes to every
# import check and no to this. It is asked in a THROWAWAY process because
# starting the CUDA plugin preallocates most of the device by default, which
# would leave the worker that needs it nothing to run in.
_GPU_PROBE = "import frx; print(any(d.platform == 'gpu' for d in frx.devices('cuda')))"


def _gpu_device_exists() -> bool:
    """Whether the probe process found a device. A probe that cannot run at all
    answers no, so a host without the plugin drops the GPU leg rather than
    failing the sweep."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", _GPU_PROBE],
            capture_output=True,
            text=True,
            timeout=180,
            env=dict(
                os.environ,
                FRX_PLATFORMS="cuda",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
                PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
            ),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    # The last token: the probe's stdout can carry load-time chatter ahead of
    # the answer.
    tokens = out.stdout.split()
    return bool(tokens) and tokens[-1] == "True"


def available() -> tuple[str, ...]:
    """The legs this machine can run. A leg needing no device always can."""
    legs = [leg.name for leg in LEGS if not leg.needs_device]
    if _gpu_device_exists():
        legs += [leg.name for leg in LEGS if leg.needs_device]
    return tuple(legs)


def pin_to_one_core() -> None:
    """Restrict this process to the lowest core it is already allowed to use.

    Lowest rather than any: a repeat run on the same host lands on the same
    core, so two rows differ by the arm rather than by which core the scheduler
    handed out.
    """
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("the one-core leg needs sched_setaffinity (Linux)")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
