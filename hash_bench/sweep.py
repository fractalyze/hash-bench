# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The driver: one command, one row per (hash, batch, backend, arm).

Two processes deep, because what a run selects can only be chosen before a
backend exists. `XLA_FLAGS` — which is how an arm is requested — is parsed once,
when the plugin loads, and the CPU affinity mask a leg sets has to be in place
before the backend spawns its threads. So an orchestrator process fans out one
WORKER per (backend leg, arm), each started with the environment that pair needs,
and collects the rows they emit. Running the matrix in one process would
silently measure the first combination's flags for all of them.

    bazel run //hash_bench:sweep -- --out results/$(hostname).jsonl

Narrow it with `--hash`, `--batch`, `--backend` and `--arm`; each is repeatable
and defaults to everything available.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from typing import Any, TextIO

from hash_bench import arms, backends, machine, registry, results, roofline, timing

DEFAULT_BATCHES = (1, 256, 4096, 65536)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hash-bench", description=__doc__)
    p.add_argument(
        "--hash",
        action="append",
        default=[],
        metavar="NAME",
        help=f"repeatable; default every row ({', '.join(registry.names())})",
    )
    p.add_argument(
        "--batch",
        action="append",
        type=int,
        default=[],
        metavar="N",
        help=f"repeatable; default {', '.join(map(str, DEFAULT_BATCHES))}",
    )
    p.add_argument(
        "--backend",
        action="append",
        default=[],
        metavar="LEG",
        help=f"repeatable; default every available ({', '.join(backends.names())})",
    )
    p.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="ARM",
        help=f"repeatable; default every arm ({', '.join(a.value for a in arms.Arm)})",
    )
    p.add_argument(
        "--out",
        default="",
        metavar="PATH",
        help="also append the rows here; they always go to stdout",
    )
    p.add_argument("--warmup", type=int, default=timing.DEFAULT_WARMUP)
    p.add_argument("--reps", type=int, default=timing.DEFAULT_REPS)
    p.add_argument("--target-rep-ns", type=int, default=timing.DEFAULT_TARGET_REP_NS)
    p.add_argument(
        "--quiet", action="store_true", help="do not render the summary table on stderr"
    )
    # The orchestrator sets this on the children it spawns.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def _worker_env(leg: backends.Leg, arm: arms.Arm) -> dict[str, str]:
    """The environment one (leg, arm) needs. `XLA_FLAGS` is appended to rather
    than replaced, so a caller's flags survive into every worker."""
    env = dict(os.environ)
    env["FRX_PLATFORMS"] = leg.frx_platforms
    flags = [env.get("XLA_FLAGS", ""), *leg.xla_flags, *arm.xla_flags]
    env["XLA_FLAGS"] = " ".join(f for f in flags if f).strip()
    env.update(leg.env)
    # The worker re-enters this module by name, so it needs this checkout on the
    # path whether it came from a wheel, a Bazel runfiles tree or a source tree.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


def _worker_argv(args: argparse.Namespace, leg: str, arm: str) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "hash_bench.sweep",
        "--worker",
        "--backend",
        leg,
        "--arm",
        arm,
        "--warmup",
        str(args.warmup),
        "--reps",
        str(args.reps),
        "--target-rep-ns",
        str(args.target_rep_ns),
    ]
    for name in args.hash:
        argv += ["--hash", name]
    for batch in args.batch:
        argv += ["--batch", str(batch)]
    return argv


def _run_worker(
    argv: list[str], env: dict[str, str], out: TextIO | None
) -> tuple[list[dict[str, Any]], int]:
    """Run one worker to exit, returning its rows and its exit status.

    Rows are forwarded as they arrive rather than after the worker exits: a leg
    can spend minutes on one row (a declined region compiles the whole round
    schedule), and a run that shows nothing until then looks hung. The worker's
    stderr is inherited, so its warnings arrive live too.
    """
    rows: list[dict[str, Any]] = []
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        # stdout carries the rows and stderr the table, so `sweep > rows.jsonl`
        # is a complete run and `--out` is the convenience of appending to a
        # file at the same time.
        print(line, flush=True)
        if out is not None:
            out.write(line + "\n")
            out.flush()
    return rows, proc.wait()


def orchestrate(args: argparse.Namespace) -> int:
    legs = args.backend or backends.available()
    requested_arms = [arms.Arm(a) for a in (args.arm or [a.value for a in arms.Arm])]
    rows: list[dict[str, Any]] = []
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = open(args.out, "a") if args.out else None
    try:
        for leg_name in legs:
            leg = backends.get(leg_name)
            for arm in requested_arms:
                worker_rows, status = _run_worker(
                    _worker_argv(args, leg_name, arm.value),
                    _worker_env(leg, arm),
                    out,
                )
                rows += worker_rows
                if status != 0:
                    sys.stderr.write(f"[{leg_name}/{arm.value}] worker failed\n")
                    return status
    finally:
        if out is not None:
            out.close()
    if not args.quiet and rows:
        print(results.table(rows), file=sys.stderr)
    return 0


def _row(
    *,
    spec: registry.HashSpec,
    batch: int,
    leg_name: str,
    arm: arms.Arm,
    call: registry.Call,
    compile_ns: int,
    measurement: timing.Measurement,
    observed: arms.Lowering,
    callees: tuple[str, ...],
    memory: roofline.MemoryPeak,
    method: timing.Method,
    machine_record: dict[str, Any],
) -> results.Row:
    seconds = measurement.ns_per_call * 1e-9
    hashes_per_s = call.hashes / seconds
    bytes_per_s = call.bytes_moved / seconds
    if call.ops is None:
        ops_per_hash = None
        ops_per_s = None
        ops_unit = None
        ops_note = None
        arith = None
    else:
        ops_per_hash = call.ops.count
        ops_per_s = call.ops.count * hashes_per_s
        ops_unit = call.ops.unit
        ops_note = call.ops.note
        arith = roofline.arith_peak(call.x.dtype, call.ops.unit, method)
    return results.Row(
        hash=spec.name,
        batch=batch,
        backend=leg_name,
        arm_requested=arm.value,
        arm_observed=observed.arm_name,
        kernels=callees,
        compile_ns=compile_ns,
        ns_per_hash=measurement.ns_per_call / call.hashes,
        ns_per_hash_min=measurement.ns_per_call_min / call.hashes,
        spread=measurement.spread,
        hashes_per_s=hashes_per_s,
        bytes_moved=call.bytes_moved,
        bytes_per_s=bytes_per_s,
        ops_per_hash=ops_per_hash,
        ops_per_s=ops_per_s,
        ops_unit=ops_unit,
        ops_note=ops_note,
        roofline=roofline.fractions(bytes_per_s, ops_per_s, memory, arith),
        method=dict(
            dataclasses.asdict(measurement.method),
            observation=(
                "the arm and compile_ns were read from a second lowering of the "
                "same computation; the timed calls go through the jit cache"
            ),
        ),
        machine=machine_record,
    )


def work(args: argparse.Namespace) -> int:
    """One (backend leg, arm), in the environment the orchestrator built for it."""
    if len(args.backend) != 1 or len(args.arm) != 1:
        raise SystemExit(
            "--worker runs exactly one --backend and one --arm; the orchestrator "
            "fans out the rest, because the flags each arm needs are read when "
            "the plugin loads"
        )
    leg = backends.get(args.backend[0])
    arm = arms.Arm(args.arm[0])
    if leg.one_core:
        backends.pin_to_one_core()

    method = timing.Method(
        warmup=args.warmup, reps=args.reps, target_rep_ns=args.target_rep_ns
    )
    machine_record = machine.describe()
    memory = roofline.memory_peak(method)
    batches = args.batch or DEFAULT_BATCHES
    for spec in registry.rows(tuple(args.hash)):
        for batch in batches:
            call = spec.call(batch)
            started = time.perf_counter_ns()
            compiled = call.fn.lower(call.x).compile()
            compile_ns = time.perf_counter_ns() - started
            observed, callees = arms.observe(compiled.as_text())
            measurement = timing.measure(call.fn, call.x, method)
            row = _row(
                spec=spec,
                batch=batch,
                leg_name=leg.name,
                arm=arm,
                call=call,
                compile_ns=compile_ns,
                measurement=measurement,
                observed=observed,
                callees=callees,
                memory=memory,
                method=method,
                machine_record=machine_record,
            )
            results.write(row, sys.stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return work(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
