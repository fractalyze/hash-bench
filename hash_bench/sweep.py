# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The driver: one command, one row per (hash, batch, backend, arm).

Two processes deep, because what a run selects can only be chosen before a
backend exists. `XLA_FLAGS` — which is how an arm is requested — is parsed once,
when the plugin loads, and the CPU affinity mask a leg sets has to be in place
before the backend spawns its threads. So an orchestrator process fans out one
WORKER per (backend leg, arm), each started with the environment that pair needs,
and collects the rows they emit. Running the matrix in one process would
silently measure the first combination's flags for all of them.

Each leg runs one extra worker first, which measures that leg's roofline
ceilings and prints them. Every arm worker on the leg is then handed the same
ceilings, so a leg's three arms are fractions of one denominator rather than of
three separately-noisy ones.

    bazel run //hash_bench:sweep -- --out results/$(hostname).jsonl

Narrow it with `--hash`, `--batch`, `--backend` and `--arm`; each is repeatable
and defaults to everything available.

`--arm` spans two vocabularies. The three hash-frx lowerings (`arms.py`) are
requested through `XLA_FLAGS`; the external CPU references (`references.py`) are
pinned implementations reached through `ctypes`, and are arms of the same sweep
so that a reference row and a hash-frx row are one measurement apart rather than
one harness apart. A reference is offered on the CPU legs only and produces rows
only for the hashes its codebase implements.
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

from hash_bench import (
    arms,
    backends,
    machine,
    references,
    registry,
    results,
    roofline,
    timing,
)

DEFAULT_BATCHES = (1, 256, 4096, 65536)


def all_arms() -> tuple[str, ...]:
    """Every arm name a run may ask for: the hash-frx lowerings then the
    references, which is also the order a default sweep runs them in."""
    return tuple(a.value for a in arms.Arm) + references.names()


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
        help=f"repeatable; default every arm ({', '.join(all_arms())})",
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
    # The orchestrator sets these on the children it spawns.
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--probe-peaks", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--peaks", default="", help=argparse.SUPPRESS)
    return p


def _worker_env(leg: backends.Leg, arm: str) -> dict[str, str]:
    """The environment one (leg, arm) needs. `XLA_FLAGS` is appended to rather
    than replaced, so a caller's flags survive into every worker.

    A reference adds no `XLA_FLAGS` of its own — it does not lower through the
    plugin — but it takes the leg's environment in full, because
    `OMP_NUM_THREADS` is how the one-core leg bounds a reference's own thread
    pool the way the affinity mask bounds the backend's.
    """
    env = dict(os.environ)
    env["FRX_PLATFORMS"] = leg.frx_platforms
    arm_flags = () if references.is_reference(arm) else arms.Arm(arm).xla_flags
    flags = [env.get("XLA_FLAGS", ""), *leg.xla_flags, *arm_flags]
    env["XLA_FLAGS"] = " ".join(f for f in flags if f).strip()
    env.update(leg.env)
    # The worker re-enters this module by name, so it needs this checkout on the
    # path whether it came from a wheel, a Bazel runfiles tree or a source tree.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return env


def _worker_argv(
    args: argparse.Namespace, leg: str, arm: str, peaks: str = ""
) -> list[str]:
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
    if peaks:
        argv += ["--peaks", peaks]
    return argv


def _run_worker(
    argv: list[str], env: dict[str, str], out: TextIO | None
) -> tuple[list[dict[str, Any]], int]:
    """Run one worker to exit, returning its rows and its exit status.

    Rows are forwarded as they arrive rather than after the worker exits: one row
    can outlast the whole rest of its leg (a declined region hands the backend
    the entire round schedule to compile), and a run that shows nothing until
    the worker exits looks hung. The worker's stderr is inherited, so its
    warnings arrive live too.
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


def _probe_peaks(args: argparse.Namespace, leg: backends.Leg) -> str:
    """Measure one leg's ceilings, returning them as the JSON its arm workers
    take. Run under the `routed` arm because the probes carry no hash marker for
    an arm to change; what matters is that all three arms then divide by this
    one measurement rather than by three of their own."""
    argv = _worker_argv(args, leg.name, arms.Arm.ROUTED.value) + ["--probe-peaks"]
    out = subprocess.run(
        argv,
        env=_worker_env(leg, arms.Arm.ROUTED.value),
        stdout=subprocess.PIPE,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"[{leg.name}] roofline probe failed")
    return out.stdout.strip()


def _offered(arm: str, leg_name: str) -> bool:
    """Whether this leg offers this arm.

    Only references narrow: they are the CPU frontier, and a GPU reference is a
    different set of codebases and a different pin. A hash-frx arm is offered on
    every leg and, where the pin cannot reach it, the row says which arm ran
    instead — that substitution is the harness's subject, so it is never
    filtered out here.
    """
    return not references.is_reference(arm) or leg_name in references.CPU_LEGS


def orchestrate(args: argparse.Namespace) -> int:
    legs = args.backend or backends.available()
    known_arms = all_arms()
    requested_arms = args.arm or list(known_arms)
    unknown = [a for a in requested_arms if a not in known_arms]
    if unknown:
        raise SystemExit(
            f"no such arm: {', '.join(unknown)}; have {', '.join(known_arms)}"
        )
    rows: list[dict[str, Any]] = []
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = open(args.out, "a") if args.out else None
    try:
        for leg_name in legs:
            leg = backends.get(leg_name)
            peaks = _probe_peaks(args, leg)
            for arm in requested_arms:
                if not _offered(arm, leg_name):
                    continue
                worker_rows, status = _run_worker(
                    _worker_argv(args, leg_name, arm, peaks),
                    _worker_env(leg, arm),
                    out,
                )
                rows += worker_rows
                if status != 0:
                    sys.stderr.write(f"[{leg_name}/{arm}] worker failed\n")
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
    arm: str,
    call: registry.Call,
    compile_ns: int | None,
    measurement: timing.Measurement,
    observed: str,
    callees: tuple[str, ...],
    peaks: roofline.Peaks,
    machine_record: dict[str, Any],
    observation: str,
    reference: dict[str, Any] | None = None,
) -> results.Row:
    """One row, from whichever arm produced it.

    Shared by the hash-frx arms and the references on purpose: traffic, the op
    model, the roofline fractions and the statistics all have to be computed the
    same way on both sides, or the comparison the table exists for is between
    two harnesses rather than between two implementations.
    """
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
        arith = peaks.arith_for(call.ops_dtype, call.ops.unit)
        if arith is None:
            # The leg's probe measures exactly the units its rows declare, so a
            # miss is a wiring bug. Falling through would print
            # "no arithmetic model declared for this hash" over a row that
            # declares one — a row lying about why it has one ceiling.
            raise SystemExit(
                f"{spec.name}: no {call.ops.unit} ceiling in {call.ops_dtype} was "
                "probed for this leg"
            )
    return results.Row(
        hash=spec.name,
        batch=batch,
        backend=leg_name,
        arm_requested=arm,
        arm_observed=observed,
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
        roofline=roofline.fractions(bytes_per_s, ops_per_s, peaks.memory, arith),
        method=dict(dataclasses.asdict(measurement.method), observation=observation),
        machine=machine_record,
        reference=reference,
    )


_FRX_OBSERVATION = (
    "the arm and compile_ns were read from a second lowering of the same "
    "computation; the timed calls go through the jit cache"
)

_REFERENCE_OBSERVATION = (
    "the arm is the pinned implementation named in `reference`, and the kernel "
    "is the symbol resolved out of its shared object; there is no compile step "
    "at run time, so compile_ns is null"
)


def _frx_rows(
    leg: backends.Leg,
    arm: arms.Arm,
    specs: list[registry.HashSpec],
    batches: tuple[int, ...],
    method: timing.Method,
    peaks: roofline.Peaks,
    machine_record: dict[str, Any],
) -> None:
    """The rows one hash-frx lowering produces on this leg."""
    for spec in specs:
        for batch in batches:
            call = spec.call(batch)
            started = time.perf_counter_ns()
            compiled = call.fn.lower(call.x).compile()
            compile_ns = time.perf_counter_ns() - started
            observed, callees = arms.observe(compiled.as_text())
            row = _row(
                spec=spec,
                batch=batch,
                leg_name=leg.name,
                arm=arm.value,
                call=call,
                compile_ns=compile_ns,
                measurement=timing.measure(call.fn, call.x, method),
                observed=observed.arm_name,
                callees=callees,
                peaks=peaks,
                machine_record=machine_record,
                observation=_FRX_OBSERVATION,
            )
            results.write(row, sys.stdout)


def _reference_rows(
    leg: backends.Leg,
    ref: references.Reference,
    selected: tuple[str, ...],
    batches: tuple[int, ...],
    method: timing.Method,
    peaks: roofline.Peaks,
    machine_record: dict[str, Any],
) -> None:
    """The rows one pinned reference produces on this leg.

    `arm_observed` equals `arm_requested` here, and unlike on a hash-frx arm
    that is not an assumption: the shared object either resolves the symbol or
    the worker fails, so there is no silent substitution for the row to have to
    report. What the row does have to carry is the provenance, which is the
    reference's equivalent of the revisions a hash-frx row reads off the wheels.
    """
    provenance = ref.provenance.to_json()
    method = method.synchronous()
    for spec in references.rows(ref, selected):
        for batch in batches:
            call = ref.call(spec, batch)
            row = _row(
                spec=spec,
                batch=batch,
                leg_name=leg.name,
                arm=ref.name,
                call=call,
                compile_ns=None,
                measurement=timing.measure(
                    call.fn, call.x, method, block=timing.block_none
                ),
                observed=ref.name,
                callees=(ref.symbol(spec.name),),
                peaks=peaks,
                machine_record=machine_record,
                observation=_REFERENCE_OBSERVATION,
                reference=provenance,
            )
            results.write(row, sys.stdout)


def work(args: argparse.Namespace) -> int:
    """One (backend leg, arm), in the environment the orchestrator built for it."""
    if len(args.backend) != 1 or len(args.arm) != 1:
        raise SystemExit(
            "--worker runs exactly one --backend and one --arm; the orchestrator "
            "fans out the rest, because the flags each arm needs are read when "
            "the plugin loads"
        )
    leg = backends.get(args.backend[0])
    arm_name = args.arm[0]
    if leg.one_core:
        backends.pin_to_one_core()

    method = timing.Method(
        warmup=args.warmup, reps=args.reps, target_rep_ns=args.target_rep_ns
    )
    selected = tuple(args.hash)
    specs = list(registry.rows(selected))
    if args.probe_peaks:
        units = [c for c in (spec.arith_ceiling() for spec in specs) if c is not None]
        print(json.dumps(roofline.measure_peaks(units, method).to_json()))
        return 0

    if not args.peaks:
        raise SystemExit("--worker needs --peaks; the orchestrator probes per leg")
    peaks = roofline.Peaks.from_json(json.loads(args.peaks))
    machine_record = machine.describe()
    batches = tuple(args.batch) or DEFAULT_BATCHES
    if references.is_reference(arm_name):
        _reference_rows(
            leg=leg,
            ref=references.get(arm_name),
            selected=selected,
            batches=batches,
            method=method,
            peaks=peaks,
            machine_record=machine_record,
        )
    else:
        _frx_rows(
            leg=leg,
            arm=arms.Arm(arm_name),
            specs=specs,
            batches=batches,
            method=method,
            peaks=peaks,
            machine_record=machine_record,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return work(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
