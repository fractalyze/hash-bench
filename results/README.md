# results/

One JSON Lines file per sweep, named for the machine that produced it —
`results/<host>.jsonl`, or `results/<host>-<what-changed>.jsonl` when the same
host is measured under two pins. A file is append-only: re-running the sweep
adds rows rather than replacing them, and the revisions inside each row are what
separates one run from the next.

Nothing in a row refers to its file. A single line can be quoted on its own and
still says what it measured, on what, and how.

## Reading a row

```
hash              the registry row: a permutation name, or a byte hash at the
                  message length its row declares
batch             hashes per call — the axis a permutation is vmapped over, and
                  the leading axis of a byte hash's message array
backend           cpu-1core | cpu | gpu   (see `hash_bench/backends.py`)
arm_requested     what the run asked for: one of hash-frx's lowerings
                  (routed | generic | declined) or a pinned external reference
                  (`hash_bench/references.py`)
arm_observed      what actually ran — the lowering the COMPILED MODULE showed,
                  or, on a reference arm, the reference itself
as_requested      arm_observed == arm_requested
kernels           the evidence arm_observed was read from: the custom-fusion
                  computations the module contained, or the symbol resolved out
                  of the reference's shared object
```

**Read `arm_observed`, not `arm_requested`.** A marker no pass claims inlines
silently and still computes the right bytes, so a request is not evidence that
the arm ran. A pin does not owe every arm on every backend — a hash-frx family
names the backends each of its emitters was written for — so where the two
differ, the number belongs to `arm_observed` and the requested arm did not run.

A reference arm's `arm_observed` always equals its `arm_requested`, and unlike
on a hash-frx arm that is not a claim being made: the shared object either
resolves the symbol or the worker fails, so there is no substitution the row
could have to report.

```
compile_ns        wall time to lower and compile the call once. Compile cost is
                  an axis the arms differ on — a declined region inlines its
                  whole round schedule — so an arm that wins at run time while
                  costing more to compile is a different trade, not a free one.
                  Null on a reference arm, which compiles nothing at run time —
                  null rather than zero, which would read as a free compile
ns_per_hash       the reported number: the median call time divided by batch
ns_per_hash_min   the fastest rep, same division — the noise floor of this row
spread            (max - min) / median across reps. A large spread is a row
                  whose median is not worth quoting, visible without the reps
hashes_per_s      batch / call time
```

```
bytes_moved       bytes one call reads and writes at the call boundary, from the
                  actual input and output arrays — not a model
bytes_per_s       bytes_moved / call time
ops_per_hash      the dominant arithmetic operation per hash, where the row
                  declares a model; null where none is declared
ops_unit          what ops_per_hash counts (`field_mul`), because the ceiling it
                  is compared against is measured in the same unit
ops_note          how the count was derived from the primitive's parameters
```

```
roofline.memory_fraction   bytes_per_s / peak_bytes_per_s
roofline.arith_fraction    ops_per_s / peak_ops_per_s, or null
roofline.bound             which ceiling the row is nearer, and so which one to
                           hold it to
roofline.*_probe           what each ceiling was measured with
```

Both ceilings are **measured on the machine that ran the sweep**, not read off
a datasheet, and they are two different kinds of roof. The memory probe streams
an array no cache holds, so it stands for this machine's DRAM bandwidth. The
arithmetic probe is a multiply chain compiled by frx, so it is frx's multiply
rate — a compiler's roof, not the hardware's — and a native reference can run
above it. A row that exceeds a probe gets no `bound` from it: `bound` reads
`none (...)` and names the probe it outran, and both fractions are still
reported, because a verdict read off a ceiling the row is already above would
hold it to a roof that is not one. A row whose
`bound` reads `memory (no arithmetic model declared for this hash)` has only one
candidate ceiling because nobody has written down what that hash costs; it is
not a claim that the hash is memory-bound. Declaring a model is one field on the
registry row.

```
reference         null on a hash-frx arm. On a reference arm, the pinned
                  implementation this row measured:

                    revision          the tag or commit `MODULE.bazel` pins
                    source            the upstream repository
                    implementation    which of the upstream's implementations
                                      the build selected
                    library           the shared object the row was measured
                                      through
                    compilation_mode  the Bazel mode the whole reference was
                                      built under — the shim and every
                                      upstream target beneath it
                    flags             a C shim's own copts and linkopts, the
                                      rustc flags the Rust reference applies
                                      to its shim and every crate under it
                                      (which is what decides the upstream's
                                      packed field), or a CUDA shim's nvcc
                                      flags and the rules_cuda architectures
                                      setting it was built under. Generated
                                      from the same values the build applied
                    patches           the patches applied to the pinned
                                      upstream before it compiled; empty for
                                      an upstream measured as released
                    runtime_dispatch  what the loaded library reports about
                                      itself, where it offers an answer — some
                                      upstreams pick their kernel from CPUID
                                      rather than from a flag, so the flags
                                      alone do not say what ran; a CUDA shim
                                      reports the nvcc and architectures its
                                      kernel was compiled with and the device,
                                      driver and runtime it found. Null where
                                      the selection is entirely a build-time
                                      choice
                    note              a caveat a reader of the row needs
```

`reference` is to a reference row what `machine.revisions` is to a hash-frx row:
without it the number names no implementation and is comparable to nothing.

```
method            warmup, reps, iters, the statistic, how calls were dispatched,
                  and the timer — enough to re-run the same measurement
machine           host, OS, CPU model, core count, the affinity mask in effect,
                  the backend and its devices, the environment knobs
machine.revisions every version that decides what the number means
```

`machine.revisions["frx-cuda12-plugin"]` **is** the Fractalyze XLA revision. A
row measured against a different one is not comparable to this one however alike
the machines are.

hash-frx takes two entries, and which one names the commit that ran depends on
how it arrived. `hash-frx-pin` is the commit `MODULE.bazel` pins and is always
present. `hash-frx-sha` is filled only when hash-frx came from a git checkout on
`sys.path` — a local dev run — and is then the commit that ran; a
`git_override`-fetched module has no `.git` to read, so a pinned run leaves it
null and `hash-frx-pin` is the answer.

## What a comparison needs

Two rows are comparable when they agree on `hash`, `batch`, `backend`,
`machine.host` and every entry of `machine.revisions`. That is the whole
comparison contract: everything else in the row is either the measurement or the
evidence behind it.

A reference row joins that contract on the same terms, with `reference.revision`
standing in for the hash-frx entries of `machine.revisions` — its number is a
property of that revision built with those flags, and re-pinning either produces
a row that is not comparable to this one. Both sides are timed by the same
`timing.measure` and divided by the same leg `Peaks`. On a CPU reference the one
field that differs is `method.dispatch`, because a native call returns complete
and has no queue to block on; a CUDA reference launches asynchronously, as
hash-frx does on the GPU, and its method matches the hash-frx rows' exactly,
with a stream synchronize as the block.

That holds only for rows from one sweep. Each invocation probes its own ceilings
per leg, and run-to-run drift on a shared machine is neither small nor uniform,
so a reference row from one run over a hash-frx row from another compares two
machine states and reports the difference as a property of the code. Rows from
one sweep carry a leg's `roofline.peak_bytes_per_s` exactly equal; two rows that
differ there came from different sweeps and are not a share of one ceiling,
whatever their revisions say.

The gate this harness exists for — deleting a hand-written emitter — is the
`generic` row meeting or beating the `routed` row at **every** batch on **both**
backends, with `as_requested` true on both sides.
