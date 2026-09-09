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
arm_requested     the lowering the run asked for: routed | generic | declined
arm_observed      the lowering the COMPILED MODULE showed
as_requested      arm_observed == arm_requested
kernels           the custom-fusion computations the module contained; this is
                  the evidence arm_observed was read from
```

**Read `arm_observed`, not `arm_requested`.** A marker no pass claims inlines
silently and still computes the right bytes, so a request is not evidence that
the arm ran. A pin does not owe every arm on every backend — a hash-frx family
names the backends each of its emitters was written for — so where the two
differ, the number belongs to `arm_observed` and the requested arm did not run.

```
compile_ns        wall time to lower and compile the call once. Compile cost is
                  an axis the arms differ on — a declined region inlines its
                  whole round schedule — so an arm that wins at run time while
                  costing more to compile is a different trade, not a free one
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

Both ceilings are **measured on the machine that ran the sweep**, through the
same frx and the same plugin as the rows, so a fraction is a fraction of what
this hardware and this compiler reach — not of a datasheet. A row whose
`bound` reads `memory (no arithmetic model declared for this hash)` has only one
candidate ceiling because nobody has written down what that hash costs; it is
not a claim that the hash is memory-bound. Declaring a model is one field on the
registry row.

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

The gate this harness exists for — deleting a hand-written emitter — is the
`generic` row meeting or beating the `routed` row at **every** batch on **both**
backends, with `as_requested` true on both sides.
