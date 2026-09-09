# hash-bench

One harness, one machine: hash-frx's routed, declined, and generic (`static_while`)
lowerings run beside pinned external reference implementations, reporting ns/hash
per (hash, batch, backend) and the fraction of the memory or issue roofline, with
the method and every revision recorded.

References are pinned as external dependencies at fixed revisions, not vendored
copies:

- CPU: XKCP, the official BLAKE3 implementation, OpenSSL (SHA-NI path), Plonky3
- GPU: ICICLE, sppark, Plonky3-CUDA, hashcat where it applies

The harness is the gate for every hand-written hash emitter deletion in hash-frx
and xla: an emitter goes only after the generic path measures at or above it at
every batch on both backends.

The external references above are rows of the same shape, measured by the same
driver against the same ceilings; `hash_bench/registry.py` is the list of rows
that exist.

## Running it

```sh
bazel run //hash_bench:sweep -- --out results/$(hostname).jsonl
```

That is the whole sweep: every hash in the registry, at every batch, on every
backend leg this machine has, through all three arms. Narrow it with `--hash`,
`--batch`, `--backend` and `--arm`, each repeatable:

```sh
bazel run //hash_bench:sweep -- --hash poseidon2-koalabear16 --backend gpu \
    --arm routed --arm generic --batch 65536
```

Rows are JSON Lines, one per (hash, batch, backend, arm), on stdout — so
`... > results/rows.jsonl` is a complete run and `--out` appends to a file at
the same time. The summary table goes to stderr, and `--quiet` drops it.
`results/README.md` documents every field; the short version is that a row is
self-contained, and `arm_observed` — not `arm_requested` — says what ran.

`bazel test //...` is the harness's own suite, and it runs on CPU only: it
asserts that rows are produced, classified and modelled correctly, and it
measures nothing anyone should quote. GPU routing is exercised by running the
sweep, not by the suite.

## The three arms

Each arm is one hash's one-call unit reaching the machine a different way. They
are selected per run, and each is **verified from the compiled module** rather
than assumed, because a marker no pass claims inlines silently and still
computes the right bytes.

| arm | what claims the region | what the module shows |
|---|---|---|
| `routed` | the plugin's dedicated emitter | a kCustom fusion named for the hash |
| `generic` | the `while`-to-`for` conversion over hash-frx's own decomposition | one `static_while` kCustom fusion |
| `declined` | nothing | ordinary fusions, no kCustom |

The arms are requested by disabling compiler passes (`hash_bench/arms.py`), so
the module frx emits is identical across all three and only the emitter changes
— which is what a pin without that emitter does.

Whether a (hash, backend) pair can reach an arm is a property of the pin. Where
it cannot, the row records the arm that actually ran and flags the substitution
instead of relabelling the number.

## Adding a hash

One entry in `hash_bench/registry.py`. The row names the hash and how to build
it; traffic is read off the arrays its call actually moves, and an arithmetic
model — optional, and computed from the primitive's own parameters rather than
written down — earns the row an arithmetic roofline alongside the memory one.

## The pins

`MODULE.bazel` pins hash-frx at a commit and the frx wheels (which carry the
Fractalyze XLA plugin) through `requirements_lock_3_11.txt`. That lock is copied
from the pinned hash-frx commit rather than compiled here, so both modules
resolve one frx and one zk_dtypes into a runfiles tree; `requirements.in`
carries the reason. Bumping the commit means copying that commit's lock over
ours.
