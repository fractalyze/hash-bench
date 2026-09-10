// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
//
// When a reference shim spreads its batch across cores.
//
// Every shim parallelises so that the `cpu` leg means the same thing on both
// sides of a comparison: hash-frx's row reaches every core through the XLA
// thread pool, and a serial reference would report this machine's one-core
// number in the all-core row.
//
// Below a threshold that inverts. Opening a parallel region costs more than a
// handful of hashes are worth, and a reference that paid it unconditionally
// would lose the small batches to its own thread team rather than to its hash —
// a strawman, not a frontier. So the shims hand OpenMP an `if` clause, which
// runs the identical loop on a team of one. XLA makes the same call by batch
// size; the harness's job is to compare two implementations each doing the
// sensible thing.
//
// The threshold is where the two costs cross on the fastest hash measured here:
// Keccak-f[1600] is ~190 ns per permutation on one core, and a region costs a
// few microseconds to open, so a few dozen hashes pay for it and 256 clears it
// with room to spare. It is a batch count rather than a byte count because what
// a region amortises over is iterations.

#ifndef HASH_BENCH_REFERENCE_PARALLEL_H_
#define HASH_BENCH_REFERENCE_PARALLEL_H_

#define HASH_BENCH_PARALLEL_MIN 256
#define HASH_BENCH_GO_WIDE(batch) ((batch) >= HASH_BENCH_PARALLEL_MIN)

#endif  // HASH_BENCH_REFERENCE_PARALLEL_H_
