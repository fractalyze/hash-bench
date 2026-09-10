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
// The threshold is a batch count rather than a byte count because what a
// region amortises its opening cost over is iterations, and it sits where that
// cost stays small beside the batch's hashing even on the cheapest hash the
// registry carries.

#ifndef HASH_BENCH_REFERENCE_PARALLEL_H_
#define HASH_BENCH_REFERENCE_PARALLEL_H_

#define HASH_BENCH_PARALLEL_MIN 256
#define HASH_BENCH_GO_WIDE(batch) ((batch) >= HASH_BENCH_PARALLEL_MIN)

#endif  // HASH_BENCH_REFERENCE_PARALLEL_H_
