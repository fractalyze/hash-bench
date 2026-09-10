// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
//
// XKCP behind the harness's call ABI: a batch in, a batch out, one OpenMP
// parallel loop over the batch.
//
// Out-of-place, and that is the point rather than a convenience. The hash-frx
// row this is compared against reads an input array and writes an output array,
// and `registry` counts the traffic off those two arrays; a shim that permuted
// in place would move half the bytes and be reported against the same memory
// ceiling.
//
// The parallel loop is what makes the `cpu` leg mean the same thing on both
// sides: hash-frx's row reaches every core through the XLA thread pool, so a
// serial reference would report this machine's one-core number in the all-core
// row. OpenMP reads `OMP_NUM_THREADS`, which `backends.LEGS` already sets to 1
// on the one-core leg, so the two legs need nothing else here.

#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hash_bench/reference/parallel.h"

#include "KeccakP-1600-SnP.h"
#include "SimpleFIPS202.h"

// One Keccak-f[1600] state is 25 lanes of 64 bits. hash-frx carries a lane as
// an adjacent `(lo, hi)` uint32 pair (`hash_frx/keccak/lane.py` — the toolchain
// cannot hold a uint64 safely), and on a little-endian machine that pair is the
// same 8 bytes as the uint64 XKCP reads. So the two layouts are bit-identical
// and the batch is reinterpreted rather than converted.
#define KECCAK_STATE_BYTES 200

void hash_bench_xkcp_keccak_f1600(const uint8_t *in, uint8_t *out,
                                  size_t batch) {
#pragma omp parallel for schedule(static) if (HASH_BENCH_GO_WIDE(batch))
  for (ptrdiff_t i = 0; i < (ptrdiff_t)batch; ++i) {
    uint8_t *state = out + (size_t)i * KECCAK_STATE_BYTES;
    memcpy(state, in + (size_t)i * KECCAK_STATE_BYTES, KECCAK_STATE_BYTES);
    KeccakP1600_Permute_24rounds((KeccakP1600_state *)state);
  }
}

void hash_bench_xkcp_sha3_256(const uint8_t *msg, size_t msg_bytes,
                              uint8_t *out, size_t batch) {
#pragma omp parallel for schedule(static) if (HASH_BENCH_GO_WIDE(batch))
  for (ptrdiff_t i = 0; i < (ptrdiff_t)batch; ++i) {
    SHA3_256(out + (size_t)i * 32, msg + (size_t)i * msg_bytes, msg_bytes);
  }
}
