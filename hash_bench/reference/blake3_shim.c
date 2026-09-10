// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
//
// The BLAKE3 team's own C implementation behind the harness's call ABI.
//
// The upstream picks its SIMD kernel by CPUID at run time rather than by a
// compile flag, so what this row measured is not fully described by the copts
// the build records. `hash_bench_blake3_capabilities` reports back what the
// loaded library actually is, and `references.py` puts it in the row.

#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "c/blake3.h"

#include "hash_bench/reference/parallel.h"

void hash_bench_blake3(const uint8_t *msg, size_t msg_bytes, uint8_t *out,
                       size_t batch) {
#pragma omp parallel for schedule(static) if (HASH_BENCH_GO_WIDE(batch))
  for (ptrdiff_t i = 0; i < (ptrdiff_t)batch; ++i) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, msg + (size_t)i * msg_bytes, msg_bytes);
    blake3_hasher_finalize(&hasher, out + (size_t)i * BLAKE3_OUT_LEN,
                           BLAKE3_OUT_LEN);
  }
}

const char *hash_bench_blake3_capabilities(void) {
  // The version the linked library reports, which is the one fact about the
  // run-time selection the upstream exposes; the SIMD kernel it dispatches to
  // has no public accessor.
  static char described[64];
  snprintf(described, sizeof(described), "blake3_version=%s", blake3_version());
  return described;
}
