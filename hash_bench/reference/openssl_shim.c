// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
//
// OpenSSL's SHA-256 behind the harness's call ABI.
//
// The SHA-NI path is not a compile flag. libcrypto carries every SHA-256 core
// it has and picks one from CPUID at run time, so "the SHA-NI path" is a claim
// about the machine rather than about the build — which is why
// `hash_bench_openssl_capabilities` reports the CPU bit that branch tests and
// `references.py` puts the answer in the row.
//
// `SHA256()` rather than the EVP interface, and not for brevity. EVP reaches
// the same assembly core through a provider and wraps every digest in
// per-message bookkeeping — an algorithm reference and a provider context —
// that is no part of SHA-256. The row is meant to be OpenSSL's SHA-256, not
// OpenSSL's dispatch machinery, so it calls the function that is only the hash.
// `SHA256` is deprecated in the 3.0 API and still the whole of what it always
// was; suppressing the deprecation is the cost of measuring it.

#define OPENSSL_SUPPRESS_DEPRECATED

#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <openssl/crypto.h>
#include <openssl/sha.h>

#include "hash_bench/reference/parallel.h"

void hash_bench_openssl_sha256(const uint8_t *msg, size_t msg_bytes,
                               uint8_t *out, size_t batch) {
#pragma omp parallel for schedule(static) if (HASH_BENCH_GO_WIDE(batch))
  for (ptrdiff_t i = 0; i < (ptrdiff_t)batch; ++i) {
    SHA256(msg + (size_t)i * msg_bytes, msg_bytes, out + (size_t)i * 32);
  }
}

const char *hash_bench_openssl_capabilities(void) {
  // libcrypto exports no accessor for the core it picked — `sha256_block_data
  // _order` is one symbol that branches on CPUID internally — so this reports
  // the CPU bit that branch tests. `sha_ni=1` is therefore the condition for
  // the SHA-NI core, read from the same feature word, not a readback of the
  // choice. `sha_ni=0` is the useful direction: it says outright that this row
  // measured OpenSSL's fallback.
  static char described[128];
  snprintf(described, sizeof(described), "%s, cpu sha_ni=%d, cpu avx512f=%d",
           OpenSSL_version(OPENSSL_VERSION_STRING),
           __builtin_cpu_supports("sha") ? 1 : 0,
           __builtin_cpu_supports("avx512f") ? 1 : 0);
  return described;
}
