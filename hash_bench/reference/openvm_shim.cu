// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
//
// OpenVM's KoalaBear Poseidon2 behind the harness's device-buffer ABI.
//
// The permutation is the upstream's `kb_poseidon2::poseidon2_mix`, with the
// constant tables `third_party/openvm_poseidon2_kb_plonky3_0_7_0.patch` puts in
// its header. The kernel around it is the upstream benchmark's
// (`benchmarks/fields/cuda/src/poseidon2_bench.cu`): one thread per state, its
// block size, the state loaded into registers and stored back. It differs in
// two ways, both to meet the call shape every row has: it reads one buffer and
// writes another instead of permuting in place, and it permutes once instead
// of `reps` times.
//
// The state crosses as raw Montgomery words. `Kb` is `mont32_t` with R = 2^32,
// the representation of Plonky3's `MontyField31` and of hash-frx's
// `koalabear_mont`, so the upstream's canonical-to-Montgomery init kernel is not
// run. `testing/references_test.py` is what checks that the two agree.

#include <cstddef>
#include <cstdint>

#include "hash_bench/reference/gpu.h"
// Last: the field header redefines `inline` for device code.
#include "koala_bear/poseidon2_kb.cuh"

namespace {

constexpr int kWidth = 16;
// The upstream benchmark's P2_BLOCK_SIZE.
constexpr int kBlockSize = 512;

__global__ void permute(const Kb *in, Kb *out, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  Kb cells[kWidth];
  size_t base = idx * kWidth;

#pragma unroll
  for (int i = 0; i < kWidth; i++) {
    cells[i] = in[base + i];
  }

  kb_poseidon2::poseidon2_mix(cells);

#pragma unroll
  for (int i = 0; i < kWidth; i++) {
    out[base + i] = cells[i];
  }
}

}  // namespace

extern "C" int hash_bench_openvm_poseidon2_koalabear16(const void *in,
                                                       void *out,
                                                       size_t batch) {
  if (batch == 0) return cudaSuccess;
  cudaStream_t stream;
  cudaError_t error = hash_bench_gpu_stream(&stream);
  if (error != cudaSuccess) return error;
  unsigned grid = static_cast<unsigned>((batch + kBlockSize - 1) / kBlockSize);
  permute<<<grid, kBlockSize, 0, stream>>>(static_cast<const Kb *>(in),
                                          static_cast<Kb *>(out), batch);
  return cudaGetLastError();
}

extern "C" const char *hash_bench_openvm_capabilities(void) {
  return hash_bench_gpu_describe(HASH_BENCH_NVCC_BUILD);
}
