// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0

#include "hash_bench/reference/gpu.h"

#include <cstdio>

cudaError_t hash_bench_gpu_stream(cudaStream_t *out) {
  // Created on first use rather than at load, so loading the library creates no
  // CUDA context in a process that never calls it. Non-blocking so that the
  // plugin's own streams, in a worker that also started frx, cannot serialise
  // against it.
  static cudaStream_t stream = nullptr;
  static cudaError_t status =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  *out = stream;
  return status;
}

const char *hash_bench_gpu_describe(const char *build) {
  static char described[512];
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    snprintf(described, sizeof described, "%s; no CUDA device", build);
    return described;
  }
  int driver = 0;
  int runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  snprintf(described, sizeof described,
           "%s; %s (sm_%d%d), driver %d.%d, runtime %d.%d", build, prop.name,
           prop.major, prop.minor, driver / 1000, driver % 1000 / 10,
           runtime / 1000, runtime % 1000 / 10);
  return described;
}

namespace {

// Both copies go through the reference's stream and wait for it: that stream
// does not order against the legacy default one a plain cudaMemcpy uses, so a
// plain copy could race a kernel still running on it.
cudaError_t copy_and_wait(void *dst, const void *src, size_t bytes,
                          cudaMemcpyKind kind) {
  cudaStream_t stream;
  cudaError_t error = hash_bench_gpu_stream(&stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  }
  return error != cudaSuccess ? error : cudaStreamSynchronize(stream);
}

}  // namespace

extern "C" {

int hash_bench_gpu_device_count(int *count) {
  return cudaGetDeviceCount(count);
}

int hash_bench_gpu_alloc(void **ptr, size_t bytes) {
  return cudaMalloc(ptr, bytes);
}

int hash_bench_gpu_free(void *ptr) { return cudaFree(ptr); }

int hash_bench_gpu_upload(void *dst, const void *src, size_t bytes) {
  return copy_and_wait(dst, src, bytes, cudaMemcpyHostToDevice);
}

int hash_bench_gpu_download(void *dst, const void *src, size_t bytes) {
  return copy_and_wait(dst, src, bytes, cudaMemcpyDeviceToHost);
}

int hash_bench_gpu_synchronize(void) {
  cudaStream_t stream;
  cudaError_t error = hash_bench_gpu_stream(&stream);
  return error != cudaSuccess ? error : cudaStreamSynchronize(stream);
}

const char *hash_bench_gpu_error_string(int error) {
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}

}  // extern "C"
