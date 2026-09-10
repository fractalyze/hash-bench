// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
//
// The device-buffer ABI every CUDA reference exports, and what it reports.
//
// A CUDA reference's input and output live on the device, as a hash-frx GPU
// row's do, so `references.py` allocates, fills and reads them through these
// entry points instead of handing the shim host memory: a row that copied its
// batch across PCIe inside the timed region would be measuring the bus.
//
// Every launch goes onto one stream, and `hash_bench_gpu_synchronize` drains
// it. That is the reference's `frx.block_until_ready`, so a rep launches
// `iters` kernels and blocks once, the way the hash-frx GPU rows are timed.
//
// Each shim links this statically and is loaded without RTLD_GLOBAL, so the
// `hash_bench_gpu_` names every CUDA shim shares cannot collide between two
// references in one worker. Every `int` return is a `cudaError_t`.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>

// The stream every launch and copy goes onto, created on first use. The
// creation status comes back with it, so a failure surfaces at the first call
// that needs the stream instead of every call quietly running on the legacy
// default stream.
cudaError_t hash_bench_gpu_stream(cudaStream_t *stream);

// "<build>; <device> (sm_XY), driver X.Y, runtime X.Y". `build` is the caller's
// HASH_BENCH_NVCC_BUILD, expanded in the kernel's own translation unit so that
// it names the compiler and the architectures that built that kernel.
const char *hash_bench_gpu_describe(const char *build);

#define HASH_BENCH_STR_(x) #x
#define HASH_BENCH_STR(x) HASH_BENCH_STR_(x)
#define HASH_BENCH_NVCC_BUILD                                            \
  "nvcc " HASH_BENCH_STR(__CUDACC_VER_MAJOR__) "." HASH_BENCH_STR(       \
      __CUDACC_VER_MINOR__) "." HASH_BENCH_STR(__CUDACC_VER_BUILD__)     \
      ", compiled for " HASH_BENCH_STR(__CUDA_ARCH_LIST__)

extern "C" {
int hash_bench_gpu_device_count(int *count);
int hash_bench_gpu_alloc(void **ptr, size_t bytes);
int hash_bench_gpu_free(void *ptr);
int hash_bench_gpu_upload(void *dst, const void *src, size_t bytes);
int hash_bench_gpu_download(void *dst, const void *src, size_t bytes);
int hash_bench_gpu_synchronize(void);
const char *hash_bench_gpu_error_string(int error);
}
