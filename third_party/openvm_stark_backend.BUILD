# OpenVM's stark-backend — its KoalaBear Poseidon2 CUDA kernel, as the GPU
# reference hash-bench measures `poseidon2-koalabear16` against.
#
# Headers only. The permutation is `kb_poseidon2::poseidon2_mix` in
# `benchmarks/fields/cuda/include/koala_bear/poseidon2_kb.cuh`, over the field in
# `koala_bear/kb.h`, whose Montgomery core is `ff/mont32_t.cuh` from
# `crates/cuda-common`. The upstream compiles these through cargo and its own
# `cuda-builder`; the shim includes them into its own translation unit, so the
# two include roots that build uses are all this target has to provide.

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "poseidon2_koalabear",
    hdrs = glob([
        "benchmarks/fields/cuda/include/**/*.cuh",
        "benchmarks/fields/cuda/include/**/*.h",
        "benchmarks/fields/cuda/include/**/*.hpp",
        "crates/cuda-common/include/**/*.cuh",
        "crates/cuda-common/include/**/*.h",
        "crates/cuda-common/include/**/*.hpp",
    ]),
    includes = [
        "benchmarks/fields/cuda/include",
        "crates/cuda-common/include",
    ],
    visibility = ["//visibility:public"],
)
