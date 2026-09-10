# XKCP — the Keccak team's own Keccak-f[1600] and FIPS 202, as the reference
# hash-bench measures Keccak against.
#
# XKCP's own build generates a target-specific `config.h` and a bundle from XML
# target descriptions, which is a build system this workspace would have to run
# under `rules_foreign_cc` to reach one permutation. Naming the files directly
# is both smaller and more precise about what is being measured: the AVX-512
# assembly below is a choice recorded in the row, not whatever a target
# description happened to select on this host.
#
# `KeccakP-1600-AVX512.s` is hand-written assembly with no runtime dispatch, so
# it is the whole reason this target carries a `target_compatible_with`: an
# x86-64 machine without AVX-512 links it and faults on the first permutation.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# XKCP's generated header, which only its makefile writes. Every module below
# tests these to decide which implementations exist; the set here is exactly
# what the FIPS 202 path over KeccakP-1600 needs.
genrule(
    name = "config_h",
    outs = ["config.h"],
    cmd = """cat > $@ <<'CONFIG'
#ifndef _XKCP_config_h_
#define _XKCP_config_h_
#define XKCP_has_KeccakP1600
#define XKCP_has_Sponge_Keccak_width1600
#define XKCP_has_FIPS202
#endif
CONFIG""",
)

cc_library(
    name = "keccak",
    srcs = [
        "lib/high/Keccak/FIPS202/KeccakHash.c",
        "lib/high/Keccak/FIPS202/SimpleFIPS202.c",
        "lib/high/Keccak/KeccakSponge.c",
        "lib/low/KeccakP-1600/AVX512/KeccakP-1600-AVX512.s",
    ],
    hdrs = [
        "config.h",
        "lib/common/PlSnP-common.h",
        "lib/common/SnP-common.h",
        "lib/common/align.h",
        "lib/common/brg_endian.h",
        "lib/common/load-store.h",
        "lib/high/Keccak/FIPS202/KeccakHash.h",
        "lib/high/Keccak/FIPS202/SimpleFIPS202.h",
        "lib/high/Keccak/KeccakSponge.h",
        "lib/high/Keccak/KeccakSponge.inc",
        "lib/low/KeccakP-1600/AVX512/KeccakP-1600-AVX512.h",
        "lib/low/KeccakP-1600/AVX512/SnP/KeccakP-1600-SnP.h",
        "lib/low/KeccakP-1600/plain-64bits/KeccakP-1600-plain64.h",
    ],
    includes = [
        ".",
        "lib/common",
        "lib/high/Keccak",
        "lib/high/Keccak/FIPS202",
        "lib/low/KeccakP-1600/AVX512",
        "lib/low/KeccakP-1600/AVX512/SnP",
        "lib/low/KeccakP-1600/plain-64bits",
    ],
    target_compatible_with = ["@platforms//cpu:x86_64"],
)
