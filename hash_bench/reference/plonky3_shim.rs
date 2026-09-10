// Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
//
// Plonky3's Poseidon2 over KoalaBear behind the harness's call ABI.
//
// The state crosses the boundary as raw Montgomery-form `u32`s, reinterpreted
// rather than converted. `MontyField31` is `#[repr(transparent)]` over its
// `u32` and asserts `MONTY_BITS == 32`, and hash-frx's `koalabear_mont` is the
// same representation, so the two agree byte for byte and a conversion would be
// two changes of basis that cancel. That the representations really do agree is
// not asserted here: `testing/references_test.py` runs both sides over one
// input and compares the outputs, which is the only check that can see it.
//
// Which SIMD backend this reaches is a build flag rather than a run-time
// choice — `p3-koala-bear` selects its packed field on `target_feature` — so
// `hash_bench_plonky3_capabilities` reports what the compiler actually enabled,
// and a build whose flags did not land says `scalar` instead of claiming AVX.

use std::ffi::c_char;
use std::sync::OnceLock;

use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear, default_koalabear_poseidon2_16};
use p3_symmetric::Permutation;
use rayon::prelude::*;

const WIDTH: usize = 16;

// The threshold `reference/parallel.h` states for the C shims, repeated because
// a header cannot cross into Rust. Same rule, same reason: below it a thread
// team costs more than the hashes it would spread.
const PARALLEL_MIN: usize = 256;

/// The permutation, built once. Its round constants are fixed by the parameter
/// set, so rebuilding it per call would measure the constant tables.
fn permutation() -> &'static Poseidon2KoalaBear<WIDTH> {
    static PERMUTATION: OnceLock<Poseidon2KoalaBear<WIDTH>> = OnceLock::new();
    PERMUTATION.get_or_init(default_koalabear_poseidon2_16)
}

fn permute_all(states: &mut [KoalaBear]) {
    let permutation = permutation();
    let apply = |state: &mut [KoalaBear]| {
        let state: &mut [KoalaBear; WIDTH] = state.try_into().expect("width-16 chunk");
        permutation.permute_mut(state);
    };
    if states.len() / WIDTH >= PARALLEL_MIN {
        states.par_chunks_exact_mut(WIDTH).for_each(apply);
    } else {
        states.chunks_exact_mut(WIDTH).for_each(apply);
    }
}

/// # Safety
///
/// `input` and `output` must each address `batch * 16` readable (respectively
/// writable) `u32`s and must not overlap. `references.py` allocates both.
#[no_mangle]
pub unsafe extern "C" fn hash_bench_plonky3_poseidon2_koalabear16(
    input: *const u32,
    output: *mut u32,
    batch: usize,
) {
    let elements = batch * WIDTH;
    // Out of place, so the row's traffic is the read plus the write its
    // hash-frx counterpart also pays.
    let source = unsafe { std::slice::from_raw_parts(input as *const KoalaBear, elements) };
    let states = unsafe { std::slice::from_raw_parts_mut(output as *mut KoalaBear, elements) };
    states.copy_from_slice(source);
    permute_all(states);
}

#[no_mangle]
pub extern "C" fn hash_bench_plonky3_capabilities() -> *const c_char {
    // What `p3-koala-bear`'s own `cfg` gates saw, evaluated here under the same
    // flags, so this names the backend that was compiled in rather than the one
    // the build meant to ask for.
    let described: &[u8] = if cfg!(all(target_arch = "x86_64", target_feature = "avx512f")) {
        b"packed field: x86_64_avx512\0"
    } else if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        b"packed field: x86_64_avx2\0"
    } else {
        b"packed field: scalar\0"
    };
    described.as_ptr() as *const c_char
}
