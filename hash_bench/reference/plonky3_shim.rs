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
//
// The permutation runs over `[KoalaBear::Packing; 16]`, which is the form
// `Poseidon2KoalaBear`'s own documentation says to use "wherever possible", and
// the only one that reaches the packed field at all: handing it
// `[KoalaBear; 16]` compiles and computes the right state one permutation at a
// time — a scalar program, which is not what this reference is for.
//
// Reaching it costs a transpose. A packed lane holds element `j` of `WIDTH`
// DIFFERENT states, while the harness's array is one state per row, so each
// group is gathered on the way in and scattered on the way out. That work is
// inside the timed region deliberately: the row's input and output layouts are
// the ones its hash-frx counterpart is handed, and what an implementation does
// between them is the thing being measured.

use std::ffi::c_char;
use std::sync::OnceLock;

use p3_field::{Field, PackedValue};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear, default_koalabear_poseidon2_16};
use p3_symmetric::Permutation;
use rayon::prelude::*;

/// Plonky3's vector of KoalaBears for this target — one AVX-512 register under
/// the features the reference build applies to every crate, and a one-element
/// array under none.
type Packed = <KoalaBear as Field>::Packing;

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

/// Permute `Packed::WIDTH` states at once, transposing in and out.
fn permute_group(permutation: &Poseidon2KoalaBear<WIDTH>, group: &mut [KoalaBear]) {
    let lanes = group.len() / WIDTH;
    let mut packed: [Packed; WIDTH] =
        std::array::from_fn(|j| Packed::from_fn(|lane| group[lane * WIDTH + j]));
    permutation.permute_mut(&mut packed);
    for (j, column) in packed.iter().enumerate() {
        for (lane, value) in column.as_slice()[..lanes].iter().enumerate() {
            group[lane * WIDTH + j] = *value;
        }
    }
}

/// The states a group short of `Packed::WIDTH`, one at a time.
fn permute_each(permutation: &Poseidon2KoalaBear<WIDTH>, tail: &mut [KoalaBear]) {
    for state in tail.chunks_exact_mut(WIDTH) {
        let state: &mut [KoalaBear; WIDTH] = state.try_into().expect("width-16 chunk");
        permutation.permute_mut(state);
    }
}

fn permute_all(states: &mut [KoalaBear]) {
    let permutation = permutation();
    let group_elements = Packed::WIDTH * WIDTH;
    let batch = states.len() / WIDTH;
    let groups = states.len() / group_elements;
    let (whole, tail) = states.split_at_mut(groups * group_elements);
    if batch >= PARALLEL_MIN {
        whole
            .par_chunks_exact_mut(group_elements)
            .for_each(|group| permute_group(permutation, group));
    } else {
        whole
            .chunks_exact_mut(group_elements)
            .for_each(|group| permute_group(permutation, group));
    }
    permute_each(permutation, tail);
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
    // The packing width `p3-monty-31` resolved, not this crate's view of the
    // target features. The two are different questions and the difference is
    // the whole failure mode: `target_feature` is read when EACH crate
    // compiles, so a build that sets the features on this shim alone reports
    // avx512 from a `cfg!` here while `KoalaBear::Packing` is still the
    // one-element scalar array the dependency resolved. A width of 1 is that
    // build, said out loud.
    let described: &[u8] = match Packed::WIDTH {
        16 => b"packed field: 16 KoalaBears per vector (avx512)\0",
        8 => b"packed field: 8 KoalaBears per vector (avx2)\0",
        1 => b"packed field: scalar, 1 KoalaBear per vector\0",
        _ => b"packed field: unrecognized width\0",
    };
    described.as_ptr() as *const c_char
}
