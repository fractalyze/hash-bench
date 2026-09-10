"""A reference implementation as the harness loads it: a shared object plus the
provenance of what went into it.

The provenance is generated from the SAME values the build applies, so "which
flags produced this number" is answered by the build rather than by a parallel
claim in a comment, a `.bazelrc` or a Python table. That is the same argument
`arms.py` makes for reading the compiled module instead of trusting the request:
a recorded flag that nothing compiled with is not evidence.

Every reference is built through one transition, which is where the settings
that have to reach the UPSTREAM — not just the shim — are applied. Two do:

- The compilation mode. A shim's copts reach the shim alone, so under Bazel's
  default mode the pinned upstream beneath it compiles unoptimised, while the
  hash-frx side it is compared with arrives as an optimised wheel.
- The Rust reference's target features. `target_feature` is read when each
  crate compiles, and the packed field Plonky3 uses is resolved in its own
  crates, so features set on the shim alone leave the shim compiling against a
  scalar packing.

A shared object rather than a Python extension because the harness reaches it
through `ctypes` and needs no interpreter API — the entry points take pointers
and lengths, and `references.py` hands them numpy buffers.

The provenance also names the library file it describes, which is what lets a
reference be built by whichever rule suits its language: the C shims produce
`lib<name>.so` and the Rust one does not, and `references.py` reads the name
rather than assuming one.
"""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_rust//rust:defs.bzl", "rust_shared_library")

# One value, read by the transition and written into the provenance.
_COMPILATION_MODE = "opt"

_COMPILATION_MODE_SETTING = "//command_line_option:compilation_mode"
_RUSTC_FLAGS_SETTING = "@rules_rust//rust/settings:extra_rustc_flags"

def _reference_build_impl(_settings, attr):
    return {
        _COMPILATION_MODE_SETTING: _COMPILATION_MODE,
        _RUSTC_FLAGS_SETTING: attr.rustc_flags,
    }

_reference_build = transition(
    implementation = _reference_build_impl,
    inputs = [],
    outputs = [_COMPILATION_MODE_SETTING, _RUSTC_FLAGS_SETTING],
)

def _reference_library_impl(ctx):
    library = ctx.attr.library
    if type(library) == "list":
        library = library[0]
    return [DefaultInfo(files = library[DefaultInfo].files)]

# The shared object, rebuilt under `_reference_build` together with everything
# it links.
_reference_library = rule(
    implementation = _reference_library_impl,
    attrs = {
        "library": attr.label(cfg = _reference_build, mandatory = True),
        "rustc_flags": attr.string_list(),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def _applied_once(flags):
    """`flags` with repeats dropped, in first-use order.

    A shim's compile and link flags are recorded as one list because they are
    one fact about the binary, and `-fopenmp` legitimately appears in both. A
    row that listed it twice would read as a build that said it twice.
    """
    applied = []
    for flag in flags:
        if flag not in applied:
            applied.append(flag)
    return applied

def _bundle(
        name,
        library_target,
        library_file,
        flags,
        revision,
        source,
        implementation,
        note,
        rustc_flags = []):
    """The provenance JSON for one library, and the filegroup the harness loads.

    `library_target` builds it and `library_file` is what it lands as — the two
    differ for the Rust rule, which derives its own `lib<target>.so`.
    `rustc_flags` go to the transition, which applies them to every crate.
    """
    built = "%s_library" % name
    _reference_library(
        name = built,
        library = ":" + library_target,
        rustc_flags = rustc_flags,
    )
    provenance = "%s.provenance.json" % name
    write_file(
        name = "%s_provenance" % name,
        out = provenance,
        content = [json.encode({
            "reference": name,
            "library": library_file,
            "revision": revision,
            "source": source,
            "implementation": implementation,
            "compilation_mode": _COMPILATION_MODE,
            "flags": _applied_once(flags),
            "note": note,
        })],
    )
    native.filegroup(
        name = name,
        srcs = [
            ":" + built,
            provenance,
        ],
        visibility = ["//visibility:public"],
    )

def reference_shim(
        name,
        srcs,
        deps,
        copts,
        revision,
        source,
        implementation,
        linkopts = [],
        note = None):
    """Build a C reference's shared object beside the provenance of what made it.

    Args:
      name: the reference's arm name, as it appears in a row.
      srcs: the shim translation units.
      deps: the pinned upstream `cc_library` targets.
      copts: compile flags for the shim alone; recorded verbatim in the
        provenance. The upstream's optimisation comes from the transition's
        compilation mode, not from these.
      revision: the upstream revision the pin resolves to — a tag or a commit.
      source: the upstream repository URL.
      implementation: which of the upstream's implementations this selects.
      linkopts: link flags for the shared object.
      note: a caveat a reader of the row needs, or None.
    """
    library = "lib%s.so" % name
    cc_binary(
        name = library,
        srcs = srcs,
        copts = copts,
        linkopts = linkopts,
        linkshared = True,
        deps = deps,
    )
    _bundle(
        name = name,
        library_target = library,
        library_file = library,
        flags = copts + linkopts,
        revision = revision,
        source = source,
        implementation = implementation,
        note = note,
    )

def rust_reference_shim(
        name,
        srcs,
        deps,
        rustc_flags,
        revision,
        source,
        implementation,
        edition = "2021",
        note = None):
    """The same, for a reference whose upstream is a Rust crate.

    Args:
      name: the reference's arm name, as it appears in a row.
      srcs: the shim's Rust sources.
      deps: the pinned upstream crates.
      rustc_flags: flags the transition applies to the shim and every crate
        under it; recorded verbatim in the provenance. They reach the upstream
        crates, which is what decides the packed field they compile in.
      revision: the crate version the pin resolves to.
      source: the upstream repository URL.
      implementation: which of the upstream's implementations this selects.
      edition: the Rust edition the shim is written against.
      note: a caveat a reader of the row needs, or None.
    """
    shim = "%s_shim" % name
    rust_shared_library(
        name = shim,
        srcs = srcs,
        edition = edition,
        deps = deps,
    )
    _bundle(
        name = name,
        library_target = shim,
        library_file = "lib%s.so" % shim,
        flags = rustc_flags,
        revision = revision,
        source = source,
        implementation = implementation,
        note = note,
        rustc_flags = rustc_flags,
    )
