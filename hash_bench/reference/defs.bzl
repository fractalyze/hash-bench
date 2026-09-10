"""A reference implementation as the harness loads it: a shared object plus the
provenance of what went into it.

The provenance is generated from the SAME flag list the shim is compiled with,
so "which flags produced this number" is answered by the build rather than by a
parallel claim in a comment or a Python table. That is the same argument
`arms.py` makes for reading the compiled module instead of trusting the request:
a recorded flag that nothing compiled with is not evidence.

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

def _bundle(
        name,
        library_target,
        library_file,
        flags,
        revision,
        source,
        implementation,
        note):
    """The provenance JSON for one library, and the filegroup the harness loads.

    `library_target` builds it and `library_file` is what it lands as — the two
    differ for the Rust rule, which derives its own `lib<target>.so`.
    """
    write_file(
        name = "%s_provenance" % name,
        out = "%s.provenance.json" % name,
        content = [json.encode({
            "reference": name,
            "library": library_file,
            "revision": revision,
            "source": source,
            "implementation": implementation,
            "flags": flags,
            "note": note,
        })],
    )
    native.filegroup(
        name = name,
        srcs = [
            ":" + library_target,
            "%s.provenance.json" % name,
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
      copts: compile flags for the shim; recorded verbatim in the provenance.
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
      rustc_flags: flags for the shim; recorded verbatim in the provenance.
        Unlike a C shim's copts these also decide which backend the upstream
        compiles in, the packed-field selection being a `target_feature` gate.
      revision: the crate version the pin resolves to.
      source: the upstream repository URL.
      implementation: which of the upstream's implementations this selects.
      edition: the Rust edition the shim is written against.
      note: a caveat a reader of the row needs, or None.
    """
    rust_shared_library(
        name = "%s_shim" % name,
        srcs = srcs,
        edition = edition,
        rustc_flags = rustc_flags,
        deps = deps,
    )
    _bundle(
        name = name,
        library_target = "%s_shim" % name,
        library_file = "lib%s_shim.so" % name,
        flags = rustc_flags,
        revision = revision,
        source = source,
        implementation = implementation,
        note = note,
    )
