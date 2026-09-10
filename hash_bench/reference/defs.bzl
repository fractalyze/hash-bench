"""A reference implementation as the harness loads it: a shared object plus the
provenance of what went into it.

The provenance is generated from the SAME `copts` list the shim is compiled
with, so "which flags produced this number" is answered by the build rather than
by a parallel claim in a comment or a Python table. That is the same argument
`arms.py` makes for reading the compiled module instead of trusting the request:
a recorded flag that nothing compiled with is not evidence.

A shared object rather than a Python extension because the harness reaches it
through `ctypes` and needs no interpreter API — the entry points take pointers
and lengths, and `references.py` hands them numpy buffers.
"""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_cc//cc:defs.bzl", "cc_binary")

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
    """Build one reference's shared object beside the provenance of what made it.

    The outputs are `lib<name>.so` and `<name>.provenance.json`, in a filegroup
    named `name` that a `py_library` carries as data.

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
    cc_binary(
        name = "lib%s.so" % name,
        srcs = srcs,
        copts = copts,
        linkopts = linkopts,
        linkshared = True,
        deps = deps,
    )
    write_file(
        name = "%s_provenance" % name,
        out = "%s.provenance.json" % name,
        content = [json.encode({
            "reference": name,
            "revision": revision,
            "source": source,
            "implementation": implementation,
            "copts": copts,
            "linkopts": linkopts,
            "note": note,
        })],
    )
    native.filegroup(
        name = name,
        srcs = [
            "lib%s.so" % name,
            "%s.provenance.json" % name,
        ],
        visibility = ["//visibility:public"],
    )
