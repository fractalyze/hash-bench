# hash-bench

One harness, one machine: hash-frx's routed, declined, and generic (`static_while`)
lowerings run beside pinned external reference implementations, reporting ns/hash
per (hash, batch, backend) and the fraction of the memory or issue roofline, with
the method and every revision recorded.

References are pinned as external dependencies at fixed revisions, not vendored
copies:

- CPU: XKCP, the official BLAKE3 implementation, OpenSSL (SHA-NI path), Plonky3
- GPU: ICICLE, sppark, Plonky3-CUDA, hashcat where it applies

The harness is the gate for every hand-written hash emitter deletion in hash-frx
and xla: an emitter goes only after the generic path measures at or above it at
every batch on both backends.
