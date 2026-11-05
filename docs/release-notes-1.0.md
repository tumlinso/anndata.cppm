# AnnData++ 1.0.0

The release uses the user-approved two-machine qualification policy: Linux x86-64 requalification of the compatibility changes, together with the disclosed native Apple AArch64 results. The ARM limitations below are accepted release limitations, not passing qualification gates.

## Changes

- Repair malformed-axis handling, record validation adapters, Zarr path admission, consolidated-only mutation rejection and category-page recovery.
- Add direct borrowed dense descriptors and focused retained record acquisition, with explicit metadata and numerical lifetimes.
- Simplify dense/sparse selection copies, packed text, categorical recoding, record representation and backend staging while preserving AnnData alignment, native types and explicit copies.
- Define worker-context, conflict-unit, failure/retry and allocation-statistic contracts. Adaptive streaming remains the default.
- Supply an independent dependency-free core package target and matching module-scanner checks. Preserve existing facade imports and targets.
- Add the fixed ten-scenario performance suite, separate timing and diagnostic observers, source/fixture identities and bounded execution.

The [49-finding disposition table](v1-status.md) records fixes and evidence-backed retention decisions. The [performance report](v1-performance.md) records the measured workloads and their limits; it is not a general speed guarantee. Regression investigations and superseded candidates remain in ignored local evidence rather than being overwritten.

## Compatibility and support

See [migration-1.0.md](migration-1.0.md) for pre-1.0 source changes and [support.md](support.md) for the capability and platform boundaries. The 1.x promise covers documented source APIs and data contracts. Consumers rebuild compatible module interfaces; binary ABI and cross-toolchain BMI compatibility are not promised.

The existing core/H5AD/Zarr/streaming capability boundary is retained. Extra formats, encoding families, GPU execution and FP16 storage are outside this cleanup. The package remains independently usable without Python or downstream application runtimes.

## Qualification

The historical cleanup build and private-edit matrix is recorded in [v1-build.md](v1-build.md). Final Linux requalification is recorded in [v1-release-linux.md](v1-release-linux.md); the [qualification index](v1-qualification.md) preserves source-specific scopes. ASan/UBSan, ThreadSanitizer and independent Python/Scanpy checks have distinct scopes; local ASan runs do not claim leak qualification.

The [native ARM qualification](v1-arm-qualification.md) is conditional: ASan and TSan could not execute reliably; three performance cases flagged regressions under host contention; the consumer-build comparison is incomplete. The functional completeness follow-up omitted macOS's unavailable address-space limit. Linux-only mmap/fault-injection checks, some downstream consumers, and complete restored-binary reproducibility were unavailable on ARM. These limitations remain visible in the release support contract and receipts.

Portability fixes split mixed native integer declarations, explicitly select the address-bearing HDF5 object-info API, and adapt test/benchmark observers to Darwin. They introduce no public module interface changes.
