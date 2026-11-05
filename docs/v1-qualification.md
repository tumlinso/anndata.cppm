# AnnData++ qualification

The 1.0.0 release uses the user-approved two-machine policy: Linux x86-64 requalification of the compatibility fixes plus native Apple AArch64 evidence with explicitly accepted limitations. Final Linux results are in [v1-release-linux.md](v1-release-linux.md). The cleanup history below retains its original source identities and test scopes.

The [49-finding disposition](v1-status.md), [support contract](support.md), [migration guide](migration-1.0.md) and [bounded performance report](v1-performance.md) describe the resulting API and its limits. The compact machine-readable receipt is [v1-local.json](../tests/results/v1-local.json). Raw logs, generated fixtures and source snapshots remain local under ignored `build/v1-cleanup/`.

## Correctness and package checks

Qualification proceeded against frozen snapshots. Source 2 contains the integrated cleanup; source 3 adds the measured sparse-gather repair; source 4 adds feature-detected checked-arithmetic intrinsics with the same portable fallback. The table distinguishes full suites from affected checks after those private changes.

| Gate | Source 2 full suite | Source 3 follow-up | Source 4 follow-up |
| --- | --- | --- | --- |
| Native Debug | 20/20 | 20/20 | 20/20 |
| Native Release | 20/20 | 20/20 | 20/20 |
| ASan/UBSan | 20/20 | 20/20 | 20/20 |
| Independent Python and Scanpy | 31/31 | 31/31 | 31/31 |
| Focused ThreadSanitizer | 4/4 | 4/4 | Earlier concurrency gate retained |
| Core-only | 12/12 | Arrays and installed consumer: 2/2 | Earlier component gate retained |
| HDF5-only | 16/16 | Arrays and installed consumer: 2/2 | Arrays and installed consumer: 2/2 |
| Zarr-only | 16/16 | Arrays and installed consumer: 2/2 | Earlier component gate retained |
| Shared Debug | 20/20 | Arrays and installed consumer: 2/2 | Earlier component gate retained |
| Shared Release | 20/20 | Arrays and installed consumer: 2/2 | Earlier component gate retained |

ASan used `detect_leaks=0`; these are address/undefined-behavior checks, not leak qualification. The restricted runner's documented Zarr event-loop wakeup workaround was enabled for interoperability tests. The final arithmetic helpers additionally passed an independent signed-128 oracle on 200 boundary/sign cases for each of the intrinsic and portable implementations. The source-4 change affects private checked add/multiply operations; it adds no shared state, interface or storage dependency.

Independent Python producers/readers used AnnData 0.13.4, NumPy 2.2.6, pandas 2.3.3, SciPy 1.17.1, h5py 3.16.0, Zarr 3.4.2, numcodecs 0.17.1 and Scanpy 1.11.5. Native fixed-string padding was also checked against an independently generated physical HDF5 fixture. Adversarial padding cases establish decoding behavior, not universal AnnData acceptance.

The source-4 HDF5-only installation passed nine downstream CTests covering preprocess.cppm, Cellerator host modules and installed SDK consumers, CellShard ingestion, and velocity's installed consumer. Four additional CellShard installed executables exited successfully. These are CPU consumer checks; they make no GPU qualification claim.

Private implementation edits to `records.cpp`, `backed.cpp` and later `arrays.cpp` left ten active public BMIs and two consumer objects unchanged. Restoring the source reproduced binary hashes. Core-only consumption compiled the five core interfaces without discovering optional storage dependencies. Installed checks include sibling consumers and downstream re-exporting targets. An injected scanner with a mismatched compiler major was rejected.

The final bounded suite passed all 38 untimed operations for both revisions, followed by exactly five alternating pairs per scenario and separate diagnostic processes. All ten scenarios stayed within their budgets; the final run had no regression flags under the declared median/range rule. Caller-provided contiguous, strided and gathered dense copies are included. Earlier checked-acquisition differences and repaired sparse regressions remain documented; overlapping sample ranges do not establish equal performance.

One Clang 18 frontend crash occurred while compiling `records.cpp`. A retry and independent reproduction build succeeded without a source workaround. The diagnostic and reproducer remain local; its cause was not established.

## Provenance and reproduction

The receipt binds the frozen baseline/candidate manifests, qualification log manifests, downstream results, arithmetic oracle and performance summary by SHA-256. Later documentation and receipt edits do not change the tested production source. The baseline includes the preexisting package-boundary work, which was validated and committed separately.

Use [development.md](development.md) for environment setup and the full independent reference command. Native/package gates use these existing presets:

```sh
cmake --preset debug
cmake --build --preset debug --parallel 2
ctest --preset debug
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release
cmake --preset sanitizers
cmake --build --preset sanitizers --parallel 2
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 ctest --preset sanitizers
cmake --workflow --preset core-only
cmake --workflow --preset hdf5-only
cmake --workflow --preset zarr-only
cmake --workflow --preset shared
cmake --workflow --preset shared-release
```

Configure `thread-sanitizers` and `interop-check` with an explicit isolated reference Python, as documented in [development.md](development.md). Coordinate an exclusive timing window for the separate [performance recipe](v1-performance.md). Do not treat a passing native suite as storage interoperability or a bounded performance result as cross-device portability.

The [ARM qualification report](v1-arm-qualification.md) records the Apple host's conditional results and limitations. The user explicitly accepted unavailable ARM sanitizers, contended timing flags, incomplete consumer-build evidence, and resource-limit/reproducibility coverage gaps after successful Linux requalification. Historical receipts retain the device policy recorded at the time of their runs; the two-machine release decision supersedes that policy. Source/module sources and compact reports are the release artifacts. Compiler-specific BMIs are not portable distribution artifacts.
