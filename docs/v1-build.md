# Package and build dispositions

The package version remains 0.1 until the release qualification is complete.
The initial qualified pairing is Clang 18 and matching clang-scan-deps on Linux;
CMake rejects a scanner whose reported major does not match the compiler. Other
compiler and platform combinations need their own qualification.

`find_package(anndatapp CONFIG REQUIRED)` retains the full
`anndatapp::anndatapp` facade. To build and link only the dependency-free core:

```cmake
find_package(anndatapp CONFIG REQUIRED COMPONENTS core)
target_link_libraries(my_target PRIVATE anndatapp::core)
```

The core component installs and builds the five `anndatapp.core` interfaces
and its own library. It does not discover HDF5, compression libraries, or the
Zarr dependencies. The facade links core and supplies backed, streaming and
configured storage modules. Both targets are also available after
`add_subdirectory`. Consumers still build their own module interfaces; set
`CMAKE_CXX_EXTENSIONS OFF` and use a matching module toolchain. The installed
consumer checks core without optional dependency discovery, separate consumer
subdirectories, a re-exporting downstream target, and core followed by facade
lookup. Each native preset runs that check for its configured library and
backend combination. R01 is a package boundary change; it does not by itself
establish a build-time or size improvement.

R02 additionally rejects an explicitly supplied scanner with the wrong major;
a Clang 18 configure and an injected LLVM 17 scanner rejection were checked.
A matching major is necessary, not a claim that all versions or platforms have
been qualified.

S01: `tools/compile_benchmark.py` explicitly configures HDF5 and Zarr via
`--[no-]hdf5` and `--[no-]zarr`, checks the resulting cache, and reports the
enabled modules. Core and facade probes link their respective targets. Reports
hash build inputs, including source files that may be disabled in a particular
configuration. No fresh timing or size claims follow from this correction.

S02: `tools/record_measurements.py` is a historical checkout-specific campaign
recovery script, not a portable benchmark. Its fixed paths and pre-existing
artifacts are intentional for that interrupted campaign; it records one
sample and requires `--acknowledge-campaign-recovery`. Preserve its old reports
as source-specific historical evidence. Use a separately designed, controlled
benchmark for new comparisons.

Early packaging checks against the frozen pre-cleanup source with the package
changes overlaid passed for both static and shared full-backend builds. These
checks exercise the installed consumers and are not final candidate validation.

The private-rebuild checker collects public BMIs from both component targets
and locates the edited implementation object and its corresponding library.
Run it on an isolated, fully built candidate copy so its temporary source edit
cannot interfere with other work. It checks unchanged public artifacts and
restores the source and binary hashes; it is not a compile-time comparison.

On 2026-11-05, Clang 18 candidate qualification passed: Debug 20/20,
Release 20/20, ASan/UBSan 20/20, core-only 12/12, HDF5-only 16/16,
Zarr-only 16/16, shared 20/20, and shared Release 20/20. Full Python
interoperability including Scanpy 1.11.5 records passed 31/31; focused
ThreadSanitizer passed 4/4. Debug and HDF5-only additionally passed the late
arrays test update. ASan used `detect_leaks=0`; it is not a leak check.
The local evidence manifest `build/v1-cleanup/build-qualification-sha256.txt`
has SHA-256 `0959b4b73fa4d65d33fab7d1ba3873b7bcba8494b7f62f11bbcade5470c8cf07`;
it binds the candidate snapshot manifest, logs, and both private-edit receipts.
The `records.cpp` and `backed.cpp` private edits each left ten public BMIs and
two consumer objects unchanged, and restoration reproduced binary hashes.
These correctness checks do not establish performance or platform portability.
