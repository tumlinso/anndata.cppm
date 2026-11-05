# Modules, packaging, and portability review

This review uses the frozen source identified by [source-manifest.json](source-manifest.json). It covers the top-level CMake project and presets, all four `cmake/` files, and CI. The package-provider changes were already uncommitted when review began; they are assessed as supplied work, not authored or accepted by this review. No build configuration or implementation was changed in the live tree.

## Retain the compiled boundary

`CMakeLists.txt:60–85` separates public `.cppm` interfaces, the private backed partition, and implementation units. Backend headers and nlohmann JSON stay private. `anndatapp.core` exports its data partitions; the facade exports enabled storage and streaming modules. Ordinary numerical pointers remain usable without a backend call or a library lock. Retain this design, move-only ownership, explicit copies, CMake dependency scanning, and consumer-built BMIs. Installing interface sources is appropriate: BMIs are not a portable binary format across arbitrary toolchains and flags.

The one consumer-local interface provider (`cmake/anndatappModuleProvider.cmake:39–63`) has a concrete purpose: consumers of the installed package share interface compilation rather than generating duplicate CMake 3.28 BMI rules. It preserves static/shared binary identity, imported configurations, dependency link metadata and compile requirements. The helper's loops are explicit property forwarding rather than numerical overhead. Replacing them with a generic target-cloning framework would add machinery without an established benefit. `FindBlosc.cmake` preserves shared-library dependency identity and runtime paths; the config avoids asking consumers for the private JSON dependency. Retain these boundaries.

The frozen Debug build and its installed multi-consumer test passed during this review. That confirms this supplied configuration on this host; it does not independently qualify all preset combinations or the provider on other CMake releases. No new private-edit invalidation or compilation-speed comparison was run. Historical receipts remain source-specific.

## R01 — One installed provider compiles the whole enabled interface set (P2, measured build work; candidate split)

The public package target links one static interface provider whose `CXX_MODULES` file set contains all installed public modules, regardless of which module the executable imports (`cmake/anndatappModuleProvider.cmake:39–45`; `CMakeLists.txt:60–85`). Thus the small logical `import anndatapp.core;` interface is distinct from the total build work performed by that installed package target. Core-only configuration disables the storage engines, but there is no independently exported core binary/component target.

A fresh installed consumer containing only a core import was built for this review. Its provider generated all ten enabled public BMIs, including H5AD, Zarr and streaming; see [validation.json](validation.json). This establishes which work ran, not a compile-time regression or a claim that a split is faster. Consumer-built module implementation objects are intentional package mechanics; their link/code-size effects need assessment before changing them.

**Recommendation:** retain the working single target while measuring whether explicit core/backed/format components reduce common consumer rebuild/import work enough to justify package complexity. If adopted, keep `anndatapp::anndatapp` as the ordinary facade and add explicit component targets with clear dependencies, rather than requiring callers to manage BMI order or link internals. Test static/shared, core-only and single-backend installations, multiple executables and separate consumer subdirectories. Measure fresh provider work, consumer-only rebuilds, public/private edit invalidation, BMI bytes and binary size on identical settings. No component split is implemented here.

## R02 — Toolchain qualification is narrower than source portability (P3, explicit coverage limit)

`CMakeLists.txt:8–18` intentionally admits Clang >=18 and Ninja, locating a versioned scanner with an unversioned fallback. Presets pin Clang 18; CI uses Ubuntu 24.04 and that toolchain. These are useful reproducible qualification boundaries. They do not establish support for GCC, MSVC, AppleClang, Windows, different standard libraries, or every newer Clang. The scanner fallback also needs a matching-major check before broadening qualification; a found executable alone is insufficient evidence of compatibility.

Keep C++20 modules and the current qualified pairing. Make platform assumptions and optional mapping/spill capabilities explicit, and isolate POSIX operations behind small private boundaries if a new target needs them. Compiler overflow builtins, `off_t` ranges, host pointer/index widths, native object lifetime, alignment, byte order, dependency discovery and mapping semantics need target-specific checks. Do not replace standard containers or exceptions merely to appear portable. Additional target acceptance requires actual configure/build/native/installed-consumer tests, independent storage round trips, and applicable concurrency/sanitizer checks.

## Verification design

`CMakePresets.json` supplies native, core-only, HDF5-only, Zarr-only, static/shared, sanitizer and interop paths. CI exercises them with independent Python fixtures and focused TSan work. `cmake/test-installed-consumer.cmake` tests all enabled consumer modules and real storage operations, not just successful import. Keep this matrix and distinguish its historical/CI coverage from checks actually executed for this review.

The installed-consumer helper uses fixed scratch paths below its build tree and uncapped `execute_process` calls. They are reasonable for ordinary correctness tests, but not bounded performance-campaign orchestration. Use a fresh build/prefix for qualification and apply an outer timeout in a bounded campaign. The benchmarking-tool scope issues are [S01/S02](support.md), rather than evidence that current native tests are invalid.

Public interface changes require migration notes and rebuild measurements; private changes should preserve BMI and consumer-object identity where the existing module contract promises it. Neither shorter source nor fewer includes alone proves a faster import or executable.
