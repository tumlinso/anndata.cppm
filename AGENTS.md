# Working on anndata.cppm

Prioritize correct, maintainable delivery and shorter completion time within the user's usage and resource limits. Account for handoff, coordination, and integration costs; use stronger ownership when it materially improves decisions or independent progress.

Use the user's configured cooperative agent profiles (`scout`, `engineer`, `executor`, `optimizer`, `integrator`, `verifier`, `investigator`, `reviewer`, `principal`). Root retains intent, integration, workflow lifecycle, and final acceptance. Delegate substantial diagnosis/design/implementation to engineer; established bounded work to executor. Terra optimizer owns performance diagnosis and interpretation; Terra integrator owns combined behavior and validation. Root may allocate multiple principals to independent substantial objectives. Use Codex subagents, never Project Control local coding workers. Preserve concurrent edits and unknown state. Start separable work promptly and coordinate shared compute, especially measurements.

## Build and architecture

- Follow [STYLE.md](STYLE.md): compact numerical/HPC code with visible memory operations, concise local notation, minimal machinery, and small compiled interfaces.
- Also follow the written [Cellerator coding style](/home/tumlinson/Cellerator/coding_style.md), rather than Cellerator's current implementation as an example. For `anndata.cppm`, fast module imports and consumer recompilation times remain **very high priorities**; preserve small public interfaces and implementation isolation when applying its hot-path guidance. Measure and report runtime/build tradeoffs explicitly. Keep the referenced guide unchanged.
- C++20 named modules are mandatory. Public consumption is `import anndata;` or `import anndata.core;`, with `.cppm` interfaces. Do not introduce a public umbrella header or require `import std;`.
- Initial toolchain: Clang 18 plus matching clang-scan-deps, CMake >=3.28, Ninja >=1.11. Use CMake CXX_MODULES file sets and scanning; do not hand-order BMI commands.
- Run `cmake --preset debug`, `cmake --build --preset debug`, `ctest --preset debug`. Sanitizers use the `sanitizers` preset. See docs/development.md for optional Python semantic tests and release benchmarks.
- Keep storage dependencies private and optional. The core has no dependency on Python, HDF5, Arrow, Torch, or Cellerator. Downstream integration never reverses that dependency direction.
- H5AD and Zarr v2/v3 are enabled by default through private storage adapters. `cmake --workflow --preset core-only` verifies the dependency-free configuration; `hdf5-only`, `zarr-only` and `shared` verify single-backend and shared-library installations. See docs/storage-dependencies.md for installation commands. Python interoperability tests independently reopen C++ output and inspect physical encodings.
- Preserve move-only owners, explicit deep copies, retained read-only views, native pointers/strides and raw sparse buffers. Borrowing and unchecked access are deliberate capabilities with documented contracts.
- Keep numerical element loops free of type-erased dispatch and library synchronization. Callers own parallel execution. Choose representations by evidence, not blanket preferences for or against the standard library.
- Follow AnnData/Scanpy data conventions. Consult pinned upstream specifications/source and independently generated fixtures before adding formats. Do not claim file compatibility from an in-memory test.
- Keep implementation-only changes out of public module interfaces. Measure rebuilds and direct-memory performance before making performance claims.
- Checkpoint performance evidence before stopping a campaign. Start with geometry-faithful probes capped at 30 seconds; count calls, allocations and bytes, profile the mechanism, and stop repeating established slow cases. Larger confirmations need explicit time and I/O budgets. Preserve the baseline and compare identical workloads and settings.
- Keep generated biological stores, binaries, raw campaign reports and profiler outputs ignored. Commit reproducible measurement sources and compact summaries with artifact hashes; preserve raw evidence locally.

## Workflow and preservation

Project Control workspace/repository alias: `anndata-cppm`. Use its supported workflow routes and the installed Todo Orchestrator skill for native authority operations. `.todo-orchestrator/authority.sqlite3` is semantic authority; never hand-edit it or generated projections. Private credentials stay outside the repository. Registration, authority creation, and full host workflow readiness require separate checks.

Do not reset, clean, force-push, delete authority state, overwrite others' work, or commit private runtime artifacts. Preserve exact source/measurement provenance. Record partial setup and unsupported behavior honestly.
