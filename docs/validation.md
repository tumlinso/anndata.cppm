# Validation

## Adaptive backed block I/O

The [validation receipt](../tests/results/streaming-validation.json) records final
source hashes, commands and raw log identities. Debug passed 21/21 groups,
including all five independent Python gates; Release passed 16/16. ASan/UBSan
passed 17/17, including the expanded Python streaming gate. Focused
ThreadSanitizer runs passed parallel Zarr v2 and sharded v3 reads/writes with
independent Python readback. LeakSanitizer was excluded for runner restrictions.

The [final package matrix](../tests/results/streaming-matrix.final.json) passed
core-only 11/11, HDF5-only 15/15, Zarr-only 12/12, and shared 16/16. Each includes
an independently installed streaming consumer. The earlier shared-link failure
is preserved; adding private map/variant headers fixed template emission without
changing data algorithms or public interfaces.

A private edit to `streaming.cpp` preserved all ten public BMIs and two actual
consumer objects by hash and timestamp. Exact source, object, library and tested
executable hashes were restored afterward. [Performance evidence](streaming-performance.md)
separates measured call/byte reductions from persistent shard-write amplification
and single-observation timing limits. Raw evidence is ignored and preserved under
`build/evidence/streaming-20261104` in the primary checkout.

Qualification remains Linux/Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1 and the pinned
reference environment. This is source and package qualification; Project Control
returned an active-task conflict without a workflow handle, so no native Todo
acceptance is claimed.

## Native records and refactor integration

The record implementation and compact-selection refactor were combined in
merge commit `421e4f8928f1b8f236338524a1b5a282d7862a4a`. The [integration
receipt](../tests/results/records-integration.json) identifies source hashes,
commands, log hashes and the independently recorded configuration and Python
results. Qualification used Linux, Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1,
HDF5 C 1.10.10, AnnData 0.13.4 and Scanpy 1.11.5.

| Check | Result |
|---|---|
| Debug | 13/13 CTest groups passed |
| ASan + UBSan | 13/13 passed, including installed consumer |
| Core-only | 8/8 passed |
| Zarr-only | 9/9 passed |
| HDF5-only | 12/12 passed |
| Shared library | 13/13 passed |
| Independent Python interoperability gates | 4/4 passed: semantics, records, H5AD and storage |

The installed consumer imports `anndata.core` and exercises borrowed
interleaved records. Native tests additionally cover record ownership, byte
strides, external storage, pointer-backed text, bounded HDF5 string transfers,
and records retained through nested dataset slices. Python generates real Scanpy
ranking results, independently reads C++ output, and verifies physical encodings,
sparse matrices, metadata and raw data. H5AD supports the declared primitive
numerical/text record subset; Zarr record codecs and nested/subarray fields are
not yet supported. The [record contract](records.md) states further limits.

A single post-merge private `records.cpp` edit rebuilt that implementation and
relinked its consumers. All nine public BMI hashes and timestamps and both
measured consumer-object hashes and timestamps remained unchanged. Restoring
the exact source and rebuilding reproduced the original implementation-object,
static-library and executable hashes. The [probe](../tools/check_private_rebuild.py)
retains raw logs and the original source under an ignored build directory. This
is dependency evidence, not a new timing campaign. The [record performance
report](records-performance.md) identifies the separate pre-merge timing inputs.
LeakSanitizer was disabled due to the runner's ptrace restriction; ASan and
UBSan remained active.

## Backed storage and Zarr milestone

The [storage validation record](../tests/results/storage-validation.json) identifies the final source, actual commands, independent fixtures and limits. Qualification uses Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1 and HDF5 C 1.10.10 on Linux. The supported storage/codec subset is explicit in [storage.md](storage.md); the [dependency report](storage-dependencies.md) contains the installation commands.

| Final configuration | CTest result |
|---|---|
| Debug | 9/9 passed |
| Native workflow, Python disabled | 9/9 passed |
| Core-only | 7/7 passed; backend cases skip storage operations |
| HDF5-only | 9/9 passed |
| Zarr-only | 7/7 passed |
| Shared library | 9/9 passed |
| Release | 9/9 passed |
| ASan + UBSan | 9/9 passed |
| Pinned Python interoperability | 12/12 passed |

All native configurations were rebuilt and tested on the final implementation, including the focused dense-transfer fix. The complete interoperability suite was rerun after correcting its Python fixture's automatic categorization and raw chunk geometry; strict dtype, missingness and value assertions remain. Installed tests compile consumers of `anndata.core`, `anndata.backed`, the facade and enabled format modules, then exercise actual file I/O. Shared installation includes a non-system Blosc dependency without requiring a manual library-path environment setting. A private writer workaround avoids a Clang 18 module-view move-constructor linker defect; it does not change public ownership or numerical loops.

The focused fix preserved all eight public BMI hashes and timestamps plus the import-only example object. Profiled small cases confirmed repeated codec and allocation work before changes. A same-instrumentation sharded-export pair improved from 1.89 to 0.054 seconds; this is one observation per candidate. Four subsequent 64 MiB export/reindex operations passed in 0.406–0.535 seconds under explicit time/I/O caps, with complete independent value/axis comparisons. Full workload conditions and instrumentation differences are in the [compact performance record](../benchmarks/results/storage-summary.json).

The AnnData 0.13.4/Zarr 3.4.2 suite independently creates and reopens H5AD and Zarr v2/v3 fixtures. Native and Python checks cover caller buffers and strides, sparse widths/duplicates, lazy metadata, structural growth/preflight/reindexing, stale and retained descriptors, mappings, same-store extension preservation, cross-store rejection, codecs, sharding and corruption diagnostics. An independent category matrix passed 66 checks plus Python reopening of 12 exports/reindexes, including pruning, recoding, color alignment, nullable masks, NA/NaN distinctions and raw alignment. Permanent tests also cover packed string ownership/validation and nullable category dictionaries.

A read-only check used the supplied modern `GSE147520_all_cells_raw_counts.h5ad`: 68,008 by 26,587 CSR with 114,455,556 float32 entries. Python independently reopened C++ selected output. Unique-variable gathers match exact CSR arrays; repeated-variable gathers match values, labels, tables, duplicate multiplicities and row pointers, with the documented [internal tie-order difference](memory.md) from SciPy. Core and backed C++ representations match each other exactly. The original fixture hash was unchanged.

The [storage performance report](storage-performance.md) separates selected reads, full materialization, caller buffers, mappings, parallel sessions and compilation measurements. Its memory observations apply to the recorded workloads, not every schema or codec. Direct structural updates are not whole-store crash transactions. Mapped typed access is explicitly unsafe and qualified only on Linux/Clang. At this storage milestone, compound/ragged/complex/half/legacy encodings and cloud stores were unsupported; the later record milestone above adds a compound subset. ASan and UBSan were active; LeakSanitizer was disabled for the local ptrace restriction. GitHub CI is authored, not remotely executed here.

Project Control returned `workflow_task_already_active` without a handle; its observer reported unavailable semantic workflow state. [The observation](../planning/storage-workflow-observation.json) records this separately. Existing authority and active tasks were preserved; these local results are not a native Todo outcome acceptance receipt.

## H5AD milestone

The [H5AD validation record](../tests/results/h5ad-validation.json) identifies final source hashes, commands and coverage. Local qualification uses Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1 and HDF5 C 1.10.10. The [H5AD encoding matrix](h5ad.md) states the supported subset and upstream limitations.

| Check | Result |
|---|---|
| Native HDF5-enabled workflow | 6/6 CTest groups passed |
| Dependency-free core-only workflow | 5/5 passed; no HDF5 in the example's dynamic dependencies |
| Shared-library workflow | 6/6 passed |
| Release build and CTest | 6/6 passed |
| ASan + UBSan build and CTest | 6/6 passed, including installed consumers |
| Pinned AnnData 0.13.4 interoperability workflow | 8/8 passed, including the existing 13 in-memory scenarios and independent bidirectional file tests |
| AnnData 0.12.10 compatibility mode | Shared-subset file round trip passed |
| Independent final-source sanitizer review | Nine targeted reproductions passed; no remaining confirmed defects in the reviewed scope |

Installed-package tests build multiple consumers together in a C++-only project, covering `anndata.core`, the convenience facade, and the H5AD module. HDF5-enabled consumers actually read and write files. Static, shared, core-only and sanitizer installations pass. Additional independent probes qualified sibling-directory package discovery and a downstream library's exported dependency.

File tests independently inspect encodings/dtypes with h5py and read C++ output using Python AnnData. Cases include compact sparse integer widths, unsigned categorical codes, raw and empty dimensions, UTF-8 arrays, nullable sentinels, selected views, physical byte order, explicit native conversion, compression, validation opt-out and path-specific failures. Limits for raw without X and foreign-endian sparse data are recorded rather than counted as universal Python interoperability.

The installed Project Control run-free context adapter failed before a usable workflow handle was returned. Its expired attempt was released through the supported recovery API after root observed producer completion. [The workflow receipt](../planning/h5ad-workflow-receipt.json) distinguishes this recovery from outcome acceptance; the results here are local evidence. GitHub CI has been authored but not run remotely.

## Historical in-memory foundation

The first foundation was qualified locally on Linux with Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1, and C++20 without language extensions. The [validation record](../tests/results/validation.json) identifies the source inputs and actual commands. GitHub CI is configured but was not executed remotely in this session.

| Check | Result |
|---|---|
| `cmake --workflow --preset native-check` | 4/4 CTest groups passed; no Python required |
| `cmake --workflow --preset interop-check` | 5/5 groups passed, including 13 semantic scenarios against AnnData 0.13.4 |
| Release configure/build/CTest | 4/4 groups passed |
| ASan + UBSan configure/build/CTest | 4/4 groups passed, including installed-package consumer |
| Same final semantic driver with AnnData 0.12.10 | 13/13 scenarios passed |
| Separate installed client using `import anndata;` | Configured, built and ran successfully; the regular installed CTest independently imports `anndata.core` |

The semantic oracle creates real H5AD fixtures in Python, independently reopens them, and compares corresponding C++ in-memory outcomes. Coverage includes dense/CSR/CSC selections, reordered/repeated/empty selections, aligned arrays and tables, nullable and categorical metadata, raw's independent variables, table-valued raw varm, category colors, and scalar dtype preservation. This is not C++ file-I/O interoperability.

Native tests cover pointer identity, move-only ownership, explicit deep copies, retained lifetimes, custom deleters, const external buffers, independent sparse widths, strides and gathers, construction/validation boundaries, explicit validation opt-out, and unchecked structure access. Independent review reproduced defects before repairs and retested those contracts; it additionally exercised randomized dense and compressed sparse selection cases.

LeakSanitizer was disabled for the local sanitizer runs because it cannot operate under this execution sandbox's ptrace. AddressSanitizer and UndefinedBehaviorSanitizer remained enabled. The ordinary sanitizer preset keeps the compiler's normal defaults for execution outside that sandbox.

These observations do not claim support for all AnnData encodings or formats. See the [compatibility matrix and representation limits](architecture.md) and [performance report](performance.md). Local command evidence is distinct from a Todo workflow outcome acceptance receipt.
