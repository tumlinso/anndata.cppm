# Compact selections and direct metadata access

The shared `Axis` selector keeps all/affine mappings inline and retains owned
gathers. Ordinary AnnData slice syntax is unchanged. The old low-level
nested-vector selector API is replaced; see [memory.md](memory.md) for migration,
pointer/rank access, ownership, and materialization costs.

Dataset slicing resolves and composes observation/variable maps once, reusing
them across numerical, string, table, pairwise, and raw observation slots. Current
category dictionaries still determine recoding and color projection. Direct
`obs_column()`/`var_column()` acquisition retains one column without building a
full table view. Full table snapshots retain their existing bindings. Aligned
table index comparison now reads rows without constructing copied name lists.

## Correctness

Clang 18.1.3, CMake 3.28.3, and Ninja 1.11.1 on Linux qualified these checks:

| Configuration | Passed groups | Coverage |
|---|---:|---|
| Debug | 9 | Native data, arrays, strings, backends, import and installed consumer |
| Core-only | 7 | Dependency-free core and installed modules |
| Shared | 9 | Shared library and independently installed consumer |
| ASan/UBSan | 9 | Native and installed-consumer checks |
| Python interoperability | 12 | Native plus semantics, H5AD and backed HDF5/Zarr round trips |

The semantic oracle now checks 17 scenarios, including nested ranges and
gathers, reversed axes, repetitions, empty selections, categories/colors,
nullable values, and raw's independent variable axis. Native tests additionally
cover caller selector lifetimes, owner destruction/replacement, alias-visible
writes, string identity/packed access, sparse widths/order, and integer limits.
Compact repeated-major selectors that cannot produce addressable sparse output
are rejected before allocating or traversing output offsets.

The initial Python run encountered a reproducible exception in the pinned
AnnData 0.13.4/SciPy 1.17.1 pair: slicing a repeated-row view again with reversed
columns raised `ValueError: cannot reshape array of size 2 into shape (2,2)`.
Only that Python reference intermediate is now materialized before the second
slice. The C++ path remains nested, and all value, metadata, and physical sparse
buffer comparisons remain enabled. The affected semantic group passed on
recheck; the already-passing H5AD/storage groups were not repeated. Original
failure and successful recheck logs are retained separately.

The runner used `ASAN_OPTIONS=detect_leaks=0`,
`UBSAN_OPTIONS=halt_on_error=1`, and `ANNDATA_INTEROP_ZARR_WAKEUP=1` as documented
in [development.md](development.md). ASan/UBSan remained active; LeakSanitizer
was excluded because of runner restrictions.

## Bounded allocation evidence

One Debug probe run completed successfully under a 30-second cap. Fixture setup
and reporting were excluded; the measured scope included acquisition and
release of returned descriptors. Counts intercept C++ `new`/`delete`, including
aligned forms. Requested bytes are cumulative allocations, not peak memory.

| Rank-two acquisition | Logical lengths | Allocation calls | Requested bytes |
|---|---|---:|---:|
| All, forward range, or reverse range | 16; 1,048,576; 4,294,967,296 | 2 each | 136 each |
| Nested affine ranges | Same lengths | 4 | 272 |

The fixture borrows one float with zero strides so that increasing logical
length isolates descriptor costs. The counts stayed fixed; this is not a
large-payload benchmark or a runtime speedup claim.

| Columns per axis | Direct obs/var calls / bytes | Full-table route calls / bytes |
|---:|---:|---:|
| 8 | 1 / 136 | 13 / 1,944 |
| 128 | 1 / 136 | 137 / 29,784 |
| 1,024 | 1 / 136 | 1,036 / 237,656 |

Both routes returned identical pointers and contents. Direct acquisition still
allocates a retained descriptor and scans ordered names; only its allocation
volume is independent of unrelated columns. Category processing, irregular
gathers, and materialization retain their documented traversal/allocation costs.
No new cache, scheduler, storage engine, or numerical-loop checks were added.

Reproduce with `ANNDATA_BUILD_BENCHMARKS=ON`, build the
`anndata_selection_probe` target, and run
`timeout 30s build/debug/anndata_selection_probe`. Other build configurations
may produce different counts and must be identified separately.

The [allocation records](../benchmarks/results/selection.json) contain source,
binary, and raw-report hashes. The [validation receipt](../tests/results/selection-validation.json)
records commands, input hashes, logs, and the qualified Python recheck. Raw logs
remain under ignored `build/`; previous campaign evidence was preserved.

No dependencies were added or installed, so this step required no installation
commands. Existing setup commands remain in
[storage-dependencies.md](storage-dependencies.md).
