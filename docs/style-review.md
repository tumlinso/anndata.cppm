# Source and memory transparency review

Reviewed source: `2ddf57559412c90297e5b525bd6a5b6d9a0c36d1`.

The criterion is the [project coding style](../STYLE.md): compact numerical/HPC
code whose computation, storage, and execution are immediately understandable.
This is a source review of public interfaces, core metadata handling, backed
numerical transfers, and representative Zarr traversal. The findings describe
style and operation visibility; they are not newly established correctness
defects or measured performance regressions. The original review changed no
implementation; source links below are pinned to that reviewed revision.

## Implementation disposition

The compact-selection step implements findings 1 and 2: shared affine/gather
axes, direct column acquisition, and row-wise index comparison. It preserves
full table snapshots, retained ownership, category behavior, and raw slicing.
The low-level selector API was deliberately replaced; see
[memory.md](memory.md) for migration and acquisition costs.

Findings 3, 5, and 6 remain separate work. Finding 4's traversal/materialization
costs are now documented; packed string copying was not redesigned. No new
numerical dependency or application was introduced. Current test and allocation
evidence is recorded in [selection-validation.md](selection-validation.md).

## Historical findings, in suggested order

### 1. Regular selections become index-list machinery

`ArrayView::select()` exposes
`vector<optional<vector<size_t>>>` in
[anndata.arrays.cppm](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/anndata.arrays.cppm#L301). It builds and composes
stored index lists in [arrays.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/arrays.cpp#L417). Acquiring a dense view
then scans those lists to discover whether they describe an arithmetic
progression ([arrays.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/arrays.cpp#L461)). The higher-level `range`
selection also expands into positions
([data.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/data.cpp#L388)).

A simple strided subarray therefore travels through a representation much more
elaborate than its pointer, shape, and stride. A compact regular-range path is
the most useful representation change to investigate before further expansion.
Keep explicit gathers for arbitrary ordering and repetitions, and preserve
negative steps, empty dimensions, sparse behavior, and category pruning.
Changing this shared representation requires deliberate module dependencies and
existing semantic tests; an additional generic indexing framework would add
little value.

### 2. Single-column access constructs a whole table snapshot

`data.obs().column("batch")` first calls `Table::view()`. That allocates a table
view, copies its index-name string, and copies every column name while creating
an allocated view descriptor for every column
([data.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/data.cpp#L752),
[column view construction](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/data.cpp#L560),
[AnnData accessors](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/data.cpp#L1000)). It does not copy the column payloads,
but the expression conceals work proportional to the number of columns.

The full snapshot serves a real purpose: retained bindings survive structural
replacement. Preserve that contract. Consider a focused column-acquisition path
that avoids constructing unrelated descriptors. Existing callers can acquire
and reuse a table view outside a loop. Similarly, `set_obsm(Table)` compares
indexes by materializing both lists of strings
([data.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/data.cpp#L1033)); direct row-wise comparison would express
the required operation more plainly without copying both indexes.

### 3. Dense external-buffer entry points still require STL metadata

`Array::borrow_dense()` requires vectors for shape and strides
([anndata.arrays.cppm](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/anndata.arrays.cppm#L389)), and creates an allocated
wrapper state ([arrays.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/arrays.cpp#L314)). This borrows the numerical
buffer; it does not make descriptor creation allocation-free. Constructing the
lighter `BorrowedDenseView` directly requires a public constructor parameter
named `detail::DenseDimensions`
([anndata.arrays.cppm](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/anndata.arrays.cppm#L120)). `DenseBuffer` likewise
owns vector metadata ([anndata.backed.cppm](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/anndata.backed.cppm#L22)).

Consider a small pointer/rank/shape/stride entry point with an explicit metadata
lifetime contract. It should complement retained owners, remain easy to use
with stack arrays, and avoid exposing implementation-detail constructors. This
is an interface simplification opportunity, not evidence that descriptor setup
dominates applications.

### 4. Some string accessors and copies hide complete traversals

A selected `StringArrayView::byte_size()` visits every logical string
([strings.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/strings.cpp#L179)). Selected `copy()` stages a vector of
individual strings and then repacks it
([strings.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/strings.cpp#L251)). Categorical `ColumnView::codes()` can
also allocate and materialize recoded values despite returning `ArrayView`
([data.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/data.cpp#L621)); its interface already mentions recoding.

Document these acquisition/materialization costs where callers encounter them.
Selected string copies could size and fill the final packed byte/offset buffers
directly, retaining order, duplicates, empty strings, and UTF-8 behavior. The
practical benefit needs measurement if implemented; the extra representation
and traversal are visible in the source already.

### 5. Internal sparse transfers repeatedly resolve dtype and strategy

`scan_sparse()` calls dtype-switching `read_index()` and `write_index()` for
stored and emitted entries
([backed_array.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/backed_array.cpp#L314),
[inner loop](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/backed_array.cpp#L449)). `MinorLookup::each()` also chooses
its selection strategy per source index; its bounded-memory fallback can scan
the entire explicit selector for each stored index
([backed_array.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/backed_array.cpp#L346)).

These are internal transfer loops, distinct from the caller's acquired pointer
loops. Consider selecting typed and selector-specific transfer routines once
per operation or block. Preserve mixed sparse widths, safe unaligned loads,
required range checks, duplicate selections, and bounded scratch. Avoid a large
cross-product of template instantiations. Compiler optimization may already
remove some source-level dispatch; runtime and build effects need measurement.

Sparse preparation itself scans the selected structure to count output entries,
then filling scans again. Reusing `PreparedRead` can amortize preparation;
convenience `read_into()` prepares anew each call
([backed_array.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/backed_array.cpp#L483),
[read_into](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/backed_array.cpp#L895)). Make that cost clear. The separate
validation pass before sparse writes protects against partial writes and should
not be collapsed casually.

### 6. Some short source is difficult to read as an algorithm

`Store::resize()` packs chunk traversal, allocation of fill data, nested cell
traversal, and writeback onto one long line
([zarr_engine.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/zarr_engine.cpp#L429)). Its `coordinates()` helper uses
a type-erased callback ([zarr_engine.cpp](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/src/zarr_engine.cpp#L207)).

Unfold the existing control structure into concise visible loop bodies when
this code is next touched. Keep related arithmetic close together and make
buffer lifetimes apparent. A local templated traversal may be simpler than the
erased callback, but no runtime cost is established by this review. This calls
for local clarification rather than a new traversal abstraction or a repository
formatting sweep.

## What already fits

- Typed dense views expose a pointer, extents, and signed element strides.
  Sparse views expose the actual buffers and independent integer widths.
- Owners are move-only, deep copies are explicit, and retained versus borrowed
  lifetimes are deliberate. The common small dense ranks use inline dimensions.
- Caller-owned backed buffers, explicit materialization, and restricted mapping
  accurately distinguish addressable storage from encoded storage.
- Dense transfers visibly separate direct rectangles from bounded packing;
  recent fixes reuse staging storage. There is no application scheduler or
  synchronization in the caller's acquired numerical loops.
- Public module file sets and private storage linkage in
  [CMakeLists.txt](https://github.com/tumlinso/anndatapp/blob/2ddf57559412c90297e5b525bd6a5b6d9a0c36d1/CMakeLists.txt#L57) keep HDF5/JSON/codec headers behind the
  implementation boundary. The core has no third-party storage dependency.

The existing compilation reports demonstrate private-edit isolation for their
recorded source and toolchain ([storage-performance.md](storage-performance.md)).
This review does not attribute a build bottleneck to any particular standard
header or claim that removing it would improve compilation. Keeping interfaces
small and template specialization contained remains the appropriate discipline.

## Recommended scope before extending formats

Start with the regular-selection representation and focused metadata
acquisition. They most directly improve the visibility of ordinary library
usage. Packed string copying and internal transfer dispatch are separable
follow-ups. Keep the working owners, memory descriptors, module boundaries,
and interoperability tests. No new application, numerical dependency, or broad
standard-library replacement is needed to address these findings.

The original review used document/source-reference and Git diff checks; native
tests and benchmarks were not rerun for those documentation-only edits. The
subsequent implementation has separate validation evidence linked above. Native
workflow acceptance is not inferred from local validation; the sanitized
[workflow observation](../planning/selection-workflow-observation.json) records
the available Project Control route without modifying authority state.
