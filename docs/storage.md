# Backed AnnData storage

`anndata.backed` provides lazy descriptors without changing the in-memory `Array` contract. Format modules provide `open_h5ad()` and `open_zarr()`; the default `anndata` facade exposes both. Local HDF5 files and local Zarr v2/v3 directories are supported for the declared encoding subset.

```cpp
import anndata;

int main() {
    auto disk = anndata::open_h5ad("atlas.h5ad");
    auto cells = disk.slice(anndata::range{0, 4096}, anndata::all);
    auto block = cells.x().read();
    auto values = block.edit().dense<float>();
    const float* p = values.data();
    // Compute directly with p and the descriptor's extents/strides.
}
```

## Explicit I/O and ownership

Opening reads schemas, shapes and encoding metadata. Numerical arrays, table indexes/columns, categorical dictionaries and nested values load only when requested. An all/range selection is compact. Label lookup scans stored indexes; label-list selection retains AnnData's source-uniqueness rule, while a single label can select duplicate names.

`BackedArray::read()` creates an owning in-memory array. `read_into(DenseBuffer)` writes into caller-provided storage, and `prepare_read()` reports sparse NNZ, shapes and required dtypes/capacities before `read_into(SparseBuffer)`. Numerical loops after acquisition use ordinary pointers without library dispatch, bounds checks, locks or an execution pool.

`DenseBuffer` strides are signed byte strides, including negative strides, and its pointer denotes the first logical cell. Empty strides request contiguous row-major output. Acquired typed in-memory views continue to expose strides in elements.

The dataset owner is move-only. Retained storage descriptors keep its session alive after owner destruction. Explicit `close()` invalidates storage operations. Structural replacement/deletion invalidates affected descriptors instead of redirecting them to new data at the same path; captured dataset views are invalidated by structural changes. Already materialized arrays keep their independent allocations.

Dataset views and slices are read-only, including when obtained from a read/write session. Direct array writes are available from the owning dataset's handles after opening with `BackedOptions::mode = AccessMode::read_write`.

Explicit dataset slices prune unused categorical values, recode their dictionaries, and keep associated color palettes aligned, including lazy access and streamed export. An unselected owner/view retains its complete dictionary. Ordinary categorical codes/dictionaries are read-only; `unsafe_codes()` and `unsafe_categories()` expose physical writable encodings on an owning writable handle. Callers maintain invariants and discard derived pruned views after unsafe edits, then validate explicitly.

## Working memory and parallelism

`scratch_bytes` defaults to 64 MiB and `decoded_cache_bytes` to zero. Application-level decoded caching is not implemented; a nonzero cache request is rejected. Streaming export and reindexing operate in bounded tiles; they do not materialize a complete matrix. Caller-owned output storage is separate from scratch. A codec or string value that cannot fit its working allocation produces an explicit diagnostic rather than a full-array fallback.

The initial Zarr codec engine uses complete inner-chunk decode buffers, with conservative subdivisions of the scratch budget. Raise the budget or rechunk stores with oversized inner chunks. Shard reads use byte ranges, and shard rewrites stream old content rather than loading a whole shard. HDF5's own metadata/codec working allocations are separate from library scratch; variable-length attribute allocation can only be accounted after HDF5 reads it.

Forward dense copies use destination codec geometry and one reusable bounded rectangle buffer. Packed forward reads/writes can use the caller's buffer directly; arbitrary gathers retain the bounded fallback. Scratch controls transfer payload/staging, rather than total process RSS; metadata, fixed stack storage and third-party internal allocations are reported separately where measured.

Callers synchronize access and choose parallel execution. HDF5's installed thread-safety capability can be queried. Zarr chunks within one shard share a write-conflict unit. Array shapes, selections, and public source geometry let callers partition work; block streams retain ownership of conflict units across bounded fragments. The library launches no execution pool, and numerical buffers can be processed independently using ordinary C++ concurrency rules.

## Structural updates

For source geometry, reusable worker resources, and caller-controlled adaptive
or manual block execution, see [streaming](streaming.md). Physical geometry is
now public through `storage_geometry()`; it describes source coordinates rather
than promising contiguous memory for a selection.

`edit_structure()` collects replacements/deletions and validates the final axis relationships before applying changes. It supports slots, nested metadata, table columns, sparse replacements and coordinated observation/variable changes. Axis growth requires explicit replacement data for affected aligned slots; no biological fill values are invented. Reindexing transforms layers, multidimensional annotations, both pairwise dimensions and raw observation alignment; raw keeps its independent variable axis.

Updates apply directly in place. Per-object temporary storage permits a bounded rewrite when an HDF5 contiguous array or a sparse structure must be replaced. This is not a whole-store crash transaction. Failures report paths and may leave partial changes; call `flush()` explicitly. Zarr structural changes invalidate consolidated metadata and rebuilding it occurs during flush.

Untouched extension children and attributes remain in the same store. Same-store rewritten objects preserve opaque attributes where the backend can do so faithfully. Export to a new store rejects unsupported children and opaque attributes instead of silently omitting them. The eager in-memory model still has the previously documented H5AD attribute-preservation limits.

## Formats and mappings

Zarr v2 support includes C/F order, chunk separators, fills/missing chunks, Blosc/LZ4, gzip/zlib, Zstandard and variable-length UTF-8. Zarr v3 support includes endian-aware bytes, transpose, UTF-8 strings, compression, CRC32C and indexed sharding. Unknown required codecs/types are diagnosed. Trailing outer codecs after a shard are explicitly unsupported in the initial range-access engine. New Zarr output defaults to v3 with unsharded chunks; existing supported sharded stores can be read and updated.

The logical element subset remains the numerical/string/table/categorical/nullable/mapping/raw types documented in [h5ad.md](h5ad.md). Table indexes are strings. Compound, ragged, complex/half, legacy unversioned encodings and arbitrary extension schemas remain outside this milestone.

`map_readonly()` is explicit and never falls back to an allocating read. Eligible HDF5 storage must be an allocated, unfiltered, contiguous native representation in a regular local SEC2 file. Eligible Zarr mapping is for directly readable native raw chunks, not an arbitrary multi-chunk matrix. Mappings retain the actual file descriptor and memory mapping independently of the store session.

Mapped storage exposes read-only bytes. `unsafe_data<T>()`/`unsafe_dense<T>()` opt into the qualified Linux/Clang typed interpretation; C++20 does not establish ordinary typed-object lifetime merely by mapping file bytes. Callers coordinate overlapping writes, replacement and truncation. Writable mmap coherence is not promised.

See [storage-dependencies.md](storage-dependencies.md) for the exact dependency installation commands and [storage-performance.md](storage-performance.md) for measured workloads and limits.
