# Memory access and ownership

Packed UTF-8 `StringArray` uses the same move-only ownership and retained-view model as numerical arrays. Its view exposes `bytes()` and 64-bit `offsets()` for unselected packed storage. A lazy selection exposes its original allocation through explicitly named `storage_bytes()` and `storage_offsets()`; use `copy()` to pack the logical selection. Mutable bytes and `unsafe_offsets()` are available through `edit()`. Validation is explicit after unsafe structural edits.

Packed construction moves the byte and offset vectors into one retained private
owner without moving their allocations. External sharing and adoption use that
same lifetime field; borrowing retains only the descriptor. Buffer pointer
constness selects mutable versus read-only access, independently of the lifetime
token's pointee type. Empty owned byte storage exposes a null pointer.

`Column::from_packed_strings()` and `from_nullable_packed_strings()` move caller-provided byte, native-size offset and validity vectors into metadata storage. They preserve empty nullable columns and the NA/NaN sentinel distinction. Full construction validates UTF-8 and offsets; trusted construction can omit that scan. Backed column materialization writes directly into these final packed buffers instead of staging a complete vector of strings.

Numerical `ByteOrder` serialization hints describe preferred file representation, independently of the native byte order used by typed memory access. They do not add checks to numerical loops. The erased sparse-byte descriptor supports codecs and external adapters without generating a template specialization for every combination of value/index/offset types.

## Owners, retained access, and borrowing

`Array` and `AnnData` are move-only owning resources. `copy()` produces independent numerical data and metadata. Moving an owner transfers it without copying buffers. Read-only `ArrayView` and dataset slices retain the referenced storage. Mutable numerical access is explicit through `edit()` or a dataset's `edit_x()`.

```cpp
import anndata;

int main() {
    auto data = anndata::AnnData::empty(1000, 2000);
    data.set_x(anndata::Array::empty<float>({1000, 2000}));
    auto x = data.edit_x().dense<float>();
    float* p = x.data();
    auto sr = x.stride(0), sc = x.stride(1);
    for (int r = 0; r < 1000; ++r)
        for (int c = 0; c < 2000; ++c)
            p[r * sr + c * sc] = 1.0f;
}
```

Typed acquisition checks dtype/layout once. Pointer access and element traversal have no per-element type-erased dispatch, lock, or allocation. Extent/stride lookups check their axis; cache these or use the pointers directly in a hot loop. `empty<T>()` leaves numerical bytes uninitialized; initialize them before reading. `zeros<T>()` requests initialization explicitly.

The no-copy acquisition contract applies to directly addressable array storage. Metadata normalization is a separate operation: obtaining categorical `codes()` after dictionary pruning may allocate and recode a numerical buffer. Use the per-row code accessors or an unpruned column view when that conversion is unwanted. Explicit unsafe changes to parent categorical codes can invalidate a captured pruned dictionary; checked access then diagnoses the stale mapping instead of fabricating missing values.

Retained access descriptors keep allocations alive after the original owner is destroyed or its slot is replaced. They are aliases to bytes, not immutable snapshots of those bytes. A dataset slice captures bindings; later structural replacement does not retarget those bindings. Concurrent reads and disjoint writes can use the caller's threads, OpenMP, SIMD, or other execution systems. Aliasing, overlap, and structural synchronization are the caller's responsibility.

`borrow_dense()` and `borrow_sparse()` wrap externally managed allocations without taking ownership. `share_dense()`/`share_sparse()` retain an explicit external owner. `adopt_dense()`/`adopt_sparse()` accept a C-style deleter and context, supporting caller-chosen allocators without requiring an STL numerical container. For adopted sparse storage the callback receives the values pointer and context; use the context to release separately allocated structural buffers when necessary.

All six wrapping factories accept a final `Validation` argument. The default `Validation::full` checks the supplied buffers; sparse validation scans indices and offsets. Pass `Validation::none` for already trusted data to omit those checks and that scan. The caller then supplies valid metadata, supported types, sufficient storage, and a valid compressed structure. `validate()` remains available explicitly. Ownership/deleter and read-only checks still apply; neither mode inserts validation in the caller's numerical loop.

```cpp
// values, indices and offsets remain owned by the caller.
auto a = anndata::Array::borrow_sparse(
    anndata::Layout::csr, rows, columns, nnz,
    values, indices, offsets, anndata::Validation::none);
auto csr = a.edit().csr<float, int, long>();
float* p = csr.values();
// Run your own kernel directly on p and the structural pointers.
```

Here the typed acquisition must match the actual native types of the supplied buffers. A borrowed typed descriptor can also be constructed directly from compatible pointers and dimensions, without wrapping them in an owning `Array`.

Descriptors obtained from borrowed arrays retain the wrapper's metadata, not the external allocation. The external owner must still keep its bytes alive. The explicitly borrowed typed descriptors returned by `borrow()` also omit allocation retention; keep the referenced allocation and required descriptor metadata alive for their use.

## Layout and selections

Dense strides are signed and measured in **elements**. The data pointer denotes the first logical element, including negative-stride views. External memory must satisfy the declared shape, strides, alignment, element type, and allocation lifetime. Wrapping a pointer cannot establish the size of an unknown allocation.

`data()` exposes the actual strided base pointer. `contiguous_data()` requires row-major contiguous layout. A slice whose selected positions form an arithmetic progression can expose a strided dense view. An arbitrary gather has no single strided pointer representation; call `copy()` to materialize it. Empty dimensions and repeated selections retain their specified semantics.

`Axis` describes one positional selection: `all()`, `stride(first, count,
step, reverse)`, or `gather(positions, count)`. Coordinates and step magnitude
are unsigned; reverse selects `first - i * step`. Gather construction copies
the positions immediately, including the initializer-list form. Returned views
retain their mapping and numerical storage independently of selector lifetimes.

```cpp
auto v = data.x().select({anndata::Axis::stride(8, 4, 2, true),
                         anndata::Axis::all()});
auto x = v.dense<float>();  // rows 8, 6, 4, 2; signed element strides

const anndata::Axis a[] = {anndata::Axis::gather({2, 0, 2}),
                           anndata::Axis::all()};
auto gathered = data.x().select(a, 2);
auto owned = gathered.copy();
```

All and affine mappings compose without expanding their positions. Arbitrary
gather composition builds retained positions; checking whether such a gather
can expose dense strides scans its coordinates. Bounds and arithmetic are
checked when acquiring a selection. A zero low-level stride repeats a position;
a zero step in Python-style `range` remains invalid. Empty and singleton axes
do not evaluate unused progression arithmetic. `select(nullptr, 0)` applies to
rank-zero arrays; positive rank requires a selector pointer and matching count.

**Source change:** the old nested-vector `ArrayView`/`StringArrayView::select`
interface is replaced by pointer/rank and initializer-list `Axis` arguments.
`ColumnView::select` and `TableView::select` take one `Axis`, and
`Selection::resolve` returns an `Axis`. Use `Axis::gather(...)` for former index
lists and `Axis::all()` for former absent-axis selectors. Ordinary
`AnnData::slice(all/range/indices/mask/labels)` calls keep their syntax and
semantics. Backed selection continues to use its existing `Selection` interface.

Native C++ type compatibility also matters: equal-width distinct types do not always permit aliasing. Use the stored element type for checked typed acquisition. Explicit unchecked access leaves its documented type/alignment preconditions to the caller.

## Metadata acquisition

`data.obs_column("quality")` and `data.var_column("symbol")` acquire one
retained column directly. Both owners and dataset views provide these methods.
They match `obs().column(...)` and `var().column(...)` without first constructing
a complete table snapshot. Lookup still scans ordered column names; the direct
route avoids allocating descriptors for unrelated columns.

Existing `obs()`/`var()` table snapshots retain their captured column and index
bindings. A direct column also survives owner moves, destruction, and column
replacement. Numerical writes through an existing alias remain visible. Use
`copy()` for independent data. Missing columns retain the table lookup error
contract.

Table/column selection shares the same `Axis` mapping as arrays. AnnData slices
still prune and recode categories and project associated colors, including for
an all-axes slice. These operations can traverse selected rows and allocate
dictionaries or recoded buffers; compact range mappings do not make the whole
dataset slice constant-time. Selected string `byte_size()` also traverses its
logical strings, and selected string `copy()` materializes packed storage.

## Compressed sparse buffers

```cpp
#include <cstdint>
import anndata;

int main() {
    auto a = anndata::Array::csr<float, std::int32_t, std::int64_t>(2, 3, 2);
    auto csr = a.edit().csr<float, std::int32_t, std::int64_t>();
    csr.values()[0] = 1.0f;
    csr.values()[1] = 2.0f;
    csr.unsafe_indices()[0] = 0;
    csr.unsafe_indices()[1] = 2;
    csr.unsafe_offsets()[0] = 0;
    csr.unsafe_offsets()[1] = 1;
    csr.unsafe_offsets()[2] = 2;
    a.validate();
}
```

Allocation creates writable buffers for construction; populate the compressed structure before treating it as valid data. Normal typed access exposes writable values and read-only index/offset pointers. Explicit unsafe structural access supports assembly and in-place editing. Validation checks bounds and monotone offsets, not canonical ordering or uniqueness. Duplicate entries and explicit zeros are preserved. The library does not silently sort, coalesce, densify, or narrow integer widths.

Selecting sparse arrays stays lazy. Accessing selected compressed buffers requires an explicit copy that constructs the selected CSR/CSC representation; this traversal currently validates the source. An unselected sparse copy transfers directly into its final buffers without a contents scan or O(nnz) staging vectors. It preserves the structural bytes, including bytes deliberately changed through unsafe access; call `validate()` when a structural check is wanted.

Gathering a repeated axis can expand one stored entry into several output positions. The C++ core and backed readers use the same deterministic expansion order. SciPy may place those equal-value copies in a different internal compressed-buffer order; logical coordinates, values, duplicates and requested axis order agree. File reads and unselected copies preserve the stored sparse ordering.

## Boundaries

Standard-library adapters are conveniences rather than a numerical storage requirement. Internal metadata and retained ownership do use standard-library facilities; their acquisition/storage costs are measured separately from pointer loops. In-memory arrays expose host memory. [Backed descriptors](storage.md) require explicit reads into owned or caller-provided buffers; eligible read-only mappings are a separate capability. Future device storage likewise must advertise its actual access capabilities rather than promise a host pointer to data requiring I/O or conversion.
