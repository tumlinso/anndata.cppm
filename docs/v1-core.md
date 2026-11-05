# V1 core cleanup

The cleanup preserves move-only owners, retained bindings, native C++ scalar
identity, explicit deep copies, sparse widths/order/duplicates, packed UTF-8,
and caller-controlled parallelism. It changes the six arrays/strings/records
source files; core/facade module names and partitions are unchanged.

The baseline is the preserved working tree under
`build/v1-cleanup/baseline-source`. Review findings remain historical in
`docs/review/core.md`; this document records their implementation disposition.
Runtime and build claims require the separately coordinated paired campaign.
Checked arithmetic keeps optimized compiler intrinsics behind feature detection;
portability does not require replacing them with recurring division on supported
compilers. Both paths preserve the same overflow diagnostics.

| Finding | Disposition |
|---|---|
| C01 | Selected strings and string-view initializer lists pack directly into final byte/offset buffers; record text copies also size and fill final buffers. Internal selected traversal no longer recomputes shape products through checked scalar access. Final selected-copy UTF-8 validation remains. |
| C02 | Owning dense copies allocate their result then use the same `copy_into` traversal. The existing trusted owning-copy contract is preserved by omitting whole-source validation on that route, including huge zero-stride source geometries. |
| C03 | Selected sparse copies use typed count/fill traversal. All-minor selections copy complete stored runs; affine filters use arithmetic; gathers use bounded direct duplicate chains for compact axes and a sorted duplicate-aware vector for large/gapped axes. Output-sized source-position/index staging and hash-node vectors are removed. |
| C04 | `borrowed_dense` accepts pointer/rank/shape/signed element strides, copies metadata inline through rank four and owns larger metadata, and exposes checked/trusted validation. Internal descriptor constructors are private. |
| C05 | All-selector composition returns the existing array/string binding before allocating selection state, including rank zero and already selected views. |
| C06 | `RecordArray` has focused const `array`, `string_array`, `member<T>` and `text` access. Mutable member acquisition also avoids a full-schema snapshot. Numerical member metadata uses inline dimensions without a temporary stride vector. |
| C07 | Record schema has one ordered field list. Item size/alignment are scalar state; interleaved records do not allocate fieldwise payload vectors. `record_layout()` constructs its explicit returned metadata. |
| C08 | One private allocation owner retains the native record buffer and stable copied text arena. Copies create no extra text-owner/owner-pair layer, including numeric-only and fixed-text records. |
| C09 | Native member construction dispatches per field. Record copies use a cached byte stride at rank zero/one and an incremental coordinate walk at higher rank. |
| C10 | Checked array add/multiply use compile-time feature-detected overflow intrinsics with standard limit-test fallbacks; other range checks remain standard arithmetic. Compile-time checks declare the supported eight-bit-byte, one-byte-bool, IEEE binary32/binary64 scalar representation. Additional compiler/platform qualification remains explicit. |

## Public acquisition and migration

```cpp
const std::size_t shape[]{rows, columns};
const std::ptrdiff_t strides[]{std::ptrdiff_t(columns), 1};
auto x = anndatapp::borrowed_dense(p, 2, shape, strides);
```

The factory returns `BorrowedDenseView<T>` with pointer constness following `T`.
It copies shape/stride metadata, so the stack arrays can expire or change.
It borrows the numerical bytes: the caller still owns their lifetime.
`Validation::full` checks pointer alignment, nonempty storage and reachable
address arithmetic; `Validation::none` trusts storage preconditions while
copying and checking representable descriptor metadata. No numerical allocation
or State wrapper is created. Ranks above four allocate descriptor metadata.

Replace direct construction using `detail::DenseDimensions` with this factory.
Descriptor-taking constructors of retained dense/sparse/record views are now
private. Obtain them through `Array::view/edit`, `RecordArray::member`, or the
captured-view accessors. These are deliberate pre-v1 source changes; there are
no compatibility shims or header-based public alternatives.

```cpp
auto x = records.member<double>("x");
auto labels = records.text("label");
```

Focused numerical descriptors retain the selected numerical allocation or the
interleaved record/text owner; text retains one captured field. Replacing,
erasing or destroying the owner does not retarget these bindings. Full
`records.view()` still captures every field when that is requested. Interleaved
fields still require byte-strided member access or an explicit copy; focused
`array`/`string_array` do not materialize incompatible storage. Byte-stride
conversion normalizes unused singleton axes, and every axis of an empty field,
to zero so that valid unused element strides need not fit after scaling.

## Source-cost evidence and validation

Selected string copy has no intermediate `vector<string>` or individually
owned string payloads. Sparse selected copy retains one major-offset vector and,
only for gathers, either bounded `head`/`next` arrays or one `(source, output)`
coordinate vector; it no longer retains
two `size_t` staging arrays per emitted nonzero. Typed structural specialization
is private and excludes the numerical-value dtype cross product. The deliberate
tradeoff is a count/fill source scan for filtered sparse copies and private code
growth, to be judged by paired measurements rather than assumed faster. Direct
lookup is limited to at most 1,048,576 source coordinates and a source extent
no greater than four times the requested gather count; the comparison avoids
multiplication overflow. Reverse chain construction preserves output order
within repeated source coordinates. Large/gapped sources do not allocate that
source-sized table.

Interleaved schema names are stored once instead of both a layout list and
field list. Numerical/fixed-text copies retain one allocation owner instead of
constructing two additional shared objects. Rank-one focused member acquisition
uses direct retained ownership without allocating a whole captured schema.
These statements describe source mechanisms, not measured runtime gains.

Regression additions cover copied stack metadata and rank-zero/high-rank
borrowing, negative/zero strides, checked/trusted errors, gathered owning-copy
parity with `copy_into`, direct packed-string output and invalid UTF-8 detection,
and focused record lifetimes through replacement, erase and owner destruction.
The existing raw sparse-copy, native-identity, serialization-hint, malformed
layout and empty/broadcast geometry tests remain enabled. Integrated build/test
results and standardized paired evidence are recorded by their owning campaign.

Concurrent read-only acquisition and disjoint pointer writes retain ordinary
C++ memory rules. Gathers, broadcast strides and external aliases can overlap;
callers synchronize structural mutation and aliasing. The cleanup adds no
thread pool, shared mutable cache, or per-element library synchronization.
