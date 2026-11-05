# Core memory and record abstraction review

This review covers all 2,514 lines of the eight assigned source files: the core
and facade modules, and the arrays, strings, and records interface/implementation
pairs. Evidence refers to the frozen working tree at
`build/thin-review-20261105/source`, created 2026-11-05T14:24:14Z from HEAD
`d2260c2abb37e9028fa42c95d1712ffcaa282b0e` **with the recorded uncommitted changes**.
The per-file hashes and complete abstraction/operation inventory are in
[core-ledger.json](core-ledger.json). Source links below are convenient checkout
links; frozen hashes, rather than HEAD alone, identify the reviewed bytes.

No production code changed. This review ran no runtime timings, allocation
probes, or native tests. Costs described as current are visible operations and
representations in the frozen source, not measured latency or an established
application bottleneck. Prior benchmark results are explicitly historical.

## Assessment and retained design

The numerical boundary is already close to the requested library: an acquired
dense or sparse descriptor exposes the real pointers, extents and strides;
caller loops require no library execution engine, per-element type dispatch,
allocation or synchronization. Large owners are move-only, deep copies are
explicit, arbitrary gathers remain representable, and retained descriptors
survive owner destruction or replacement. Borrowed handles deliberately leave
storage lifetime with the caller. These contracts earn their abstractions.

Keep `DType` and `NativeType` distinct. Equal-width native integer types need not
be interchangeable C++ objects. Keep byte descriptors for codecs/adapters and
independent sparse value/index/offset types. `RecordMemberView` correctly uses
byte strides and live member objects rather than pretending a record member is
a `T[]` across rows. Keep the cached first stride: historical disassembly and a
bounded probe motivated it; the measured work is documented in
[records-performance.md](../records-performance.md).

`Axis` is a useful compact shared selection vocabulary, not a framework to
replace. Its affine representation and gathered order/repetitions are necessary
capabilities. The core/facade sources themselves contain only module exports;
keep these boundaries and implementation isolation. Core annotations and
AnnData composition are covered by the separate metadata review.

The repeated tiny accessors in retained, borrowed, typed and byte views express
different lifetime, mutability and stride-unit contracts. A common base class
or generic descriptor framework would need to reduce total machinery without
obscuring those distinctions; no such replacement is justified by this review.
Compact forwarding methods that preserve the acquisition boundary are retained.

The main opportunities concern acquisition and materialization: intermediate
representations, whole-schema snapshots for focused record access, repeated
shape/stride work, and duplicated traversal implementations. Removing useful
views or replacing every standard container would not address those costs.

## Findings

Priorities rank useful follow-up work, not correctness severity: P1 is a strong
source-supported simplification/performance candidate; P2 needs a local design
or portability decision and targeted evidence. All findings below are current
in the frozen source and have **unmeasured runtime impact**.

### C01 — P1: selected strings pass through individual owned strings

**Evidence:** [strings.cpp:246](../../src/strings.cpp#L246),
[strings.cpp:294](../../src/strings.cpp#L294), and
[strings.cpp:318](../../src/strings.cpp#L318).

Selected `copy()` builds `vector<string>`, copying every selected value, then
`from_strings()` sums their sizes, copies their bytes into the final packed
buffers, and validates the packed result. The initializer-list string-view
factory also creates individual owned strings before packing. Long values may
allocate separately; short values still incur object storage and the extra
copy. The repeated `size()` in the selected-copy loop recomputes the selected
shape product. Checked `string(i)` separately computes the product and unravels
coordinates on every call (lines 98–128).

Existing core passage:

```cpp
std::vector<std::string> values;
values.reserve(size());
for (std::size_t i = 0; i < size(); ++i) values.emplace_back(string(i));
return StringArray::from_strings(shape(), std::move(values));
```

Compact candidate, schematic at the packing boundary:

```cpp
const auto n = size();
std::vector<std::uint64_t> offsets(n + 1, 0);
for (std::size_t i = 0; i < n; ++i) offsets[i + 1] = checked_add(offsets[i], string(i).size());
std::vector<char> bytes(offsets.back());
for (std::size_t i = 0; i < n; ++i) {
    auto s = string(i);
    if (!s.empty()) std::memcpy(bytes.data() + offsets[i], s.data(), s.size());
}
return StringArray::from_packed(shape(), std::move(bytes), std::move(offsets));
```

The sketch removes the intermediate owned-string representation; it does not
claim to be a compiled patch. An internal traversal may acquire shape/mapping
once and walk coordinates incrementally, provided that it replaces repeated
work rather than adding a generic iterator framework. Keep final full UTF-8
validation where the existing selected copy performs it: aliases can have
changed bytes since construction. Preserve the existing trusted unselected
copy behavior separately. No public API change is required.

**Acceptance:** empty/scalar/high-rank selections, reverse/stride/gather and
repetitions, embedded NUL, multi-byte UTF-8 and invalid bytes after unsafe edits,
zero-copy input adoption, independent output lifetime, and checked size
addition. Count allocations/requested bytes for short and long strings; then
pair identical workloads before claiming runtime benefit.

### C02 — P1: dense owning copy maintains a second, more expensive walker

**Evidence:** [arrays.cpp:553](../../src/arrays.cpp#L553) versus
[arrays.cpp:621](../../src/arrays.cpp#L621).

`copy_into()` chooses contiguous transfers and trailing runs, resolves scalar
width once, and uses ordinary coordinate arithmetic after boundary validation.
The owning `copy()` has another dense walker: rank/selection work and checked
signed multiplication/addition recur for every element. It copies one scalar
at a time for a non-contiguous view, even when the last dimension forms a
contiguous run. Shape and traversal logic must be understood and maintained
twice. It also uses exception handling to discover a non-stridable gather.

The owning fallback currently includes:

```cpp
for (std::size_t pos = 0; pos < n; ++pos) {
    std::ptrdiff_t offset = 0;
    for (std::size_t a = 0; a < rank(); ++a)
        if (state_->strides[a] != 0)
            offset = checked_signed_add(offset, checked_signed_mul(
                signed_index(original_index(selection_, a, coordinate[a])), state_->strides[a]));
    std::memcpy(output + pos*width,
                input + checked_signed_mul(offset, std::ptrdiff_t(width)), width);
    // Advance the same logical coordinates used by copy_into().
}
```

Candidate structure:

```cpp
auto out = allocate_dense(type, native, shape, false);
copy_dense_into(source, out.edit(), existing_validation_contract);
return out;
```

`copy_dense_into` here means one private traversal shared by the two public
operations, with acquisition/validation outside its element loops. Merely
calling the existing public `copy_into()` could change validation behavior;
extract the common traversal with explicit preconditions instead. Keep native
type identity, serialization hints, empty/broadcast arithmetic and error
behavior. Do not alter the separate raw sparse-copy contract: the existing
[test at test_arrays.cpp:394](../../tests/test_arrays.cpp#L394) deliberately
preserves invalid trusted unselected sparse buffers byte-for-byte.

**Acceptance:** existing arrays and copy-into suites, all scalar widths,
rank 0/1/2/>4, contiguous trailing runs, negative/zero strides, arbitrary gathers,
empty dimensions, huge broadcast indices, and ownership independence. Compare
copy versus caller-buffer acquisition and transfer costs separately; verify
private implementation edits leave consumer BMIs/objects unchanged.

### C03 — P1: sparse selection materializes output-sized staging plus a hash map

**Evidence:** [arrays.cpp:671](../../src/arrays.cpp#L671), especially lines
679–710; helper dispatch at lines 173–210.

For a selected sparse copy, `selected_minor` stores one hash node and a vector
for each selected source coordinate. `source_positions` and `output_indices`
then retain two `size_t` values per emitted entry, plus an offsets vector, while
the final sparse buffers are allocated. This happens even when the minor
selector is affine. Reading and writing indices/offsets also repeatedly calls
runtime-dtype switches inside entry loops. Compiler hoisting is possible; the
source alone does not establish emitted instruction cost.

Current representation:

```cpp
std::unordered_map<std::size_t, std::vector<std::size_t>> selected_minor;
std::vector<std::size_t> source_positions, output_indices, output_offsets;
// Discover output entries, allocate result, then replay the staging vectors.
```

Candidate structure:

```cpp
// Select the all/affine/gather strategy and integer types once.
count_selected_entries(source, selection, final_offsets);
allocate_final_buffers(final_offsets.back());
fill_selected_entries(source, selection, final_buffers);
```

This is a candidate, not an assertion that two scans are always faster.
Compare direct count/fill with the existing one-scan-plus-replay strategy.
Specialize affine filtering arithmetically and retain a duplicate-aware gather
lookup where necessary. Factor numerical byte copies from index/offset
specialization; avoid a value×index×offset template cross product. Keep useful
small-width indices and offsets rather than coercing everything to `size_t`.

**Acceptance:** CSR/CSC; all 8–64-bit signed/unsigned structural widths;
unsorted indices, duplicate stored entries and explicit zeroes; repeated and
reversed selections; empty output; overflow diagnostics; serialization hints;
raw unselected-copy behavior. Measure peak live scratch as well as cumulative
allocation bytes and traversal counts. Public API need not change.

### C04 — P2: the light dense entry point exposes internal dimensions

**Evidence:** [anndatapp.arrays.cppm:156](../../src/anndatapp.arrays.cppm#L156),
[anndatapp.arrays.cppm:431](../../src/anndatapp.arrays.cppm#L431),
[arrays.cpp:364](../../src/arrays.cpp#L364).

`Array::borrow_dense()` borrows numerical bytes, but takes owning vector
metadata and allocates a `State`. Direct construction of `BorrowedDenseView`
avoids that state yet requires a public constructor accepting
`detail::DenseDimensions`. The type is exported with its mutable shared
large-rank overflow state. This is more representation knowledge than a
pointer/shape/stride caller should need.

Current shape of direct construction:

```cpp
BorrowedDenseView<T>(p, detail::DenseDimensions{/* implementation layout */}, contiguous);
```

Candidate interface shape:

```cpp
auto v = borrowed_dense(p, rank, shape, element_strides); // public metadata contract
```

Investigate a minimal factory or public descriptor with explicit metadata
lifetime/copy semantics. Preserve inline small ranks and arbitrary-rank
support; making the common path allocation-free must not silently borrow
temporary shape arrays. Keep retained wrappers for callers who need them.
Avoid replacing vector requirements with a larger span/view framework merely
to rename the dependency.

**Acceptance/migration:** compile stack-array consumers; cover rank 0 and >4,
negative/zero strides, constness and lifetime, native type identity, and checked
versus trusted boundaries. Compare allocation counts and public BMI/consumer
compile cost. Any replacement of the existing direct constructor is a source
API change with an explicit migration example; no binary portability promise
is implied by C++ modules.

### C05 — P2: identity selection allocates before finding that it does nothing

**Evidence:** [arrays.cpp:467](../../src/arrays.cpp#L467),
[strings.cpp:224](../../src/strings.cpp#L224).

Array selection always allocates a shared selection object and initializes its
axes vector, even for all axes; positive rank also allocates vector storage,
and rank zero still allocates the object. String selection allocates a shared axes vector
before normalizing identity axes and possibly discarding it. The earlier
[selection measurement](../selection-validation.md) reports fixed acquisition
allocations; that is historical evidence, not a zero-allocation guarantee.

Current array opening:

```cpp
auto result = std::make_shared<ArraySelection>();
result->axes.resize(rank());
```

Safe narrow candidate after pointer/rank checks:

```cpp
bool same = true;
for (std::size_t a = 0; a < count; ++a) same = same && axes[a].is_all();
if (same) return *this;
```

This returns the existing validated mapping for an all-selector composition,
including nested selected views. Broader affine/gather identity normalization
may be useful, but must account for the traversal needed to establish identity.
Do not add a mutable cache to remove a small boundary cost.

**Acceptance:** rank-zero null-selector call, identity on selected and
unselected views, invalid count/null pointer, empty and gathered identities,
packed-string access, binding lifetime, and allocation counts. This is private
implementation work; update any allocation evidence rather than rewriting its
historical numbers.

### C06 — P2: focused record acquisition snapshots every field

**Evidence:** [records.cpp:276](../../src/records.cpp#L276),
[records.cpp:394](../../src/records.cpp#L394), and
[records.cpp:194](../../src/records.cpp#L194).

`RecordArray::view()` allocates an entire captured schema: shape, layout,
strides, field names/metadata and views. Even `owner.edit_member<T>("x")`
passes through it. Acquiring a fieldwise numeric descriptor then constructs a
separate stride vector despite the final descriptor's inline rank-four
storage. `validate()` also constructs the full view. A retained snapshot has a
real purpose, but the cost of focused access scales with unrelated fields.

Current focused mutable path ends with:

```cpp
return view().member_descriptor(name, type, ref);
```

Candidate structure:

```cpp
// Validate mutability/type, resolve one field, retain that binding only.
return focused_member_descriptor(owner, name, type, ref);
```

The private helper must retain the **selected field's allocation**, not just a
mutable record container whose field may later be replaced. For interleaved
storage, retain the record/text owner and copy the required dimensions. Reuse
the inline descriptor directly instead of allocating temporary small-rank
stride vectors. Full `view()` remains valuable when callers want every captured
binding. A focused const owner accessor, if added, needs an explicit small API
proposal; do not introduce a record-access hierarchy.

**Acceptance:** retained member after field replacement/erase/owner destruction,
read-only external records, fieldwise and interleaved rank 0/1/>4, mixed numeric
and text fields, pointer identity, independent field counts, and acquisition
allocations. Historical rank-one member-loop timings excluded snapshot
construction and do not qualify this change.

### C07 — P2: interleaved schema metadata is stored twice

**Evidence:** [records.cpp:21](../../src/records.cpp#L21),
[records.cpp:31](../../src/records.cpp#L31),
[records.cpp:40](../../src/records.cpp#L40), and
[records.cpp:410](../../src/records.cpp#L410).

Both owner and captured view keep `layout.fields` and another fields vector
whose `info` repeats each `RecordField` (including its owned name). Interleaved
field entries also reserve numeric/string handle slots they never use.
`set_field_serialization_byte_order()` must update both copies. No new
measurement is needed to establish duplicated stored state; whether a new
representation wins overall still needs evidence.

Current invariant:

```cpp
p->layout = std::move(layout);
for (const auto& f : p->layout.fields) p->fields.push_back({f, {}, {}});
```

Candidate invariant:

```cpp
// Keep one ordered field schema; item size/alignment remain scalar state.
// Build the returned-by-value RecordLayout only when record_layout() is called.
```

Prefer one schema source of truth over a new generic variant/registration
framework. Keep field buffers only where they carry fieldwise payloads. Review
complete owner, view, validation, conversion and mutation code before selecting
a representation. This stays private; returned `RecordLayout` semantics need
not change.

**Acceptance:** field order/names, copies and byte-order edits, retained schema
snapshots, replacement, empty schemas, duplicate/overlap diagnostics and
interleaved round trips. Count requested/live metadata bytes by field count and
measure acquisition separately from traversal.

### C08 — P2: every interleaved copy constructs an unused text ownership layer

**Evidence:** [records.cpp:490](../../src/records.cpp#L490), especially
lines 510–514. `Allocation` at line 257 already contains record and text fields.

After allocating the record owner, `copy_interleaved()` unconditionally creates
another `Allocation` for text and an `Owners` object retaining both. Numeric-only
and fixed-text copies have `bytes == 0` and need neither extra object/control
block. Empty pointer-text fields also need no text arena. These are explicit
source-level constructions, not measured allocation counts.

Current passage:

```cpp
auto allocation = std::make_shared<Allocation>(); allocation->text.resize(bytes);
auto owners = std::make_shared<Owners>(Owners{out.impl_->owner, allocation});
out.impl_->owner = owners;
```

Narrow candidate:

```cpp
std::shared_ptr<Allocation> text;
if (bytes) {
    text = std::make_shared<Allocation>(); text->text.resize(bytes);
    // Retain records and this nonempty text arena together.
}
```

A larger private change could allocate final record and text storage under one
existing `Allocation`, eliminating the extra wrapper even for nonempty text.
Compare the complete construction path before adopting it. Keep the arena
non-growing once `Utf8Ref` pointers are installed.

**Acceptance:** numeric-only, fixed UTF-8, all-empty and nonempty pointer text;
source mutation/destruction; copied text addresses; empty record count;
alignment; adoption/deleter lifetime. Count ordinary and aligned allocations,
not only ordinary `new`, because native record buffers use aligned allocation.

### C09 — P2: record row loops repeatedly resolve invariant layout work

**Evidence:** [records.cpp:443](../../src/records.cpp#L443),
[records.cpp:459](../../src/records.cpp#L459),
[records.cpp:516](../../src/records.cpp#L516),
[anndatapp.records.cppm:74](../../src/anndatapp.records.cppm#L74).

`empty_interleaved()` calls `visit_numeric(f.dtype, ...)` inside the per-row
loop. Copying numeric members uses `flat_ptr(i)` for every rank, unraveling
coordinates with division/remainder even for the common rank-one case. Typed
acquired rank-one access already has the cached first stride; the copy paths do
not choose that simpler traversal. The compiler may optimize invariant work;
inspect code generation and measure before calling this a runtime regression.

Current construction shape:

```cpp
for (const auto& f : layout.fields) for (std::size_t i = 0; i < n; ++i) {
    auto* p = data + i*item_size + f.offset;
    if (f.kind == RecordFieldKind::numeric)
        visit_numeric(f.dtype, [&]<class T> { ::new(static_cast<void*>(p)) T; });
    // Other representations follow.
}
```

Candidate shape:

```cpp
for (const auto& f : layout.fields) {
    if (f.kind == RecordFieldKind::numeric)
        visit_numeric(f.dtype, [&]<class T> {
            for (std::size_t i = 0; i < n; ++i)
                ::new(static_cast<void*>(data + i*item_size + f.offset)) T;
        });
    // Utf8Ref/fixed-text loops retain their distinct lifetime/initialization work.
}
```

Choose rank-one byte-stride copying once per field and use an incremental
coordinate walk for other ranks where it is demonstrably useful. Keep live
member object construction and `std::launder` where required; replacing them
with unchecked struct casts would lose the existing native-object contract.

**Acceptance:** all canonical field types; rank-zero and zero extents;
rank-one and high-rank negative strides; padding and fixed text initialization;
boolean representation checks; pointer/reference text lifetimes. Compare
assembly/call counts and bounded copy/construction probes, including build/code
size if specialization expands.

### C10 — P2: compiler and scalar-representation assumptions need a declared boundary

**Evidence:** [arrays.cpp:95](../../src/arrays.cpp#L95),
[arrays.cpp:271](../../src/arrays.cpp#L271),
[records.cpp:174](../../src/records.cpp#L174), and
[anndatapp.arrays.cppm:138](../../src/anndatapp.arrays.cppm#L138).

Overflow checks directly require `__builtin_add_overflow` and
`__builtin_mul_overflow`. The numerical dtype map assumes the usual native
`float`/`double` widths and byte-sized fixed-width integer storage. The qualified
baseline is Clang 18 on Linux, not arbitrary C++20 compilers or scalar ABIs.
`NativeType` already protects equal-width distinct native types; do not remove
that protection in the name of compactness.

Current overflow boundary:

```cpp
if (__builtin_mul_overflow(a, b, &result)) fail(...);
```

Portable candidate boundary:

```cpp
// One private checked-multiply operation: compiler intrinsic when available,
// otherwise equivalent limit tests that avoid overflowing signed arithmetic.
```

Define the supported compiler/platform matrix first. For conventional HPC
platforms, compile-time representation assertions may be clearer than silently
accepting an unimplemented scalar ABI. Keep portable fallback code private and
small; do not add checks to acquired numerical element loops.

**Acceptance:** compiler feature detection, min/max signed overflow boundaries,
zero/negative strides and empty dimensions; native-type identity under relevant
LP64/LLP64 models; module/import and install checks on each declared platform.
If additional compilers/platforms are unavailable, record them as unqualified
rather than claiming portability from source inspection.

## Parallel-access contract

The reviewed core operations contain no shared mutable cache, thread pool,
mutex, or parallel runtime. Independent read-only descriptors can share retained
storage. Disjoint writes through acquired numerical pointers use ordinary C++
memory rules, with the caller responsible for non-overlap and any simultaneous
reads. Reference-count operations occur at retained-handle boundaries; they do
not make shared numerical bytes immutable or synchronize structural mutation.

String offsets, category/record metadata and serialization hints are structural
state. Mutating them while another thread acquires or traverses a descriptor
requires caller synchronization. `Utf8Ref` rebinding transfers no lifetime.
Avoid implying that two distinct logical selections are disjoint: gathers,
zero strides and overlapping external buffers can alias the same memory.
Backend thread-safety is a separate reviewed contract.

Acceptance for future core changes should include concurrent read-only
acquisition/traversal, disjoint writes with serial-oracle comparison, retained
lifetimes across joined workers, and explicit exclusion of concurrent
structural mutation. Use a race detector where the compiler/module/backend
configuration supports it, with limitations reported. Do not add a library
lock to every accessor to manufacture a broader guarantee.

## Coverage and evidence limits

The ledger explicitly records every declared named type/alias and all public
operation families in these eight files, including detail descriptors and local
aliases, private state and helpers, ownership/deleter paths, validation and
conversion. Declarations completed in another assigned file are linked to their
implementation entries. Repeated overloads are grouped with explicit names and
locations. Standard-header occurrence analysis belongs to the repository-wide
include ledger and is not duplicated here.

Historical evidence remains useful for selecting probes: direct acquired-memory
loops in [performance.md](../performance.md), selection allocation counts in
[selection-validation.md](../selection-validation.md), packed-owner consolidation
and bounded ASCII validation in [compliance-fixes.md](../compliance-fixes.md), and
record stride/copy evidence in [records-performance.md](../records-performance.md).
Those source identities and workloads differ from the present frozen tree;
none establishes current whole-library latency, complete Unicode performance,
portable compiler qualification, or backend parallel safety.
