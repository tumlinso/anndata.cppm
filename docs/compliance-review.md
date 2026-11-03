# Repository compliance review

The findings below describe the preserved pre-cleanup audit snapshot. For the
subsequent implementation and checks, see [compliance-fixes.md](compliance-fixes.md).
Historical source hashes and measurement inputs remain unchanged.
R1–R5 and S1–S4/S6/S7 are implemented; R6/R7, category-state tagging and S5
remain deferred.

Reviewed the current working tree against the updated written
[Cellerator coding guide](/home/tumlinson/Cellerator/coding_style.md), including
its representation and review requirements. Cellerator's implementation was
not used as an example. **Assessment: partial compliance.** The module boundary
and numerical access contracts are strong; compact representation and direct
expression of algorithms are inconsistent.

Compact, clearer code does not require a speedup for acceptance. It must retain
the required semantics and show no demonstrated degradation. Fast imports and
consumer recompilation remain very high priorities. A smaller call site alone
does not justify more machinery elsewhere.

This was an audit, not a refactor. No implementation, tests, build files,
measurement inputs, or protected guidance were changed. Alternatives below are
source-review fragments or explicitly marked design sketches, not compiled
patches. There are no new runtime, footprint, or compilation measurements.

## Representation findings

### R1. Zarr attributes are owned twice, with one copy unused

[zarr_engine.hpp:26](../src/zarr_engine.hpp#L26) contains both:

```cpp
struct ArrayMetadata { /* parsed array fields */ Json attributes, original; };
struct Node { /* node state */ Json attributes; ArrayMetadata array; };
```

`parse_array` fills the array copy, and `inspect` assigns it again from the
node. The examined source has no reads of `ArrayMetadata::attributes`.
Compact equivalent: remove that member and its two assignments; retain
`Node::attributes` and `ArrayMetadata::original`. The latter is needed for
faithful JSON rewriting during resize. This removes redundant live state and
copies without changing the format model. Allocation-failure points disappear;
attribute preservation and v2/v3 rewrites still need checks if implemented.
This is private storage implementation, with no public module expansion.

### R2. Packed strings have three lifetime handles for one retained object

[StringArrayImpl](../src/strings.cpp#L18) stores:

```cpp
std::shared_ptr<std::vector<char>> owned_bytes;
std::shared_ptr<std::vector<std::uint64_t>> owned_offsets;
std::shared_ptr<void> external_owner;
```

[from_packed](../src/strings.cpp#L317) separately allocates both vector owners.
The compact representation candidate is one lifetime token and one owned pair:

```cpp
struct PackedBuffers {
    std::vector<char> bytes;
    std::vector<std::uint64_t> offsets;
};
// In StringArrayImpl:
std::shared_ptr<void> owner;
```

Move both input vectors into one retained `PackedBuffers`; use its buffers for
the existing pointers and sizes. External share/adopt paths use the same owner
field. This changes two owner/control-block allocations into one, while leaving
the vector payload allocations alone. Preserve empty byte-pointer behavior,
writability, validation, custom deleters, and failure cleanup. The token need
not point at either numerical buffer. The change can remain entirely private.

### R3. Backed index comparison retains a table twice

[backed_structure.cpp:274](../src/backed_structure.cpp#L274):

```cpp
struct Index {
    std::size_t size = 0;
    std::function<std::string(std::size_t)> name;
    std::optional<BackedTable> backed;
    std::optional<TableView> native;
};
```

Both constructors capture the table in `name` as well as storing its concrete
handle. All backed comparison combinations already have dedicated paths;
the callback fallback is native/native. A smaller private representation is:

```cpp
struct Index {
    std::size_t size;
    std::variant<BackedTable, TableView> table;
};
```

Select the existing comparison path once. The native/native loop becomes:

```cpp
const auto& a = std::get<TableView>(left.table);
const auto& b = std::get<TableView>(right.table);
for (std::size_t i = 0; i < left.size; ++i)
    if (a.name(i) != b.name(i))
        fail(ErrorCode::shape_mismatch, path, "dataframe index differs from its aligned axis");
```

This removes callback state, duplicate retention, and native string construction.
Retain the size check, three budgeted backed combinations, and error-path
remapping. A tagged alternative is an option; removing the redundant callback
is the essential change. No new public templates are needed.

### R4. Every chunk copies full array metadata to override two properties

[Zarr chunk reads/writes](../src/zarr_engine.cpp#L277) start with:

```cpp
auto payload_meta = m;
payload_meta.scratch_bytes = scratch_bytes_;
// Sharded branch also substitutes payload_meta.codecs.
```

That copies shape, chunks and full source JSON, although the changed information
is the active codec pipeline and budget. A private borrowed-view sketch is:

```cpp
struct PayloadMeta {
    const ArrayMetadata& array;
    const Json& codecs;
    std::size_t scratch_bytes;
};
// Synchronous codec call:
decode_payload({m, spec.codecs, scratch_bytes_}, shape, payload);
```

Use `m.codecs` for an unsharded payload. Keep `spec` alive through decoding;
the current read function decodes after its branch, so merely saving a reference
to a branch-local spec would be wrong. Preserve original JSON for rewriting,
codec order, exceptions and scratch admission. This is a private codec-interface
candidate, not a proven runtime bottleneck. It must simplify the complete codec
path rather than add forwarding layers.

### R5. Category plans retain transient and unused fields

[CategoryPlan](../src/backed_category.cpp#L187) includes:

```cpp
std::size_t source_count = 0, selected_count = 0, budget = 0;
mutable std::size_t page_begin = /* sentinel */, page_count = 0;
```

The plan's `budget` is assigned but never read. Its `page_count` is used only
during the refill that sets it. Keep only persistent state:

```cpp
std::size_t source_count = 0, selected_count = 0;
mutable std::size_t page_begin = /* existing sentinel */;
```

Make the refill count local at the existing assignment point. Keep the actual
capacity-based scratch accounting, page cache, file ownership, range guards,
and update/read order. This finding concerns these fields specifically, not
the separate `Bitset` cache whose count persists for flushing.

### R6. Metadata elements reserve storage for every alternative

[ElementImpl](../src/data.cpp#L64) has all four scalar fields, a string, eight
optional owner/view payloads, and a mapping simultaneously. Only one semantic
kind is active. A representation sketch is:

```cpp
struct ElementImpl {
    ElementKind kind;
    DType scalar_dtype;
    ByteOrder scalar_byte_order;
    Payload value; // One active scalar, string, owner, retained view, or mapping.
};
```

Here `Payload` means an explicit tagged union or a private variant of the
existing alternatives, not a new general framework. Preserve owning versus
selected-view alternatives, scalar widths/byte order, mutation, and captured
mapping bindings. Compare total construction/destruction/access code before
choosing a mechanism: manual lifetime switches or a large visitor framework
could defeat the source simplification. This is a promising design candidate,
not an established net win. Keep it inside `data.cpp`; consumer BMIs need not
acquire its recursive types. No byte-size reduction has been measured.

### R7. Pruning maps occupy every column view and are copied with it

[ColumnViewImpl](../src/data.cpp#L38) always contains:

```cpp
std::vector<std::size_t> category_rows;
std::vector<std::int64_t> recode;
bool pruned = false;
```

These are unused for numeric/string and unpruned categorical views. An optional
shared immutable pruning record could replace them:

```cpp
struct Pruning {
    std::vector<std::size_t> category_rows;
    std::vector<std::int64_t> recode;
};
std::shared_ptr<const Pruning> pruning;
```

Null means unpruned; a nonnull empty record represents a pruned empty dictionary.
This reduces inline state and allows map sharing, but introduces an allocation
and indirection for a newly pruned result. Preserve original-versus-recoded
coordinates, category order, snapshots and stale-code detection. This tradeoff
needs a short isolated check before adoption; fewer fields alone is insufficient.

## Source and algorithm findings

### S1. Aligned-view wrapping repeats the same operation three times

[data.cpp:1228](../src/data.cpp#L1228) repeats allocation, kind assignment and
insertion in the array, string-array and table branches:

```cpp
if (e.kind() == ElementKind::array) {
    auto ep = std::make_shared<detail::ElementImpl>();
    ep->view_array = select_first(e.array(), selected);
    ep->kind = ElementKind::array;
    dest.emplace_back(name, ElementView(std::move(ep)));
} // The following two branches repeat the wrapper construction and insertion.
```

Compact equivalent for the loop body:

```cpp
const auto& [name, e] = source[i];
auto ep = std::make_shared<detail::ElementImpl>();
ep->kind = e.kind();
if (ep->kind == ElementKind::array) ep->view_array = select_first(e.array(), selected);
else if (ep->kind == ElementKind::string_array) ep->view_string_array = select_first(e.string_array(), selected);
else ep->view_table = current[i].second.table().with_rows(selected, true);
dest.emplace_back(name, ElementView(std::move(ep)));
```

One wrapper is the common operation; only its selected payload differs. Retain
the **current** table branch, needed for existing pruning state, and the same
one allocation and exception propagation. No public interface change is needed.

### S2. Scalar copying dispatches twice and contains unreachable cases

[ElementView::copy](../src/data.cpp#L934) handles four numeric kinds before a
switch, then repeats those cases inside it:

```cpp
if (kind() == ElementKind::boolean || /* the other three scalar kinds */) {
    // Existing metadata-preserving field copy.
    return e;
}
switch (kind()) {
case ElementKind::boolean: return Element(boolean());
// Three more scalar cases, unreachable from the valid scalar path.
```

Group those cases around the existing copy body in the single switch. Keep its
assignments to kind, dtype, byte order and scalar storage; returning through
scalar constructors instead would lose metadata. This exposes one dispatch and
one scalar-copy rule without new helpers or allocations.

### S3. String flattening repeats impossible zero-extent fallbacks

[source_index](../src/strings.cpp#L109) already rejects `flat_index >= count`,
and `view_size` returns zero for any empty selected dimension. The later code:

```cpp
const auto coordinate = extent ? flat_index % extent : 0;
flat_index = extent ? flat_index / extent : 0;
const auto original = (*selection)[axis].index(coordinate);
result += original * stride;
```

can retain the guard and expose the mixed-radix calculation directly:

```cpp
const auto n = selected_extent(*impl, selection, axis);
const auto coordinate = flat_index % n;
flat_index /= n;
result += (*selection)[axis].index(coordinate) * stride;
```

Keep `stride *= impl->shape[axis]` after this block. All values remain `size_t`,
the original ordering is retained, and rank-zero skips the loop. This relies on
the established positive-extent invariant, not on weakening input checks.

### S4. Writer tiles reconstruct coordinates that already exist

[backed_writer.cpp:345](../src/backed_writer.cpp#L345) copies `start` into
`transfer_start`, decomposes `outer` again for a byte offset, and builds another
coordinate vector in the gather branch:

```cpp
auto coordinate = outer;
for (std::size_t axis = shape.size() - 1; axis > 0; --axis) {
    const auto previous_extent = shape[axis - 1];
    const auto index = coordinate % previous_extent;
    coordinate /= previous_extent;
    offset += static_cast<std::ptrdiff_t>(index) * bytes->stride_bytes(axis - 1);
}
```

`start` already contains those coordinates. The corresponding calculation is:

```cpp
std::ptrdiff_t offset = 0;
for (std::size_t a = shape.size() - 1; a > 0; --a)
    offset += static_cast<std::ptrdiff_t>(start[a - 1]) * bytes->stride_bytes(a - 1);
```

Reuse `start` for gather selectors and backend writes as well. Retain descending
accumulation, signed conversion, fallback ownership and the current position
after `packed.resize`; hoisting checks can change exception precedence. This
removes repeated decomposition and copied descriptors, not an abstract layer.

### S5. Two bounded string readers duplicate a traversal

[backed_codec.cpp:182](../src/backed_codec.cpp#L182) and
[277](../src/backed_codec.cpp#L277) repeat coordinate mapping, adjacent-run
discovery, rectangle construction, bounded reading and prefix validation:

```cpp
const auto logical = coordinate(output_shape, flat);
auto start = physical_coordinate(axes, logical, value->path);
std::vector<std::size_t> rectangle(logical.size(), 1);
std::size_t run = 1;
// Same adjacent-run/read/prefix traversal in both readers.
```

A compact organization, **pseudocode**, is:

```cpp
string_batches(value, axes, output_shape, count, budget, [&](auto& strings) {
    // Existing output-specific validation and move/pack operations.
});
```

The helper must contain that complete common traversal once, privately and with
a concrete callable, not a type-erased callback or new iterator framework.
Keep distinct output construction: moving individual strings versus packing
bytes/64-bit offsets. Preserve scalar/empty rectangles, actual prefix length,
validation flags, scratch limits and both packed-size overflow checks. Compare
the complete source before extracting; distinct traversal requirements would
justify retaining duplication.

### S6. Dense formatting hides branch structure and buffer lifetime

The entire resize traversal is on [zarr_engine.cpp:429](../src/zarr_engine.cpp#L429).
Its actual operation is: find affected chunks, clear cells whose global
coordinates are outside the new shape, write chunks, then publish shape.
The existing single line contains nested `coordinates` callbacks, both loops,
fill allocation, removal tests and writeback. Its opening, verbatim, is:

```cpp
coordinates(first,stop,[&](const Shape& coord) { bool touches=false; for(std::size_t i=0;i<shape.size();++i) touches=touches || coord[i]*inner[i]+inner[i]>shape[i]; if(!touches) return;
```

A local equivalent layout of the traversal is:

```cpp
coordinates(first, stop, [&](const Shape& c) {
    bool touches = false;
    for (std::size_t i = 0; i < shape.size(); ++i)
        touches = touches || c[i]*inner[i] + inner[i] > shape[i];
    if (!touches) return;
    auto value = chunk(path, m, c), empty = filled(m, inner, scratch_bytes_);
    Shape zero(shape.size(), 0);
    coordinates(zero, inner, [&](const Shape& p) {
        bool removed = false;
        for (std::size_t i = 0; i < shape.size(); ++i)
            removed = removed || c[i]*inner[i] + p[i] >= shape[i];
        if (!removed) return;
        auto k = flat(p, inner);
        if (m.type == Type::string) value.strings[k] = empty.strings[k];
        else std::memcpy(value.numeric.data() + k*m.item_size,
                         empty.numeric.data() + k*m.item_size, m.item_size);
    });
    chunk(path, m, c, value);
});
```

The same allocations and read/clear/write order remain. No new helper is needed.
Separately, the preceding `overlap` vector at line 424 is unused and can be
removed with its population loop; unlike layout alone, that removes an
allocation-failure point. Similar dense branch packing obscures nullable-string
validation at [backed_copy.cpp:308](../src/backed_copy.cpp#L308). Existing:

```cpp
if(code=="nullable-string-array"){if(values_code!="string-array")fail(ErrorCode::unsupported_encoding,values->path,"nullable string values require string-array");auto it=info.attrs.find("na-value");if(it!=info.attrs.end()){auto value=string_attr(info,"na-value",source->path);if(value!="NA"&&value!="NaN")fail(ErrorCode::unsupported_encoding,source->path,"unsupported nullable string sentinel");}}
```

Equivalent branch structure:

```cpp
if (code == "nullable-string-array") {
    if (values_code != "string-array")
        fail(ErrorCode::unsupported_encoding, values->path,
             "nullable string values require string-array");
    if (info.attrs.contains("na-value")) {
        auto s = string_attr(info, "na-value", source->path);
        if (s != "NA" && s != "NaN")
            fail(ErrorCode::unsupported_encoding, source->path,
                 "unsupported nullable string sentinel");
    }
}
```

This replaces the existing one-line nested branch, preserving validation before
target creation and values-before-mask copying. Layout should reveal these
relations, not expand arithmetic into a chain of trivial helpers.

### S7. A benchmark adds heap ownership solely to cross a branch boundary

[benchmarks/h5ad.cpp:135](../benchmarks/h5ad.cpp#L135):

```cpp
std::unique_ptr<anndata::AnnData> output;
if (action == "seed" || action == "write") output = std::make_unique<anndata::AnnData>(make_data());
// The same condition is tested again before using output.
```

Keep construction and use in one branch:

```cpp
if (action == "seed" || action == "write") {
    auto output = make_data();
    sum = checksum(output.x());
    payload = payload_bytes(output.x());
    // Existing seed/write operations and returns.
}
```

This removes a nullable wrapper and one allocation for the `AnnData` object.
Construction remains before the write timer; no timed-I/O speed claim follows.
Historical measurements keep their original source identity.

## Smaller and conditional findings

- [test_strings.cpp:234](../tests/test_strings.cpp#L234) calls
  `metadata.at("signed");` immediately before the same lookup assigned to
  `signed_scalar`. Delete only the unused first call; the value/type assertion
  remains.
- [examples/foundation.cpp:8](../examples/foundation.cpp#L8) uses a two-line loop
  and `static_cast<float>(i)`. Its compact equivalent is
  `for (int i = 0; i < 6; ++i) p[i] = float(i);`. The explicit arithmetic
  conversion and exact values are unchanged.
- [test_data.cpp:918](../tests/test_data.cpp#L918) packs a 5-by-3 matrix into
  `{0,1,2, 10,11,12, ...}`. Preserve the literal values but show row groups as
  `{ 0, 1, 2,` then `10, 11, 12,` and so on. This exposes row-major geometry;
  formatting is not a reason for a benchmark campaign.
- [storage_interop_driver.cpp:359](../tests/storage_interop_driver.cpp#L359)
  discards a selected view before reindexing the same rectangle. A candidate is
  to retain the helper's dimension `require` directly and omit the view, but
  first confirm the call is not intended to check that slicing itself succeeds.
  It is conditional debt, not a confirmed redundant test.
- `memory.cpp` repeats invariant fields in trial records, but those fields may
  intentionally make rows self-contained. Do not shrink the result schema
  without checking consumers and its evidence contract.
- `CategoryPlan` also combines leaf vector/file/page storage with composed-plan
  pointers. A private `Leaf`/`Composition` tagged alternative could expose the
  exclusivity. This is exploratory: preserve capacity-based scratch charging,
  recursive ownership, file/page behavior and signed recoding checks. Do not
  introduce a visitor framework solely to reduce member count.
- [STYLE.md:3](../STYLE.md#L3) describes compact forms as a preference, while the
  canonical guide requires them by default and requires explicit compact
  alternatives during review. A faithful summary would start: "Use compact,
  explicit numerical/HPC code by default; additional source must contribute
  concrete computational or contractual meaning." The stronger referenced
  guide already applies; the local summary should not weaken it. The protected
  Cellerator guide was not modified.

## What should remain

- Small named-module interfaces, private storage headers, independently compiled
  implementations, and the installed-consumer module provider. The CMake 3.28
  proxy has a documented qualification reason; replacing it cosmetically would
  endanger the import/build contract.
- Independent sparse-buffer owners. Values, indices and offsets may come from
  different allocations; one undifferentiated owner would lose that contract.
- Captured table/index/column bindings. Aliasing a mutable `TableImpl` directly
  would make old views change when columns or names are replaced.
- Explicit owner/borrowed distinctions, mapped-file lifetime/identity handling,
  full sparse prevalidation before writes, and bounded category/string staging.
  Those classes and checks enforce specific contracts.
- Meaningful numerical temporaries and checked arithmetic. Do not trade explicit
  dtype/byte-order/overflow semantics for shorter but different expressions.
- Independent test oracles and exact historical campaign inputs. The ten frozen
  C/C++ probe sources and their Python/shell scripts remain unchanged. Repeated
  instrumentation may be factored for new probes, not rewritten into old evidence.

## Coverage and next cleanup order

All 61 current C/C++ files were read: 29 production files, one example, 15 test
and consumer files, six active benchmark files, and ten historical C/C++ probe
inputs. The four Python tools, three Python test files, 19 historical Python/shell
scripts, CMake files/presets and CI were also reviewed. The C++ guide was not
mechanically applied as Python syntax rules. No actionable infrastructure
rewrite was supported: provenance, cancellation, process isolation and CMake
module support account for most of that machinery. Authored guidance and design
documents were checked for applicable contracts; JSON evidence was inventoried,
not treated as executable source. Generated builds, binary/biological stores,
private credentials and Todo authority were excluded from style review.

Start with R1, R2 and R3, the unused plan state in R5, and local source reductions
S1/S2/S4. R4 requires a codec lifetime/budget check. R6/R7 and category-state
tagging need a separate representation design comparison before implementation.
Formatting/local expression changes need semantic review and affected tests;
allocation/ownership/traversal changes need short targeted runtime/build guards.
No cleanup requires a speedup merely to justify genuinely clearer compact code.

The [audit inventory](../planning/compliance-audit.json) records source and guide
hashes, scope, and exclusions. This review makes no new file-compatibility or
performance claim and does not constitute native Todo acceptance. Validation
was limited to source-reference, inventory, preservation and document checks.
