# Data subsystem review

Reviewed the complete frozen [`anndatapp.data.cppm`](../../src/anndatapp.data.cppm) and [`data.cpp`](../../src/data.cpp). The companion [ledger](data-ledger.json) covers every declared type and public operation family, plus every private state and helper family; grouped overloads are named in their entries. Source links use repository paths; line references and hashes identify the frozen reviewed source, which can differ from later checkout contents. This is a source review, not a benchmark, file-format qualification, or a proof of concurrent mutation safety. No production code was changed. Frozen SHA-256: interface `887693cc4a7a2e7cd17cf39baefa90af741d4ee2da90ec512334e1881dcb82a9`; implementation `c6870730ed08a4855b2da9dda3fa07ecaa3ed5f299f223534fc896807703fe88`.

The overall separation is sound: move-only owners retain payloads through views, array access supplies native data through the array API, and explicit `copy()` materializes data. Alignment checks at setters and validation protect the biological container's observation, variable, pairwise and raw relationships. Snapshots deliberately preserve structural bindings while underlying mutable payloads remain shared. The implementation has no library synchronization for those payloads; callers must coordinate mutations, concurrent structural changes, and overlapping writes. Read-only use after acquisition must also exclude concurrent writes to shared state. In particular, shared ownership is a lifetime mechanism, not a data-race guarantee. `Validation::none` is an intentional trusted-input boundary, not evidence that invalid offsets or codes can safely be used.

## Findings

### D01 — Medium: a null C-string selector enters undefined behavior

The public [`Selection(const char*)`](../../src/anndatapp.data.cppm#L85) constructs `std::string_view(v)` unconditionally ([implementation](../../src/data.cpp#L354)). A null pointer does not designate a valid C string; length discovery is undefined. By contrast, the nearby metadata C-string constructor explicitly rejects null ([data.cpp:830](../../src/data.cpp#L830)). The API declares no null precondition. Reject null with `ErrorCode::invalid_argument` before constructing the string view, preserving the normal label path and error conventions. Acceptance: a null-selector regression must throw the documented error without dereferencing, and ordinary empty and nonempty labels and duplicate-label semantics must remain unchanged. If maintainers intentionally choose a non-null precondition instead, document it explicitly; this would retain the caller hazard.

### D02 — Medium: pruning plans are copied before they are rebuilt

Every [`ColumnViewImpl`](../../src/data.cpp#L38) embeds two vectors plus a flag, even for numeric, string and unpruned categorical views. [`select`](../../src/data.cpp#L629) copies the whole descriptor, then [`select_all`](../../src/data.cpp#L638) copies it again, scans selected codes, and replaces the copied pruning vectors. [`TableView::with_rows`](../../src/data.cpp#L793) has a similar descriptor copy. The scan first calls `code(i)` for validation and then reads the original code again. This retains important stale-code detection, but duplicates map copying and code acquisition on a pruned path.

Build a new descriptor from the source and composed rows, validate the original code against the old plan once, and build the new plan directly; preserve behavior for `prune_categories=false` and current dictionary-relative mappings. Separately evaluate the earlier R7 proposal for an immutable shared pruning record: it removes inactive inline vectors and shares existing plans, but adds allocation and indirection on pruning. A nonnull empty plan must still differ from no plan. Acceptance: compare all/mask/gather/repeated/reversed/empty and nested selections, missing codes, dictionary ordering, signed and unsigned widths, stale-code detection after edits, retained lifetimes and `codes()` identity cases. Count allocations and map bytes in a bounded representative probe before selecting a representation. No speedup is established here.

### D03 — Medium: categorical copies stage and copy recoded codes more than once

For nonidentity pruning, [`codes()`](../../src/data.cpp#L612) builds an `int64_t` vector, then [`make_codes`](../../src/data.cpp#L208) copies the selected typed array and overwrites each element. [`ColumnView::copy()`](../../src/data.cpp#L658) subsequently calls `.copy()` on that returned view. Besides the temporary values vector, that path creates a typed output and then creates another typed copy. For identity or unpruned codes, the selected view is borrowed and one deep copy is required.

Investigate a private typed materialization operation that allocates the final output once, writes validated mapped codes directly, and preserves original dtype and byte-order metadata. The public `codes()` still needs a retained result for a recoded view; `copy()` can consume a separately returned owner or write its own final output. Retain the identity fast path's stale-code validation when dictionary truncation leaves an identity prefix. Acceptance: cover negative missing codes, every integer width and unsigned range errors, nested/stale selections, selected strides, empty columns, ordered flag and byte order, plus ownership after source destruction. Compare allocation/copy counts and identical-workload timings before claiming benefit.

### D04 — Medium: selected strings are unpacked into individual strings before repacking

[`ColumnView::copy()`](../../src/data.cpp#L670) builds a vector of owning strings and calls `Column::strings` or `nullable_strings`, which iterates those strings again to build packed bytes and offsets ([data.cpp:456](../../src/data.cpp#L456)). The initializer-list factories similarly create individual strings before packing ([data.cpp:470](../../src/data.cpp#L470), [485](../../src/data.cpp#L485)). The operations and their intermediate allocations are not obvious from the signatures.

Investigate a direct packed-copy path that computes or accumulates offsets and copies each selected byte range into the final buffers; use the existing packed factory or a private shared pack routine without introducing public template machinery. Preserve selection order and duplicates, zero-length strings, null values distinct from empty values, sentinel `NA`/`NaN`, UTF-8 validation boundary and exception behavior. Full validation of previously unchecked input must not be silently omitted. Acceptance: compare packed bytes, offsets, validity and errors for trusted/full construction and reverse/gather/empty selections; count allocations and copied bytes on representative inputs before drawing performance conclusions.

### D05 — Medium: structural snapshots eagerly scale with all bindings

[`Table::view()`](../../src/data.cpp#L738) allocates a table-view descriptor, copies index and column names, and acquires a separately allocated descriptor for every column. [`AnnData::view()`](../../src/data.cpp#L1127) snapshots all aligned slots and wraps each aligned element, while [`slice()`](../../src/data.cpp#L1218) builds selected bindings, prunes categorical columns and adjusts category-color metadata. These allocations are structural, not deep copies of numerical payloads; they serve the retained binding contract. They can matter for repeated acquisition on wide tables.

Keep snapshot semantics and the existing focused [`obs_column` / `var_column`](../../src/data.cpp#L995) path. Investigate sharing immutable binding records or reducing redundant descriptor work only if bounded wide-table and repeated-slice evidence justifies its extra ownership/mutation complexity. Acceptance: old views must retain original bindings and names across owner replacement, selected maps and category colors must compose correctly, raw observation alignment must remain intact, and value edits must retain their documented visibility. Measure acquisition allocations/bytes and consumer build effects separately; callers can already reuse a view outside numerical loops.

### D06 — Low: categorical validity resolves the same row twice

[`valid`](../../src/data.cpp#L565) calculates `r` before calling `code(row)`, which independently checks and resolves that row. The existing passage:

```cpp
auto r = impl_->row(row);
if (kind() == ColumnKind::categorical) return code(row) != -1;
return impl_->source->validity.empty() || impl_->source->validity[r] != 0;
```

A direct equivalent keeps the numeric/string row check and lets `code` perform the categorical check once:

```cpp
if (kind() == ColumnKind::categorical) return code(row) != -1;
auto r = impl_->row(row);
return impl_->source->validity.empty() || impl_->source->validity[r] != 0;
```

This preserves the preceding null guard, error ordering for valid kinds, and stale-code checks. It changes no ownership or allocation. Verify out-of-bounds and stale categorical behavior; this is a source-level simplification, not a measured speed claim.

### D07 — Medium: public cost and lifetime contracts are incomplete at acquisition points

The interface documents possible recoding for `codes()` but not its possible scan and materialization ([interface:127](../../src/anndatapp.data.cppm#L127)); nor does it describe eager table/AnnData binding snapshots, `names()` and key-list copies, pruning during `select`, or the lifetime of returned `string_view`s ([interface:121](../../src/anndatapp.data.cppm#L121), [196](../../src/anndatapp.data.cppm#L196), [401](../../src/anndatapp.data.cppm#L401)). Also, the retained read-only views do not freeze shared values against writes through an owner or edit handle. This makes it harder for callers to put acquisition outside parallel loops or reason about string-view lifetime and synchronization.

Add short boundary comments describing the actual work and invalidation/retention rules, especially that string views borrow bytes and require their retained owner/view to remain alive, descriptor snapshots retain bindings rather than freeze contents, and concurrent mutation needs caller coordination. Check the underlying array and string contracts before specifying cross-object details; do not promise immunity to changes made through borrowed external storage. Acceptance: documentation agrees with construction, replacement and edit behavior, and test coverage demonstrates the distinctions without requiring library synchronization or making public implementation bodies larger.

### D08 — Low: inactive representation state and an avoidable empty owner merit isolated evaluation

[`ElementImpl`](../../src/data.cpp#L64) simultaneously contains all scalar fields, string, optional owner/view alternatives and a mapping, although one semantic kind is active. [`ColumnImpl`](../../src/data.cpp#L23) and public [`Selection`](../../src/anndatapp.data.cppm#L63) likewise carry inactive kind-specific fields. Separately, [`Table::with_index`](../../src/data.cpp#L717) default-constructs a shared empty names vector and immediately replaces it with the supplied one. These are source-visible state/allocation facts, not measured costs or proof that a variant is better.

Evaluate a private active-payload representation for `ElementImpl` and direct initialization of table index ownership, keeping total lifetime switches and recursion understandable. Public `Selection` changes additionally affect ABI, BMI/import and consumers; do not fold them into a private-only cleanup. A private tagged union or variant may reduce inactive state but can increase branching, generated code or implementation complexity. Preserve scalar widths and byte order, owned versus selected-view payloads, mapping mutation, retained snapshots, nullable empty columns and error behavior. Acceptance: compare `sizeof`, construction/destruction/access code, allocation counts, module rebuild effects and representative timings; test every alternative and lifetime, and adopt only a net justified change.

## Reconciliation with prior records

The older [style review](../style-review.md) identified whole-table work for single-column access and materialized table-index comparisons. In this frozen source, `AnnData::obs_column` and `var_column` call `Table::column` directly, and `same_index` compares the underlying vectors directly ([data.cpp:162](../../src/data.cpp#L162), [754](../../src/data.cpp#L754), [995](../../src/data.cpp#L995), [1028](../../src/data.cpp#L1028)). The general `obs().column()` expression still acquires a whole table snapshot, so D05/D07 describe the remaining cost without reopening the fixed direct-access and index issues. The prior note that `codes()` can recode remains applicable and is expanded in D03/D07.

The [compliance review](../compliance-review.md) deferred R6 active metadata payload and R7 pruning representation; D08 and D02 retain these as candidates, not approved rewrites. The [fix record](../compliance-fixes.md) says S1 aligned wrappers and S2 scalar dispatch were fixed. The current slice has one aligned wrapper allocation per element ([data.cpp:1234](../../src/data.cpp#L1234)); scalar copy uses one switch and preserves dtype and byte order ([data.cpp:941](../../src/data.cpp#L941)). Those issues are not repeated. Past validation and timings apply to their recorded source and workload; this review did not rerun them or conduct timed experiments.
