# Backed access and streaming review

This is a source review of the frozen snapshot identified by `build/thin-review-20261105/baseline.json` (HEAD `d2260c2abb37e9028fa42c95d1712ffcaa282b0e`, plus the recorded working-tree state). All source and test references below are relative to `build/thin-review-20261105/source/`; they identify the reviewed bytes, not necessarily later live files. The companion `access-ledger.json` lists each public abstraction/operation family and the private state and algorithm families reviewed. No production code was changed. Findings marked candidate need a reproduction or measurement before a change; priorities concern review order, not an implementation order or a general performance claim.

The central design is worth retaining: metadata and backend dispatch occur at acquisition and transfer boundaries; applications receive numerical buffers and schedule their own work. Explicit worker contexts have private backend resources, and storage geometry separates logical selection from physical conflict units. The two general access routes have very different costs: a one-shot `read`/`prepare_read` re-describes the schema and may scan sparse selections; `prepare_transfer` retains schema, worker backend, and scratch for repeated blocks. The latter does not make metadata acquisition, all allocations, or backend operations free. Dense packed rectangles, sparse native-major transfers, irregular gathers, string metadata, category transforms and unsharded/sharded storage must be judged separately.

## Findings and proposed acceptance

### A01 — malformed axis can index an empty shape (high; reproduced)

`backed.cpp:111–141` returns an empty shape for a validly encoded non-array such as `dict` or `null`; `make_store` at `263–270` unconditionally reads `obs[0]` and `var[0]`. A malformed on-disk AnnData root whose `/obs` or `/var` has that encoding can cause undefined behavior instead of a contextual exception. The same startup code also accepts a non-dataframe encoding with a nonempty first shape without establishing the required axis schema. A focused frozen Debug probe opens a minimal Zarr v2 store with both axes encoded as `dict`; it exits with signal 11 (exit status 139), with core dumps disabled.

**Candidate:** explicitly inspect and validate both axes as dataframe encodings and validate one-dimensional index shape before reading element zero, rejecting malformed roots with an `Error`. Audit other indexed uses of `logical_shape` under external mutation and enforce existing synchronization requirements. **Acceptance:** malformed dict, null, scalar, multidimensional and missing axes in both storage backends fail deterministically under ASan/UBSan; valid empty dataframes still open; test independently produced stores. No intended API migration.

### A02 — one worker context is not a general shared-session guarantee (high; contract and coverage)

`backed.cpp:143–170` can insert into shared `Session::revisions` while creating an apparently read-only handle; `backed_array.cpp:1369–1382` calls `describe` and `storage_geometry` using the original backend before forking it. `Session::open`, its revision map and revision values are ordinary unsynchronized objects (`anndatapp.backed.storage.cppm:211–220`). `docs/streaming.md:16–20` requires context preparation to be serialized with structural mutation, but does not clearly say whether independent concurrent preparations and other ordinary backed operations on the same session are safe. Direct `read`, `read_into`, selection and metadata access reuse the original backend. The existing parallel integration tests at `tests/streaming_interop_driver.cpp:128–155,185–215` use independently opened sessions and prepare contexts serially. They do not establish safe simultaneous same-session handle creation, preparation, or direct API use.

**Candidate:** document an explicit, conservative operation-by-operation concurrency contract now, including serialized descriptor/preparation and session lifecycle operations unless qualified. If concurrent acquisition is desired, design worker-private resources or controlled synchronization outside acquired numerical loops; do not simply add a lock around every element. **Acceptance:** TSan and correctness tests for serially prepared contexts from one session, simultaneous preparation/descriptor access under any newly promised guarantee, direct APIs, close/edit exclusion, overlapping outputs, and independent HDF5/Zarr workers. Treat HDF5 library capability and physical write conflicts separately. A stronger public guarantee may need a deliberately designed API change.

### A03 — category plans have shared mutable pages (high; static risk, coordinated with model review)

Spilled `CategoryPlan` objects contain mutable `page` and `page_begin` (`backed_category.cpp:187–195`); `category_position` changes and reads these through `shared_ptr` (`224–243`). A derived backed category handle may retain and share such a plan (`backed_codec.cpp:765–775`; `anndatapp.backed.storage.cppm:181–207`). Concurrent access to the same spilled plan has a data-race risk, even for nominally read-only operations. General callers are already told to synchronize access (`docs/storage.md:40`); `prepare_transfer` rejects recoded/color overrides, so this is not evidence that independently prepared ordinary numerical worker contexts are unsafe. The risk matters if a stronger read-only shared-handle guarantee is intended.

**Candidate:** specify the restriction and test it; if parallel shared category views are required, supply per-worker paging or an immutable/direct access strategy, with any synchronization outside per-element numerical loops. **Acceptance:** force spilling at a small bounded budget, test same-plan reads across pages and concurrent readers with TSan for whichever usage is promised, preserve nested pruning and bounded scratch. No runtime race test was run for this review.

### A04 — sparse selection dispatch and fallback deserve a measured fast path (medium; candidate)

`scan_sparse` (`backed_array.cpp:525–604`) calls runtime `read_index` for each stored minor index and `write_index` for selected entries; both switch on dtype (`426–465`). `MinorLookup` (`468–523`) optionally builds a hash map, but under its budget threshold fallback may scan the explicit selector for every source index. `write_sparse` scans the selected structure twice to validate before modifying values (`638–706`). Those choices preserve widths, duplicates and validation; they may dominate irregular or opposite-orientation sparse access. They do not prove a slowdown without geometry-specific measurements or compiler inspection.

**Candidate:** dispatch stored and destination index widths once per sparse operation, retain the existing bounded fallback, and test a compact sorted/indexed lookup where it reduces repeated scans within the scratch cap. **Acceptance:** bounded probes covering CSR/CSC, compressed and opposite orientation, gathered/repeated minor selection, wide widths and low budgets; record calls, visited cells, allocated bytes, wall time and source/fixture identity. Preserve logical order, duplicates, explicit zeros, overflow diagnostics, pre-write validation and build/import isolation. No API migration is needed for a private specialization.

A schematic compact direction is to select the integer type once, then keep the inner loop typed; the existing validation and duplicate handling must remain:

```cpp
// Current loop, simplified from scan_sparse
for (std::size_t j = 0; j < len; ++j) {
    auto index = read_index(data + j * width, stored_dtype, node);
    /* select and validate index */
}
// Candidate outline: dispatch once per buffer/operation.
visit_integer(stored_dtype, [&]<class Index> {
    for (std::size_t j = 0; j < len; ++j) {
        auto index = unpack_index<Index>(data + j * sizeof(Index), node);
        /* same selection and validation */
    }
});
```

### A05 — repeated descriptor work and owner forwarding allocate (medium; candidate)

Each `BackedArray::{layout,dtype,rank,extent,shape,index_dtype,offset_dtype}` calls `describe` (`backed_array.cpp:70–125,982–999`), which re-inspects metadata and for sparse arrays enumerates/sorts children and inspects three physical datasets. `read_array` describes again through `read_prepared` (`888–928`); `BackedArray::read_into` goes through a heap-owned `PreparedRead` and re-describes (`956–979,1017–1025`). `BackedAnnData::view()` always makes a shared dataset copy and owner `n_obs`, `n_vars` and many other forwarding operations construct that view (`backed.cpp:381–405`). These are control-plane costs rather than per-value cost, but repeated small reads/accessors are a plausible concern.

**Candidate:** reuse a validated descriptor within each one-shot call and access already-owned dataset state directly for trivial owner queries; explore an explicit, retained prepared descriptor only if workload evidence supports it. Keep structural revision checks and do not cache across uncoordinated mutations. **Acceptance:** count metadata inspections, allocations and bytes separately for one-shot versus prepared dense/sparse operations; compare identical small and large workloads, correctness and stale-handle/close behavior; record BMI and consumer rebuild impact. No implicit global cache.

Illustrative local change, subject to tests and measurement:

```cpp
// Current
std::size_t BackedAnnData::n_obs() const { return view().n_obs(); }
// Candidate: preserve the same open/revision check without allocating a view.
std::size_t BackedAnnData::n_obs() const {
    backed_detail::require_dataset(impl_);
    return impl_->obs.size;
}
```

### A06 — stream planning adds per-request metadata work (medium; candidate)

`BlockStream` owns a vector of lanes, each with multiple dynamic vectors and a retained `BlockRequest` (`streaming.cpp:39–65`). `next` returns that lvalue request by value (`315–339`), copying its shape/conflict vectors, while `find` linearly scans lanes on completion (`212–219`); tuning sorts interval samples in bounded windows (`133–145`). These costs occur per block/control event, not inside caller numerical loops, and have not been isolated from I/O. Resource fields in `StreamSettings` are admission arithmetic, not actual reserved buffers (`streaming.cpp:84–91`); a caller must allocate and enforce them.

**Candidate:** benchmark small, cached transfers and planner-only operations, including allocation count as workers/rank change. If significant, explore caller-supplied request storage, contiguous per-lane metadata, or direct request lookup without making lifetime/conflict safety opaque. Preserve out-of-order completion and serialized stream methods. **Acceptance:** exact grid coverage, conflict-unit ownership across fragments, adaptive transitions, split/configure and no-regression probes; measure control overhead and public BMI/import effects before adopting interface changes. Any changed request ownership requires explicit migration.

### A07 — gathered sparse writes can materialize the complete selection (medium; confirmed path)

`BackedArray::write` catches a non-contiguous sparse source and calls `source.copy()` before validating and writing (`backed_array.cpp:1134–1154`). The comment makes that choice explicit. This allocation scales with selected sparse data and is outside the bounded packing approach used for dense gathered sources; it is not evidence of a full-store copy or a defect in the documented sparse transfer interface.

**Candidate:** expose or implement an incremental sparse selected-source traversal if representative workloads need it; otherwise document the allocation and preserve the straightforward implementation. **Acceptance:** caller-visible peak allocation, duplicate/reordered selection, widths, mismatched topology, zero entries and failure behavior; prove validation occurs before mutation or document any changed transaction semantics. Compare against existing source copy and native-major cases.

### A08 — scratch statistics are not general allocation counters (medium; contract clarity)

`TransferContext::stats` adds backend counters and `!p.tile.empty()` (`backed_array.cpp:1356–1366`). The tile can be replaced or resized in grouped reads (`243–270`), so the boolean reports presence of a current nonempty tile, not the number of tile allocation events. Temporary axis vectors and some third-party allocations are not counted. `docs/streaming.md:68–80` already limits scope but public `scratch_allocations` does not describe its exact semantics.

**Candidate:** name and document the existing statistic precisely, or count actual tile allocation events and define every included/excluded source. **Acceptance:** exercise tile growth, replacement, empty operations and repeated reads; compare against a scoped allocator observation, report retained capacity separately from cumulative allocations. A public rename is a compatibility decision; do not silently change metric meaning.

### A09 — worker failure and stream ownership need an explicit recovery contract (medium; design gap)

`next()` marks a lane pending, while `complete()` and sparse-only `split()` release it (`streaming.cpp:315–371`). There is no explicit cancel/failure transition. A caller can retry an issued request where safe or abandon the stream; it must not complete a failed block as successful. Storage errors can leave earlier completed shards changed (`docs/streaming.md:91–96`). Caller code must drain other workers before structural changes and account for partial writes; there is no universal rollback.

**Candidate:** document valid retry/abandon behavior and the status of partial output first; add a cancellation/reset operation only if users need recovery without reconstructing a stream. **Acceptance:** injected backend read/write errors, retry and abandonment with out-of-order workers, sharded partial failure, no duplicate coverage or accidental release of a conflict unit, plus independently reopened output. Any new recovery API must define whether a failed write can be replayed.

### A10 — distinguish essential API from forwarding and bridge surface (low; maintainability)

`BackedAnnData` and `BackedAnnDataView` repeat extensive accessor surfaces (`anndatapp.backed.cppm:249–289,318–366`), and `read`/`to_memory` and `unsafe_data`/`unsafe_dense` are aliases. The exported `detail` bridge functions and public `BackedAnnData(std::shared_ptr<detail::BackedDataset>)` (`366–378`) are implementation plumbing observable in the public module. Separate read-only view versus owning writable access carries meaningful capability; aliases and bridges should not be removed solely to reduce line count.

**Candidate:** inventory actual installed-consumer use, move bridges to an implementation-only import boundary if C++20 module linkage supports it, and consider removing redundant aliases only in an explicitly approved API migration. **Acceptance:** installed core/backed/facade consumers and optional backends, public module build/import timing, lifecycle and const/writable semantics. Record any source break and migration spelling.

## Parallel access matrix

| Operation | Current contract and evidence | Review disposition |
| --- | --- | --- |
| Acquired in-memory numerical buffers | Ordinary C++ memory rules; no library lock or backend dispatch in the caller's loops (`docs/storage.md:22,40`). | Retain direct pointer/stride access; shared writes require caller coordination. |
| One `TransferContext` | One per worker; it holds mutable staging, counters, sparse requirement cache and backend resources (`backed_array.cpp:1173–1185`). | Never use one concurrently without external serialization. |
| Multiple contexts from one session | Serial preparation forks backend resources; each context retains the shared session/revision state. Structural edits/close require exclusion. Threaded integration tests use separate sessions. | Qualify serial-preparation same-session execution separately; concurrent preparation remains unqualified. Probe status below. |
| Ordinary direct backed reads, selects and metadata on one session | May access the original shared backend and mutate the revision map; no general concurrent guarantee established. | Serialize unless a narrower operation/backend combination is separately qualified. |
| Independent sessions and contexts | Real parallel integration executes disjoint Zarr read/write work; HDF5 capability must be queried and HDF5 write streams are serial (`docs/streaming.md:98–115`). | Preserve backend-specific scope; tests are evidence for fixtures, not all filesystems or all APIs. |
| Dense Zarr writes | One stream owns a chunk, or whole shard, across its fragments; separate streams/external writers coordinate overlap (`docs/streaming.md:105–110`). | Require non-overlapping physical units; no cross-stream transaction. |
| Sparse access | Prepared transfers support full minor axis and consecutive native major blocks; no parallel sparse structural writes (`docs/streaming.md:28–38`). Cached requirements must be discarded after unsafe index/offset changes. | Keep output buffers private and synchronize any structural edit. |
| Close, structural edit, external mutation, mappings | Caller drains/synchronizes workers; close and affected replacement invalidate contexts. Mappings have independent lifetime but replacement/truncation requires coordination (`docs/storage.md:26,61–63`). | Test explicit errors and retained mappings without assuming filesystem atomicity. |
| Transformed categories | Spilled plans can share a mutable page; direct same-plan parallel access is unqualified (A03). | Explicitly synchronize or provide private page state if a broader promise is chosen. |

Scratch is per worker for library-owned payload staging and certain grouping/index/codec allocations; caller output, persistent schema metadata, HDF5 caches and codec internals are excluded (`docs/streaming.md:68–80`). Failure may leave partial results, notably completed shards in a write. A geometry capability bit does not authorize overlapping writes, remove HDF5's own constraints, or guarantee speedup.

## Evidence and next validation

The source tests cover transfer dtype/shape/capacity errors, retained lifetime and invalidation (`tests/test_transfer_context.cpp`), stream coverage/adaptation/out-of-order completion (`tests/test_streaming.cpp`), and independent-worker read/write interoperability (`tests/streaming_interop_driver.cpp`). Focused TSan presets exist (`docs/development.md:65–88`); prior receipts and benchmark summaries apply to their own recorded source hashes, not this snapshot. `docs/gather-read-performance.md` records grouped sharded-read observations, so the earlier gather screen is not the current implementation baseline. No runtime, build, allocation or performance measurements are asserted here unless listed in the probe record below.

A focused executable linked against the frozen Debug static library used one read/write Zarr session, prepared four independent X read contexts and four contexts for separate layer arrays serially, then ran ten concurrent read/write iterations per worker with complete value checks. Both Zarr v2 and v3 passed. This is a correctness observation for those stores and distinct physical arrays; no race detector was used, and it does not qualify concurrent preparation, shared contexts, overlapping physical write units, HDF5, or arbitrary metadata access. The same executable reproduced A01 against a manually constructed minimal Zarr v2 hierarchy; the process exited 139 before an exception could be printed.

The reproducible [probe source and setup](probes/access.md) are tracked with this review; the ignored binaries and fixtures are in `build/thin-review-20261105/access-probes/`. SHA-256: `probe.cpp` `d224efdbede5632bd461cdcd686ab8eb8235d8a9200ccb397f825ab0a4417c4f`; `probe` `c2da7b6f256e57052df0ac50caf13fcd66abfefa1340c60b7c9069011b3834b1`; frozen Debug `libanndatapp.a` `dcb67df70229b3c2df1eaeb7e7ea5a0de998b365a3de794456c74e884ba46c9f`. The probe used four workers and ten iterations without performance timing. No production code changes or new suite test were made.
