# HDF5, Zarr and format-extension review

The backend layer is substantially direct and appropriately private, with focused simplification work still warranted. The most consequential boundary finding is **F01**, missing validation of Zarr root paths before filesystem operations. **F02** reproduces loss of metadata discoverability after a failed structural operation on a child accepted through the engine's consolidated-only fallback. The other findings identify concrete source/style costs or bounded investigations; they are not measured performance regressions.

This review fully read the eleven assigned files, **5,078 lines**, in `build/thin-review-20261105/source/src`. The [ledger](backends-ledger.json) records **177 type/helper/operation families**, their source ranges, purpose, costs, disposition and finding references, including retained code. Its SHA-256 manifest identifies the actual frozen bytes: the snapshot includes dirty work and is not described by Git HEAD alone. Baseline HEAD is `d2260c2abb37e9028fa42c95d1712ffcaa282b0e`. All source line references below refer to that snapshot. Production source, build files, protected guidance and other reviewers' artifacts were preserved.

The review applied frozen `AGENTS.md`, `STYLE.md` and the written `/home/tumlinson/Cellerator/coding_style.md`. Public interface/build/include acceptance and the shared backed model have separate owners. No general benchmark, sanitizer or upstream interoperability campaign was run here. Two isolated engine correctness probes were compiled and executed against the frozen source; their scope and reproducibility are recorded below.

## Findings

### F01 — Reject embedded NUL in Zarr roots before filesystem work (P1, boundary correctness)

`backed_zarr.cpp:128–134` passes the root to `stat` before checking its contents; `prepare_destination` (`:239–249`) checks emptiness but not embedded NUL before `exists`, root-marker checks, `remove_all` and `create_directories`. `reject_source_overlap` (`:259–265`) likewise constructs filesystem paths first. The engine validates **node** strings (`zarr_engine.cpp:50–56`) but not its root (`:235–239`). Root and node paths are different boundaries. POSIX C path calls terminate at NUL, while the C++ string/path retains its suffix.

The isolated root `existing_store + "\0unrequested"` reached file I/O and reported `basic_filebuf::underflow error reading the file: Is a directory`, rather than an argument diagnostic. The saved probe catches `std::exception` and does not establish its dynamic type. `zarr_call` catches engine, filesystem and JSON exceptions (`backed_zarr.cpp:38–43`); public-wrapper error translation remains untested for this input. The probe did not invoke truncation. Destructive destination behavior is explicitly untested: its risk is inferred from the truncate path calling removal after root-path checks that omit NUL rejection, not from a demonstrated deletion.

Use the existing HDF5 boundary (`h5ad.cpp:1376–1380`) as the compact form:

```cpp
// Before: filesystem operations consume the unvalidated root string_view.
const auto location = std::filesystem::path(path);
if (location.empty()) throw Error(...);

// After, schematic: one shared private root validator, called first.
if (path.empty() || path.find('\0') != std::string_view::npos)
    throw Error(ErrorCode::invalid_argument, std::string(path), "invalid Zarr root path");
const auto location = std::filesystem::path(path);
```

Apply admission to open/read and both export paths before canonicalization, directory identity, destination preparation or metadata access. Preserve overlap rejection and record preflight. Migration: valid paths keep their behavior; invalid paths receive a contextual public error. Acceptance: embedded-NUL open/read/eager-write/backed-write cases fail with `invalid_argument`; hash the existing prefix destination and source before/after to prove preservation, including truncate/exclusive modes. Empty roots receive the same deliberate policy. Catching more exceptions alone does not fix the path ambiguity.

### F02 — Preserve accepted consolidated metadata until structural admission succeeds (P2, recovery)

`Store::inspect` accepts children found only in `.zmetadata` or inline `consolidated_metadata` (`zarr_engine.cpp:248–267`). `Store::mutation` removes those sources immediately (`:287–291`). `attributes` then calls `inspect` (`:715–716`); create/rename also inspect after mutation (`:678–686,:750–754`). For such a child, the operation destroys the source that its own inspection needs.

The engine fixture reproduced this for v2 and v3: `inspect("X").exists` was true, `attributes("X", ...)` threw `Zarr element does not exist`, and a subsequent inspection returned false. No payload was involved. This is deliberately bounded to metadata the engine's fallback accepts; this review does not assert that every upstream format version promises writable stores with only consolidated child metadata. Ordinary stores with independent child metadata are a different case.

Choose a small explicit write policy: materialize the supported consolidated-only metadata before invalidation, or reject that writable mode/operation before changing the store. Capture/admit metadata needed by the operation before discarding consolidation; merely moving one `inspect` above `mutation` does not preserve unrelated consolidated-only children. Retain explicit reconsolidation at `flush`, opaque JSON and existing partial-update semantics. Migration: document the selected write policy while preserving read fallback. Acceptance: v2/v3 consolidated fallback children with attributes/create/rename/erase either mutate and reopen correctly, or reject with original store hashes unchanged. Include a normal store with independent metadata as a control.

### F03 — Make Zarr row traversal use direct offsets and reusable rank state (P2, style and allocation mechanism)

`coordinates` accepts `std::function<void(const Shape&)>` and owns a rank vector (`zarr_engine.cpp:222–232`). Rectangular transfer uses it twice and creates `Shape local(rank), out(rank)` inside the copied-row callback (`:369–405`). Thus every copied row constructs two owning vectors and recomputes its flattened offsets through them. `resize` also uses the type-erased visitor per cell (`:731–744`). The numerical rectangle body already copies contiguous runs with `memcpy`; preserve that useful calculation. This is not a claim of per-element dtype dispatch in ordinary numeric transfer: the callback there is per chunk/row, while resize's inner callback is per cell.

Existing form and a private direct-offset sketch:

```cpp
// Before, within each copied row:
Shape local(rank), out(rank);
for (std::size_t i = 0; i < rank; ++i) {
    local[i] = absolute[i] - coord[i]*shape[i];
    out[i] = absolute[i] - start[i];
}
auto source_index = flat(local, shape), target_index = flat(out, count);

// After: same C-order mixed-radix calculation, no row-local vectors.
std::size_t source_index = 0, target_index = 0;
for (std::size_t i = 0; i < rank; ++i) {
    source_index = source_index*shape[i] + (absolute[i] - coord[i]*shape[i]);
    target_index = target_index*count[i] + (absolute[i] - start[i]);
}
```

`template<class F> void coordinates(..., F&& visit)` is a small private alternative to the type-erased parameter. Keep the complete traversal together; avoid an iterator framework. Reuse required rank scratch outside row loops and include it in admission. The checked transfer boundary establishes valid coordinates; preserve rank-zero/empty behavior and unsigned arithmetic invariants. Existing `Hdf5Numeric::select` (`backed_hdf5.cpp:726–747`) demonstrates reusable rank vectors and a retained file/memory space at this layer.

Migration: private implementation only, with templates confined to this translation unit. Acceptance: scalar/empty/rank-three/chunk-edge/fill/gather/reverse cases, scratch-limit behavior and complete output hashes; compare allocation counts on identical rectangles, and preserve public BMI/consumer isolation. Timing and build effects require measurements before claims. The earlier review's compressed resize formatting has already been expanded in current `:722–748`; that old formatting finding is not repeated here. The remaining callback and state-allocation costs are independently visible.

### F04 — Make dtype translation static and independent of enum ordinals (P2, style and extension)

`parse_type` constructs an owning `vector<string>` for ten or twelve dtype names, and `dtype_name` constructs **both** tables even when it uses one (`zarr_engine.cpp:72–92`). Name-to-enum conversion derives ordinals from table positions. The adapter also uses ordinal casts between independent `ze::Type` and public `DType` (`backed_zarr.cpp:166,:208,:236`). They currently match the frozen declarations (`zarr_engine.hpp:19`; `anndatapp.arrays.cppm:16`); this is an extension coupling, not a currently observed wrong dtype.

```cpp
// Before:
const std::vector<std::string> names = {"i1", "u1", /* ... */};
for (std::size_t i = 0; i < names.size(); ++i)
    if (dtype == names[i]) return static_cast<Type>(i + 1);

// After, representative private table:
static constexpr std::string_view names[] = {"i1", "u1", /* ... */};
// Use an explicit associated Type table or switch, not an inferred ordinal.
```

Use explicit `Type`/`DType` mapping functions with an unsupported case; a new public dtype must not silently reinterpret a private engine enum. Keep names as non-owning constant data. Migration: no public signature change; a dtype addition intentionally extends all relevant format/type maps. Acceptance: every supported width, bool, endian and v2/v3 encoding; unsupported dtype rejection before destructive destination preparation; metadata-operation allocation counts and consumer-rebuild isolation. No new registry is justified.

### F05 — Remove avoidable ordinary eager HDF5 string staging (P2, style and memory mechanism)

`read_strings` decodes into `vector<string>` (`h5ad.cpp:224–258`); `read_string_array` retains it while sizing and filling final packed bytes/offsets (`:421–440`). VLEN reads temporarily hold HDF5 text plus pointers and copied strings; fixed reads hold raw bytes plus copied strings. Packing then adds another complete output representation. Eager loading necessarily materializes the final result, but this intermediate representation is a separate cost. The record path already appends raw strings directly to final packed storage under a reclaim guard (`:458–503`), so a concrete local exemplar exists.

```cpp
// Before: values.values owns every decoded string while these buffers fill.
for (const auto& s : values.values) {
    bytes.insert(bytes.end(), s.begin(), s.end());
    offsets.push_back(bytes.size());
}

// Candidate shape, not a compiled patch:
read_text_tiles(..., [&](std::string_view s) {
    append_checked(bytes, s); offsets.push_back(bytes.size());
});
```

Only extract a concrete private traversal if the complete decoder and call sites become smaller. Attributes and string columns still require their own output contracts. Preserve fixed-padding semantics, UTF-8 policy, exception cleanup and final packed overflow checks. The 1 MiB/64 MiB **record** caps do not currently bound ordinary eager strings, and this finding does not imply they do. Writing ordinary text (`:1058–1091`) likewise copies strings to obtain NUL-terminated VLEN input; assess direct references only where actual lifetime/NUL guarantees allow them.

Migration: private implementation; full eager owner size remains proportional to the result. Acceptance: independently generated fixed/VLEN, Unicode, empty/scalar, filtered and failure fixtures; identical validation/errors; compare allocations and peak live bytes before a memory-improvement claim. Preserve raw reports locally if measured.

### F06 — Reuse eager HDF5 tile metadata and staging (P3, style and allocation mechanism)

`write_tile` constructs start/count vectors, opens file space and creates memory space for every run (`h5ad.cpp:950–961`). `write_dense_data` recomputes outer coordinates for each row and constructs axes plus a fresh copied numerical owner per gathered tile (`:981–1006`). The direct contiguous path (`:971–974`) and bounded strided pack are sensible; retain them.

Before: `write_tile(c, dataset, type, shape, outer, first, len, data, path)` reconstructs the transfer descriptor on every call. A private per-dataset state could own `start`, `count`, file space, reusable memory space and one bounded numerical tile; a call updates only last-axis start/count and writes it. Existing `Hdf5Numeric` provides this ownership pattern without another hierarchy. For gather, reuse a bounded destination through existing selected-copy operations where possible; keep its actual allocation visible. This removes repeated metadata acquisition, not the required value gathering or HDF5 conversion.

Migration: private only. Acceptance: contiguous, signed-stride, gathered, rank-three, scalar/empty and byte-order outputs, matching diagnostics and no full dense materialization. Measure if claiming a runtime/build benefit. Selected sparse export currently may copy the **entire selected sparse result** (`:1038–1044`); record that separately and do not describe all eager writes as scratch-bounded.

### F07 — Establish fixed-string semantics before sharing duplicated decoding (P2, investigate)

The eager fixed-string path distinguishes `NULLTERM`, `NULLPAD` and `SPACEPAD`: `NULLPAD` removes trailing zeroes only (`h5ad.cpp:245–251`; record path `:495–499`). Backed ordinary/record fixed-string decoding cuts at the first NUL for all non-space padding (`backed_hdf5.cpp:240–245,:679–684`). For stored bytes `a\0b\0` under `NULLPAD`, these source algorithms produce `a\0b` and `a` respectively. This difference is source-proven; its status under the intended AnnData/HDF5 text contract was not independently established here, so it is not presented as a reproduced valid-file compatibility failure.

Create an independent fixture to decide accepted/rejected embedded-NUL and padding semantics across eager/backed, attributes/ordinary/record fields. Then share only a proven small padding/UTF-8 primitive if it reduces total implementation. `valid_utf8` is independently implemented in eager HDF5 (`:201–222`), backed HDF5 (`:201–222`) and Zarr codecs (`zarr_codecs.cpp:258–276`), in addition to shared-model reviewers' paths. Keep different boundary diagnostics, validation modes and HDF5's output NUL prohibition explicit.

Before: each boundary owns a complete UTF-8 byte decoder. After, schematic: `if (full && !is_valid_utf8(value)) boundary_fail(...);`, using a private shared predicate while retaining each boundary's policy. Do not unify the capped record allocator with ordinary VLEN string staging merely because both use linked headers: their output representations, budgets and coexisting copies differ. Migration may affect formerly inconsistent input interpretation and must follow the fixture decision. Acceptance includes all padding modes, interior/trailing NUL, valid Unicode, truncated/overlong/surrogate/out-of-range encodings and full/none policy.

### F08 — Pass codec error categories explicitly (P3, style and diagnostics)

`zarr_codecs.cpp:17–21` infers `Failure` from words in the message (`unsupported`, `unknown`, `working-memory`, `budget`). A wording edit can therefore change the public `ErrorCode` through `translate`. This obscures the contract without reducing codec work.

```cpp
// Before: category is inferred from the sentence.
fail("unsupported Zarr codec: " + n);
// After, schematic:
fail(Failure::codec, "unsupported Zarr codec: " + n);
```

A category parameter or a few direct named failure helpers is enough. Do not introduce an exception taxonomy/registration framework. Migration: private implementation, retaining intentionally chosen error codes and paths. Acceptance: malformed/truncated/checksum payloads, unsupported codecs/configurations, insufficient scratch and I/O diagnostics keep their intended public categories even if messages change.

### F09 — Remove unused shard owner/state (P3, style)

`TempFile` (`zarr_sharding.cpp:35–40`) is never instantiated; the actual writer owns `OutputFile` (`:208–213`). `ShardWorkspace::encoded` (`zarr_engine.hpp:107`) is never populated or read as payload; only capacity sums and reset sites mention it (`zarr_engine.cpp:340,:545`). The general encoded index in `ShardWriter::commit` is a separate local (`zarr_sharding.cpp:383–387`).

Before: `Bytes index, encoded;` plus a dead RAII owner. After: `Bytes index;` and remove `TempFile`, its dead capacity term and reset calls. Preserve active `OutputFile` failure cleanup and the real general-codec index allocation. Migration: private layouts only. Acceptance: backend configuration builds, shard replacement/failure tests and meaningful scratch statistics; there is no runtime speed claim from deleting unused declarations.

### F10 — Measure codec-state construction before retaining more state (P3, investigate)

The reusable numeric path reuses **byte capacity**, but still calls `deflateInit2`/`inflateInit2` and ends the stream per chunk (`zarr_codecs.cpp:57–83`), or creates/frees `ZSTD_CCtx`/`ZSTD_DCtx` per chunk (`:85–115`). Fallback payload decoding additionally constructs its `Pipeline` on every chunk (`:382–400`). This is source-visible acquisition, with no measurement here establishing dominance. Blosc uses explicit context functions with one internal thread (`:136,:147`), respecting caller execution policy.

Keep the simple implementation unless a representative bounded probe justifies persistent codec state. If justified, attach it to an exclusively used prepared transfer/worker backend, preserve parameter changes and failed-call recovery, and identify retained third-party memory separately from admitted byte buffers. No global cache, library scheduler or lock is implied. Migration: optional private design only. Acceptance: identical codec output/decoding, settings, malformed input recovery, fork isolation and memory limits; count allocations/context calls and bytes with identical geometry before timing claims. No campaign was launched for this optional finding.

## Retained mechanisms and practical costs

| Area | Retain | Cost/limit that callers and extension authors must see |
|---|---|---|
| Public format facades | Small C++20 modules; concrete owner/view overloads; private HDF5/JSON/codec headers | No public umbrella header or `import std`; implementation-only proposals stay behind compiled boundaries |
| Eager HDF5 numeric/record reads | Typed allocation once, direct H5Dread; field projections; explicit boolean canonicalization | Eager owns full payload; one projected numerical read per record field; HDF5 cache/codec memory separate |
| Prepared HDF5 transfers | Dataset/type/file-space retention, reusable vectors and equal-shape memory space | Mutable descriptor belongs to one worker; unprepared calls reacquire metadata and filters |
| Native mapping | Actual HDF5 SEC2 descriptor duplicated; Zarr eligible file opened/fstat'd before mapping; retained read-only owner | Native uncompressed geometry only, bool restrictions; excludes general chunked/gathered layouts; source mutation still needs caller coordination |
| HDF5 extension preservation | Opaque attribute datatype/dataspace preserved on same-store replacement; references rejected | HDF5 1.10 H5Aread has no custom allocator; opaque VLEN budget checked after backend allocation, then reclaimed; cross-format export is not universal opaque preservation |
| Zarr payload transforms | Explicit serializer/byte codec/permutation order; native output; `PayloadMeta` borrows source metadata | General fallback copies encoded bytes and may hold transform output alongside input; encode deliberately copies a const caller chunk; static raw/single-codec paths avoid some copies |
| Selected sharded numeric writes | Per-axis plans, bounded points/maps, logical-last duplicate choice; one physical writer per touched shard | O(sum gathered axis lengths), sort maps, pack/scatter tile; capacity admission precedes shard writes; callback is per tile |
| Shard replacement | Positional range reads, overflow/EINTR/short-transfer handling, unique mkstemp output and rename | Unchanged encoded payload is copied; one shard replacement is atomic publication, not dataset transaction or fsync durability |
| Missing chunk/fill | Typed native fill and explicit inner-chunk shape | Full inner chunk may be decoded/filled for small selection; final caller buffers/materialized outputs are not scratch |
| Metadata flush | Explicit consolidation invalidation/rebuild | Directory-tree scan and complete JSON metadata tree; no background scheduler; node discovery may repeatedly parse metadata |

`Store::resize` (`zarr_engine.cpp:722–748`) keeps value and fill chunks and visits the old geometric grid. It writes cleared intersecting/outside chunks, including potentially absent fill chunks; the comment's phrase “delete chunks” does not describe a file-removal implementation. No call site in the assigned adapter exposes this operation. Treat its future extension as a separate structural geometry question, rather than inferring performance from the bounded size of an individual chunk. This review makes no measured resize claim.

## Portability, extension and caller concurrency

The qualified target remains Linux/Clang 18. `backed_hdf5.cpp` selects SEC2, obtains its POSIX `int*` driver handle, uses device/inode identity and duplicates the actual descriptor for mapping. `backed_zarr.cpp` uses POSIX stat/open/mmap; `zarr_sharding.cpp` uses pread/pwrite/mkstemp/fchmod/rename. These are useful explicit mechanisms on this target, not portable-standard-C++ capabilities. A non-POSIX extension needs a small private file/mapping implementation and target qualification, preserving large offsets, partial transfers, cleanup, identity and publication semantics. HDF5 version-specific object/link APIs and format-library availability also require an actual target build. No portability rewrite is needed merely to satisfy the currently qualified platform.

New storage backends fit the existing private `Backend`/`NumericTransfer` boundary. They must report physical read/write conflict units separately from logical selection; native-endian rectangle semantics, scratch ownership, retained sessions, explicit close and structural invalidation remain shared-model contracts. Dynamic dispatch at **transfer acquisition/rectangle/tile** boundaries is justified; the acquired caller numerical loop is ordinary pointer/stride work. New formats should reuse element codecs for tables/categories/nullable/sparse/records and provide their actual container-root relationships. The Zarr byte engine is independent of AnnData types; no generic plugin registry or second numerical model is warranted by current source. Zarr structured rec-array remains an explicit preflight rejection, with no lossy group substitution (`backed_zarr.cpp:273–281`). Parameterized dtypes, unknown required transforms, unsupported codec positions and link types must keep contextual rejection.

Callers own scheduling. A `Store` mutates reusable byte/map buffers even during const reads; `ZarrNumeric` references that same store. A single `Hdf5Numeric` mutates spaces, shape and counters. Thus sharing one prepared/context object concurrently is not established by `parallel_reads` flags. Worker forks provide independent backend scratch/state; callers still exclude structural changes and physical write conflicts. Zarr numerical payload writes deliberately avoid metadata mutation, allowing independent chunks/shards under those conditions. Fixed `.anndatapp-tmp` payload/metadata names require exclusive access to the physical file; unique shard temp names do not prevent lost updates when two writers replace the same shard. `exclusive` root creation currently uses an existence check followed by `create_directories`, so it is not a proven atomic multi-creator admission primitive. These are caller coordination/qualification limits, not permission to add library locks.

`hdf5_library_threadsafe` and the inspected thread-safe flag describe backend correctness capability. They do not prove simultaneous I/O execution or throughput scaling: HDF5 can synchronize internally. Neither acquired numerical loops nor the engine introduce a scheduler; Blosc's calls request one internal thread. No multithreaded speed claim follows from the flags or from this source review.

## Historical evidence reconciliation

The old screen in `docs/head-to-head-performance.md` is explicitly historical. Its 8,192 logical gather reads and observed native deficit were subsequently addressed in private `backed_array.cpp`, outside these eleven files. `docs/gather-read-performance.md` records grouped per-axis reads of touched sharded inner chunks: the two eight-repeat fixtures fell from 8,192 to 640/1,520 logical reads, with candidate single warm-cache times 63.31/163.33 ms versus earlier 735.44/719.21 ms. Python values were retained from the old screen, not rerun. These are existing source/fixture-bound observations, not new measurements by this review and not language-wide or all-layout claims. No finding here reopens the old point-read deficit as current.

The correction's larger logical `TransferStats::read_bytes` counts expanded rectangle outputs, not compressed physical I/O or useful result bytes; retained scratch excludes temporary per-axis maps. It covers **sharded numeric reads**, with fallback for low budget or aliasing output strides, and no cross-call cache. The current backend still performs its ordinary inner-chunk decode/range operation when asked; grouped transfer above it changes request granularity. F03's row metadata, F10's codec-state construction and sharded index acquisition are distinct mechanisms, whose current cost needs separate evidence before optimization.

`docs/compliance-fixes.md` records R1/R4 as implemented: current `ArrayMetadata` no longer has the unused attribute copy, and `PayloadMeta` borrows array/codecs (`zarr_engine.hpp:26–43,:62–66`; `zarr_engine.cpp:298–318`). Do not re-report them as unresolved full metadata copies per chunk. `Store::write_selection` does copy metadata once for an operation (`zarr_engine.cpp:528`), which is a different scope. The old S6 resize formatting expansion is also present. The review retains these fixes and identifies only the remaining source mechanisms.

## Executed evidence and acceptance boundary

The byte-identical preserved [probe source](probes/backends.cpp), originally `/tmp/anndatapp-backend-review/repro.cpp`, compiled directly with Clang 18 and the frozen three non-module engine implementations, linking system zlib/Zstd/Blosc. It created only disposable stores under `/tmp/anndatapp-backend-review/stores` and reported inspected before/after values. The complete output is `/tmp/anndatapp-backend-review/result.json`; v2/v3 each reported `before: true`, `after: false` with the expected missing-element exception, and the NUL-root test captured the file-buffer error message. These checks validate F01/F02's bounded engine paths; they do not validate upstream semantics or public wrapper error handling.

The [probe recipe](probes/backends.md) records the exact dependencies, fixture paths, frozen-source conditions, output and failure expectations. From the repository root:

```sh
mkdir -p /tmp/anndatapp-backend-review
clang++-18 -std=c++20 -O0 -I build/thin-review-20261105/source/src \
  docs/review/probes/backends.cpp \
  build/thin-review-20261105/source/src/zarr_engine.cpp \
  build/thin-review-20261105/source/src/zarr_codecs.cpp \
  build/thin-review-20261105/source/src/zarr_sharding.cpp \
  -lzstd -lz -lblosc -o /tmp/anndatapp-backend-review/repro
/tmp/anndatapp-backend-review/repro
```

| Artifact | SHA-256 |
|---|---|
| Reproducer source | `eb72728aae623f47a54b281576536163c84cd91cbbd6a68a82b1690b4032085f` |
| Reproducer executable | `e910426961d45c58d8f26f5ec1171a22925fb700c3f9c651b165ff9afb40dde1` |
| Complete result JSON | `040e626913c7f7b0fc4832f917e1c9a8e1004c097b4502af54a4aeac33b23816` |

This deliverable is a review, not an implementation: production fixes, public-wrapper reproductions, independent fixed-string fixtures, sanitizer/concurrency coverage and any optional measurements remain acceptance work for a subsequent authorized patch. Other review owners' native build/test results are not claimed as tests run by this owner. No performance or build speedup is claimed from the source findings or the correctness fixture.
