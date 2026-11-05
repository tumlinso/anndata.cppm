# 1.0 cleanup status

The 1.0.0 release uses the user-approved two-machine policy: final Linux x86-64 requalification and the disclosed native Apple AArch64 results. The ARM sanitizer, timing, consumer-build and coverage limitations were explicitly accepted for release and remain documented. Source compatibility applies to the documented 1.x API. Binary ABI compatibility is not promised. Adaptive streaming remains the default.

All 49 review findings close through a fix, a tested API decision, or evidence supporting retention. Historical review documents remain unchanged as baseline observations. Generated inventories and raw campaigns stay ignored. See the [qualification record](v1-qualification.md) for exact test scopes and the remaining release gates.

The pre-cleanup working tree is preserved under `build/v1-cleanup/baseline-source/`, with exact hashes in `build/v1-cleanup/baseline.json`. Project Control returned `no_work` with no workflow handle; no native acceptance or release is inferred.

| Finding | Workstream | Status | Resolution/evidence |
| --- | --- | --- | --- |
| [A01](review/access.md) | malformed axis can index an empty shape (high; reproduced) | Resolved | Fixed: dataframe/index schema admission; malformed H5AD/Zarr regressions. |
| [A02](review/access.md) | one worker context is not a general shared-session guarantee (high; contract and coverage) | Resolved | Qualified contract: serial preparation/lifecycle, independent worker contexts; TSan coverage. |
| [A03](review/access.md) | category plans have shared mutable pages (high; static risk, coordinated with model review) | Resolved | Retained caller-serialized category plans; no shared-plan parallel promise; B02 repairs paging recovery. |
| [A04](review/access.md) | sparse selection dispatch and fallback deserve a measured fast path (medium; candidate) | Resolved | Specialized stored-width dispatch; retained remaining width conversions with measured code-size tradeoff. |
| [A05](review/access.md) | repeated descriptor work and owner forwarding allocate (medium; candidate) | Resolved | Removed repeated descriptions, prepared-owner allocation and trivial owner-view forwarding. |
| [A06](review/access.md) | stream planning adds per-request metadata work (medium; candidate) | Resolved | Lane-encoded opaque IDs; removed duplicate retained requests; adaptive default retained. |
| [A07](review/access.md) | gathered sparse writes can materialize the complete selection (medium; confirmed path) | Resolved | Retained explicit selected-sparse materialization; allocating fallback is documented and tested. |
| [A08](review/access.md) | scratch statistics are not general allocation counters (medium; contract clarity) | Resolved | Counts real tile allocation events; metadata/third-party exclusions are explicit. |
| [A09](review/access.md) | worker failure and stream ownership need an explicit recovery contract (medium; design gap) | Resolved | Tested retry/abandon contract; failed requests stay outstanding and partial writes are explicit. |
| [A10](review/access.md) | distinguish essential API from forwarding and bridge surface (low; maintainability) | Resolved | Private implementation-state constructor; internal detail bridges excluded from stable API; useful aliases retained. |
| [B01](review/backed-model.md) | Record arrays cannot traverse validation adapters (P1, correctness) | Resolved | Fixed complete record validation adapters; both mode overrides and malformed-text regressions. |
| [B02](review/backed-model.md) | Failed category page refill can publish an invalid cache page (P2, recovery) | Resolved | Fixed publish-after-success paging with injected-I/O-error retry regression. |
| [B03](review/backed-model.md) | Duplicate bounded string traversal (P2, style; prior S5) | Resolved | Shared concrete bounded string traversal with destination-specific validation. |
| [B04](review/backed-model.md) | Three private UTF-8 decoders duplicate the same invariant (P3, style) | Resolved | Shared private UTF-8 invariant; 18,153,739-string differential check plus boundary tests. |
| [B05](review/backed-model.md) | Temporary spill paths are POSIX-specific (P3, portability) | Resolved | Private POSIX spill boundary documented; future platform qualification remains a release gate. |
| [B06](review/backed-model.md) | Categorical validation performs dtype dispatch per code (P3, style/dispatch) | Resolved | Integer type selected outside streamed code loops; all eight widths and error cases tested. |
| [B07](review/backed-model.md) | Writer repeats validation after whole-dataset preflight (P3, investigate) | Resolved | Complete dataset preflight reused; direct edits retain validation and synchronous source stability is explicit. |
| [C01](review/core.md) | P1: selected strings pass through individual owned strings | Resolved | Direct final packed buffers; selected traversal and UTF-8 validation preserved. |
| [C02](review/core.md) | P1: dense owning copy maintains a second, more expensive walker | Resolved | Shared dense-copy traversal preserving checked/trusted and raw sparse-copy contracts. |
| [C03](review/core.md) | P1: sparse selection materializes output-sized staging plus a hash map | Resolved | Bounded direct duplicate chains repair the measured opposite-axis regression; compact/gapped CSR/CSC regressions and paired timing evidence. |
| [C04](review/core.md) | P2: the light dense entry point exposes internal dimensions | Resolved | Public borrowed_dense factory with copied inline/owned metadata and external numerical lifetime. |
| [C05](review/core.md) | P2: identity selection allocates before finding that it does nothing | Resolved | Identity array/string selection reuses captured binding before allocation. |
| [C06](review/core.md) | P2: focused record acquisition snapshots every field | Resolved | Focused retained record owner acquisition; inline field dimensions. |
| [C07](review/core.md) | P2: interleaved schema metadata is stored twice | Resolved | One record schema; payload vectors only for fieldwise representation. |
| [C08](review/core.md) | P2: every interleaved copy constructs an unused text ownership layer | Resolved | One native/text allocation owner; no redundant owner-pair objects. |
| [C09](review/core.md) | P2: record row loops repeatedly resolve invariant layout work | Resolved | Field-level construction dispatch; rank-aware incremental traversal. |
| [C10](review/core.md) | P2: compiler and scalar-representation assumptions need a declared boundary | Resolved | Feature-detected overflow intrinsics plus independently checked portable fallback; explicit scalar guards. Descriptor source/codegen investigated; final checked-acquisition sample ranges overlap. |
| [D01](review/data.md) | Medium: a null C-string selector enters undefined behavior | Resolved | Null C-string selector rejected contextually. |
| [D02](review/data.md) | Medium: pruning plans are copied before they are rebuilt | Resolved | Immutable shared pruning plans; one source-code load and no old-map copies during rebuilding. |
| [D03](review/data.md) | Medium: categorical copies stage and copy recoded codes more than once | Resolved | Recoded categorical output fills one final native-typed array. |
| [D04](review/data.md) | Medium: selected strings are unpacked into individual strings before repacking | Resolved | Selected/initializer string columns pack directly; nullable and trusted-input contracts preserved. |
| [D05](review/data.md) | Medium: structural snapshots eagerly scale with all bindings | Resolved | Retained captured binding snapshots with focused/no-op fast paths; no new mutation/COW framework. |
| [D06](review/data.md) | Low: categorical validity resolves the same row twice | Resolved | Removed duplicate categorical row resolution. |
| [D07](review/data.md) | Medium: public cost and lifetime contracts are incomplete at acquisition points | Resolved | Documented acquisition costs, borrowed strings and shared-payload lifetimes. |
| [D08](review/data.md) | Low: inactive representation state and an avoidable empty owner merit isolated evaluation | Resolved | Private active Element payload (344 to 56-byte layout probe); shared pruning descriptor; public selection layout retained. |
| [F01](review/backends.md) | Reject embedded NUL in Zarr roots before filesystem work (P1, boundary correctness) | Resolved | Root-path admission precedes filesystem work; public read/write mode preservation tests. |
| [F02](review/backends.md) | Preserve accepted consolidated metadata until structural admission succeeds (P2, recovery) | Resolved | Fallback-only metadata remains readable; unsupported structural mutation rejects before invalidation. |
| [F03](review/backends.md) | Make Zarr row traversal use direct offsets and reusable rank state (P2, style and allocation mechanism) | Resolved | Concrete coordinate visitor and direct row offsets; no row-local coordinate vectors. |
| [F04](review/backends.md) | Make dtype translation static and independent of enum ordinals (P2, style and extension) | Resolved | Static non-owning dtype tables and explicit engine/public enum conversions. |
| [F05](review/backends.md) | Remove avoidable ordinary eager HDF5 string staging (P2, style and memory mechanism) | Resolved | Eager packed HDF5 output avoids full intermediate string vector with reclaim guards. |
| [F06](review/backends.md) | Reuse eager HDF5 tile metadata and staging (P3, style and allocation mechanism) | Resolved | Reusable private tile dataspaces/coordinates and bounded numerical scratch. |
| [F07](review/backends.md) | Establish fixed-string semantics before sharing duplicated decoding (P2, investigate) | Resolved | Independent physical padding fixture; consistent native decoding and explicit binding differences. |
| [F08](review/backends.md) | Pass codec error categories explicitly (P3, style and diagnostics) | Resolved | Explicit codec failure categories instead of message-derived classification. |
| [F09](review/backends.md) | Remove unused shard owner/state (P3, style) | Resolved | Removed unused shard owner/workspace state. |
| [F10](review/backends.md) | Measure codec-state construction before retaining more state (P3, investigate) | Resolved | Retained per-call contexts after scoped lifecycle-cost diagnostic; no unmeasured persistent-state framework. |
| [R01](review/build.md) | One installed provider compiles the whole enabled interface set (P2, measured build work; candidate split) | Resolved | Independent core target/component, five core BMIs and no optional discovery; installed/downstream exports tested. |
| [R02](review/build.md) | Toolchain qualification is narrower than source portability (P3, explicit coverage limit) | Resolved | Matching scanner admission and explicit qualified platform boundaries. The two-machine release plan awaits the user's requalification on the other machine. |
| [S01](review/support.md) | Current module-build benchmark encodes a stale no-storage architecture | Resolved | Correct backend scope/dependency prefix reporting; bounded standard and consumer probes replace broad campaign. |
| [S02](review/support.md) | `record_measurements.py` is a checkout-specific continuation recipe, not a portable current benchmark | Resolved | Historical recovery tool explicitly labeled/guarded; current measurements use the bounded reproducible runner. |

Implementation evidence: [core](v1-core.md), [data](v1-data.md), [access](v1-access.md), [backed model](v1-backed-model.md), [backends](v1-backends.md), [package/build](v1-build.md), and [standard performance](v1-performance.md). Current support and source-compatibility boundaries are in [support.md](support.md); source migration is in [migration-1.0.md](migration-1.0.md).
