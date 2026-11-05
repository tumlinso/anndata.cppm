# Thin-library review of AnnData++

The library's direct numerical access boundary is worth keeping. The strongest opportunities are around acquiring descriptors, copying selected data, recoding metadata, and moving storage blocks. Those paths contain repeated discovery, intermediate representations, and dispatch that the direct-pointer benchmark does not measure.

This is the complete **review phase**, with recommendations and acceptance criteria. No production implementation or public API was changed. The reviewed artifact is a frozen working tree based on `d2260c2abb37e9028fa42c95d1712ffcaa282b0e`, including supplied uncommitted package and Zarr changes; [source-manifest.json](source-manifest.json) identifies its bytes. Existing and concurrent work was preserved. Local raw evidence is under ignored `build/thin-review-20261105/`.

## Coverage and navigation

All 36 production C++ files were read across the subsystem reviews. The ledgers inventory public operation families, private types/state/helpers, retained choices, and proposed changes. Overloads and closely related helpers are explicitly grouped rather than treating each declaration as a separate abstraction. A syntax inventory cross-check supplements this human review; neither a lexical match nor a ledger count proves semantic correctness.

| Review | Contents | Coverage ledger |
| --- | --- | --- |
| [Core](core.md) | Types, selections, dense/sparse storage, descriptors, packed strings, fieldwise/interleaved records | [core-ledger.json](core-ledger.json) |
| [Data](data.md) | Selectors, nullable/categories, tables, recursive metadata, AnnData and raw alignment | [data-ledger.json](data-ledger.json) |
| [Access](access.md) | Backed handles, prepared/context transfers, geometry, streams, concurrency and failure rules | [access-ledger.json](access-ledger.json) |
| [Backed model](backed-model.md) | Element codecs, category/color transforms, index spills, copying, validation, writing and structural edits | [backed-model-ledger.json](backed-model-ledger.json) |
| [Backends](backends.md) | HDF5, Zarr v2/v3, chunk/shard I/O, codecs and format boundaries | [backends-ledger.json](backends-ledger.json) |
| [Build](build.md) | Modules, package provider, private dependencies, presets and CI | [build-ledger.json](build-ledger.json) |
| [Supporting code](support.md) | Tests, fixtures, examples, probes, benchmark tools and historical records | [support-ledger.json](support-ledger.json) |
| [Includes](includes.md) | Include assessment and reproduction of every literal include occurrence | Generated locally; see the reproduction command |

[library-choices.md](library-choices.md) assesses all 31 distinct production C++ standard headers; [aliases-ledger.json](aliases-ledger.json) records the local aliases found during the root cross-check. [coverage.json](coverage.json) records file coverage and validation of ledger references. Include-site classifications and missing/unused-header suggestions are **lexical candidates**, not compiler-verified edits. No include-removal or header-attributed compile-speed claim is made.

## Findings that should lead the next phase

Finding IDs are stable within this review. Subsystem priorities express local importance; the order below separates correctness from unmeasured optimization work. Detailed reports supply source locations, compact alternatives, migration notes, and acceptance cases.

| Order | Findings | Evidence and recommended outcome |
| --- | --- | --- |
| 1 | **A01** malformed backed axes | Reproduced Zarr v2 process crash, exit 139. Validate required dataframe/index shape before indexing dimensions; malformed stores must throw contextual errors. |
| 2 | **B01** record validation adapters | Reproduced rejection of valid HDF5 records by `validate_all()` and changed-validation export. Complete record forwarding/sink behavior; retain independent validation semantics. Same-mode exports round-trip successfully. |
| 3 | **F01/F02, B02, D01** input/recovery boundaries | Engine probes reproduce malformed-path diagnostics and loss of accepted consolidated-only discovery after a failed mutation. Category-page publish-before-read and null C-string selection are source-supported hazards. Resolve them with preservation and failure tests; do not infer whole-format failures from bounded engine cases. |
| 4 | **A02/A03, A09** parallel and failure contract | Clarify which handles may be shared, which acquisition/lifecycle calls must be serialized, and valid retry/abandon behavior. Qualify same-session workers with TSan before broadening guarantees. No supported-use race was reproduced here. |
| 5 | **C01/C02, D03/D04, F05** copies and packing | Remove intermediate string/code owners and duplicated dense walkers where a private shared traversal preserves validation. Compare final-buffer filling against the current implementation. Runtime benefit remains unmeasured. |
| 6 | **C03/A04, B06, C09/F03** sparse and repeated inner work | Evaluate geometry-specific sparse count/fill, dispatch outside entry loops, incremental coordinates and concrete callbacks. Preserve duplicate/order/width semantics and validation before writes. Avoid a template cross product. |
| 7 | **C05–C08, D02/D05, A05/A06** acquisition and representation | Reduce redundant snapshots, schema lists, empty owner layers, pruning copies, and request/shape allocations. Retained binding semantics must survive owner replacement. Measure acquisition separately from pointer loops. |
| 8 | **F04/F08/F09, B03/B04, D06/D07** compactness and explicit contracts | Prefer static non-owning dtype tables, explicit error categories, removal of proven dead private state, concise calculations, and shared invariants where the complete source becomes simpler. Keep destination-specific diagnostics. |
| 9 | **R01/R02, C04/C10, D08, F07/F10, S01/S02** deliberate design/evidence work | Evaluate package components, a minimal public borrowed descriptor, target assumptions, active metadata payloads, fixed-string semantics and codec-context retention. Repair benchmark scope descriptions before collecting new build comparisons. These are bounded proposals rather than approved framework rewrites. |

The full reports also cover gathered sparse-write materialization, scratch-statistic meaning, repeated writer validation, public bridge/alias surface, POSIX spills, and other retained choices. An apparent local simplification must be judged with its complete stored state, ownership paths, generated code and build consequences.

## What the current evidence establishes

The frozen Debug configure/build completed and **20/20 native tests passed**, including the installed multi-consumer package test. A targeted independent assessment checked consequential source claims in six draft reports and found no material issue; it did not independently verify every ledger row or finding.

Fresh observations, recorded in [validation.json](validation.json), are deliberately narrow:

- The existing allocation probe returned **one allocation / 136 requested bytes** for direct observation/variable column acquisition at 8, 128 and 1,024 columns per axis. At 1,024 columns, the whole-table route made **1,036 allocations / 237,656 requested bytes**. Both paths returned identical pointers and values. These are Debug acquisition/release counts, not time or process-memory measurements.
- Affine all/forward/reverse selection made two allocations / 136 requested bytes at logical lengths 16, 2²⁰ and 2³²; nested affine selection made four / 272. This confirms length-independent metadata size for those cases, while showing that compact does not mean allocation-free.
- An installed executable importing only `anndatapp.core` still built **ten public BMIs totaling 44,615,920 bytes** with the supplied default-enabled package provider. This measures artifact work/size, not import latency or process memory.
- Four serially prepared worker contexts from one session passed concurrent repeated reads and writes to **separate arrays** for Zarr v2/v3. This did not test concurrent preparation, shared contexts, overlapping chunks/shards, HDF5, or arbitrary metadata access.
- [Access](probes/access.md), [record validation](probes/record_validation.md), and [backend](probes/backends.md) probes preserve reproducible sources and bounded defect observations.

No new optimization timing comparison, full ASan/UBSan matrix, TSan qualification, or complete Python/Scanpy interoperability campaign was run. Existing historical reports remain attached to their original source and workload. The later grouped-Zarr-read correction supersedes that portion of the earlier Python/C++ screen. The rejected custom gather-buffer experiment remains a useful negative result: fewer allocations did not establish a better overall representation.

## Interfaces, extensions, and implementation acceptance

Retain move-only large owners, explicit deep copies, retained versus borrowed lifetime choices, checked versus trusted boundaries, signed strides, native-type identity, sparse structural widths, AnnData alignment, and C++20 modules. Keep backend headers private and caller numerical loops free of library dispatch and synchronization. A blanket STL replacement would not address the identified repeated work.

Breaking API recommendations are allowed with explicit migration. The strongest public candidates are a small pointer/shape/stride borrowed-entry contract, focused record acquisition without a whole-schema snapshot, clearer request/metric ownership, and an operation-specific concurrency contract. Most ranked work can remain private; public interfaces should grow only when they give callers a meaningful new capability.

For **biological composition**, a new named module should compose arrays/tables/records/AnnData with its own relationship and dataset-root rules. Demonstrate one representative container's construction, selection, ownership and serialization path before generalizing the core. No new biological format is claimed implemented or specified by this review.

For **storage/encoding extension**, keep backend-native details behind the existing compiled boundary and implement the relevant private backend/transfer capabilities. Walk the capability through eager/backed access, read/write, copying, validation adapters, independent output reopen and unsupported-destination errors. B01 demonstrates why adding a virtual operation only in the physical backend is insufficient. An explicit capability checklist or compiler-enforced adapter completeness is preferable to a new public registry/plugin ABI without a consumer requirement.

Every selected change needs its report's acceptance cases and proportionate native/sanitizer/concurrency/interoperability/installed-consumer checks. Cost-changing candidates begin with geometry-faithful probes capped at 30 seconds, counts of calls/allocations/bytes, and identical source-hashed workload pairs. Preserve raw evidence locally and compact summaries in Git. Larger confirmations require a time and I/O budget. Measure public BMI/import/rebuild effects when interfaces change and verify private-edit isolation when claimed.

The next implementation plan should cover all findings. Start with correctness and concurrency, then remove repeated copy/acquisition work and resolve the representation, portability and build questions with focused evidence. Findings marked investigate need a decision supported by measurements or fixtures rather than indefinite deferral. Ordering sets the execution sequence, rather than selecting a smaller scope.
