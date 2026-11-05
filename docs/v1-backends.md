# Backend cleanup

This implements the backend review against the preserved dirty baseline in `build/v1-cleanup/baseline-source`. It preserves the existing Zarr mapping guard and keeps format/codec details behind compiled implementations. Public format interfaces remain unchanged.

| Review | Disposition and resulting behavior |
|---|---|
| F01 | Fixed. One private root validator rejects empty/NUL-containing paths before Zarr stat, canonicalization, open or destination preparation. Public eager/backed owner/view writes, both creation modes, and reads/opens have invalid-root regressions with prefix/source file snapshots. |
| F02 | Fixed with an explicit write limit. Reads retain the consolidated-only fallback. Structural changes require independent metadata files for every consolidated entry; missing files reject before metadata invalidation. No implicit metadata materialization was introduced. |
| F03 | Simplified. `coordinates` has a concrete private callable. Numerical rectangle rows compute source/output offsets directly; the two row-local vectors are gone. Existing contiguous-run copies, rank-zero behavior and shared scratch ownership remain. |
| F04 | Simplified. Dtype names are static non-owning entries with associated explicit engine types. Adapter conversion uses exhaustive named `Type`/`DType` switches rather than matching enum ordinals. |
| F05 | Simplified. Ordinary eager HDF5 string arrays append decoded text directly to final packed bytes/offsets. Attribute/column outputs retain their required string vectors. VLEN reclaim still encloses all append/validation failures. |
| F06 | Simplified. A per-dataset `TileWriter` retains rank coordinates and file/memory dataspaces. It resizes its own memory space when run length changes. Dense strided/gathered writes reuse one typed numerical allocation; gathered tile descriptors still allocate metadata, visibly, without allocating another numerical tile. String writers reuse their batch descriptor capacities. |
| F07 | Fixed and independently qualified for the selected native policy. One private `h5ad_text.hpp` handles fixed padding for eager/backed, ordinary/compound paths. Shared `utf8.hpp` replaces the three backend UTF-8 decoders while local diagnostics/validation policies remain. |
| F08 | Simplified. Every codec failure passes an explicit `Failure` category. Diagnostic wording no longer selects the public error category. |
| F09 | Removed. Unused `TempFile` and `ShardWorkspace::encoded`, including dead capacity/reset terms, are gone. Active shard publication/cleanup owners remain. |
| F10 | Retained with evidence. The authorized source-2 Zstd diagnostic attributed approximately 1.88%/2.42% of operation CPU to intercepted context creation/free in the gathered-read/partial-write cases. Keep simple per-call contexts and reusable byte buffers; the small measured fraction does not justify added persistent-state lifetime, reset/failure handling and retained-memory complexity for these workloads. |

The F02 limit is intentional: a store may be readable through consolidated metadata without being structurally writable by this implementation. Rejection checks the complete consolidated metadata source before changing it, preserving unrelated fallback-only nodes as well as the operation's target. Ordinary stores with independently stored metadata keep invalidation/reconsolidation behavior. Numerical payload writes do not invalidate consolidation. This does not add a whole-store transaction or durability guarantee.

The F05 result still materializes the requested eager output and temporarily retains HDF5 raw strings/pointers or fixed bytes. Packed output vectors may grow as text arrives. It removes the complete intermediate `vector<string>` from packed-array acquisition, not all allocations or all copies. Ordinary eager text remains distinct from the bounded 1 MiB target/64 MiB cap used for record staging. Selected sparse eager output may still materialize the full selected sparse result. No peak-memory or speedup claim follows from these source changes.

`TileWriter` is private acquisition state, not a new public writer abstraction. Contiguous numerical output still calls H5Dwrite directly. Strided and gathered output remains limited to the existing row-fragment geometry; the new rank-three regression crosses a 1 MiB numerical tile plus a short tail in four reversed-stride rows. The borrowed gathered destination exists only while its retained numerical scratch is alive. Caller-acquired numerical loops gain no dispatch, locks or synchronization.

## F10 codec-context decision

The authorized initial diagnostic used `candidate-source-2`, source identity `18ade68593a3db62994c4726072ea2d3461529349d7ad4268ebb61707b085fac`, against baseline identity `96dd8c96fe516d7f71d76960cea3be29609fd66392a613bd8a8350edf1f9e08c`. Its preserved [raw comparison](../build/v1-cleanup/performance-run/comparison-with-consumer.json) has SHA-256 `c649d1ffb9d0941c835d2687fa68a90bbbb3ed9d84896385eafc868fadaebbeb`. These are **source-2 observations**, not relabeled source-3 measurements. The later arrays repair leaves the backend implementation unchanged: all thirteen backend/format/helper file hashes, including `utf8.hpp` and `h5ad_text.hpp`, matched source 2 when this decision was closed. Final whole-suite acceptance for source 3 is recorded separately.

The fixed Zarr v3 store is 256 × 256 float32, with 64 × 64 inner chunks, 128 × 128 shards and Zstd. Its fixture manifest identity is `0756f4141bb99ed73caffc178077965d06329ed1b0b7ce8b23a0d2f22f51b180`; logical X hash is `4d65c4487a88c242c8610d5477cace2fd10571dda5122adab731ac216e5b6d74`. Each operation has four repetitions. Counts came from one separate instrumented diagnostic per operation/revision, excluded from the five paired uninstrumented timing samples.

| Source-2 diagnostic | Context create/free pairs, same in baseline and candidate | Baseline lifecycle CPU / operation CPU | Candidate lifecycle CPU / operation CPU |
|---|---|---|---|
| Gathered read | 64 decoder pairs; 64 `decompressStream` calls | 123,121 / 6,636,359 ns ≈ 1.86% | 123,877 / 6,580,192 ns ≈ 1.88% |
| Partial shard write | 16 encoder and 16 decoder pairs; 16 `compress2` and 16 `decompressStream` calls | 109,703 / 5,208,821 ns ≈ 2.11% | 136,914 / 5,666,719 ns ≈ 2.42% |

The numerator sums intercepted Zstd create/free call thread CPU; the denominator is operation process CPU. Clock/interposition overhead, different clock scopes and unobserved internal calls limit attribution. C-heap allocation bytes are excluded, so these numbers do not quantify codec workspace size or all allocator work. Allocation or reusable work inside `compress2`/`decompressStream` is not isolated: the percentages are not an upper bound on every possible retained-context benefit. No retained-context variant was measured. Baseline and source-2 overall timing ranges overlapped for both operations; they resolve no overall change for these cases. This is not evidence that context creation is negligible in every geometry or codec, and it does not qualify a persistent zlib context design.

Retain the tested per-call codec context ownership for this cleanup. The measured create/free cost was small in these cases, and no measured persistent variant establishes a benefit that justifies adding retained third-party memory and reset, failure recovery and worker/fork lifetime rules. Reusable byte capacity remains independent of codec-state lifetime. A later workload that identifies a material context bottleneck can support a focused alternative; no global cache, lock or scheduler is introduced here.

## Fixed-string evidence

[`tests/h5ad_padding_fixture.py`](../tests/h5ad_padding_fixture.py) independently generates an AnnData base file, then uses low-level h5py datatype/data APIs to install explicit fixed-width string bytes in ordinary columns, string arrays, scalar/empty arrays, semantic attributes and flat compound record fields. It records exact logical bytes read back with the same file datatype, high-level binding results and native expectations. Generated stores and the evidence JSON remain ignored under `build/v1-cleanup/padding-fixture`.

The selected native interpretation preserves the existing eager rule and makes backed reads agree:

| HDF5 padding tag | Native interpretation |
|---|---|
| NULLTERM | Stop at the first NUL. |
| NULLPAD | Remove trailing NUL bytes; preserve interior NUL bytes. |
| SPACEPAD | Remove trailing spaces; preserve other bytes, including NUL before that padding. |

For samples `A\0B`, `tail\0`, `mid\0x`, the native expected values are respectively `A, tail, mid` under NULLTERM; `A\0B, tail, mid\0x` under NULLPAD; and `A\0B, tail\0, mid\0x` under SPACEPAD. The helper returns a borrowed view into the current staging bytes; no backend allocation escapes into a native view.

Observed fixture environment: Python 3.12.3, AnnData 0.13.4, h5py 3.16.0, h5py's HDF5 2.0.0, NumPy 2.2.6 and pandas 2.3.3. Its same-type low-level read preserves the supplied fixed-width bytes. h5py's high-level SPACEPAD result drops the terminal NUL before space padding in `tail\0`; this native policy deliberately preserves that byte. AnnData 0.13.4 rejects the adversarial embedded-NUL compound fixture with `UFuncTypeError`. This fixture therefore establishes native consistency and physical-byte handling, not universal AnnData-valid text interoperability or an external format-standard claim. HDF5 output still explicitly rejects embedded NUL in VLEN text.

The independently generated fixture's captured SHA-256 is `4a1891c3d276690efd849155ccc6694abd449c4095b6829f4ff5f1584fe0c102`; the generator/evidence JSON record their own hashes and environment. Reproduction from the repository root:

```sh
.venv/bin/python tests/h5ad_padding_fixture.py build/v1-cleanup/padding-fixture
# After the coordinated candidate Debug build:
build/debug/test_records_h5ad \
  build/v1-cleanup/padding-candidate-run \
  "$(realpath build/v1-cleanup/padding-fixture/fixed-string-padding.h5ad)"
```

`test_records_h5ad` normally runs without the optional fixture argument; with it, the test compares eager and backed ordinary columns, packed arrays, scalars, empty arrays, individual compound fields and complete compound records in all three padding modes. The generator is optional Python qualification; native tests do not acquire a Python dependency.

## Validation

Affected native groups are `h5ad`, `backed_h5ad`, `records_h5ad`, `record_zarr_rejection`, `zarr_sharding`, `gather_transfer` and `shard_selection`. The standalone engine tests verify byte-for-byte preservation after rejected v2/v3 consolidated-only attributes/create/erase/rename/resize operations, retained read/reopen fallback, invalid roots and explicit codec failure categories. Public root-path tests cover eager/backed owner/view writes in truncate/exclusive modes and confirm that the existing prefix/source store's complete file snapshot is unchanged.

The standalone private byte engine and `test_zarr_sharding.cpp` compiled sequentially with Clang 18, C++20 and `-O0`, linking system zlib/Zstd/Blosc, then passed (`shard writer tests passed`). The ignored receipt `build/v1-cleanup/backend-engine-validation.json` records the exact command, exit 0, source/binary hashes and scope. This covers the new consolidated/root/category cases and existing shard replacement, checksum, endian, budget and injected-I/O failures. It does not replace public-module, HDF5 or sanitizer checks.

The source-2 coordinated Debug run passed all 20 groups, including `h5ad`, `backed_h5ad`, `records_h5ad`, `backed_records`, `record_zarr_rejection`, `zarr_sharding`, `shard_selection`, `gather_transfer`, `backed` and the installed consumer (`build/v1-cleanup/debug-final-ctest.log`). Its HDF5-only configuration also passed all 16 groups and the installed consumer (`build/v1-cleanup/hdf5-final-ctest.log`). Earlier validation/counter failures were repaired; their initial log remains preserved. These source-2 receipts do not substitute for the final source-3 suite after the arrays repair.

The optional padding command above also passed with `record H5AD tests passed`; its stdout was observed but not retained in a separate log. The independent fixture result is a correctness check, not a performance measurement.

Future platform work still needs an actual target build: POSIX mapping, identity, positional I/O and shard replacement remain Linux-qualified mechanisms. Contexts and backend scratch remain worker-local; callers exclude structural changes and physical write conflicts.
