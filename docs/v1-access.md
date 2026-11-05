# Backed access cleanup

This note records the dispositions of access findings A01–A10 in
[the frozen review](review/access.md). Runtime and build performance are not
claimed without a matched measurement.

- **A01:** Opening validates that `obs` and `var` are dataframes with
  one-dimensional indices before reading their lengths. Malformed Zarr and
  H5AD axis fixtures exercise the error path; empty valid axes remain supported.
- **A02–A03:** Backed handle acquisition, context preparation, direct session
  access, edits and close are caller-serialized. Independently prepared worker
  contexts can execute concurrently subject to backend capabilities and physical
  conflict units. Shared transformed categorical handles require serialization;
  see [streaming](streaming.md) and [storage](storage.md).
- **A04:** Sparse source indices dispatch by stored integer type once per loaded
  index buffer for selection, copying, direct updates and prepared validation.
  Other width conversions and source-topology checks retain checked runtime
  dispatch. Further specialization needs measurement against added code and
  compilation cost.
- **A05:** One-shot reads reuse the description already acquired; direct
  caller-buffer reads avoid allocating a prepared-read owner. Owner `n_obs()`
  and `n_vars()` check their retained dataset without allocating a view.
- **A06:** Stream request lookup uses its lane-encoded opaque ID; lanes retain
  only the ID and reconstruct expected request coordinates, avoiding a second
  owned request descriptor. The adaptive default and caller-owned scheduling
  are unchanged. Request vectors remain necessary for arbitrary rank.
- **A07:** The public write contract explicitly identifies the allocation for a
  non-contiguous sparse source. The tested materialization preserves the
  existing sparse write and validation behavior; streaming sparse-source
  traversal is outside this access contract.
- **A08:** Context tile replacements increment an allocation-event counter;
  backend staging events remain backend-specific. Temporary metadata and
  third-party allocations remain excluded, and retained scratch bytes are not a
  peak or cumulative byte counter. Read-only contexts reserve only read
  workspace, permitting grouped reads when the admitted budget fits; a focused
  regression checks grouping and tile reuse.
- **A09:** A failed request remains outstanding. The caller may retry it only
  when replay is safe, or drain other workers and abandon the stream; partial
  writes are not rolled back. Existing coverage and conflict units are retained.
- **A10:** Retain the owner/view distinction and aliases. The constructor taking
  private dataset state is no longer public; internal session construction uses
  a private friend factory. Internal `detail` bridges remain available to connect
  C++20 named modules and are excluded from the public API contract. Code using
  the implementation-only constructor must use the public open functions.

Independent worker contexts are not a universal concurrent-handle guarantee.
No synchronization was added to caller numerical loops.

## Sparse dispatch code generation

The matched baseline and candidate Release builds used Clang 18 and
`-O3 -DNDEBUG`. `llvm-size-18` reported 172,104 and 207,051 bytes of `text`,
respectively, for `backed_array.cpp.o` (+34,947 bytes, 20.3%). `llvm-nm-18`
reported 4,990 and 11,413 bytes for the local `scan_sparse` symbol; candidate
output also contains multiple separately emitted typed callbacks. The baseline
has an out-of-line `read_index` dispatcher used within sparse operations; the
candidate still uses it for offsets and some source topology checks.

This is a whole-object comparison: the candidate also changes direct reads,
scratch accounting and admission, and is built against the changed core module.
It does not isolate the size of the dtype change or establish runtime speed.
Further cross-product specialization of input and output widths would add
instantiations and is retained as a measurement-dependent decision.

The ignored objects are
`build/v1-cleanup/performance-baseline/anndatapp-build/CMakeFiles/anndatapp.dir/src/backed_array.cpp.o`
(SHA-256 `3f0b65ab18c2e122411d072425770279514b3fc3b59c2e5e942d1ecbc7e974b1`)
and the corresponding path under `performance-candidate`
(SHA-256 `873ba7d7778afd91a44d727f99e2495e0989cb663c0e4b04d9f623c4509a02c2`).
Source hashes are `ec8cf5b0438bfe772cdcc2d8900bc2bf826f89d59b75ac29477707ae8d5438e2`
for baseline and `7ae11939ec5db5b7f8c7bfc6c2f44157b1ed7e7ce7ad04f04b55ec7b87eb9673`
for candidate. Reproduce the figures with `llvm-size-18` on the two objects and
`llvm-nm-18 --print-size --size-sort -C` filtered for `scan_sparse`.
