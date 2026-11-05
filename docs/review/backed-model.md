# Backed model and structural operations audit

Scope: full source read of the eight backed-model files in `build/thin-review-20261105/source/src`; lines below refer to that frozen snapshot, not necessarily the live tree. The accompanying [ledger](backed-model-ledger.json) inventories every helper/flow family, including retained ones. This is a source audit, not a test run or performance measurement. Storage backends and public streaming contracts have separate owners. Existing dirty live files were preserved.

The material issue is **B01**: record arrays are omitted from the private validation adapters. The remaining findings are bounded error-recovery, portability, or source-clarity work. The source contains deliberate bounded staging, type dispatch at transfer boundaries, and retained owners; none justifies a blanket rewrite. Backend calls and allocations are observable in the ledger; their runtime cost is not measured here.

## Findings

### B01 — Record arrays cannot traverse validation adapters (P1, correctness)

`backed_copy.cpp:154–189` wraps the original backend in `BorrowedBackend` when requested export validation differs from the open session. It forwards ordinary numeric and string reads, but not record-field reads; `Backend` defaults for these operations throw unsupported encoding (`anndatapp.backed.storage.cppm:128–146`). `copy_node_impl` dispatches `rec-array` to `copy_record` (`backed_copy.cpp:282`), which creates the record then reads its fields (`backed_records.cpp:221–255`). Thus an HDF5 record export with a changed validation setting reaches the unsupported record-read default. Independently, `BackedAnnData::validate_all()` uses `Sink` (`backed_copy.cpp:191–210,379–382`), whose record-creation/write defaults throw before validating the record payload. The existing record test checks same-validation export and readback (`tests/test_backed_records.cpp:204–245`), which does not exercise these adapter paths. An independent [targeted reproduction](probes/record_validation.md) confirms that a valid HDF5 record opened with `Validation::none` fails both `validate_all()` and full-validation export with `unsupported_encoding`; same-mode none/none and full/full exports round-trip successfully. With an invalid UTF-8 byte (`0xff`), the mismatched validation paths fail with `unsupported_encoding` before the real diagnostic; full/full export reports `invalid_argument` for invalid UTF-8, while none/none export succeeds. Raw logs are retained in `build/thin-review-20261105/record-verification/`.

Add forwarding for both record-read methods in `BorrowedBackend` with the same retained-session open check, and implement no-op record creation and writes in `Sink`, retaining real source reads and validation. Audit the complete virtual interface when adding capabilities so defaults cannot silently bypass the adapters' purpose. Acceptance: nested numeric and variable-string record arrays validate via `validate_all()`; export an HDF5-backed source opened with `Validation::none` with full export validation, reopen and compare fields; include the inverse setting if supported, malformed UTF-8/full-validation rejection, empty/scalar record cases, and preserve unsupported destination behavior. This is private implementation work, with no public migration.

### B02 — Failed category page refill can publish an invalid cache page (P2, recovery)

`category_position` updates `page_begin` before `read_exact` (`backed_category.cpp:236–243`). If a temporary-file read fails or short-reads after filling only part of the page, the exception escapes while that page remains marked current; retrying the same plan/page skips refill and may return stale or partially read positions. The file-backed mapping is reached when the selected dictionary exceeds its in-memory threshold (`:299–310`). Invalidate `page_begin` before refilling and publish the new marker only after the full read succeeds; the previous page contents may be overwritten by a partial read. A fault-injection test must fail a refill, retry the same page, and verify either successful refill with exact positions or a repeated error; preserve successful paging, scratch accounting, and file lifetime. Private fix, no API migration. This finding concerns recovery after an I/O error, not evidence of an observed production failure.

### B03 — Duplicate bounded string traversal (P2, style; prior S5)

`read_strings_all` and `read_string_array` independently implement the same logical-coordinate, adjacent-run, bounded-prefix loop (`backed_codec.cpp:172–218,267–320`); the distinct destinations (string vector versus packed bytes/64-bit offsets) are necessary. Existing form, abbreviated:

```cpp
const auto logical = coordinate(output_shape, flat);
auto start = physical_coordinate(axes, logical, value->path);
std::vector<std::size_t> rectangle(logical.size(), 1);
// discover consecutive last-axis run, read bounded prefix, advance flat
```

A private concrete-callable traversal could express that operation once:

```cpp
string_batches(value, axes, output_shape, count, budget, [&](auto& batch) {
    // move strings or append checked packed bytes and offsets
});
```

This is pseudocode, not a claim that an extraction is automatically shorter: include the complete helper and both call sites in the comparison. Preserve rank-zero and empty shapes, prefix sizes, selection order/duplicates, conditional validation and both packed overflow checks. Exercise independent H5AD/Zarr fixtures, irregular/reversed selections, UTF-8, scalar/empty values and tight scratch. Measure build/runtime effects before claiming improvement. Private change; no migration. This is the previously deferred S5, not a newly completed fix.

### B04 — Three private UTF-8 decoders duplicate the same invariant (P3, style)

`backed_codec.cpp:128–170`, `backed_copy.cpp:39–52,213`, and `backed_writer.cpp:77–105` each decode UTF-8; the writer additionally forbids embedded NUL and uses a different error code, while the copy path reports more specific reasons. Separate the decoder's boolean/diagnostic policy from boundary-specific errors only if the total private implementation becomes smaller and diagnostics remain intentional. For example, replace the writer's private boolean loop with a single shared check:

```cpp
// before: valid_utf8(text) contains a complete decoder
if (validation == Validation::full && !valid_utf8(text)) fail(...);
// after: private shared decoder; writer still applies its own error and NUL rule
if (validation == Validation::full && !is_valid_utf8(text)) fail(...);
```

A concrete extraction must preserve error code, path, validation mode, NUL policy and decoder acceptance; compare malformed continuation, overlong, surrogate, out-of-range, truncated and valid Unicode sequences for each caller. No public migration or speed claim.

### B05 — Temporary spill paths are POSIX-specific (P3, portability)

`backed_category.cpp:19–20,35–77,86–95` uses `unistd.h`, `fileno`, `pread`, `pwrite`, `ftruncate`, and `off_t`; `backed_index.cpp:19,36–44` uses `fseeko`/`ftello` and `off_t`. These are legitimate bounded spill mechanisms on the initial Linux toolchain, but are not portable standard C++ facilities. No supported non-POSIX target is established here. Document the platform assumption or, if a non-POSIX target is adopted, isolate a private positional-I/O adapter and retain partial-transfer/EINTR, range checks, cleanup, concurrency and scratch behavior. Acceptance for an extension requires a real target build and spill/large-offset/failure tests. No migration on current supported setup.

### B06 — Categorical validation performs dtype dispatch per code (P3, style/dispatch)

`backed_copy.cpp:80–107` resolves `category_code`'s switch for each loaded value, whereas `backed_category.cpp:115–144,170–183` already selects a type once outside its loop. Keep bounded runs, signed/unsigned range checks, missing sentinel and validation semantics; a small private typed scan can make the transfer boundary clear. Existing and proposed shape:

```cpp
// before
for (std::size_t j = 0; j < run; ++j) check(category_code(data.data()+j*width, info.dtype, path));
// after (schematic)
visit_integer(info.dtype, path, [&]<class T> { for (std::size_t j = 0; j < run; ++j) check(decode<T>(data.data()+j*width, path)); });
```

Compare the full source and generated code before sharing machinery: `decode_code` rejects values below -1 while current `signed_value` leaves that check to its caller. Test all integer widths, unsigned overflow, -1, invalid negatives and upper bounds in full/none modes. No speed claim; measurement is required for a runtime or build claim. Private change only.

### B07 — Writer repeats validation after whole-dataset preflight (P3, investigate)

`initialize_store` validates the dataset (`backed_writer.cpp:742–745`), then `write_table` validates each table again (`:566–568`); `write_column` can rescan categorical dictionaries and codes (`:505–530`). The repeated traversals and hash-set allocations are source-visible, but their cost is unmeasured and separate entry points need their own validation. Investigate carrying a private already-validated condition only if it preserves mutation, exception ordering, path diagnostics, mode semantics and direct `write_element` use. Test malformed categories, indexes and strings for both modes and record operation/allocation counts on identical workloads before any performance claim. No public migration is presumed.

## Contracts, reconciliation, and limits

- `docs/compliance-fixes.md` records R1–R5 and S1–S4/S6/S7 as implemented in its identified baseline; R5's removal of unused category-plan fields is visible in this snapshot (`backed_category.cpp:187–195`), R3's tagged index comparison in `backed_structure.cpp:273–325`, and S4's reuse of writer coordinates in `backed_writer.cpp:348–366`. Do not reopen them as unresolved. R6/R7 and category-state tagging remain deferred design candidates outside this assignment; S5 remains B03.
- `docs/style-review.md` records regular selection and focused metadata fixes as completed, with its items 3, 5 and 6 separate. Its string-materialization issue remains documented, not a claim that all string copies were redesigned. `read_packed_column_strings` now explicitly uses two bounded passes and one final packed byte allocation (`backed_codec.cpp:224–264`); those two passes perform backend reads twice and validate via the enclosing owner, and should not be conflated with completed general string-copy work.
- Geometry-aware dense bulk-copy measurements in `docs/storage-performance.md` are bounded to their recorded source, fixtures, settings and instrumentation. This review ran no performance experiments and does not extend them. String and record paths have bounded staging but their final materialized owners can scale with result size; `scratch_bytes` is not total RSS.
- `docs/storage.md` requires callers to synchronize access. A spilled category plan shared by retained handles has mutable `page` and `page_begin` (`backed_category.cpp:187–195,224–243`); unsynchronized calls on the same plan race even when payload buffers differ. Separate independently opened sessions and backend thread-safety conditions remain relevant; no universal parallel-safety claim follows. Numerical work on acquired buffers is separate from backed access. Revision and structural mutation coordination also remains the caller's responsibility.
- Record string and category spill paths depend on backend and OS behavior, and full storage compatibility requires independent fixture reopening, not source inspection. The audit ran no performance experiments; a separate verifier added the targeted B01 reproduction described above. No fault injection or thread-sanitizer testing was performed for this review.
