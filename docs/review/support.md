# Supporting code review

Scope: supporting code and records in frozen `build/thin-review-20261105/source`; the library implementation and top-level CMake/CI are reviewed separately. I read the frozen `AGENTS.md` and `STYLE.md`; no files in the frozen source were changed. No build, test, benchmark, or timed experiment was run for this review.

## Findings

### S01 — Current module-build benchmark encodes a stale no-storage architecture

`tools/compile_benchmark.py` labels `optional_format_modules` as “none implemented” at lines 240–241 and repeats at 389 that no HDF5/Zarr or optional module is implemented. The same frozen source tree contains H5AD/Zarr modules, storage adapters, and backend tests; the report also hashes those sources into `build_input_hashes` (lines 121–141). Its configure command (166–171) does not disable storage backends, so its cold build may compile them while its module-layout and notes explicitly declare their costs unmeasured. This makes the report internally inconsistent and the stated coverage unsuitable for current import/build-cost conclusions. The checked-in `benchmarks/results/module-build.json` records a different source root (`/home/tumlinson/anndata.cppm`) and is historical evidence, not a current measurement.

**Requirement relevance:** fast module imports and consumer rebuild time are high priorities; implementation and interface build costs must be measured against the configuration being described.

**Fix / acceptance:** derive module-layout and enabled-backend descriptions from the configured CMake cache/targets, or give the benchmark explicit backend-disable options and scope its claim to that configuration. Confirm reported targets and build inputs agree, then produce a fresh source-hashed report from this snapshot. Until then retain the old result only as historical evidence.

### S02 — `record_measurements.py` is a checkout-specific continuation recipe, not a portable current benchmark

The script pins output to `ROOT/build/record-measurements` and dependency prefix `/home/tumlinson/anndatapp/build/dependencies/usr` (lines 13–15), then its resume/runtime/candidate operations assume pre-existing `final2-work`, `final2-review`, and baseline binaries/reports (38–60, 86–143). Its normal route takes one sample (195–200), hardcodes Clang 18 and the dependency prefix (190–194), and copies the supplied current `src/` files into a pre-existing work tree (112–143). These are legitimate recovery steps for the named interrupted campaign but do not reproduce a clean baseline/candidate experiment on another checkout. Running the `final` route is also stateful: it refuses an existing target (170–173), creates a persistent copied tree, logs and reports, and appends probe targets to that copy. Treat its outputs as campaign-specific; do not cite them as a general benchmark recipe.

**Requirement relevance:** measurement claims need portable, attributable source/configuration identity and explicit repetition/cache conditions; cleanup or path assumptions must not invalidate preserved evidence.

**Fix / acceptance:** label/document the script as a frozen campaign recovery utility and keep it out of generic benchmark entry points. For a reusable benchmark, require explicit output/dependency paths, verify the source and baseline identities, expose toolchain/configuration, and report the one-sample limitation. No campaign was run during this review.

## Supporting coverage and observations

- The test suite includes direct expected-value assertions in C++ and independent Python reopens/physical-format checks. The Python drivers explicitly check semantics and storage representations; those are stronger file-format oracles than same-library round trips. The compact public consumer examples exercise module imports, but they are smoke examples rather than downstream installed-package compatibility proof.
- Bounded probes generally carry explicit caps and independently validate outputs. For example `tools/streaming_probe.py:52-55,85-97` records 30-second and intercepted-I/O bounds, checks a single counter record, and validates sparse hashes; write readback is outside the timed/counter cap by design (`:110-128`). The completeness/record probes similarly separate preparation and checking from measurement. These boundaries must remain attached to any interpretation of reported times.
- The repository contains extensive benchmark/test JSON, workflow observations, and receipts. They are source-specific historical records, not fresh validation. Several carry explicit source hashes; others have no root timestamp/source identity fields. No claim here treats them as current test results or current timings. Coverage is limited to static inspection; test execution and artifact revalidation were not part of this assignment.
- Style review found no broad reportable style defect in the inspected probe/test support: code is mostly explicit about allocations, counters, timing boundaries, and expected values. Some one-off campaign scripts are long and repeated across geometry cases, but they are campaign-local and no maintenance or correctness requirement was shown to justify refactoring them during this review.

## Coverage

`support-ledger.json` enumerates 169 supporting files under `tools/`, `benchmarks/`, `tests/`, `examples/`, and `planning/`: 10, 86, 59, 1, and 13 files respectively. It records each file's purpose, evidence disposition, and any finding reference. The review did not inspect production library internals, top-level CMake, CI, or rerun any captured campaign. The 169-file count is an inventory count, not a claim that historical test/benchmark evidence was re-executed.
