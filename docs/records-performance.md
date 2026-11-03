# Record measurements

This is a bounded observation of the native records implementation, not a
forecast of application performance. The baseline is the exact archived commit
`2ddf57559412c90297e5b525bd6a5b6d9a0c36d1`. Raw logs and build artifacts are
retained locally in the ignored `build/record-measurements/` directory. The
compact machine-readable record is `benchmarks/results/records-summary.json`.
These measurements precede integration of refactor commit `70d8698`; their
source and binary identities are preserved separately from combined validation.

The compile experiment uses Clang 18, CMake/Ninja, Release (`-O3 -DNDEBUG`),
two build jobs, both storage adapters and fresh source/build copies. It imports
`anndata.core` and `anndata` in otherwise identical consumers after building
the library. The OS page cache is not cleared. These are single samples;
concurrent compilation and run-to-run variation limit timing interpretation.
The final build also appends a private declaration in a disposable copy of
`src/records.cpp`, then adds a public constant to its record interface, and
compares BMI and consumer object hashes before and after each edit.

The native loop probe uses 32,768 rows of `{double x, double y, Utf8Ref label}`,
100 passes of `x += 0.125*y`, and identical initialized storage for each path.
It compares direct struct access, the acquired rank-one member operator, and
acquired pointer plus byte stride, checks every result and reports allocations
from global `operator new`. Separate measurements cover 10,000 retained member
acquisitions and a deep copy of 48-byte pointer-backed text. It checks that the
copied text has distinct addresses and survives mutation of the original. The
allocation counter covers ordinary global `new` and `new[]` calls; it does not
intercept `malloc`, aligned allocation, or third-party allocators. Initialization
and descriptor acquisition are outside the loop timings. Values are one
single-thread observation each and establish neither variance nor a general
speedup. The workload is synthetic and not an HDF5/I/O measurement.

On the measured Xeon E5-2698 v4 with Ubuntu Clang 18.1.3, the single baseline
consumer observations were 0.342 s for core and 0.358 s for the facade. The
updated snapshot measured 0.371 s and 0.410 s respectively, with possible
concurrent builds; the observations do not establish a timing distribution.
The nine baseline BMIs occupied 39,523,092 bytes; the ten measured BMIs occupied
46,428,536 bytes, including 6,182,324 bytes for the record partition. A private
record implementation edit left all measured BMI hashes and the facade
consumer object hash and timestamp unchanged. Adding a public constant changed
eight dependent BMI hashes and rebuilt the facade object. The incremental
public-edit operation took 46.27 s under the stated conditions.

The independent HDF5 call-count probe used 8,192 rows, 256-row gzip-compressed
chunks, and two or eight `float64` members. A single-field backed read made one
`H5Dread` call and selected 65,536 native bytes in both cases. Whole-record
reads made two and eight calls and selected 131,072 and 524,288 native bytes,
respectively. Export of the record itself made the same number of reads and
writes. Field and export values were checked. These are API call and selected
memory-byte counts, not physical I/O bytes or decompression counts. The source,
fixture generator, and local artifact instructions are under
`benchmarks/record_io_probe/`; the raw report hash is in the JSON summary.

To reproduce the build probe from a matching checkout, use a fresh ignored
measurement directory and the recorded baseline commit:

```sh
mkdir -p build/record-measurements/baseline-source
git archive 2ddf57559412c90297e5b525bd6a5b6d9a0c36d1 | tar -x -C build/record-measurements/baseline-source
python3 tools/record_measurements.py baseline --source build/record-measurements/baseline-source
python3 tools/record_measurements.py final --source .
```

Use a new `--tag` when preserving an existing result. This recorded run required
`python3 tools/record_measurements.py final --source . --resume-final` after an
interrupted harness run; its final consumer builds use the already populated
build tree, so no final cold-build time is claimed. Logs record actual compiler
commands and raw reports include the measured snapshot's individual source and
artifact hashes. The first failed integrated build and its diagnostic are also
retained locally.

Before caching the first byte stride in the member descriptor, the runtime
probe observed 5.585 ms for direct rows, 15.299 ms through the member operator,
and 5.219 ms through the acquired byte stride. Disassembly showed two calls
to `DenseDimensions::stride` per member-loop element; raw and acquired-stride
loops contained no calls and were unrolled.

After caching that stride at acquisition, the same probe observed 5.497 ms for
direct rows, 5.624 ms through the member operator, and 5.518 ms through the
acquired byte stride. The candidate member loop contains no calls and is
unrolled two rows. These are one sequential observation per binary, not a
variance estimate or an application-wide speedup. Both runs checked every
numeric value and copied-text ownership; loop allocation counts were zero.
Ten thousand acquisitions took 0.217 ms before and 0.176 ms after, with zero
counted `new` calls in either run. Deep-copy observations were 4.879 ms and
5.083 ms, each with 19 counted calls requesting 1,575,632 bytes. These counts
exclude `malloc`, aligned allocation, and backend allocators.

The two runtime snapshots differ only in the record module interface and
implementation; the workload source, other source files, and build settings
match. The compile timings, BMI sizes, and edit-isolation checks above precede
the cached-stride interface change and do not measure its incremental build
cost. Full source and binary hashes are in the JSON summary and raw reports.

The candidate record interface and implementation hashes matched the source
when measured. The H5AD implementation changed between the earlier runtime
snapshot and the final integrated source; the numeric and text-copy probe does
not use H5AD. Run the native executable with a
30-second cap after compiling and when competing measurement activity is quiet:

```sh
python3 tools/record_measurements.py final --source . --prepare-runtime
timeout 30s build/record-measurements/final2-work/build/record_native_bench > build/record-measurements/runtime-result.json
objdump -d -C build/record-measurements/final2-work/build/record_native_bench > build/record-measurements/runtime-disassembly.txt
```

The `--prepare-runtime` command resumes the retained `final2` build from this
campaign and records a separate source snapshot and binary hash. A fresh run
uses the executable built by the `final` command instead.

The candidate rebuild and disassembly were captured without repeating the cold
build. With the retained earlier runtime binary and build tree, run:

```sh
python3 tools/record_measurements.py final --source . --prepare-candidate
timeout 30s build/record-measurements/final2-work/build/record_native_bench > build/record-measurements/candidate-result.json
objdump -d -C build/record-measurements/final2-work/build/record_native_bench > build/record-measurements/candidate-disassembly.txt
```
