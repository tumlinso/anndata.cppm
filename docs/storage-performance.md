# Backed storage qualification

The storage workload uses [storage.cpp](../benchmarks/storage.cpp) and
[storage_benchmark.py](../tools/storage_benchmark.py). The tracked report is
[storage-summary.json](../benchmarks/results/storage-summary.json). A result is meaningful only
with the recorded source snapshot, fixture identity, toolchain, settings and
individual observations. A failed or unsupported operation is a result, not a
measurement to discard. The earlier H5AD report covers a different eager API
and is not a paired baseline for these workloads.

The completed baseline and subsequent bounded probes are summarized below;
the original raw reports and fixtures remain local and are excluded from Git.
For a new investigation, start with one selected workload against an existing
fixture and executable. For example, replace the paths in this command:

```sh
.venv/bin/python tools/storage_benchmark.py \
  --output /tmp/anndata-storage-investigation.json \
  --dependency-prefix "$PWD/build/dependencies/usr" \
  --skip-module-build --storage-executable /path/to/storage_benchmark \
  --fixture zarr2:/path/to/fixture.zarr --action dense \
  --runs 1 --timeout-seconds 30 --limit-mib 512 \
  --stop-file /tmp/anndata-storage-stop
```

Use a stop-file path that does not exist initially. The driver atomically
checkpoints progress, checks the stop file before starting more work, and by
default does not repeat a timed-out fixture/action. It defaults to a 30-second
operation timeout; specify `--runs 1` for an initial single observation.
`--limit-mib` limits address space, not I/O. For bounded I/O, use the
[paired](../benchmarks/storage_probe/paired96/README.md),
[small-geometry](../benchmarks/storage_probe/geometry128k/RESULT.md), or
[large-confirmation](../benchmarks/storage_probe/large_confirm/README.md)
recipes, whose interposers enforce the stated intercepted-byte caps. Copies,
hashing and independent validation may perform additional I/O outside those
operation caps. A supplied executable is hashed; independently establish its
correspondence to the source snapshot. Use `--help` for focused options and
inspect failed or skipped observations before drawing conclusions.

Add `--readonly-sparse-fixture h5ad:/absolute/path/to/input.h5ad` to include
an existing sparse dataset. This measures opening and the two sparse
selections only; the CLI rejects mutating actions with this option. A
column selection on CSR may scan all rows and take much longer than a row
selection. Each subprocess performs preparation, an allocating comparison and
two reads into the same buffers, so process time and I/O exceed the single
timed `read_into` call. A timeout remains a failed observation.

The harness opens each fixture in a fresh subprocess with 64 MiB scratch and
the decoded cache disabled. It records open and operation times separately,
and process maximum RSS includes opening, output buffers, validation and
checking, not just the timed operation. Dense reads select a 32 by 32 tile;
sparse row and column selections retain the opposite axis. The prepared
`read_into` operations compare their bytes with an allocating read and repeat
into the same caller-owned buffers, checking pointer identity and contents.
These comparisons detect disagreement between API paths; they are not an
independent format oracle. Fixture and interoperability tests must separately
establish semantic correctness. Sparse preparation time is reported
separately from the read itself.

The mapping probe applies only to eligible full arrays, and reports failure
instead of treating other physical encodings as zero-copy. A selected write
updates a 32 by 32 float32 tile and reads it back after reopening. The
structural-column probe adds a float32 observation column and reopens it.
Reindex removes one observation from a writable disposable copy; export
writes to a new path.
Both should be run at multiple physical sizes when assessing memory scaling.
Export and reindex verify dimensions and an X tile after reopening. These
checks do not independently establish the semantics of all aligned slots:
the interoperability suite supplies broader evidence.

The large logical fill fixture is intended to have a 1 TiB float32 X, no
allocated numerical chunks and a small zero-filled selected tile. Run the
`huge` probe in a fresh process with a 512 MiB address-space limit. Record
shape, chunk/shard configuration, physical size, limit, return code and RSS;
success with the expected zero tile rules out materializing that full X in
that execution. It does not prove bounded behavior for every selection,
metadata schema, compressed chunk, export, or structural update. The full
logical array must never be written by the fixture creator. Cache state is
observational: the harness does not drop host page caches, and measurements
on a shared host can vary with other activity.

Compiler measurements use a frozen copy of the source with transient runtime,
build, Git and result directories excluded. Each configuration gets its own
fresh build tree and three trials. Report compiler and scanner versions,
dependency prefix, CMake flags, job count, source manifest and worktree
status. Core-only, HDF5-only, Zarr-only and combined configurations are
measured separately. Record BMI sizes and actions for library and actual
consumer objects separately; object hashes and timestamps establish what was
rebuilt after private implementation and public interface edits. An unchanged
hash does not by itself establish that compilation did not occur. Fresh build
trees are not cold operating-system caches. Import timings include their
documented build steps and cannot be attributed to parsing alone.

Interpret all observations within their exact fixture, host and filesystem
conditions. Three repetitions provide a median and range, not a general
performance guarantee. The parallel probe reads the same selected dense tile
from four independently opened sessions and verifies agreement; its timing
includes session opens. HDF5 is exercised this way only when the linked
library reports thread safety. The bounded workload is neither a speedup claim nor
evidence of universal memory use, compatibility, parallel safety or
zero-copy access. Caller parallelism needs a separately recorded workload and
the backend's actual thread-safety conditions; concurrent benchmark runs must
not overlap the timing campaign.

## Baseline, 3 November 2026

The full original measurements are preserved locally, but excluded from Git,
in `benchmarks/results/storage-baseline.json`, SHA-256
`e402278c5267684586f173ebbe3614b0382b86e8bbbd6befa6b2634ef47491ba`.
The source manifest is `0f33852c9b9963e24932b955a04f2692b7df2de7e9a5c98baf2f139dce910785`.
The locally retained annotated `benchmarks/results/storage.json` classifies the
two user-stopped real-data repetitions and invalid initial write attempts;
the tracked summary records their interpretation and hashes. The machine was an Intel Xeon E5-2698 v4,
Clang 18, release builds, two Ninja jobs, 64 MiB scratch, zero decoded cache,
and a 512 MiB address-space limit per storage subprocess. Page caches were
not cleared. Each completed group below has three trials unless stated.

| Configuration | Fresh build median (range), s | Private edit median, s | Interface edit median, s |
|---|---:|---:|---:|
| Core | 29.34 (29.15–29.90) | 3.90 | 28.94 |
| HDF5 only | 36.35 (36.02–36.52) | 4.09 | 35.85 |
| Zarr only | 41.50 (41.15–42.30) | 4.01 | 33.30 |
| Both | 46.94 (46.88–47.48) | 4.23 | 38.92 |

In all twelve builds, editing `backed_array.cpp` changed one library object
while preserving public BMIs and actual consumer objects by timestamp and
hash. Compiler actions show the implementation compiling and consumers
relinking. Editing the public arrays interface rebuilt 7/8/8/9 BMIs and
3/4/4/5 consumer objects for the configurations in table order; captured
compiler actions confirm those consumer compilations.

| 4096 × 4096 float32 X | Export median (range), s | Reindex median (range), s | Median process RSS for export/reindex, MiB |
|---|---:|---:|---:|
| H5AD, uncompressed contiguous | 0.449 (0.433–0.460) | 0.466 (0.457–0.478) | 18.6 / 17.5 |
| Zarr 2, 512 × 512 Blosc/LZ4 chunks | 190.37 (190.06–191.12) | 191.99 (191.70–193.05) | 17.0 / 16.6 |
| Zarr 3, 256 × 512 Zstd inner chunks | 186.56 (185.50–186.76) | 191.70 (191.13–193.61) | 16.6 / 16.6 |

These are different physical encodings and are not paired speedups. The
Zarr 3 source had 2048 × 4096 shards. On the 256 × 256 inputs, export RSS
medians were 16.9, 13.1 and 13.5 MiB respectively. The two-size observations
are consistent with bounded streaming in these cases; they do not prove a
general memory bound. The very slow Zarr bulk path is a material limitation.
Source inspection finds row-at-a-time transfers and per-chunk decode/rewrite
operations; repeated chunk work is a plausible cause, not a profiled finding.

All nine selected-tile reads of unallocated 524,288 × 524,288 float32 X
passed under the 512 MiB limit: three each for H5AD, Zarr 2 and Zarr 3. Median
RSS was 15.0–15.2 MiB. This establishes bounded access to that selected tile,
not bounded export or reindexing of the 1 TiB logical array.
Post-run metadata inspection found zero allocated HDF5 X bytes and no stored
Zarr X chunks; the complete stores occupy about 48.1, 4.2 and 0.7 MiB,
respectively, including axis metadata.

The supplied gzip CSR file contains 68,008 × 26,587 cells and 114,455,556
entries. The selected-row `read_into` median was 0.0327 s (0.0324–0.0355).
One selected-column `read_into` completed in 69.41 s; its whole process took
256.8 s because preparation and correctness checks include additional reads.
The other two column repetitions were stopped at the user's request and are
not numerical failures or timeouts. Initial write probes used a read-only
slice, so they are invalid workload attempts, not evidence of a failed
writable-owner API. Generated Zarr mapping attempts were explicitly rejected
for ineligible compressed/chunked arrays; eligible H5AD mapping passed.

Corrected writable-owner measurements used the original baseline library and
a separately identified driver; their summary is tracked, and the full
`benchmarks/results/storage-write-supplement.json` is retained locally.
The driver source SHA-256 was `c4d9a9077f9a9d0ff4d4d77b28813ac2d2d55f1815cd0b0693e473747daca66f`,
the executable `a3e5fed3ef8e383a62fd7d8656518113e5c8f6314512600a7a4a2c590ed4b997`,
and the baseline library `90d24ee98b7d760e3b42517ccd004b2867bca047ad3603c9d54a78f9ad3321f8`.
All 18 selected-write trials passed. For the larger source, median 32 × 32
write times were 0.00179 s for H5AD, 0.1267 s for Zarr 2, and 1.476 s for
Zarr 3. Files were copied outside the operation timer. These represent
different storage encodings and should not be read as paired speedups.

A separately instrumented baseline exported rows [1792, 2176) of the Zarr 2
source across a 512-row chunk boundary. Its 6 MiB useful payload took 15.65 s
with instrumentation; the output reopened and exactly matched the source
selection in Python. During the export, interposition counted 3,074 Blosc
decodes and 3,074 Zstd compressions, about 3.92 GB read and 1.08 GB written
through intercepted libc symbols, and 978,618 C++ `new`/`new[]` calls
requesting 22.49 GB cumulatively. This is not peak live memory and excludes C
`malloc`. The counters are codec or libc API calls, not logical backend calls;
they may miss direct syscalls. Ptrace tracing was denied and the six-event
hardware counter group did not run. The probe had a 30-second timeout but no
enforced byte cap: its observed I/O exceeded the initial estimate based on
distinct chunks. Instrumentation affects timing. The
[probe sources](../benchmarks/storage_probe/) and their summary are tracked;
full original logs are retained locally under
`benchmarks/results/storage-probe-baseline/`. Recorded absolute temporary
paths identify the original run environment.

A separate Zarr 3 probe crossed its 2048-row shard boundary and stopped at an
enforced 512 MiB read cap after 3.91 s. Its partial output is not a correctness
result. The run intercepted 536.8 MB read and 100.0 MB written, with 18,578
Zstd stream decode calls and 934 compress calls; a two-event hardware group
did run. The difference between full and partial probes precludes treating
their timings as a format comparison.

The first rectangular-transfer candidate was compared with the baseline on
rows [2000, 2096) of the same Zarr 3 source, crossing its shard boundary.
Both outputs matched the source exactly. With the same instrumentation, one
baseline run took 1.889 s and the candidate took 0.0522 s; this is one
instrumented pair, not a general speedup. Logical X calls fell from 96 reads
and 96 writes to one of each. Intercepted read/write bytes fell from
426.4/68.0 MB to 7.56/1.42 MB; Zstd stream-decode and compression API calls
fell from 8,426/194 to 130/4. Both runs stayed below the 30-second and
512 MiB-per-direction caps. Captured glibc `malloc` calls fell from 379,753
to 31,803; these include ordinary C++ `new`, omit un-intercepted internal
allocations, and count cumulative requests rather than live peak. Source,
instrumentation and candidate identities are in the tracked summary and
[paired probe recipe](../benchmarks/storage_probe/paired96/README.md).

On the original 6 MiB Zarr 2 selection, the candidate completed in 0.0710 s
and its output matched the baseline and source exactly. Intercepted read/write
traffic was 14.89/5.62 MB, with 18 Blosc decodes and 10 Zstd compressions.
The original driver source was reused, but the candidate used an I/O-capped
interposer whereas the earlier baseline interposer was uncapped; the 15.65 s
versus 0.0710 s timings therefore have a measurement-method difference. Both
are single instrumented runs on this fixture.

A further 512 × 512 Zarr 3 geometry probe with 32 × 64 inner chunks and
64 KiB scratch stopped while writing the destination observation index:
its string chunk exceeded the codec working-memory limit. It made zero X
backend reads or writes and provides no evidence about X chunk alignment.
The failed attempt and its limits are recorded in the summary; the
[fixture and probe recipe](../benchmarks/storage_probe/geometry64k/README.md)
are preserved for reproduction.

At 128 KiB scratch, the same 512 × 512 fixture exported successfully and
matched X and both indexes exactly. The source used 32 × 64 inner chunks and
256 × 512 shards; the actual destination chunks were 64 × 64. Candidate A2
made 32 logical X reads and 32 writes, with 478 Zstd stream-decode and 272
compression API calls across the instrumented operation. This identified
additional partial-chunk work; the codec counts include metadata activity.
The final geometry-aware candidate produced byte-identical output in 0.0381 s
versus A2's 0.0922 s in one instrumented pair. Logical calls remained 32 per
direction, but Zstd stream-decode/compress calls fell from 478/272 to 158/80,
and intercepted read/write bytes from 3.88/2.37 MB to 1.40/0.95 MB. On the
96-row shard-crossing case it retained the one-read/one-write behavior and
exact X and index values (0.0541 s versus A2's 0.0522 s, one run each).

The final candidate also completed one full-size 4096 × 4096 confirmation
per format and operation. Full X and both axes matched independently, and
neither 256 MiB-per-direction operation cap was reached.

| Final candidate, 64 MiB X | Export, s | Reindex, s | Intercepted read/write per operation, MB |
|---|---:|---:|---:|
| Zarr 2 | 0.439 | 0.406 | 59.4 / 59.8 |
| Zarr 3 | 0.535 | 0.495 | 60.0 / 59.8 |

These single candidate runs compare with the baseline's roughly 186–193 s
Zarr operations on the same source fixtures, but use different instrumentation
and are not a repeated paired benchmark. The byte caps cover the measured
operation only. Input copies and hashes, reference reads, and full Python
validation ran outside the cap; they also warmed caches. Zarr 3 input used
shards; the output used unsharded 512 × 512 chunks. The initial Zarr 2 export
validator had a harness-only axis-length error after comparing values; its
corrected validation checked the existing output without rerunning export.
