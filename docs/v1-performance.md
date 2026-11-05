# AnnData++ 1.0 standard performance suite

The final [compact receipt](../benchmarks/results/v1-summary.json) contains the
source4 library comparison: ten fixed scenarios, all **38 operations validated**
in both revisions, five alternating observer-free timing pairs and one separate
diagnostic per scenario/revision. There are 100 timing subprocesses and 20
diagnostics. Final regression flags under the stated >5% paired-median plus
separated-range rule: **none**. This is bounded evidence for these geometries,
without a general speed, compatibility or parallel-safety claim.

Charged process time was 6.552 seconds baseline and
5.639 seconds candidate. All scenario/revision caps passed. The JSON also
sums prior recorded campaigns by immutable library revision; those cumulative
30-second scenario and 300-second revision budgets passed. Baseline code digest:
`96dd8c96fe516d7f71d76960cea3be29609fd66392a613bd8a8350edf1f9e08c`.
Final candidate-source-4 digest:
`50e2f3c952ad0068505fe3a929d288db372d20fb6ba094d89de655026386f0a0`.
Raw final samples: `build/v1-cleanup/performance-run-5/samples.json`, SHA-256
`7437f7115d6503070589489defdd2c6634a873d05dd39bb600f6ac1cdf878710`. Verbose summary and prior comparisons remain local under
ignored `build/v1-cleanup/`; the compact committed JSON preserves all values.

The table reports median elapsed time for the full repetition count. Overlapping
sample ranges do not establish a resolved timing change; the JSON retains all
five samples and every one of the 38 operations.

| Operation | Baseline → candidate, ms | Diagnostic C++ allocations |
| --- | ---: | ---: |
| Borrowed descriptor,10,000 calls | 1.568 → 0.541 | 30,000 → 0 |
| Checked descriptor,10,000 calls | 0.350 → 0.299 | 0 → 0 |
| Owned strided copy,12 copies | 11.013 → 7.007 | 96 → 132 |
| Owned gathered copy,12 copies | 9.492 → 6.525 | 132 → 132 |
| Caller-provided contiguous copy,12 copies | 0.281 → 0.296 | 24 → 24 |
| Caller-provided strided copy,12 copies | 7.069 → 7.038 | 48 → 48 |
| Caller-provided gathered copy,12 copies | 6.365 → 6.407 | 48 → 48 |
| CSR native gather,8 copies | 0.246 → 0.054 | 264 → 72 |
| CSR opposite gather,8 copies | 0.856 → 0.256 | 4,392 → 88 |
| CSC opposite gather,8 copies | 0.817 → 0.227 | 4,392 → 88 |
| Category pruning,8 copies | 15.953 → 7.473 | 504 → 368 |
| Focused record,10,000 calls | 74.285 → 2.689 | 110,000 → 0 |

Source2's opposite-axis sparse copies regressed about 3.8× despite fewer
allocations. Source inspection connected the extra work to count/fill binary
lookups. Source3 added a bounded head/next lookup for dense minor-axis gathers;
source4 retains it. Final CSR/CSC opposite-gather paired ratios are
0.288/0.272; allocations fell from 4,392 to 88 and requested bytes from
852,160 to 248,064 per eight-copy operation. Large sparse lookup geometry remains
outside this timing scope.

An earlier 35-operation source4 run flagged checked descriptor acquisition:
0.296 → 0.341 ms per 10,000 calls, about 4.45 ns/call marginal-median difference.
The final 38-operation run has overlapping checked-acquisition ranges and no
flag. The earlier observation is preserved rather than erased. Controls found
zero allocations, zero descriptor division instructions after guarded overflow
builtins, and near-baseline descriptor text size (1,937/1,968 bytes). The residual
boundary variation's cause remains unresolved; it is not attributed entirely to
arithmetic and no general acquisition advantage is claimed.

The separately labeled **source2** consumer-build evidence remains historical:
one fresh core-import consumer and three source-only recompiles, against prebuilt
BMIs. Fresh compilation was 0.226/0.183 seconds baseline/source2, recompile medians
0.230/0.174 seconds, import BMI closure 26,204,908/26,191,152 bytes. Each revision
used less than one second of its 60-second build budget. These measurements are
not relabeled as source4 results; final installed-consumer/private-edit gates
are reported separately by the build workstream.

Final F10 diagnostics observed 64 decoder create/free pairs for four gathered
Zarr3 reads; lifecycle CPU was 1.87%/1.91% of diagnostic operation CPU.
Four partial writes created/freed 16 encoder and 16 decoder contexts; shares
were 2.04%/2.18%. Intercepted create/free CPU does not isolate internal codec work or
allocation and is not an upper bound on retained-context benefit. No retained
codec-state variant was measured. The diagnostic limits below apply.

The ten scenarios deliberately avoid a datatype/layout/backend cross product:

| Scenario | Fixed work |
| --- | --- |
| Direct memory | 4,194,304 FP32 elements (16 MiB per array), eight SAXPY passes, serial and four caller-owned workers |
| Acquisition | 10,000 checked/unchecked/borrowed dense acquisitions, affine/gather selections, focused/whole observation-column acquisition at 128 columns |
| Dense copy | Twelve owned and twelve caller-provided copies per geometry: contiguous, stride two and deterministic 256 × 256 FP32 gathers from a 512 × 512 owner |
| Sparse copy | Deterministic 512 × 512 CSR/CSC, eight entries per major axis, int32 indices/int64 offsets, unsorted duplicate entries and a nonmonotone 256-position gather; eight copies per native/opposite selection |
| Packed text | Eight copies of 16,384 ASCII, UTF-8 and nullable strings |
| Metadata | Eight category-pruning copies of 8,192 selected rows; 10,000 focused/whole record-member acquisitions at 128 fields |
| H5AD | Four dense reads/writes crossing chunk boundaries; four CSR/CSC native/opposite selected transfers |
| Zarr v2 | Four 64 × 64 reads/writes crossing 64 × 64 chunks of a 256 × 256 array |
| Zarr v3 | Four duplicate/nonmonotone gathered reads; four 64 × 64 partial writes crossing four 128 × 128 shards with 64 × 64 inner chunks |
| Caller execution | Four reused independent worker contexts, four reads/writes per worker to disjoint full Zarr v2 chunks; adaptive read dispatch across 128 full chunks of a 2048 × 256 fixture (2 MiB), with four reserved workers and real completion timestamps |

Numerical and text fixtures are deterministic. External H5AD/Zarr stores are
generated independently with Python and identified by per-file hashes and a
manifest recording the installed producer versions. Each benchmark process gets
a fresh disposable byte-identical fixture copy. Initialization, fixture copying,
context preparation, worker creation and correctness checks occur outside the
operation timer. Allocating operations include temporary destruction. Caller-provided outputs and edit handles are prepared outside timing; `copy_into` uses full validation and reuses the output. Adaptive
timing includes request issuance, caller coordination and completion reporting.
Explicit store flushes are outside the timer; write results measure API transfer
completion under normal filesystem caching, without a durability guarantee.
The adaptive case asserts 128 actual completions and at least one adaptation
trial, and records trial/adaptation/rollback counts and the final flight limit
for each uninstrumented sample. Its two 64-completion windows reach the normal
baseline and trial phases without synthetic latency.

There are exactly five uninstrumented baseline/candidate pairs per scenario, ordered baseline,
candidate, baseline, candidate. The baseline always supplies the first timed
sample. One separate diagnostic execution per scenario and revision supplies
allocation, RSS, backend and codec counters; its elapsed time does not enter the
timing distributions. No warmup or extra timing trial is hidden. Page caches remain in their
normal state. The runner serializes its own samples; other host activity remains
uncontrolled. A 30-second scenario/revision cap and a 300-second machine/revision
cap conservatively charge timing and diagnostic subprocess wall time including
process setup and validation. Build, fixture generation/copying and separate correctness runs have
separate budgets. The runner checkpoints each completed sample and refuses to
overwrite an existing sample receipt.

The two newly supplied acquisition capabilities use explicit compile-time
adaptations in the same driver source. Baseline borrowed acquisition constructs
an `Array` and retained descriptor; candidate acquisition calls `borrowed_dense`.
Baseline focused record acquisition obtains `owner.view().member<T>()`;
candidate acquisition calls `owner.member<T>()`. These measure equivalent
requested access through the available public contracts, including their
different acquisition lifetimes and ownership work. Each result labels its
actual path. All other operations use the same public API in both builds.
Native-axis H5AD sparse selections use reusable transfer contexts; opposite-axis
selections use prepared reads because contexts require consecutive native-major
blocks and a complete minor axis. Prepared-read backend statistics are null.

Diagnostic global C++ throwing `new`/`new[]`, including aligned forms, count successful
allocation requests and requested bytes only during each operation. C `malloc`,
system allocation, live bytes and peak heap are outside this scope. Timed runs
load no observers and use the normal global allocation functions. Diagnostic
runs load identical allocation/I/O observers in both revisions. RSS is the whole-process high-water
mark, including initialization and validation, and is not attributed to one
operation. TransferContext statistics report logical backend calls/bytes where
available; missing values are null. The optional preload interposer counts
selected libc I/O and codec calls during the operation only. Direct syscalls and
uninterposed internal calls can be missed; zero intercepted calls do not prove
zero physical I/O. HDF5's own caching can avoid libc calls on repeated reads.
For F10, diagnostics also sum thread CPU time around intercepted Zstd context
creation/free and compression/decompression calls, alongside whole-operation
process CPU time. This is approximate attribution with clock/interposition
overhead and incomplete internal-call visibility. Zstd's C heap allocations
are outside the C++ allocation counter. These CPU observations do not enter the
five uninstrumented timing distributions.

The summary retains all five elapsed samples, quartiles and ranges, paired
median ratios, separate diagnostic allocation/RSS/backend counters, immutable library/driver
source identities, executable hashes, fixture identity, toolchain and machine.
A candidate operation exceeding a 5% paired-median regression is investigated
when its five-sample range is entirely above the baseline range. Overlapping
ranges receive no claim of a resolved timing difference. Raw checkpointed
receipts and build logs stay under ignored `build/v1-cleanup/` and are referenced
by hash.

Reproduce after coordinating an exclusive measurement window:

```sh
.venv/bin/python benchmarks/v1/make_fixtures.py --output build/v1-cleanup/fixtures-final
python3 tools/v1_benchmark.py build --source build/v1-cleanup/baseline-source \
  --build build/v1-cleanup/performance-baseline
python3 tools/v1_benchmark.py build --source build/v1-cleanup/candidate-source-4 \
  --build build/v1-cleanup/performance-candidate-4 --new-core
python3 tools/v1_benchmark.py validate \
  --baseline build/v1-cleanup/performance-baseline/receipt.json \
  --candidate build/v1-cleanup/performance-candidate-4/receipt.json \
  --fixtures build/v1-cleanup/fixtures-final --work build/v1-cleanup/performance-validation-5
python3 tools/v1_benchmark.py run --timing-granted \
  --baseline build/v1-cleanup/performance-baseline/receipt.json \
  --candidate build/v1-cleanup/performance-candidate-4/receipt.json \
  --fixtures build/v1-cleanup/fixtures-final --work build/v1-cleanup/performance-run-5
```

Freeze both source directories before building. The tool refuses measurements
if source, driver, executable or interposer bytes have changed since the build.
Reconfiguration uses CMake's C++ module scanning and two build jobs; no BMI build
order is hand-written. A separate build budget of 60 seconds per revision covers
one fresh core-import consumer compilation and three source-only recompiles
against prebuilt library BMIs. It records command high-water RSS, source/object
hashes, the BMI import closure from the consumer compiler's module map, and all
already-built BMI artifacts (including private partitions); this is not a parsing-only
latency measurement. The build workstream separately checks private-edit
isolation and installed-package behavior.
