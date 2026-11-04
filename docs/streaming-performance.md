# Backed block transfer measurements

These are single instrumented observations, not a general speed guarantee.
The machine was an Intel Xeon E5-2698 v4 running Linux, Clang 18.1.3, `-O3`,
HDF5 1.10.10, Zarr 3.4.2 and AnnData 0.13.4. The OS page cache was not cleared;
fixture creation, copying and validation also warm it. Host contention was not
measured. See the [compact machine-readable report](../benchmarks/results/streaming-summary.json)
for source, library, binary, fixture and raw-report hashes.

The baseline is source commit `d9869da416ca36571421241fb10821d4c0acd19a`.
Its pristine Clang 18 Release `storage_benchmark` fails to link at both `-O3`
and `-O2`: a `std::shared_ptr<const std::vector<size_t>>` move-assignment
specialization used by `backed_writer.cpp` is unresolved. The probes link the
same separate explicit-instantiation shim into *both* executables, with no
baseline source edit. The candidate's normal Release benchmark links without
that shim. Both measured binaries use `-O3 -DNDEBUG`.

Final shared-package validation required adding private `<map>` and `<variant>`
includes to the geometry implementation so Clang emits its required standard
library templates. These measurements precede that include-only correction;
no transfer or tuning algorithm changed afterward.

The 4096 × 1024 fixture has float32 dense values and int32 CSR/CSC sparse
values. Zarr 2 uses 128 × 128 Blosc chunks; Zarr 3 uses 64 × 128 Zstd inner
chunks inside 256 × 512 shards. Sparse H5AD is gzip compressed with 32 entries
per CSR row. Dense values are checked against the generating formula; sparse
outputs match independently generated SciPy block hashes. Every Zarr parallel
write is independently reopened and its complete X compared with the source
plus 0.5. All paired cases returned identical output hashes.

| Workload | Useful payload | Existing select/prepare | Reusable context | Distinguishing counts |
|---|---:|---:|---:|---|
| Zarr 2 aligned dense reads | 16 MiB | 0.143 s | 0.0508 s | 256 calls; 392,960 → 74,308 intercepted malloc calls; 256 Blosc decodes each |
| Zarr 3 boundary dense reads | 16 MiB | 0.481 s | 0.198 s | 704 calls; 1,884,800 → 313,370 malloc calls; 1,024 Zstd decodes each |
| H5AD CSR row blocks | 1,065,216 B | 2.011 s | 0.00825 s | 20,480 → 320 H5Dread calls; 67.80 → 0.223 MB intercepted reads |
| H5AD CSC column blocks | 1,052,736 B | 0.425 s | 0.00604 s | 5,120 → 80 H5Dread calls; 15.90 → 0.214 MB intercepted reads |
| Zarr 3, four-worker shard reads | 16 MiB | 0.0423 s | 0.0365 s | 32 whole-shard requests; 512 Zstd decodes each |
| Zarr 3, four-worker shard writes | 16 MiB | 0.843 s | 0.823 s | 32 whole-shard requests; 512 Zstd encodes each; about 169 MB read and 168 MB written each |

Dense times include full value verification and hashing; separately recorded
transfer times exclude them. Parallel times exclude per-worker opening and
preparation. The write observation excludes the initial serial metadata
deconsolidation, store copy, final flush and independent readback. It reports
buffered completion, not durability. The persistent sharded write amplification
is a remaining opportunity; this measurement does not establish a write speedup.

An adaptive single-worker Zarr 3 run independently validated all values over
1,036 requests, made two tuning trials and accepted one adjustment. It took
0.199 s with 1,036 Zstd decodes and 21.93 MB intercepted reads. Its block
partition differs from the manually chosen 704-block boundary run, so the two
times are not a matched comparison. The adaptive result does not prove it is
faster than manual settings or that its choices transfer to other workloads.

H5Dread is counted at the HDF5 API boundary; its logical bytes are not physical
disk traffic. Intercepted I/O covers wrapped libc APIs, not direct syscalls,
`mmap` or hidden library calls. The malloc counts include ordinary `new` on this
toolchain, but omit allocations bypassing intercepted glibc entry points. The
reported requested bytes are cumulative, not peak live memory. Codec counters
cover only wrapped entry points; a Zstd streaming call is not universally one
complete chunk decode. Each measured operation had a 30-second timeout and
512 MiB intercepted read and write caps, independently. No cap was rejected.
Copies, hashes, fixture construction and validation outside the armed region
are excluded from the cap.

One import-only compilation per module used already-built BMIs. Baseline
core/backed/facade were 0.04/0.03/0.04 s and candidate were
0.03/0.03/0.03 s; candidate streaming import was 0.03 s. Peak RSS was 94–97 MiB.
These coarse, warm single observations cannot establish a compile-time change.
The nine baseline public BMIs totaled 37,741,660 bytes; the ten candidate
public BMIs totaled 44,705,032 bytes, including a 6,745,236-byte streaming BMI.
See the report for individual module sizes. Private-edit behavior is validated
separately by the project checks.

## Reproduction

The [probe CMake project](../benchmarks/streaming_probe/CMakeLists.txt) takes an
`ANNDATA_SOURCE` path, allowing the same driver to build against the frozen
baseline and candidate. Use `-DANNDATA_STREAM_CANDIDATE=OFF` for the baseline
and `ON` for the candidate; set the same Clang, Release flags and private
dependency prefix for both. The [fixture generator](../benchmarks/streaming_probe/make_fixture.py)
refuses to overwrite its output. Its optional
`ANNDATA_INTEROP_ZARR_WAKEUP=1` mode uses `tests/python_storage.py` for the
restricted runner; ordinary hosts need no workaround.

Build [the interposer](../benchmarks/streaming_probe/counters.c) as a shared C
library with the HDF5 include path and Blosc, Zstd, zlib, pthread and `dl`.
Then run [the checkpointing driver](../tools/streaming_probe.py) with each
probe executable, the same interposer and fixture root, and separate disposable
write roots. Pass `--baseline` for the candidate. Select `dense_sequential`,
`dense_boundary`, `sparse_csr`, `sparse_csc`, `parallel_read`, and
`parallel_write` in both runs; add `adaptive_boundary` for the candidate. The
runner checkpoints before write validation and after every case, stops after a
timeout or failed case, and does not repeat one automatically. Generated
fixtures, binaries and raw records are retained in ignored
`build/streaming-measurements/` in the measurement worktree.

The first baseline write run completed but its system-Python post-check failed
because NumPy was unavailable, before that older runner checkpointed the result.
Its disposable store remains in the raw artifacts. Only that missing case was
rerun with the pinned development environment; the runner now checkpoints the
captured result before validation.

The fixture was generated before the generator's optional restricted-runner
wakeup bootstrap was added; the calling wrapper installed the same bootstrap.
The original generator bytes were not independently hashed before that change.
The generated store manifest and every fixture file hash are retained, and the
summary identifies the current reproduction source. The runner likewise gained
explicit budget-failure exit handling after measurements; the raw reports
retain each observation and its exact command.
