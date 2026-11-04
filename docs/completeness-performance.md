# Storage completeness: bounded measurements

These measurements compare the current completeness changes with commit
`2565c3c79354b67be5aba43a84af42f8011b68d9`, built in Release without the historical
link shim. They are single observations, not statistical performance claims. The
machine used Linux x86-64, Clang 18.1.3, CMake 3.28.3 and Ninja 1.11.1. The
filesystem cache was not cleared. Test-only interception and Python validation
are not part of the library.

The immutable source archive has SHA-256
`e66c717e61a58daa942927ff8e9c2cbadd1bd0dbe94b8b41bbedd81c0c908f2e`.
The generated fixture manifest has SHA-256
`afa7ecf8b829c95d88c75967bdbc345bdd3039403fbde1d9b6a872d5fe14b26c`;
the preserved source fixture manifest has SHA-256
`a0db8a376c375923873b74b61c30d4af0eb16b12cc830f2c4aa21027061d56d4`.
Raw outputs, disposable fixture copies and complete hashes are stored locally in
ignored `build/completeness-measurements` and `build/completeness-fixtures`.

Each write case uses one public call and independently checks its full result
with Python. The ordinary fixtures have 256 × 512 shards and 4, 16, or 64 Zstd
compressed inner chunks. The partial rectangle crosses four shards. The gather
cases include nonmonotonic repeated destinations and respectively a negatively
strided external source and a gathered source view. A separate huge logical
array touches four far-apart absent chunks without allocating the array. The
fixtures include category metadata, a sparse layer and `.raw`, which the Python
validator also checks. The large-gap check reads only selected points and fill
values, avoiding materialization.

The probe intercepts libc I/O and allocation calls, shard renames, and codec API
calls during the numeric operation. Its 30-second timeout and 512 MiB limits
apply per operation and direction; fixture creation and Python validation are
outside those limits. Counts do not cover direct system calls, memory maps or
all operations internal to third-party libraries. Allocation figures are
cumulative requested bytes, not peak live memory. Scratch statistics are retained
capacity after a call, not a peak. `ru_maxrss` is inherited from the launching
Python process and is not reported as library workspace usage.

For these particular fixtures the index uses bytes plus CRC, and the only
numeric chunk codec is Zstd. Decoded and encoded chunk counts are calculated by
dividing observed Zstd output and input bytes by the known full inner-chunk
size. A single chunk can require multiple streaming API calls. The analysis
reads the immutable source shard indexes to determine changed, fully overwritten,
partial and untouched payloads; it also checks each destination shard's observed
index-read bytes, payload reads, and successful replacements. Index-end write
counts are position/size candidates, rather than universally classified writes.

## Sharded writes

| Case | Useful bytes | Time, ms (baseline → candidate) | Shard replacements | Chunks decoded | Chunks encoded | Intercepted read / write bytes, baseline → candidate |
|---|---:|---:|---:|---:|---:|---:|
| Full, 4 inner chunks | 524,288 | 7.46 → 4.87 | 4 → 1 | 4 → 0 | 4 → 4 | 1,038,891 / 916,367 → 68 / 183,805 |
| Full, 16 inner chunks | 524,288 | 14.05 → 3.45 | 16 → 1 | 16 → 0 | 16 → 16 | 5,387,666 / 5,388,078 → 260 / 339,431 |
| Full, 64 inner chunks | 524,288 | 76.04 → 4.22 | 64 → 1 | 64 → 0 | 64 → 64 | 21,785,114 / 21,694,239 → 1,028 / 327,410 |
| Partial, 16-chunk shards | 320,000 | 17.86 → 8.09 | 16 → 4 | 16 → 12 | 16 → 16 | 5,406,186 / 5,440,019 → 1,254,607 / 1,375,725 |
| Gather, strided source | 512 | 136.71 → 9.37 | 128 → 4 | 128 → 15 | 128 → 15 | 42,849,244 / 42,816,265 → 1,337,732 / 1,338,033 |
| Gather, source view | 512 | 147.08 → 10.73 | 128 → 4 | 128 → 15 | 128 → 15 | 43,047,163 / 42,816,361 → 1,388,820 / 1,338,032 |
| Four distant missing chunks | 16 | 2.25 → 1.56 | 4 → 4 | 0 → 0 | 4 → 4 | 0 / 1,136 → 0 / 1,136 |

The 64-chunk full overwrite has 64 baseline replacements and reads 64 old
chunks; the candidate performs one replacement and no old-chunk decode. The
partial case needs 12 existing chunk decodes, and both gathers need 15. The
analyzer confirms that each touched shard is replaced once, each index is read
at most once, unchanged encoded payloads are not repeatedly read, and each
changed chunk is encoded once. All seven outputs passed independent Python
checks, including the duplicate-index last-write behavior.

Batching retains more staging memory. The numeric transfer context reported
about 1.08–1.45 MB retained scratch for the candidate cases, compared with
8–4,096 bytes for corresponding baseline contexts; the gathered-view operation
does not expose those context statistics. For example, intercepted cumulative
`malloc` requests for the 64-chunk case were 91,429,240 → 10,992,893 bytes
(24,879 → 694 calls), and for the strided gather 245,219,618 → 10,835,457 bytes
(46,981 → 520 calls). These are allocator requests across the armed operation,
not live peaks or all allocations. A separate native test exercises the configured
scratch limit; the library does not claim that third-party allocations are included
in that limit.

## Caller-managed workers

The caller-owned harness reads 512 distinct 64 × 128 blocks, 16 MiB total,
using eight persistent workers and one transfer context per worker. It injects
0.5 ms latency per X-shard read, changing to 2 ms partway through the dataset;
that delay exists only in the benchmark interposer. Every resulting value and
block count is checked. Fixed modes limit outstanding requests; adaptive mode
uses measured completions to recommend a limit. No scheduler is supplied by the
library.

| Mode | Baseline seconds | Candidate seconds | Maximum concurrent intercepted reads, baseline → candidate |
|---|---:|---:|---:|
| Fixed 1 | 1.757 | 1.764 | 1 → 1 |
| Fixed 2 | 0.886 | 0.895 | 2 → 2 |
| Fixed 4 | 0.444 | 0.441 | 4 → 4 |
| Fixed 8 | 0.219 | 0.217 | 8 → 8 |
| Adaptive | 0.784 | 0.283 | 4 → 8 |

In this one run, the candidate moved 1 → 2 → 4 → 8 outstanding requests early,
then tested 4 and returned to 8 after the injected latency change. The baseline
spent most of the run at 2 and eventually settled at 4. The fixed controls,
probe mutex, warm cache and artificial latency limit what these timings say
about actual storage or other workloads. Manual fixed limits remain available.

## Import cost

Already-built Release BMIs were imported once per version with Clang 18 and
`-std=c++20 -O3 -DNDEBUG`, without clearing the OS cache. `/usr/bin/time`
reported 0.03 s for `anndata.core`, 0.02 s for `anndata.streaming`, and 0.03 s
for `anndata` in both versions, at its 0.01-second resolution. Peak compiler RSS
was approximately 96–99 MB. The measured public BMI files totaled 44,706,168
bytes at baseline and 44,705,396 bytes for the candidate. These are warm,
single-sample consumer imports, not cold full-build or general portability
measurements. Implementation-only rebuild isolation is checked separately.

## Reproduction and preserved evidence

`benchmarks/streaming_probe/make_fixture.py` generates the streaming seed;
`benchmarks/completeness_probe/make_fixtures.py` derives the fixture suite from
that seed and records hashes. The original generated stores are not redistributed;
matching package versions are needed to reproduce their exact bytes. Build the baseline and candidate with identical Clang Release
settings and use `benchmarks/completeness_probe/CMakeLists.txt` for a source
snapshot. Build `benchmarks/completeness_probe/counters.c` as a shared preload
library with the installed HDF5, Blosc, Zstd and zlib development libraries.
`tools/completeness_probe.py --help` describes the runner and its required
paths; it refuses to overwrite output, checkpoints each case before independent
validation, and stops on failure. `tools/completeness_analyze.py` compares
saved observations against the fixture indexes. `tools/completeness_compile_probe.py`
measures import-only compilations against the built module artifacts. The runner
uses a test-only Zarr wakeup when `ANNDATA_INTEROP_ZARR_WAKEUP=1` is required by
the execution environment; it does not change library behavior.

The source, executable, fixture, raw-report and compact-result hashes are in
[completeness-summary.json](../benchmarks/results/completeness-summary.json).
These observations demonstrate the tested mechanisms and their tradeoffs; they
do not qualify every codec, storage backend, device, or concurrency level.
