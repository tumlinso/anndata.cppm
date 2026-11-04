# Grouped sharded Zarr reads

The [Python/C++ screen](head-to-head-performance.md) exposed repeated work on
orthogonal Zarr v3 gathers. For eight reads of a 16 × 64 float32 selection,
the previous native path made one backend read per selected value. The grouped
path plans each axis separately, reads each touched inner chunk once per public
call, and scatters its selected values into the caller's destination. The
[compact record](../benchmarks/results/gather-read-summary.json) links the exact
baseline, candidate source, binary, fixture and raw-report hashes.

| Fixture | Earlier C++ ms | Candidate C++ ms | Earlier Python ms | C++ logical backend reads, earlier → candidate | Intercepted C++ read bytes, earlier → candidate |
|---|---:|---:|---:|---:|---:|
| 1024² gather, eight repetitions | 735.44 | 63.31 | 285.47 | 8,192 → 640 | 29.45 → 2.29 MB |
| 4096² gather, eight repetitions | 719.21 | 163.33 | 790.57 | 8,192 → 1,520 | 29.65 → 5.46 MB |
| 4096² aligned Zarr scan control | 223.81 | 215.45 | 949.99 | 32 → 32 | 7.33 → 7.33 MB |

For the smaller gather, the identical Zstd-interposing diagnostic counted 8,192
→ 640 stream decode calls, corresponding to this fixture's individual inner
chunks. The 4096² diagnostic used the libc-only counter, so codec calls are not
reported there. Counts cover intercepted APIs, not every possible syscall or
physical disk byte. Candidate and baseline returned the same independently
computed full-output hashes. The Python observations were retained from the
previous screen and were not rerun. Other Python/C++ screen rows were also not
remeasured after this private implementation change.

The earlier and candidate C++ times are one warm-cache observation each on the
same fixtures and selected CPU, with the same portable Clang 18 Release settings.
They demonstrate the observed effect but do not establish a timing distribution.
The host was shared: load averages were 3.50 and 3.38 around the short candidate
campaign, and CPU pressure and selected-CPU tick snapshots are in the raw report.
The aligned scan control has the same read and intercepted-I/O counts; its two
single timings do not establish a speedup or absence of all regressions.

For the gather, retained transfer scratch rose from 256 bytes to 32 KiB for the
reusable numerical tile; process peak stayed about 11.5 MiB in these single
observations. Retained scratch statistics exclude temporary per-axis maps and
are not process peak or cumulative allocator requests. The maps and tile are
admitted against the existing scratch budget. If they cannot fit, or if caller
strides may alias distinct output positions, the previous logical-order path is
used. Repeated source positions, reverse and nonmonotonic selections, signed
strides, chunk edges and missing fill chunks retain their semantics. Grouping
allocates O(sum of selected axis lengths) points for gathered axes, with no
Cartesian coordinate map or scan across unselected physical gaps. Each call
operates independently; there is no cross-call chunk cache or library scheduler.
The current optimization covers sharded numerical Zarr reads; other layouts keep
their existing paths.

`TransferStats::read_bytes` counts backend rectangle output bytes, so it changes
from the original 32 KiB of point requests to 20.97 / 49.81 MiB of expanded
inner-tile requests for the two gathers. It is not the logical result size, nor
compressed I/O or the earlier decoder's total output. The 1024² diagnostic shows
that actual intercepted compressed reads fell while the number of decoded chunks
fell from 8,192 to 640. Caller-acquired numerical loops have no added checks,
dispatch or synchronization.

Final-source Debug and ASan/UBSan each passed all 20 native groups, including
installed consumers and the independent gathered-transfer fixture (duplicates,
partial blocks, reverse and negative strides, overlapping outputs, low-budget
fallback, rank three, four dtypes, missing chunks and huge gaps). Three upstream
Python storage/completeness suites also passed, including threaded operations.
The [validation receipt](../tests/results/gather-read-validation.json) records
source and log identities. A private `backed_array.cpp` edit preserved all ten public BMIs and
both measured consumer objects by hash and timestamp; restoring the source
reproduced the original object, library and executable hashes. LeakSanitizer was
disabled for local runner restrictions. Existing unsupported encodings and other
format limits are unchanged.

The work was isolated from concurrent fp16 development in another checkout.
Its only production change is the private backed-read implementation; this
qualification does not cover that other agent's uncommitted changes.

The candidate used the original immutable fixtures and Python driver, the same
caller CPU and I/O-cap tooling as its corresponding baseline, and the same
source version of the C++ benchmark driver. Each diagnostic was limited to
30 seconds and 512 MiB per intercepted I/O direction, with the measurement
checkpoint written before hashing. No timeouts or cap rejections occurred. Raw
candidate outputs are under ignored `build/gather-final`; previous baseline
artifacts remain separately preserved. The generator source hash was captured
after generation and restoration of its original call; fixture content hashes
were checked before the candidate run.

To reproduce with an already built candidate library, install it to a fresh
prefix and build the [head-to-head driver](../benchmarks/head_to_head/CMakeLists.txt)
with the settings in the [baseline recipe](head-to-head-performance.md). Then,
with the preserved fixture directory and baseline campaign available:

```sh
python3 benchmarks/head_to_head/compare_gather.py \
  --root /path/to/candidate-artifacts --baseline /path/to/baseline/campaign.json \
  --python /path/to/reference/python --driver benchmarks/head_to_head/python_driver.py \
  --binary /path/to/candidate/head_to_head --library /path/to/candidate/libanndata.a \
  --counter-small /path/to/original-codec-counter.so \
  --counter-large /path/to/io-only-counter.so
python3 benchmarks/head_to_head/summarize_gather.py \
  --raw /path/to/candidate-artifacts/gather-candidate.json \
  --output benchmarks/results/gather-read-summary.json
```

`candidate-artifacts/fixtures` must point to the immutable baseline fixtures.
The tool verifies fixture and driver hashes, adopts the recorded CPU if allowed,
and refuses to overwrite a prior candidate campaign. The baseline recipe builds
the I/O-only counter; the small codec counter is built from
`benchmarks/completeness_probe/counters.c` with the same compiler and flags.
