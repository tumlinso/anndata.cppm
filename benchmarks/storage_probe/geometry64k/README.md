# A2 64 KiB geometry probe

## Result

This is one candidate-only run of the requested 512×512 float32 Zarr3 case with 64 KiB source and writer scratch budgets. It did **not** reach X transfer and is not evidence for the small-geometry A2 hypothesis. The output is an intentionally preserved partial destination; it is not a valid completed AnnData export. No retry was made.

The source fixture is `/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/source-512.zarr`. X is shape `[512,512]`, float32, with Zarr3 inner chunks `[32,64]` and shards `[256,512]`; source obs/var `_index` chunks are `[64]`. The fixture has 1,048,576 useful X bytes. Generation used seed 20261103, AnnData 0.13.4 and Zarr 3.4.2; `PollPolicy` was installed before importing AnnData/Zarr. `fixture-evidence.json` has the full X/index metadata, X shard byte sizes and hashes, and metadata hashes. X metadata SHA256 is `03811ff9d76656d9ad549544416c78164f42568e2113d033861a93572565208a`; the fixture evidence manifest digest is `94e8a285fdaa449daf06824a6652f4424951c06c31a822df97386c3b18475723`.

The single run record is `record-a2/`. The executable reported `status=partial`, exit 2, elapsed 0.004732456 s, thread CPU 4,731,218 ns. Its stderr says `write_zarr failed: string chunk exceeds the codec working-memory limit`. Intercepted I/O totals were 155,229 read bytes and 947 write bytes, with zero read or write budget rejections against 536,870,912 bytes per direction. No cap or timeout was hit.

The profile reported logical X backend reads/writes `0/0`, with `0/0` float32 elements. Thus no X codec, copy, or output counts were obtained. The libc interposer observed 84 `read` calls and 3 `write` calls. The codec interposer observed 150 Zstd decode API calls, 708 input bytes and 1 output byte; no encode calls. The allocation interposer observed 17,970 malloc calls and 2,622,200 cumulative requested bytes; no calloc/realloc or aligned allocation calls. These are ROI counters over the failed `write_zarr` attempt only. Cumulative requested bytes are not live peak memory. The per-thread perf group ran for 4,750,881 ns and reported 12,219,109 cycles and 19,132,175 instructions for the failed attempt; it says nothing about X transfer.

The partial destination `/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/output-a2.zarr` is 947 bytes and contains root metadata plus `obs` and `obs/_index` metadata, but no string chunk or X array. Exact-value validation was correctly skipped because the native command exited nonzero. Therefore there is no output geometry, byte comparison, or exact correctness result to report.

## Cause established from the preserved source

The Python fixture's index chunks are 64 labels as requested, but that does not constrain the C++ export's destination chunks. `src/zarr_engine.cpp` sets the output chunk byte target to `min(1 MiB, max(1024, scratch_bytes/8))`; with 64 KiB this is 8 KiB. String arrays use width 16 for chunk sizing, so the 512-entry obs index is assigned a 512-label output chunk. `src/zarr_codecs.cpp` gives codecs `scratch_bytes/4`, or 16 KiB, and `encode_payload` stores a `std::string` object per item before text. At 512 items, the 512 string objects consume exactly 16 KiB, leaving no room for any non-empty label, so encoding raises the observed error. This conclusion is based on source rules and the partial destination state; the failed run did not emit a chunk-level path trace.

Relevant source locations: `src/backed_copy.cpp` `copy_strings` (source-side byte-bounded batches); `src/zarr_engine.cpp` `Store::create_array` (destination chunk sizing); `src/zarr_codecs.cpp` `encode_payload` (codec memory check). The input fixture's 64-label chunks cannot bypass the destination-side 512-label chunk choice.

## Reproduction and identity

From `/home/tumlinson/anndata.cppm`, the fixture command was:

```bash
timeout 30s .venv/bin/python \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/generate_fixture.py \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/source-512.zarr
```

The driver-only build command was:

```bash
timeout 30s /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/build_geometry64k_v2.sh
```

The one measured run command was:

```bash
timeout 40s /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/run_geometry64k_once.sh \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/build-run-v2 \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/source-512.zarr \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/output-a2.zarr \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/record-a2
```

The outer timeout protects the runner; the runner itself enforces 30 seconds and loads the exact capped I/O/codec preload and allocation preload. Per-direction budget is 512 MiB for intercepted dynamic libc `read`, `pread`, `readv`, `preadv`, `write`, `pwrite`, `writev`, and `pwritev`; this is not a claim about direct syscalls or mmap traffic. There were no rejected requests. No production files or libraries were modified.

The candidate identity is `A2_budget_rectangles`, source manifest SHA256 `17ca2627c1192ad3f40bf62869a76cfcb863b10775579ebe034732c81a47dee7`, original candidate archive SHA256 `b910afcf8e0f9fc1eca3f18015943117706e639512db5cabc3ff1fd01c8dc9a6`. The measurement linked a driver against the existing profiled candidate archive SHA256 `ccb2343c7226e9ebcfd78b3c3a04b9af530ab40ee3f717e00c233deb1e1f03f8`; the profiled archive is a private copy with ROI-only logical backend counters. Candidate driver SHA256 is `ea0ccdac1ca1497824ab890370ab93a53398fe008b16036e820138dd246247d9`. The reused exact allocation preload SHA256 is `740aaf7c85d6089b93210e487c268dbf1cf36930f0f352ff2398bf18834132cb`; the reused exact capped codec/I/O preload SHA256 is `45ff53af1e7d5014278965215832113f965348711a94c03a16ce657c40c8305d`.

`build-run/` and `build_geometry64k.sh` are retained but superseded: that no-workload build compiled a second codec interposer with a different linked DSO hash. The measured run used only `build-run-v2/`, which reused the exact already-profiled interposer and is identified in the run record.

## Preserved artifacts

- `generate_fixture.py`: deterministic fixture generator and Python-side geometry/value assertions.
- `fixture-evidence.json`: actual source X and index metadata, X compressed shard sizes/hashes and manifest digest.
- `bulk_export_geometry64k.cpp`: one-shot native driver, 64 KiB source/writer scratch, ROI counters, time cap data.
- `build_geometry64k_v2.sh`: driver-only build against the frozen profiled candidate archive; no library rebuild.
- `compare_geometry64k.py`: exact source/output X comparator, run only after a successful native export.
- `run_geometry64k_once.sh`: one-run 30s, 512 MiB-per-direction capped runner; skipped comparator on the observed failure.
- `record-a2/`: raw stdout/stderr, status, source/build identity, actual counters, partial destination inventory and validation-skipped marker.
- `build-run-v2/`: compiled driver, object, source copies, archive/preload identities, and build context.
