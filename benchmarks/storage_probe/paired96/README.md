# Paired 96-row Zarr3 profile

This directory contains one bounded run each for the baseline and frozen candidate A2. Both completed and passed independent exact-value comparison. The source overlays, builds, outputs, logs, and command recipe are preserved here. A separate 64 KiB geometry probe requested afterward is not part of these results.

## Workload

The source is `/tmp/anndata-storage-campaign-20261103/comparison-4096/dense.zarr3`. Its X array is 4096×4096 float32. Zarr3 outer shards are 2048×4096; inner chunks are 256×512; the payload codec is Zstd level 0. X metadata SHA256 is `e6be8a846e58b2e005aa57f85fc8be74f6af6d67af17f77ac72ef5f68bd7b5bc`. The X shard files are `X/c/0/0` (29,896,515 B, SHA256 `37a47938d7bac081e4865e6be784e0006e5826d355362c2224dd14e0b4493e24`) and `X/c/1/0` (29,896,269 B, SHA256 `53b3745672ddfd1b68cd172dcf48b61a62050b406730f7fd016af63cc8b5ea7f`). The campaign manifest records the full store at 59,803,222 B with tree SHA256 `23db0bea7a153c189974c63e9a873722a9ec93470069b8262de24233bdc19822`.

Each run exports rows `[2000,2096)` and all 4096 columns to a fresh Zarr3 directory. The 96×4096 float32 useful payload is 1,572,864 B. The selection crosses the source row-2048 shard boundary and inner-chunk boundary, covering two source shards and sixteen 256×512 inner chunks. Both completed outputs use destination X chunks `[96,2048]` (two chunks). The independent comparator reads only this source slice and the completed destination X, then requires exact element equality.

## Instrumentation scope

- `logical-profile.patch` adds ROI-gated `ZarrBackend::read_numeric` and `write_numeric` counters for X. It reports calls attempted/completed, summed logical elements and float32 logical bytes. `bulk_logical_probe_begin/end` delimit only `write_zarr`; the source open and view construction are outside the ROI. These are backend method counts, not physical I/O or codec counts.
- `libcodec_counts_limited.so` is the existing codec and libc-I/O interposer from the 384-row probe. It caps each direction at 512 MiB and 30 seconds is enforced externally. It wraps successful dynamic libc `read`, `pread`, `readv`, `preadv`, `write`, `pwrite`, `writev`, and `pwritev` calls during the ROI. It is not path-tagged and can miss direct syscalls, hidden libc internals, or memory-mapped I/O. Thus the cap and transfer totals apply to intercepted libc calls, not all kernel or physical-device traffic.
- `alloc_counts.c` counts requested calls and bytes through glibc's public `malloc`, `calloc`, and `realloc` entrypoints via `__libc_*` delegates. Regular `libstdc++` `operator new` imports `malloc` in this environment, so it is included within the malloc line and must not be added as a separate total. `posix_memalign`, `aligned_alloc`, `memalign`, `valloc`, and `pvalloc` are reported separately. Bytes are cumulative requests, not a live peak. Calls made through glibc internal `__libc_*` paths, `mmap`/`brk`, and non-interposed allocators are outside this count. The counter preload changes allocator dispatch and adds overhead.
- The codec wrappers count the observed Blosc/Zstd API entrypoints and accumulated caller-thread CPU. For Zarr3 this includes `ZSTD_decompressStream` and `ZSTD_compress2`; it does not imply every possible codec API is interposed.
- The driver also reports wall time, thread CPU time, process user/system CPU, and a user-only cycles/instructions perf group when permitted. Earlier `strace` access was denied by ptrace policy; perf availability varies. All timing/counter results are instrumented-run observations.

## Build a lane after clearance

The build is isolated: it compiles the profile overlay to a new object, replaces only `backed_zarr.cpp.o` inside a copy of that lane's static archive, and links a new driver. It does not mutate the source tree or input archive. The output directory must not exist.

Baseline inputs:

```bash
TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96
BASE=/tmp/anndata-storage-benchmark-nppztw87
bash "$TASK/build_profile.sh" baseline \
  "$BASE/frozen-source" \
  "$BASE/storage-driver-build" \
  "$BASE/storage-driver-build/libanndata.a" \
  "$TASK/baseline/build-run-v2"
```

The baseline archive SHA256 is `90d24ee98b7d760e3b42517ccd004b2867bca047ad3603c9d54a78f9ad3321f8`. The candidate owner supplied label `A2_budget_rectangles`, source manifest SHA256 `17ca2627c1192ad3f40bf62869a76cfcb863b10775579ebe034732c81a47dee7`, source root `/home/tumlinson/anndata.cppm`, build root `/tmp/anndata-dense-transfer-candidate`, and archive `/tmp/anndata-dense-transfer-candidate/libanndata.a` SHA256 `b910afcf8e0f9fc1eca3f18015943117706e639512db5cabc3ff1fd01c8dc9a6`.

Candidate build command:

```bash
TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96
CAND=/tmp/anndata-dense-transfer-candidate
bash "$TASK/build_profile.sh" A2_budget_rectangles \
  /home/tumlinson/anndata.cppm \
  "$CAND" \
  "$CAND/libanndata.a" \
  "$TASK/candidate/build-run"
```

The patch is deliberately strict (`--fuzz=0`); if candidate source changed the backend method sites, stop and report that the instrumentation patch needs an evidence-preserving adaptation rather than forcing it. The build script writes `candidate-identity.json` and `candidate-source.sha256` into its isolated output directory when present.

The script's direct Clang commands use the lane's CMake module map for the backend object and a generated consumer map pointing at the same lane's existing BMI files, with `-O3 -DNDEBUG`, Clang 18, HDF5/Zarr defines, and original dependency paths. This also works for the candidate build, which has no `storage_benchmark` target map. It builds `alloc_counts.c` as a private `/tmp` preload. No build was run before the parent coordinator's GO.

## Run each lane once

Use distinct destination and record paths for baseline and candidate. Neither path may already exist. The run script invokes the driver exactly once with a 30-second timeout, 512 MiB intercepted read cap, and 512 MiB intercepted write cap. If the run times out or the preloader rejects an I/O request, it saves the partial destination and counters, skips correctness validation, and returns the run status. Do not retry that lane without parent coordination.

Example baseline command after clearance:

```bash
TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96
SOURCE=/tmp/anndata-storage-campaign-20261103/comparison-4096/dense.zarr3
bash "$TASK/run_once.sh" \
  "$TASK/baseline/build-run-v2" \
  "$SOURCE" \
  "$TASK/out-baseline-96.zarr" \
  "$TASK/record-baseline-96"
```

The exact candidate invocation was:

```bash
TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96
SOURCE=/tmp/anndata-storage-campaign-20261103/comparison-4096/dense.zarr3
bash "$TASK/run_once.sh" \
  "$TASK/candidate/build-run" \
  "$SOURCE" \
  "$TASK/out-candidate-96.zarr" \
  "$TASK/record-candidate-96"
```

The runner writes raw stdout/stderr, exit status, source metadata hash and file sizes, instrumentation/binary source hashes, output file sizes, and exact-comparison JSON into each fresh record directory. `destination.sha256` is a canonical manifest by relative destination path; the two complete manifests match byte-for-byte.

## Results

Each lane ran once. The same instrumented driver, alloc interposer, codec/I/O interposer, 30-second timeout, 512 MiB per-direction intercepted-I/O caps, source fixture, selected region, scratch settings, and destination format were used. Neither run hit a cap or timeout. A user-only cycles/instructions perf group ran for the full measured window in each lane.

| Measure | Frozen baseline | A2 candidate | Baseline / A2 |
|---|---:|---:|---:|
| Wall time | 1.889452 s | 0.052191 s | 36.2× |
| Thread CPU | 1.889428 s | 0.046779 s | 40.4× |
| Logical X backend read calls | 96 | 1 | 96× fewer |
| Logical X backend write calls | 96 | 1 | 96× fewer |
| Logical bytes read / written | 1,572,864 / 1,572,864 B | 1,572,864 / 1,572,864 B | same useful payload |
| Intercepted libc read bytes | 426,353,114 B | 7,562,998 B | 56.4× lower |
| Intercepted libc write bytes | 68,000,111 B | 1,416,296 B | 48.0× lower |
| Zstd stream-decode calls | 8,426 | 130 | 64.8× fewer |
| Zstd decode API input / output | 425,301,378 / 552,189,960 B | 7,478,197 / 8,503,304 B | 56.9× / 65.0× lower |
| Zstd encode calls | 194 | 4 | 48.5× fewer |
| Zstd encode API input / output | 151,053,640 / 67,987,702 B | 1,631,560 / 1,403,887 B | 92.6× / 48.4× lower |
| `malloc` calls / cumulative requested bytes | 379,753 / 3,412,598,332 B | 31,803 / 60,247,441 B | 11.9× / 56.6× lower |
| Full output store size | 1,411,567 B | 1,411,567 B | equal |

Both runs report 393,216 float32 elements on each logical X backend direction, equal to the useful 1,572,864 B payload. The baseline reports process user/system CPU of 0.952195/0.937233 s, Zstd decode/encode caller-thread CPU of 0.642817/0.169963 s, and perf cycles/instructions 3,285,867,076/5,564,964,297. A2 reports process user/system CPU of 0.023280/0.023499 s, Zstd decode/encode caller-thread CPU of 0.012168/0.004349 s, and perf cycles/instructions 75,811,624/125,185,648.

Both destination stores have shape `[96,4096]`, X chunks `[96,2048]`, the same metadata and bytes, and exact equality with source rows `[2000,2096)`. Their canonical relative-path SHA256 manifests are identical: `afa488fcd324deab432f8bd5589e89713dd4dac80aadfa966e7631acd34b3762`. Independent value checks are in `record-baseline-96/validation.json` and `record-candidate-96/validation.json`.

These are one-shot instrumented observations, not a repeated-trial distribution. The malloc counters are cumulative requested bytes, not live peak; ordinary libstdc++ `operator new` allocations enter the observed `malloc` line and are not added separately. Intercepted libc I/O is not path-tagged and is not total physical disk traffic. The profile wraps the listed dynamic libc and codec APIs; direct syscalls, glibc-internal allocation paths, mmap I/O, or unwrapped codec entrypoints may not be represented. Both I/O caps are on intercepted libc calls only.

### Binary and source identity

| Lane | Input archive SHA256 | Instrumented archive SHA256 | Driver SHA256 | Source identity |
|---|---|---|---|---|
| Baseline | `90d24ee98b7d760e3b42517ccd004b2867bca047ad3603c9d54a78f9ad3321f8` | `8ed91ca02b7366038bd98565755a80cb972607efb4f477b2960a4697f3eaf888` | `2f823997839497545ac9974ce393b5eef85239c718839d83c53bafb1af202ae6` | frozen baseline source; `src/backed_zarr.cpp` `c78c8183483521909090c1c44fcfbebdf3091b7f4cc45559c52b292c1da002b8` |
| A2 candidate | `b910afcf8e0f9fc1eca3f18015943117706e639512db5cabc3ff1fd01c8dc9a6` | `ccb2343c7226e9ebcfd78b3c3a04b9af530ab40ee3f717e00c233deb1e1f03f8` | `7b81f1b167e151d2ecb9450c8dbb9bfddc6e7ad006cbe7151fc368b1bb772d07` | manifest `17ca2627c1192ad3f40bf62869a76cfcb863b10775579ebe034732c81a47dee7`; candidate identity is copied into the build and run records |

The instrumented source object SHA256 is `6532f314287030f82d7f0e4b08b855a9235b3948902f7cc2e6eda280b41f37da` for baseline and `c1df9ae82e674217544601d1a7ac38bed1889418de3d101b7397b07a0709f3cc` for candidate. Common profile source: driver `054c5cb5ed11063e39a45e6f20411f44c2bf6bb593f0267d684e906371d7575c`; allocation interposer `cec7f23d9743272b6ce6597adc79437ad16d2bc320b494cd1cc4cf377ef3b957`; alloc preload `740aaf7c85d6089b93210e487c268dbf1cf36930f0f352ff2398bf18834132cb`; capped codec/I/O preload `45ff53af1e7d5014278965215832113f965348711a94c03a16ce657c40c8305d`.

The first two profile-object build attempts failed before any storage workload and are retained at `baseline/build-run/` and `baseline/build-run-fixed/` with brief failure notes. The successful isolated baseline build is `baseline/build-run-v2/`; candidate build is `candidate/build-run/`. Neither supplied input archive was modified.

## Fixture regeneration

The original campaign command, provided by the campaign coordinator, was:

```bash
.venv/bin/python tools/storage_benchmark.py \
  --output benchmarks/results/storage.json \
  --dependency-prefix /home/tumlinson/anndata.cppm/build/dependencies/usr \
  --generate-fixtures /tmp/anndata-storage-campaign-20261103 \
  --fixture-size 256 --comparison-fixture-size 4096 --include-huge-fixtures \
  --readonly-sparse-fixture h5ad:/home/tumlinson/data/Cellerator/data/test/reference/GSE147520_all_cells_raw_counts.h5ad \
  --build-storage-executable --keep-work
```

The exact source snapshot recorded for that campaign has manifest SHA256 `0f33852c9b9963e24932b955a04f2692b7df2de7e9a5c98baf2f139dce910785`; the full report is preserved at `benchmarks/results/storage-baseline.json` (SHA256 `e402278c5267684586f173ebbe3614b0382b86e8bbbd6befa6b2634ef47491ba`). To regenerate only the 4096 comparison fixtures and avoid the unrelated huge-fixture work, use the exact generator function from that frozen snapshot:

```bash
cd /home/tumlinson/anndata.cppm
.venv/bin/python \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96/recreate_fixture.py \
  /tmp/anndata-storage-benchmark-nppztw87/frozen-source/tools/storage_benchmark.py \
  /tmp/storage-campaign-recreated
```

This creates `/tmp/storage-campaign-recreated/comparison-4096` and refuses to overwrite it. The function uses `np.random.default_rng(20261103)`, writes the dense/CSR/CSC fixtures for the configured H5AD, Zarr2, and Zarr3 formats, and receives `size=4096`, `include_huge=False`, and the original default `chunk=1024`. Compare the generated Zarr3 X metadata and values to the recorded campaign evidence before treating regenerated compressed shard bytes as identical; physical Zstd output can depend on Python package and codec versions. The fixture used for the paired profile remains the existing read-only campaign fixture.

## Preparation artifacts

- `bulk_export_paired96.cpp`: native bounded export driver.
- `logical-profile.patch`: ROI-gated backend-call instrumentation applied to a private source copy.
- `alloc_counts.c`: glibc allocation request interposer; no live peak claim.
- `build_profile.sh`: isolated lane build/archive replacement.
- `run_once.sh`: one capped run plus value comparison and artifact capture.
- `compare_zarr3.py`: independent Python/Zarr/NumPy exact comparison with `PollPolicy` installed before importing Zarr.
- `recreate_fixture.py`: deterministic fixture-generation entrypoint using the frozen benchmark's generator.

These files were prepared under `/tmp`; no candidate build or paired workload was started.
