# A2 candidate Zarr2 export check (384 × 4096)

## Workload and validation

This is the single authorized candidate run matching the preserved uncapped baseline export. The source fixture is `/tmp/anndata-storage-campaign-20261103/comparison-4096/dense.zarr2`, X shape `[4096,4096]`, float32, chunks `[512,512]`, Blosc/LZ4. X `.zarray` SHA256 is `afffd55495c363ff9cfe58f2704b762982bf1e31897bda6a9fbab3a26dfdaeb9`. The selected rows `[1792,2176)` cross the source row-2048 chunk boundary and cover all 4096 columns, for 6,291,456 useful bytes.

The candidate wrote Zarr2 X shape `[384,4096]`, chunks `[384,512]`, Zstd level 3. The independent Python/Zarr/NumPy check reports exact float32 equality with the source slice. The candidate output has 35 files and 5,612,579 bytes. It is byte-identical to the preserved baseline output across all 35 relative paths; the canonical candidate tree manifest SHA256 is `637dd6a9d92ab5531f4a4f80f727bfbb32447d2dd396475097607ba9b573cffb`.

Candidate validation: `record-candidate-z2-384-v2/validation.json`. Preserved baseline validation: `baseline-reference-uncapped/z2-validation.json`. Both value comparisons passed.

## Results

| Counter | Preserved baseline | A2 candidate |
|---|---:|---:|
| Wall time | 15.649515 s | 0.071015 s |
| Thread CPU | 15.647190 s | 0.071013 s |
| Intercepted libc read bytes | 3,921,808,341 | 14,890,196 |
| Intercepted libc write bytes | 1,079,335,547 | 5,615,555 |
| Blosc decode calls / input / output | 3,074 / 2,847,790,613 B / 3,221,340,168 B | 18 / 14,865,532 B / 16,891,912 B |
| Zstd stream decode calls / input / output | 36,768 / 1,073,719,992 B / 2,409,627,648 B | 0 / 0 / 0 |
| Zstd encode calls / input / output | 3,074 / 2,415,981,832 B / 1,079,327,696 B | 10 / 6,354,184 B / 5,607,704 B |
| `operator new` calls / cumulative requested bytes | 978,618 / 22,493,406,246 B | 32,430 / 77,258,209 B |

The candidate run used a 30-second timeout and 512 MiB caps in each direction for intercepted dynamic libc I/O. It completed without cap rejection; counters report 14,890,196 read bytes and 5,615,555 write bytes. The cap wrappers cover `read`, `pread`, vectored equivalents, `write`, and `pwrite` dynamic libc entrypoints; direct syscalls, hidden libc internals, and mmap traffic may be outside their scope. The baseline was measured before this capped wrapper and was uncapped. The candidate cap adds wrapper overhead, so these are not strictly identical-instrumentation timings. Each result is a single observation, not a repeated-trial distribution.

There are no logical `ZarrBackend::read_numeric` / `write_numeric` counters in this Z2 pair. The allocation count is the driver's global `operator new` counter, not the separate glibc malloc profile used in paired96. Hardware perf events were unavailable for usable counting: both perf groups report `time_running=0`.

## Binary identity

- Candidate: label `A2_budget_rectangles`; source manifest SHA256 `17ca2627c1192ad3f40bf62869a76cfcb863b10775579ebe034732c81a47dee7`; candidate archive `/tmp/anndata-dense-transfer-candidate/libanndata.a`, SHA256 `b910afcf8e0f9fc1eca3f18015943117706e639512db5cabc3ff1fd01c8dc9a6`; candidate identity JSON SHA256 `38e87f989dc3bef28bacd12a3a53d05f6c84bafa661c28f417dc29e2c8bb2113`.
- Candidate native driver source SHA256 `28f855218fb08f78d1a0b5d74adad2854371299d77271ef4303d12510ddc6612`; candidate driver binary SHA256 `c97fbbcf0a00930d911a49d17bb620a46486b9fbaec3b62df86eaf718a070262`.
- Candidate capped codec/I/O preload SHA256 `45ff53af1e7d5014278965215832113f965348711a94c03a16ce657c40c8305d`.
- Preserved baseline driver source SHA256 `28f855218fb08f78d1a0b5d74adad2854371299d77271ef4303d12510ddc6612`; binary SHA256 `e15d1a1915c4273396c020b2dca3d63b8eb1ae73468d96b10e64f32e21a906cd`; original uncapped codec/I/O preload SHA256 `60f3e44d9eaa88ecb4ef7ce2bfe00d233ab0268d3002f2c20ce5cadb85aba411`. Baseline build used the frozen baseline storage archive with SHA256 `90d24ee98b7d760e3b42517ccd004b2867bca047ad3603c9d54a78f9ad3321f8`.

Candidate build details and module-map hashes are in `candidate/build/build-context.txt` and `build-hashes.sha256`. The supplied candidate archive was not modified; only the driver was compiled in an isolated `/tmp` directory. No production files were edited.

## Commands and fixture recipe

Candidate driver-only build:

```bash
bash /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired384-z2/build_candidate_driver.sh \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired384-z2/candidate/build
```

The one measured candidate run, with its original destination and record paths, was:

```bash
bash /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired384-z2/run_candidate_once.sh \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired384-z2/candidate/build \
  /tmp/anndata-storage-campaign-20261103/comparison-4096/dense.zarr2 \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired384-z2/candidate-output.zarr \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired384-z2/record-candidate-z2-384-v2
```

The runner refuses existing destination and record paths. Do not rerun it against the preserved output path; this experiment was authorized for one candidate observation.

The fixture is from the campaign's frozen `tools/storage_benchmark.py` snapshot (source manifest SHA256 `0f33852c9b9963e24932b955a04f2692b7df2de7e9a5c98baf2f139dce910785`). The original full campaign command and its generated fixture record are in `/tmp/anndata-storage-campaign-20261103` and `benchmarks/results/storage-baseline.json`. To regenerate only comparison fixtures, use the sibling paired-profile recipe, which calls the frozen generator with `size=4096`, `include_huge=False`, and original `chunk=1024`:

```bash
cd /home/tumlinson/anndata.cppm
.venv/bin/python \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96/recreate_fixture.py \
  /tmp/anndata-storage-benchmark-nppztw87/frozen-source/tools/storage_benchmark.py \
  /tmp/storage-campaign-recreated
```

This recreates the dense Zarr2 fixture alongside other comparison fixtures. Verify the regenerated X metadata and values against campaign evidence before treating compressed chunk bytes as identical; codec/package versions can change physical output.

## Preserved artifacts and scope

- `baseline-reference-uncapped/`: original baseline driver, source, preload, run logs, validation, and copied output.
- `candidate/build/`: candidate driver-only build and candidate identity.
- `candidate-output.zarr/`: candidate result, byte-identical to preserved baseline output.
- `record-candidate-z2-384-v2/`: raw candidate stdout/stderr, exit status, source metadata/hash and physical chunk inventory, binary/source hashes, destination tree manifest, exact-value validation, and profile sources.
- `record-candidate-z2-384/`: setup-only failed runner attempt. `SETUP_ATTEMPT.md` explains it failed on a record filename before invoking the native driver; no workload was run and no output was created by that attempt.
- `build_candidate_driver.sh`, `run_candidate_once.sh`, `compare_zarr2.py`: isolated compile, capped one-shot run, and independent exact-value comparison.
