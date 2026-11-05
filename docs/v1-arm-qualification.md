# Native ARM qualification — conditional

Qualification evidence for code candidate `cbb638313674ad973461ed11638aaf246955335a` on 2026-11-05. The requested original candidate `96272ff5f1bbb6abd3b2133e6c07f68b56a57fbb` failed native compilation. Focused fixes split mixed `size_t`/`uint64_t` declarations, selected the address-bearing HDF5 API explicitly, corrected native-type/POSIX test assumptions, and repaired Darwin-only observer loading. Public `.cppm` interfaces were unchanged. This is additional-device evidence, **not a clean 1.0 qualification**: sanitizer execution, uncontended performance and the consumer-build comparison remain incomplete.

Host: Apple M2, native AArch64, 8 GiB RAM, macOS 27.0 (26A428), Darwin 27.0.0 `RELEASE_ARM64_T8112`, AC power. Clang and matching scanner 20.1.3 were used at the user's request after the Clang 18 download failed; its remaining download metadata was removed. CMake 3.31.3; Ninja 1.12.1; libc++ headers 200100 and system runtime 2200.32.0; SDK MacOSX27.2. Native backends: HDF5 2.2.0, zlib 1.4, Zstandard 1.5.7, C-Blosc 1.21.7, JSON 3.12.0. Tool/executable hashes are in the raw build receipts. All BMIs were built locally with CMake scanning.

The isolated Python 3.12.2 environment used `tests/requirements-records.txt`: AnnData 0.13.4, Scanpy 1.11.5, NumPy 2.2.6, pandas 2.3.3, SciPy 1.17.1, h5py 3.16.0, Zarr 3.4.2, numcodecs 0.17.1. Full transitive pins remain in `build/arm/reference-freeze.txt`.

| Gate | Result |
| --- | --- |
| Debug / Release | 20/20 each |
| Core / HDF5 / Zarr | 12/12; 16/16; 16/16 |
| Shared Debug / Release | 20/20 each |
| UBSan alone, halt on error | 20/20 |
| Python / Scanpy full reference | 30/31; completeness functional follow-up 1/1 without address-space cap |
| ASan/UBSan preset | Built; runtime stalled before main; no pass |
| Focused TSan preset | Built; 0/4, runtime crashes/timeouts |
| Installed consumers | Passed in all ordinary native/package configurations |
| Downstream CPU | Cellerator module 1/1; adapted module-only install 2/2 |

ASan's sampled stack stayed in shadow-memory mapping enumeration before `main`; a minimal sanitized executable timed out outside the sandbox too. TSan's minimal executable also crashed outside the sandbox. These are unavailable instrumentation results, not passing checks. Normal threaded reference tests passed. No leak-detection or Zarr-wakeup workaround was enabled. Matplotlib/Numba/font caches used writable build-local paths after cache permission failures were observed.

The full reference run's sole failure was `python_completeness`: macOS rejected the 512 MiB `RLIMIT_AS` in `preexec_fn`; an isolated resource-limit probe reproduced the error. The focused rerun with `ANNDATAPP_COMPLETENESS_NO_AS_LIMIT=1` passed. That supplies functional storage evidence without a 512 MiB address-space-bound claim. Linux-only typed-mmap and syscall fault-injection subchecks were unavailable.

Cellerator `bac859c1b731f9d5ca6f0371d414886764105b58` stayed unchanged. Its full SDK install was blocked by `cuda_runtime_api.h` despite CUDA=OFF; the two installed-importer passes used an explicitly adapted module-only package. The final downstream receipt binds unchanged executable and AnnData archive hashes. No local preprocess.cppm, standalone CellShard or velocity checkout was available. The additional Cellerator style guide and Todo Orchestrator skill were unavailable; authority state was not edited.

The strict private rebuild probe preserved ten public BMIs and two consumer objects and compiled only `strings.cpp`. All nine restored core archive member objects matched the pre-probe installed archive. `/usr/bin/ar` changed archive timestamps/index metadata (six byte positions); whole-archive byte reproducibility failed. Restored executable identity was not established. This limitation is retained rather than reported as a passing reproducibility gate. The mismatched-major scanner negative check correctly rejected scanner 19 with compiler 20.

The fixed benchmark ran **once**: 38 operations validated in each revision, five alternating baseline/candidate pairs across ten scenarios, 100 observer-free timing processes and 20 separate diagnostics. Charged process time was 3.629s baseline / 3.191s candidate; all 30s scenario and 300s revision caps passed. The original baseline `cb6234a1a7a549d70a514e820085e18fb5a66403` was preserved. It also required the documented [ARM compatibility patch](../benchmarks/v1/arm-baseline-compat.patch); ARM timings compare that derived baseline with the repaired candidate, not an untouched original binary.

Production digests: original baseline `96dd8c96fe516d7f71d76960cea3be29609fd66392a613bd8a8350edf1f9e08c`; ARM baseline `eb7197adb9b13f3c3a3c2264b53f01509443694c3f7d42013b61dbd18280eaef`; ARM candidate `ff7b0b779cc9576ab656c600d932b1ceb7a838c0412ee2ac45a39e124e3ded12`. Shared driver digest `4c2ecbf1edfcbd570fd13d90d17135aba9e1848d2bc0613508cc215e181a3dfa`. Both original pinned trees remain under `build/arm/`; `baseline-arm-src` and `fixed-src` contain the disclosed repairs.

Three operations triggered the declared >5% paired-median and separated-range rule:

| Operation | Median baseline → candidate, ms | Paired median ratio |
| --- | ---: | ---: |
| `dense_copy_gather_into` | 3.377 → 3.663 | 1.0584 |
| `dense_copy_strided_into` | 4.288 → 4.642 | 1.0803 |
| `pointer_saxpy_4_workers` | 7.464 → 8.107 | 1.1126 |

Causes remain unresolved. Both copy-into diagnostics counted 48 C++ allocations / 768 requested bytes. Four-worker SAXPY uses identical driver-owned raw loops and barriers with no AnnData dispatch in the timed loop, so its flag cannot by itself be attributed to a library change. Pre-window CPU idle was only 42–58%, with iCloud/terminal/storage activity. The user granted the single window after temporary uploader-pause instructions; exclusive quiet was not verified. Disposable stores used `/private/tmp/anndatapp-arm-cbb6383/` through ignored symlinks. No repeat campaign or general speed claim was made. The original x86 timing summary and qualification receipt are retained as historical measurement records.

Consumer comparison is incomplete: the first baseline fresh consumer compiled, but BSD `time -l` failed on sandboxed `sysctl kern.clockrate`, leaving RSS unavailable. The remaining seven observations were not run. The separate 60s budget was not exhausted. Five core-named BMIs total 54,747,332 / 54,754,164 bytes; all built BMIs total 121,559,696 / 121,568,588 bytes. These are artifact inventories, not a completed consumer-build timing or candidate import-closure measurement.

Compact receipts: [qualification](../tests/results/v1-arm.json), [performance](../benchmarks/results/v1-arm-summary.json). Raw logs, binaries, fixtures and source copies remain ignored under `build/arm/`; run and consumer symlinks point to the temporary directory above. Fixture tree SHA-256 `3739e4e64f660e3844e78a4b1201c91b89b160d9761cef971d281e8771cc6394`; fixture manifest `533d69b2d188c7bf53b0aaca2e12a3976abd3dba7a196b644458951e5ad1c908`; samples `3fb0647c9b2649db9d003c7f1ec840e16d6f9069b6c3abc6a08279184368e405`. The full artifact/log hash manifest is `build/arm/artifact-manifest.json`, SHA-256 `b94806be3a4a8927cc98f3bcdb4c264475feb12e28fccf1df4d598fcd1beee81`.

The user-approved release plan covers this AArch64 host and the user's other machine. The user will requalify the pushed candidate on that other machine before the release decision. The recorded sanitizer, performance and consumer limitations remain available for that review.
