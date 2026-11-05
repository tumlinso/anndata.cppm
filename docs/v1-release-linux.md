# 1.0.0 Linux requalification

This report covers the compatibility changes from `96272ff5f1bbb6abd3b2133e6c07f68b56a57fbb` through upstream `c8d1c6c8378e3d17b4d7e3a50f2275f238f1c88a`, followed by the 1.0.0 package version and installed-consumer fixture updates. The production/build manifest digest is `bc6ceab1c711a597b3eea107d7b5640215fb2ab6b4e4ad306a4834cae5d9f96b`. Public `.cppm` interfaces are unchanged.

The host is `testlsserver`, Linux x86-64, Xeon E5-2698 v4, Clang and matching scanner 18.1.3, CMake 3.28.3, Ninja 1.11.1 and HDF5 1.10.10. Historical cleanup and ARM measurements remain separate. The compact release receipt is [v1-release-linux.json](../tests/results/v1-release-linux.json).

| Gate | Result |
| --- | --- |
| Debug / Release | 20/20 each |
| ASan/UBSan | 20/20; leak detection disabled |
| Focused TSan | 4/4 |
| Core-only | 12/12 |
| HDF5-only / Zarr-only | 16/16 each |
| Shared Debug / Release | 20/20 each |
| Independent Python / Scanpy | All 11 reference checks passed; full run 30/31, then installed-consumer rerun 1/1 |
| Private rebuild isolation | Two probes; ten public BMIs and two consumer objects unchanged in each; restored binary hashes match |
| HDF5 links | Six eager/backed rejection checks, valid-file controls and noncyclic hard-alias round trip passed |

The first installed-consumer runs requested package 0.1. CMake's `SameMajorVersion` policy correctly rejected the new 1.0 package. Current test fixtures now request 1.0; Debug, Release, ASan/UBSan and affected component suites passed their complete reruns. Interoperability retained its original 30/31 result and the successful targeted follow-up rather than repeating the reference campaign. Explicit downstream package requests require the same [migration](migration-1.0.md).

ASan used `ASAN_OPTIONS=detect_leaks=0`; UBSan used `halt_on_error=1`. The restricted runner used the documented Zarr wakeup workaround. The normal 512 MiB completeness limit and Linux syscall fault injection remained active. These results do not claim leak qualification.

The HDF5 probe generated independent files using AnnData 0.13.4 and h5py 3.16.0. Both native drivers use HDF5 1.10.10, exercising the older object-info API branch. Cycles, nested soft links and external links returned `unsupported_encoding` with the expected offending paths. A valid sibling hard alias remained readable and preserved numerical/text values through eager copying. Full storage interoperability separately checks preserved root/X attributes and opaque extension payloads. Reproduce the focused checks after building the interoperability drivers:

```sh
/path/to/reference/python tests/hdf5_link_probe.py --work build/hdf5-links
```

## Downstream and performance evidence

The nine downstream CTests and four additional installed CellShard consumers passed. Cellerator host/SDK and CellShard checks used the refreshed HDF5-only installation. Preprocess.cppm and velocity originally requested package 0.1 and correctly rejected 1.0. Isolated copies changed only their package/dependency version requests to 1.0; business, module and test sources matched the originals. External working trees were preserved. These are migrated-version consumer checks, not passes of the unchanged 0.x requests.

The [bounded comparison](../benchmarks/results/v1-release-linux-summary.json) uses the same updated driver and `--new-core` for both the previous Linux candidate and the release source. All 38 untimed operations passed before the single ten-scenario campaign: exactly five alternating pairs per scenario and one separate diagnostic per revision, totalling 100 observer-free timing processes and 20 diagnostics. No operation triggered the declared regression rule. Charged process time was 5.456 seconds baseline and 5.492 seconds candidate; all 30-second scenario and 300-second revision caps passed. This establishes no general speed advantage.

The separate consumer probe completed one fresh import and three recompiles per revision in 2.284/2.276 seconds, within its 60-second budgets. The core import BMI closure was 26,191,312/26,191,316 bytes. Primary `.text` sections were byte-identical for the four changed production units; this scoped observation does not assert identical sections, relocations or whole binaries. The two private-edit gates separately establish public BMI isolation and restored binary hashes.

Raw commands, logs, fixtures and source copies stay under ignored `build/release-1.0/` and are bound by hashes in the compact receipt. Existing x86 and ARM reports were preserved rather than relabeled as final release measurements.

## Release support decision

The user explicitly approved release after successful Linux requalification with the [ARM limitations](v1-arm-qualification.md) disclosed. Native Apple AArch64 used Clang/scanner 20.1.3 and passed its ordinary native/package checks, but ASan/TSan could not execute reliably, three timing cases flagged regressions under contention, and the consumer-build comparison is incomplete. macOS resource-limit, Linux-only fault/mmap coverage and restored-binary reproducibility gaps remain recorded. None is reported as a passing ARM gate.

The two-machine policy supersedes the earlier three-device prerequisite. Release artifacts contain source/module sources and compact qualification reports, without compiler-specific BMIs. The 1.x compatibility promise covers documented source APIs and data contracts; consumers rebuild for their compatible toolchain and no stable binary ABI is promised.
