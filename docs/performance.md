# Performance measurements

The library exposes numerical storage through typed pointers, extents and strides. The benchmark measures the same dense kernel through a retained typed view and through its raw pointer, with access acquisition outside the timed loop. It also measures that kernel partitioned across caller-owned threads. These results describe the tested machine, compiler, build flags, array size and thread count; they do not establish a general speedup.

Build and run the numerical benchmark with a Release preset:

```sh
cmake --preset release
cmake --build --preset release --target memory_benchmark
./build/release/memory_benchmark 4194304 4 7
```

Optional fourth and fifth arguments set dense-view acquisition iterations and sparse wrapping iterations:

```sh
./build/release/memory_benchmark 4194304 4 7 100000 64
```

Its first JSON line records the compiler, C++ language version and workload. Subsequent JSON lines record each paired trial, access method, execution mode, elapsed time, effective read/write bandwidth and checksum, plus separate `dense<float>()`/`unsafe_dense<float>()` acquisition and `Validation::full`/`Validation::none` sparse wrapping times. Timing distributions follow for each category. The input arrays and valid CSR buffers are initialized before timing. Pointer identity and buffer checksums are checked independently of timed regions. Thread creation and barriers are prepared before the parallel timer; the measured parallel interval includes barrier coordination.

The typed and raw numerical kernels deliberately use the same acquired memory pointers; both loop over `float*` and make no library call for each element. This isolates loop execution after acquisition and does not estimate the cost of other array APIs. Dense access trials measure `dense<float>()` versus `unsafe_dense<float>()` acquisition separately from the kernel. Sparse wrapping trials compare `Validation::full` with `Validation::none` on the same valid, preinitialized 16,384 × 16,384 CSR buffers (32,768 values); they measure wrapper construction and validation, not invalid-data behavior. These timings do not establish a general performance advantage or a safety cost for scalar access.

Measure C++ module build dependencies with:

```sh
python3 tools/compile_benchmark.py --runs 5 \
  --output benchmarks/results/module-build.json
```

The script snapshots the current source into a temporary directory and performs all generated edits and builds there. It excludes prior reports under `benchmarks/results/`, the optional Python `.venv/`, local build trees, and private runtime directories from snapshots and source identity. It measures fresh builds, a no-op consumer build, a consumer-source timestamp edit, a no-op library build, an implementation-only edit, a public-interface edit, and fresh core/facade consumer imports that reuse already-built module artifacts. The no-op consumer and consumer-edit operations use the same target set and command. The JSON report records build and compiler commands, BMI and object hashes and timestamps, a SHA-256 map of C++ and CMake build inputs plus its aggregate digest, compiler and build-tool versions, and GNU `time` maximum RSS. The manifest excludes narrative documents, so their concurrent edits do not change build-input identity. GNU `time`'s `%M` is recorded as a maximum resident-set statistic for the waited command; it is not the sum of simultaneous processes in the process tree. The operating-system file cache is not cleared.

Optional format-module costs are not measured before those modules exist. Preserve each report alongside its source hash and environment details; compare results only when the source, build configuration and workload are understood.

## Foundation baseline

Measurements were collected on Linux 6.8, Intel Xeon E5-2698 v4 at 2.20 GHz (80 logical CPUs), Clang 18.1.3, CMake 3.28.3 and Ninja 1.11.1. Both builds used Release mode and two parallel build jobs. The compile experiment used five independent temporary build trees. The OS page cache was left in its normal state. The memory program used 4,194,304 float elements, four caller-managed threads, seven paired trials, 100,000 dense acquisitions per trial and 64 sparse wraps per trial. Its dense kernel moved 32 MiB per invocation.

| Compile operation | Median wall time | Median GNU `time` max RSS |
|---|---:|---:|
| Configure | 0.386 s | — |
| Fresh library and example build | 10.076 s | 232,320 KiB |
| No-op consumer build | 0.0193 s | 17,280 KiB |
| Consumer source timestamp edit | 0.316 s | 119,680 KiB |
| No-op library build | 0.0182 s | 17,920 KiB |
| Implementation-only edit | 2.157 s | 169,248 KiB |
| Public interface edit | 9.885 s | 232,960 KiB |
| Fresh consumer importing `anndata.core` | 0.241 s | 115,840 KiB |
| Fresh consumer importing `anndata` | 0.243 s | 115,840 KiB |

The first build produced four BMIs totaling 13,855,024 bytes (about 13.2 MiB). Across all five dependency checks, no-op builds left BMI and object artifacts unchanged; a consumer timestamp touch rebuilt its object while preserving BMIs; an implementation-only edit rebuilt `arrays.cpp.o` while leaving the consumer object and BMIs unchanged; and a public interface edit changed BMIs and rebuilt the consumer. BMI checks use `.pcm`/`.gcm` files, excluding CMake's `CXXModules.json` dependency-scan metadata. The no-op consumer and consumer-edit timings use the same target and command. GNU `time` values are maximum-resident-set statistics for each waited build command, not sums across simultaneous processes.

The typed and raw dense kernels use the same acquired pointers and identical `float*` loop. Their seven-trial median timings were 1.630 ms typed and 1.705 ms raw in serial mode; with four caller threads they were 1.042 ms and 1.211 ms. The 25th–75th percentile ranges overlapped in both modes, so this run establishes no timing advantage between the two paths. The source-level kernel contains no AnnData call or type dispatch per element.

| Setup measurement | Median per call | 25th–75th percentile |
|---|---:|---:|
| `dense<float>()` acquisition | 16.51 ns | 16.17–16.64 ns |
| `unsafe_dense<float>()` acquisition | 16.04 ns | 15.90–17.68 ns |
| Sparse borrowed wrap with `Validation::full` | 36.55 μs | 36.51–36.55 μs |
| Sparse borrowed wrap with `Validation::none` | 57.20 ns | 56.67–57.31 ns |

Dense acquisition timings include creating and releasing the retained descriptor and accumulating its pointer token; the `ArrayView` handle and input checksum are prepared outside the timer. Each acquisition returned the original buffer pointer, and the pointer-token and input checksums matched. Sparse wrapping timings include wrapper-state allocation and destruction. Wrapping used the same preinitialized valid 16,384 × 16,384 CSR buffers with 32,768 values, 32-bit indices and 64-bit offsets; both validation settings preserved all three buffer pointers and their checksum. On this fixture, full wrapping scans the CSR structure while `Validation::none` skips that scan. The result describes the setup cost for this valid buffer shape; the benchmark does not exercise malformed sparse buffers or establish correctness when validation is disabled.

The raw measurements, input hashes, commands, BMI/object metadata and environment are in [module-build.json](../benchmarks/results/module-build.json) and [memory.json](../benchmarks/results/memory.json). The manifests exclude unrelated `.venv/` headers and local build-tree files; this correction changed only input maps and aggregate hashes, with all measured samples and commands preserved. Optional format-module costs remain unmeasured.
