# Bounded C++ and Python performance screen

This preserves the initial screen. The observed native gathered-read deficit
was subsequently addressed in the [focused correction](gather-read-performance.md),
with both original and updated measurements retained.

This is one paired observation per language, case and size. It identifies large
potential deficits, not stable speedups or statistical confidence intervals.
The C++ source is commit `dc946845e1457299351b0ffc4ba36ef91d49db7b`. The
[compact results](../benchmarks/results/head-to-head-summary.json) include all
samples, timings, process memory, diagnostic counts and source identities.
Raw records, fixtures, executable, shared library, logs and source snapshots are
kept locally under ignored `build/head-to-head` and `build/head-to-head-paired`.

The test uses deterministic 1024 × 1024 and 4096 × 4096 float32 matrices with
approximately 9% nonzero values. Fixtures contain ordered axis labels, categorical
columns, a separate 3%-dense CSR layer and `.raw` with 64 variables. H5AD's dense
X is contiguous and uncompressed. Zarr v3 X uses 64 × 128 Zstd inner chunks in
256 × 512 shards. CSR row and CSC column scans use files with those respective
sparse orientations. Input and requested output bytes, indices and dtypes match
within each C++/Python pair.

Clang 18.1.3 built the installed C++ library and driver in Release (`-O3
-DNDEBUG`), without `-march=native`; the driver additionally uses
`-ffp-contract=off`. The Python reference uses AnnData 0.13.4, NumPy 2.2.6,
SciPy 1.17.1, h5py 3.16.0 and Zarr 3.4.2. The environments use different
backends: C++ HDF5 1.10.10 and Zstd 1.5.5, Python h5py HDF5 2.0.0 and
numcodecs Zstd 1.5.6. These version and implementation differences confound
language attribution, especially for I/O.

Each fresh process ran on one dynamically selected CPU from the caller's allowed
affinity set. BLAS, OpenMP, Blosc, Zarr async concurrency and Zarr worker limits
were set to one; Python Zarr still reported three process threads, all under that
one-CPU affinity. Files were read to warm the OS cache immediately before each
trial, without dropping caches. The page cache, storage, compression, scheduler
activity and CPU frequency affect the results. Setup, independent verification,
copying writable stores and warming files were outside the operation timer.
No `fsync` durability guarantee is measured. Python Zarr used its normal asyncio
wakeup outside the restricted sandbox: the sandbox could not wake its event loop,
and its 10 ms polling workaround would distort this comparison.

| Workload | 1024² ms C++ / Python | 4096² ms C++ / Python | 4096² process peak MiB C++ / Python |
|---|---:|---:|---:|
| Complete AnnData H5AD read | 7.70 / 28.83 | 50.39 / 65.48 | 87.5 / 190.8 |
| Complete AnnData H5AD write | 9.02 / 28.82 | 59.23 / 73.20 | 89.7 / 191.1 |
| Dense HDF5 scan into one preallocated output | 1.89 / 3.12 | 36.46 / 31.13 | 78.7 / 180.1 |
| Backed CSR row blocks | 1.13 / 3.80 | 10.91 / 22.50 | 31.4 / 134.8 |
| Backed CSC column blocks | 1.42 / 3.78 | 14.57 / 25.69 | 31.5 / 134.9 |
| Sharded Zarr v3 scan into one preallocated output | 13.05 / 58.87 | 223.81 / 949.99 | 75.4 / 179.9 |
| Sharded Zarr v3 orthogonal gather, eight repeats | 735.44 / 285.47 | 719.21 / 790.57 | 11.6 / 116.2 |
| One complete-shard Zarr write | 1.93 / 9.13 | 1.90 / 9.77 | 13.4 / 116.2 |
| One partial write crossing four Zarr shards | 2.51 / 9.05 | 2.72 / 8.99 | 13.4 / 116.5 |
| Eight-step in-place float32 recurrence | 1.97 / 3.50 | 61.05 / 115.49 | 87.6 / 190.7 |

The full H5AD rows use `read_h5ad`/`write_h5ad` in both libraries and include
all annotations. Other rows compare prepared native backed transfers against
efficient lower-level Python paths: `h5py.Dataset.read_direct`,
`anndata.io.sparse_dataset`, or Zarr array selections. These are not complete
AnnData I/O operations. Output buffers for dense scans were allocated before
timing. C++ output blocks and sparse requirements were also prepared before
timing; Python's `sparse_dataset` returns newly allocated SciPy sparse blocks
inside its operation. Its default cached indptr was explicitly warmed during
preparation. C++ kept `Validation::full`; Python uses its ordinary API checks.

Preparation changes the sparse interpretation. At 4096², preparation plus
operation was 21.58 / 24.03 ms for C++ / Python CSR and 28.64 / 27.37 ms for
CSC. The standalone operation times in the table favor the preallocated C++
path; those preparation-inclusive single observations are much closer. For the
4096² direct dense scan, Python's 31.13 ms was below C++'s 36.46 ms, but one
pair and different HDF5 versions do not establish a reliable winner or cause.
Both instrumented pilots intercepted exactly 32 reads and 67,108,864 bytes.

The 1024² Zarr gather is an obvious observed loss for C++: 735.44 versus 285.47
ms. The 4096² times are close enough for this single-observation screen to be
inconclusive. The 4096² diagnostic nevertheless exposes repeated work: eight
16 × 64 gathers request 32,768 useful bytes, while native `TransferContext`
reports 8,192 logical reads. The bounded C++ pilot intercepted 16,384 `pread`
calls and 29.65 MB read; the Python pilot intercepted 738 reads and 9.25 MB.
The larger fixture's gather coordinates touch a different number of shards, so
these sizes are not a strict scaling experiment. The evidence points to a
native gathered-read batching opportunity; no library optimization was attempted.

The recurrence performs `x = x * 0.5f + 0.125f` in place, eight times with
matching float32 results. NumPy uses preallocated `out=` operations: two array
passes per step. C++ fuses each step into one pass. A separate warm Numba
nopython, `fastmath=False`, single-thread control on 4096² took 58.34 ms versus
61.05 ms for portable C++ and 115.49 ms for NumPy. Its separate JIT compilation
was 291.92 ms and Numba import 148.21 ms; Numba specialized to the host ISA,
whereas the C++ build did not. The near-equal warmed fused implementations
show why the NumPy result is not evidence of an intrinsic language-wide gain.

Memory columns are Linux `/proc/self/status` `VmHWM`: process peak through the
operation, **including runtime imports and setup**, not operation-only peak
allocation. The corresponding before/after `VmRSS` snapshots and startup-inclusive
wall times are in the JSON. For example, the large HDF5 scan's C++ process went
from 14.5 to 78.7 MiB and Python from 115.8 to 180.1 MiB: roughly the same
64 MiB of resident output, with different runtimes and dependencies. Numba's
301.4 MiB process peak includes LLVM/JIT preparation. Allocator retention and
lazily committed pages limit what can be inferred from these snapshots.

Every result matched an independently generated full-buffer SHA-256 (sparse
blocks include data, indices and offsets). Twenty-four write outputs were also
independently reopened and checked for full X, metadata, categories, sparse layer
and `.raw`; Zarr validation additionally compared every file outside touched
shards byte-for-byte. All 20 timed pairs and the Numba control completed. The
fixture file tree was re-hashed after the screen: 330 files, no changes.

Instrumented pilots had 30-second limits and 512 MiB per-direction **intercepted**
I/O caps, plus a 1 GiB sampled process RSS limit. The largest observed intercepted
pilot read and write were 72.65 MB and 72.63 MB; no caps were reached. Direct
syscalls, memory maps, hidden library calls, fixture setup and validation are
outside those byte counts. Counters cover selected libc I/O and malloc symbols,
not all allocations or peak memory. Timed pairs ran without the interposer after
successful pilots; diagnostic timings are not used as comparative timings.

The first pilot used an older codec-interposing counter. Python Zarr writes
failed because forwarding bundled Zstd symbols through `RTLD_NEXT` crossed
codec implementations. A small wrapper disables codec hooks while retaining
I/O and allocation instrumentation. The first independent validator also treated
AnnData's `layers[None]` X alias as an unchanged layer; the corrected validator
checks X against the specified write and each named layer independently. Both
initial failures and their outputs are preserved; successful non-write pilots
were reused only after source, binary, fixture and expected-output hashes matched.
The generator source hash was captured after fixture generation and restoration
of its original call; fixture byte hashes and independent validation, rather
than an unrecorded pre-generation source hash, establish the tested inputs.

## Reproduction

Install the dependencies in [storage-dependencies.md](storage-dependencies.md),
plus OpenSSL development headers for the benchmark-only SHA-256 checker. Prepare
the pinned Python environment including `tests/requirements-records.txt` for the
optional Numba control. Build and install the same C++ commit in Release; then,
using fresh paths and a compatible CMake prefix:

```sh
cmake -S benchmarks/head_to_head -B build/head-to-head/driver -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/anndata/install;/path/to/private/dependencies"
cmake --build build/head-to-head/driver --parallel 2
clang-18 -std=c11 -shared -fPIC -O2 -I/path/to/dependencies/include \
  benchmarks/head_to_head/io_counters.c -o build/head-to-head/io-counters.so -ldl -pthread
/path/to/reference/python benchmarks/head_to_head/make_fixture.py \
  build/head-to-head/fixtures 1024 4096
python3 benchmarks/head_to_head/run.py --root build/head-to-head \
  --python /path/to/reference/python --driver benchmarks/head_to_head/python_driver.py \
  --binary build/head-to-head/driver/head_to_head --counters build/head-to-head/io-counters.so \
  --sizes 1024 4096 --repetitions 1 --numba-control
python3 benchmarks/head_to_head/summarize.py --root build/head-to-head \
  --output benchmarks/results/head-to-head-summary.json --repetitions 1
```

The runner refuses source/binary/fixture drift on resume, checkpoints each raw
measurement before permitting validation, and stops a case after a timeout,
I/O-cap failure or correctness failure. Its default CPU is chosen from current
allowed affinity; `--cpu` selects another permitted CPU. For accurate Zarr
results, use a host where cross-thread asyncio wakeups work normally, and leave
`ANNDATA_INTEROP_ZARR_WAKEUP` unset.
