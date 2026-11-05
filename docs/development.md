# Development and validation

## Toolchain

The first qualified pairing is Clang 18.1.3, matching clang-scan-deps-18, CMake 3.28.3, and Ninja 1.11.1 on Linux. Configure rejects unqualified compiler families and generators with an actionable message. A compatible newer Clang can be selected explicitly; that does not establish a portability claim until its tests pass.

Storage defaults to `ANNDATAPP_ENABLE_HDF5=ON` and `ANNDATAPP_ENABLE_ZARR=ON`. HDF5 C (>=1.10), zlib and Zstandard are discovered through pkg-config; C-Blosc and nlohmann JSON are private Zarr build dependencies. See [storage-dependencies.md](storage-dependencies.md) for installation commands and the optional private prefix. Backend headers do not enter public module interfaces. The `core-only` workflow disables both backends. `hdf5-only`, `zarr-only` and `shared` qualify the corresponding package configurations; `shared-release` checks an optimized shared library and installed consumer in its own build directory. The normal build produces a static library.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers

cmake --workflow --preset shared-release
```

`cmake --workflow --preset native-check` configures, builds, and runs the entire native/import/installed-consumer suite in one command with Python tests disabled. It is the standalone integrated verification contract used by the Todo foundation milestone.

AnnData++ supplies boolean, fixed-width integer, FP32, and FP64 storage. Portable
binary16 conversion and velocity's explicit FP16 storage views are Cellerator-owned.
This package has no FP16 storage or CUDA component.

Compiler dependency scanning uses CMake's CXX_MODULES file sets. CMake normally detects the versioned scanner; `CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS` is an override for unusual installations. Install the scanner corresponding to the selected compiler, rather than an unrelated LLVM development library.

## Consumers and installation

```sh
cmake --install build/debug --prefix /tmp/anndatapp-install
cmake -S tests/consumer -B /tmp/anndatapp-consumer -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH=/tmp/anndatapp-install
cmake --build /tmp/anndatapp-consumer
/tmp/anndatapp-consumer/consumer
```

If dependencies were extracted to the private prefix, add that prefix to the consumer's `CMAKE_PREFIX_PATH` alongside the installed library. The installed-consumer CTest does this automatically.

The package target is `anndatapp::anndatapp`. An in-tree consumer can link the same target after `add_subdirectory()`. Set `CMAKE_CXX_EXTENSIONS OFF` in the consuming project, as in tests/consumer/CMakeLists.txt, so compiler module flags agree. The external-consumer CTest exercises installation, find_package, BMI compilation, linking, and execution.

`import anndatapp;` supports the import-only example. Consumers that name `std::vector`, `std::span`, or other standard types should include their corresponding standard headers themselves. The library does not re-export the standard library as a user namespace facility.

## Python semantic comparison

The core build and native tests do not require Python. Optional differential tests require the packages listed in `tests/requirements-interop.txt`, installed in a dedicated development environment or supplied explicitly by the caller.

```sh
cmake --preset debug -DANNDATAPP_ENABLE_INTEROP_TESTS=ON \
  -DPython3_EXECUTABLE=/path/to/python
cmake --build --preset debug
ctest --preset debug
```

`python_semantics` compares slicing/metadata outcomes with independently generated Python AnnData objects. When HDF5 is enabled, `python_h5ad` additionally exercises Python-written files through the C++ reader/writer and independently reads C++ output with AnnData and h5py. With both backends enabled, `python_storage` adds backed selection, caller buffers, structural updates, cross-backend exports and independently reopened Zarr v2/v3 output. These are separate checks: in-memory behavior alone does not establish file compatibility. The declared encoding subsets and failure contracts are documented in [h5ad.md](h5ad.md) and [storage.md](storage.md).

For a one-command reference check, prepare a workspace-local `.venv` containing the pinned development packages, then run `cmake --workflow --preset interop-check`. When using Project Control's command sandbox, the virtual environment's interpreter and dependencies must also be accessible inside its trusted mounts (for example, create it from `/usr/bin/python3`). Provisioning that optional environment is separate from the standalone native gate; the gate does not download packages.

On this restricted runner, Zarr's background event loop cannot wake itself through a socketpair. `ANNDATAPP_INTEROP_ZARR_WAKEUP=1` enables a test-only periodic wakeup in `python_storage`; it changes neither library behavior nor the normal Python/CI policy. Local sanitizer qualification uses `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1` because LeakSanitizer cannot operate under the runner's ptrace restrictions; ASan and UBSan remain active.

## Record interoperability

The `python_streaming` gate uses the ordinary interoperability requirements. It
independently checks contiguous/chunked HDF5, Zarr v2/v3 chunks and shards,
compressed/raw reusable buffers, sparse major-axis transfers, parallel worker
execution, and reopened metadata. Generated stores remain under the build tree.
The `thread-sanitizers` preset instruments the library and test executables with
ThreadSanitizer and filters CTest to `streaming`, `transfer_context`, and the
focused `python_streaming_threaded` and `python_completeness_threaded` Zarr v2/v3
fixture gates. Its build preset selects the four required executables. Supply the pinned
interop environment when configuring locally:

```sh
cmake --preset thread-sanitizers \
  -DPython3_EXECUTABLE=/path/to/reference/bin/python
cmake --build --preset thread-sanitizers
ctest --preset thread-sanitizers
```

The `python_completeness` gate covers all supported numerical dtypes, gathered
and signed-stride writes, shard boundaries, metadata preservation, caller buffers
and a huge logical array under a 512 MiB address-space limit. For ASan only,
`ANNDATAPP_COMPLETENESS_NO_AS_LIMIT=1` omits that incompatible virtual-address
limit; normal qualification retains it. TSan uses its own allocation interceptors,
so custom allocation observations are omitted from its focused tests. Ordinary
native and ASan tests retain those observations.

The record-array reference check additionally uses Scanpy 1.11.5. Keep its
development dependencies separate from application environments:

```sh
python3 -m venv /tmp/anndatapp-record-reference
/tmp/anndatapp-record-reference/bin/python -m pip install -r tests/requirements-records.txt
cmake --preset interop-check -DANNDATAPP_ENABLE_RECORD_INTEROP_TESTS=ON \
  -DPython3_EXECUTABLE=/tmp/anndatapp-record-reference/bin/python
cmake --build --preset interop-check
ctest --preset interop-check
```

`ANNDATAPP_ENABLE_RECORD_INTEROP_TESTS` defaults to `OFF`, so existing Python
checks do not acquire a Scanpy dependency. CI runs the pinned Scanpy 1.11.5
record gate with Python 3.12 and `tests/requirements-records.txt`; the existing
Python 3.13 semantic and storage job remains separate. Native records require no
Python or additional native library. `records_benchmark` is built with the
ordinary benchmark option; its acquired record-member loops are compared with
the same work on raw native structs.

## Benchmarks

The release preset builds `memory_benchmark`, `h5ad_benchmark` when HDF5 is enabled, and `storage_benchmark` when both backends are enabled. See [performance.md](performance.md) for the historical core baseline, [h5ad-performance.md](h5ad-performance.md) for eager H5AD measurements and [storage-performance.md](storage-performance.md) for backed storage and the four-configuration compile campaign. The compilation benchmark tools use disposable source copies so rebuild experiments do not edit the live checkout. Record source identity, compiler/scanner, flags, cache conditions, wall time, memory, and affected compilation commands alongside results.

`anndatapp_selection_probe` is a separate bounded allocation check, enabled by
`ANNDATAPP_BUILD_BENCHMARKS`. It counts C++ allocation calls and requested bytes
around descriptor acquisition and release; fixture setup and reporting are
excluded. It checks affine selections at fixed rank across logical lengths and
direct metadata access across table widths. It makes no runtime-speed or
process-memory claim. Run it with `timeout 30s build/release/anndatapp_selection_probe`.
See [selection-validation.md](selection-validation.md) for the recorded results.

The [retained-gather experiment](gather-buffer.md) is a separate short paired
runtime/allocation/module-build check. It retains the existing production
representation and preserves its tested alternative as an opt-in patch.

The [representation cleanup checks](compliance-fixes.md) use
`benchmarks/packed_strings.cpp` with the same isolated module fixture. To build
and run this small probe against a checkout or frozen source snapshot:

```sh
cmake -S benchmarks/gather-fixture -B build/packed-strings -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release \
  -DANNDATAPP_SOURCE="$PWD" -DANNDATAPP_PROBE="$PWD/benchmarks/packed_strings.cpp"
cmake --build build/packed-strings --target gather_probe --parallel 2
timeout 30s build/packed-strings/gather_probe 1
```

The executable reports JSON lines. Fixture generation is excluded; vector input
copies, construction, retained-view checks and destruction are included equally
in paired runs. Allocation counting and timing are separate. It counts ordinary
throwing C++ allocations, not aligned allocations, `malloc` or process memory.

## Workflow state

The Project Control workspace entry is `anndatapp`. Its immutable native repository and authority IDs retain `anndata-cppm` so workflow history remains attached. Todo SQLite is authoritative and private, with metadata under `.todo-orchestrator/`. Authored planning files are inputs, not a replacement authority. Keep credentials outside the checkout and use the installed paired runtime for bootstrap. The sanitized setup receipt distinguishes registration, authority initialization, integration provisioning, source access, and host workflow readiness.
