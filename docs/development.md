# Development and validation

## Toolchain

The first qualified pairing is Clang 18.1.3, matching clang-scan-deps-18, CMake 3.28.3, and Ninja 1.11.1 on Linux. Configure rejects unqualified compiler families and generators with an actionable message. A compatible newer Clang can be selected explicitly; that does not establish a portability claim until its tests pass.

H5AD support defaults to `ANNDATA_ENABLE_HDF5=ON` and discovers the HDF5 C library (>=1.10) through pkg-config. Neither HDF5 headers nor numerical framework headers enter public module interfaces. Configure with `-DANNDATA_ENABLE_HDF5=OFF`, or run the `core-only` workflow, to omit the storage dependency. `shared` qualifies the shared-library package and installed consumer; the normal build produces a static library.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers
```

`cmake --workflow --preset native-check` configures, builds, and runs the entire native/import/installed-consumer suite in one command with Python tests disabled. It is the standalone integrated verification contract used by the Todo foundation milestone.

Compiler dependency scanning uses CMake's CXX_MODULES file sets. CMake normally detects the versioned scanner; `CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS` is an override for unusual installations. Install the scanner corresponding to the selected compiler, rather than an unrelated LLVM development library.

## Consumers and installation

```sh
cmake --install build/debug --prefix /tmp/anndata-install
cmake -S tests/consumer -B /tmp/anndata-consumer -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DCMAKE_PREFIX_PATH=/tmp/anndata-install
cmake --build /tmp/anndata-consumer
/tmp/anndata-consumer/consumer
```

The package target is `anndata::anndata`. An in-tree consumer can link the same target after `add_subdirectory()`. Set `CMAKE_CXX_EXTENSIONS OFF` in the consuming project, as in tests/consumer/CMakeLists.txt, so compiler module flags agree. The external-consumer CTest exercises installation, find_package, BMI compilation, linking, and execution.

`import anndata;` supports the import-only example. Consumers that name `std::vector`, `std::span`, or other standard types should include their corresponding standard headers themselves. The library does not re-export the standard library as a user namespace facility.

## Python semantic comparison

The core build and native tests do not require Python. Optional differential tests require the packages listed in `tests/requirements-interop.txt`, installed in a dedicated development environment or supplied explicitly by the caller.

```sh
cmake --preset debug -DANNDATA_ENABLE_INTEROP_TESTS=ON \
  -DPython3_EXECUTABLE=/path/to/python
cmake --build --preset debug
ctest --preset debug
```

`python_semantics` compares slicing/metadata outcomes with independently generated Python AnnData objects. When HDF5 is enabled, `python_h5ad` additionally exercises Python-written files through the C++ reader/writer and independently reads C++ output with AnnData and h5py. These are separate checks: in-memory behavior alone does not establish file compatibility. The declared encoding subset and failure contracts are documented in [h5ad.md](h5ad.md).

For a one-command reference check, prepare a workspace-local `.venv` containing the pinned development packages, then run `cmake --workflow --preset interop-check`. When using Project Control's command sandbox, the virtual environment's interpreter and dependencies must also be accessible inside its trusted mounts (for example, create it from `/usr/bin/python3`). Provisioning that optional environment is separate from the standalone native gate; the gate does not download packages.

## Benchmarks

The release preset builds `memory_benchmark` and, when enabled, `h5ad_benchmark`. See [performance.md](performance.md) for the historical core baseline and [h5ad-performance.md](h5ad-performance.md) for H5AD measurements. The compilation benchmark tools use disposable source copies so rebuild experiments do not edit the live checkout. Record source identity, compiler/scanner, flags, cache conditions, wall time, memory, and affected compilation commands alongside results.

## Workflow state

The Project Control workspace/repository alias is `anndata-cppm`. Todo SQLite is authoritative and private, with metadata under `.todo-orchestrator/`. Authored planning files are inputs, not a replacement authority. Keep credentials outside the checkout and use the installed paired runtime for bootstrap. The sanitized setup receipt distinguishes registration, authority initialization, integration provisioning, source access, and host workflow readiness.
