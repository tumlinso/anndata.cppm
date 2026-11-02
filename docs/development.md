# Development and validation

## Toolchain

The first qualified pairing is Clang 18.1.3, matching clang-scan-deps-18, CMake 3.28.3, and Ninja 1.11.1 on Linux. Configure rejects unqualified compiler families and generators with an actionable message. A compatible newer Clang can be selected explicitly; that does not establish a portability claim until its tests pass.

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

The Python test creates and independently reopens real H5AD fixtures, then compares logical slicing/metadata outcomes to corresponding C++ in-memory data. This establishes a bounded semantic comparison; C++ H5AD reading/writing and file round trips belong to the subsequent milestone.

For a one-command reference check, prepare a workspace-local `.venv` containing the pinned development packages, then run `cmake --workflow --preset interop-check`. When using Project Control's command sandbox, the virtual environment's interpreter and dependencies must also be accessible inside its trusted mounts (for example, create it from `/usr/bin/python3`). Provisioning that optional environment is separate from the standalone native gate; the gate does not download packages.

## Benchmarks

The release preset builds `memory_benchmark`. See docs/performance.md for exact recorded commands, conditions, results, and limitations. `tools/compile_benchmark.py` works on a disposable source copy so rebuild experiments never edit the live checkout. Record source identity, compiler/scanner, flags, cache conditions, wall time, memory, and affected compilation commands alongside results.

## Workflow state

The Project Control workspace/repository alias is `anndata-cppm`. Todo SQLite is authoritative and private, with metadata under `.todo-orchestrator/`. Authored planning files are inputs, not a replacement authority. Keep credentials outside the checkout and use the installed paired runtime for bootstrap. The sanitized setup receipt distinguishes registration, authority initialization, integration provisioning, source access, and host workflow readiness.
