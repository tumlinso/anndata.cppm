# anndata.cppm

A standalone C++20 library for annotated biological data, with named modules and direct control of numerical memory.

The library provides **in-memory and backed AnnData data, H5AD, and Zarr v2/v3**: dense arrays, CSR/CSC, heterogeneous metadata, annotated matrices, read-only slices, and versioned codecs. Backed access loads selected data into owned or caller-provided buffers and supports direct structural updates. Python, Cellerator, Torch, and Arrow are not runtime dependencies; a core-only build requires no storage backend.

```cpp
import anndata;

int main() {
    auto data = anndata::AnnData::empty(3, 2);
    data.set_x(anndata::Array::empty<float>({3, 2}));
    auto x = data.edit_x().dense<float>();
    float* p = x.data();
    for (int i = 0; i < 6; ++i)
        p[i] = static_cast<float>(i);
    auto selected = data.slice(anndata::indices{0, 2}, anndata::all);
    auto independent = selected.copy();
    anndata::write_h5ad("selected.h5ad", independent);
    auto loaded = anndata::read_h5ad("selected.h5ad");
}
```

Owners are move-only. `copy()` creates independent data. Typed memory views retain storage by default and expose pointers, dimensions, and strides. Borrowed and unchecked paths give callers explicit control over lifetimes and validation. Parallel execution and synchronization belong to the caller.

[Reusable block I/O](docs/streaming.md) exposes physical chunk/shard geometry,
worker-local transfer contexts, and caller-executed adaptive or manual streams.
Runtime tuning uses measured transfers and explicit memory/concurrency limits;
it contains no device labels or background execution pool.

## Build

The initially qualified toolchain is Linux, Clang 18 with matching `clang-scan-deps`, CMake 3.28 or newer, and Ninja 1.11 or newer. Default builds use HDF5, zlib, Zstandard, C-Blosc and private JSON parsing. [Dependency commands](docs/storage-dependencies.md) include system and private-prefix installation.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The native test suite needs no Python. Optional interoperability tests independently generate and reopen H5AD and Zarr files with Python AnnData. See the [H5AD encoding subset](docs/h5ad.md) and [backed storage contracts](docs/storage.md).

Use `cmake --workflow --preset core-only` for a dependency-free build. `hdf5-only` and `zarr-only` qualify single-backend packages. `import anndata.core;` remains the smallest in-memory interface; `anndata.backed`, `anndata.streaming`, `anndata.h5ad`, and `anndata.zarr` are independently importable named modules.

Consumers can use `add_subdirectory()` or install the package and link `anndata::anndata` through `find_package(anndata CONFIG REQUIRED)`. `import anndata.core;` provides the independently importable core; `import anndata;` is the convenience facade. Module sources and CMake metadata are installed so consumers build BMIs with their own compatible toolchain. There is no public umbrella header and no requirement for `import std;`.

## Design and evidence

- [Architecture and compatibility matrix](docs/architecture.md)
- [Extension contracts and qualification requirements](docs/extension-contract.md)
- [Development, installation, and validation](docs/development.md)
- [Memory access and ownership](docs/memory.md)
- [Native records and compound H5AD elements](docs/records.md)
- [Compact selection and direct metadata validation](docs/selection-validation.md)
- [H5AD encodings and file access](docs/h5ad.md)
- [Backed storage, structural edits and mappings](docs/storage.md)
- [Reusable transfers, parallel block streams and runtime tuning](docs/streaming.md)
- [Storage dependency installation commands](docs/storage-dependencies.md)
- [Validation results and tested reference versions](docs/validation.md)
- [Measured performance and compilation behavior](docs/performance.md)
- [H5AD I/O and optional-module measurements](docs/h5ad-performance.md)
- [Backed storage and compilation measurements](docs/storage-performance.md)
- [Shard batching and bounded runtime-tuning measurements](docs/completeness-performance.md)
- [Matched Python and C++ runtime and memory comparisons](docs/head-to-head-performance.md)
- [Gathered Zarr read correction and paired measurements](docs/gather-read-performance.md)

AnnData and Scanpy conventions determine data alignment and biological metadata behavior. The C++ interface uses native ownership, pointers, explicit copies, and lightweight typed access rather than reproducing Python mechanics.

Licensed under the [MIT license](LICENSE).
