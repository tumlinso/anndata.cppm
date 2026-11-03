# anndata.cppm

A standalone C++20 library for annotated biological data, with named modules and direct control of numerical memory.

The library provides an **in-memory AnnData foundation and native H5AD I/O**: dense arrays, CSR/CSC, heterogeneous metadata, annotated matrices, read-only dataset slices, and versioned file codecs. Python, Cellerator, Torch, and Arrow are not runtime dependencies. The default build uses the HDF5 C library; a core-only build requires no storage backend.

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

## Build

The initially qualified toolchain is Linux, Clang 18 with matching `clang-scan-deps`, CMake 3.28 or newer, and Ninja 1.11 or newer. The default build also requires pkg-config and HDF5 >=1.10. On Ubuntu 24.04 install `clang-18 clang-tools-18 cmake ninja-build pkg-config libhdf5-dev`.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The native test suite needs no Python. Optional interoperability tests independently generate and reopen H5AD files with Python AnnData. See the [tested encoding subset and I/O contracts](docs/h5ad.md).

Use `cmake --workflow --preset core-only` for a dependency-free core build. Its `import anndata;` facade provides the in-memory API; H5AD operations require the default enabled build.

Consumers can use `add_subdirectory()` or install the package and link `anndata::anndata` through `find_package(anndata CONFIG REQUIRED)`. `import anndata.core;` provides the independently importable core; `import anndata;` is the convenience facade. Module sources and CMake metadata are installed so consumers build BMIs with their own compatible toolchain. There is no public umbrella header and no requirement for `import std;`.

## Design and evidence

- [Architecture and compatibility matrix](docs/architecture.md)
- [Development, installation, and validation](docs/development.md)
- [Memory access and ownership](docs/memory.md)
- [H5AD encodings and file access](docs/h5ad.md)
- [Validation results and tested reference versions](docs/validation.md)
- [Measured performance and compilation behavior](docs/performance.md)
- [H5AD I/O and optional-module measurements](docs/h5ad-performance.md)

AnnData and Scanpy conventions determine data alignment and biological metadata behavior. The C++ interface uses native ownership, pointers, explicit copies, and lightweight typed access rather than reproducing Python mechanics.

Licensed under the [MIT license](LICENSE).
