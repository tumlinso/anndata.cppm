# anndata.cppm

A standalone C++20 library for annotated biological data, with named modules and direct control of numerical memory.

The first milestone is an **in-memory foundation**: dense arrays, CSR/CSC, heterogeneous metadata, annotated matrices, and read-only dataset slices. H5AD serialization is the next milestone. Python, Cellerator, Torch, HDF5, and Arrow are not runtime or build dependencies of this foundation.

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
}
```

Owners are move-only. `copy()` creates independent data. Typed memory views retain storage by default and expose pointers, dimensions, and strides. Borrowed and unchecked paths give callers explicit control over lifetimes and validation. Parallel execution and synchronization belong to the caller.

## Build

The initially qualified toolchain is Linux, Clang 18 with matching `clang-scan-deps`, CMake 3.28 or newer, and Ninja 1.11 or newer. On Ubuntu 24.04 the scanner is provided by `clang-tools-18`.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The native test suite needs no Python. Optional semantic comparisons use Python AnnData to write and reopen actual H5AD fixtures; they do not claim C++ file I/O support.

Consumers can use `add_subdirectory()` or install the package and link `anndata::anndata` through `find_package(anndata CONFIG REQUIRED)`. `import anndata.core;` provides the independently importable core; `import anndata;` is the convenience facade. Module sources and CMake metadata are installed so consumers build BMIs with their own compatible toolchain. There is no public umbrella header and no requirement for `import std;`.

## Design and evidence

- [Architecture and compatibility matrix](docs/architecture.md)
- [Development, installation, and validation](docs/development.md)
- [Memory access and ownership](docs/memory.md)
- [Validation results and tested reference versions](docs/validation.md)
- [Measured performance and compilation behavior](docs/performance.md)

AnnData and Scanpy conventions determine data alignment and biological metadata behavior. The C++ interface uses native ownership, pointers, explicit copies, and lightweight typed access rather than reproducing Python mechanics.

Licensed under the [MIT license](LICENSE).
