# Validation

## H5AD milestone

The [H5AD validation record](../tests/results/h5ad-validation.json) identifies final source hashes, commands and coverage. Local qualification uses Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1 and HDF5 C 1.10.10. The [H5AD encoding matrix](h5ad.md) states the supported subset and upstream limitations.

| Check | Result |
|---|---|
| Native HDF5-enabled workflow | 6/6 CTest groups passed |
| Dependency-free core-only workflow | 5/5 passed; no HDF5 in the example's dynamic dependencies |
| Shared-library workflow | 6/6 passed |
| Release build and CTest | 6/6 passed |
| ASan + UBSan build and CTest | 6/6 passed, including installed consumers |
| Pinned AnnData 0.13.4 interoperability workflow | 8/8 passed, including the existing 13 in-memory scenarios and independent bidirectional file tests |
| AnnData 0.12.10 compatibility mode | Shared-subset file round trip passed |
| Independent final-source sanitizer review | Nine targeted reproductions passed; no remaining confirmed defects in the reviewed scope |

Installed-package tests build multiple consumers together in a C++-only project, covering `anndata.core`, the convenience facade, and the H5AD module. HDF5-enabled consumers actually read and write files. Static, shared, core-only and sanitizer installations pass. Additional independent probes qualified sibling-directory package discovery and a downstream library's exported dependency.

File tests independently inspect encodings/dtypes with h5py and read C++ output using Python AnnData. Cases include compact sparse integer widths, unsigned categorical codes, raw and empty dimensions, UTF-8 arrays, nullable sentinels, selected views, physical byte order, explicit native conversion, compression, validation opt-out and path-specific failures. Limits for raw without X and foreign-endian sparse data are recorded rather than counted as universal Python interoperability.

The installed Project Control run-free context adapter failed before a usable workflow handle was returned. Its expired attempt was released through the supported recovery API after root observed producer completion. [The workflow receipt](../planning/h5ad-workflow-receipt.json) distinguishes this recovery from outcome acceptance; the results here are local evidence. GitHub CI has been authored but not run remotely.

## Historical in-memory foundation

The first foundation was qualified locally on Linux with Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1, and C++20 without language extensions. The [validation record](../tests/results/validation.json) identifies the source inputs and actual commands. GitHub CI is configured but was not executed remotely in this session.

| Check | Result |
|---|---|
| `cmake --workflow --preset native-check` | 4/4 CTest groups passed; no Python required |
| `cmake --workflow --preset interop-check` | 5/5 groups passed, including 13 semantic scenarios against AnnData 0.13.4 |
| Release configure/build/CTest | 4/4 groups passed |
| ASan + UBSan configure/build/CTest | 4/4 groups passed, including installed-package consumer |
| Same final semantic driver with AnnData 0.12.10 | 13/13 scenarios passed |
| Separate installed client using `import anndata;` | Configured, built and ran successfully; the regular installed CTest independently imports `anndata.core` |

The semantic oracle creates real H5AD fixtures in Python, independently reopens them, and compares corresponding C++ in-memory outcomes. Coverage includes dense/CSR/CSC selections, reordered/repeated/empty selections, aligned arrays and tables, nullable and categorical metadata, raw's independent variables, table-valued raw varm, category colors, and scalar dtype preservation. This is not C++ file-I/O interoperability.

Native tests cover pointer identity, move-only ownership, explicit deep copies, retained lifetimes, custom deleters, const external buffers, independent sparse widths, strides and gathers, construction/validation boundaries, explicit validation opt-out, and unchecked structure access. Independent review reproduced defects before repairs and retested those contracts; it additionally exercised randomized dense and compressed sparse selection cases.

LeakSanitizer was disabled for the local sanitizer runs because it cannot operate under this execution sandbox's ptrace. AddressSanitizer and UndefinedBehaviorSanitizer remained enabled. The ordinary sanitizer preset keeps the compiler's normal defaults for execution outside that sandbox.

These observations do not claim support for all AnnData encodings or formats. See the [compatibility matrix and representation limits](architecture.md) and [performance report](performance.md). Local command evidence is distinct from a Todo workflow outcome acceptance receipt.
