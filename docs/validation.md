# Foundation validation

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
