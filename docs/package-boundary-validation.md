# Package boundary validation

Validated locally on 2026-11-05 with Clang 18, C++20, CMake 3.28 and Ninja.
The [fresh receipt](../tests/results/package-boundary.json) binds these results
to source and log hashes. All changes remain unpublished.

AnnData++ provides boolean, fixed-width integer, FP32 and FP64 storage. Its
active package has no FP16 storage type or CUDA component. Independently
generated Python AnnData FP16 H5AD and Zarr v2/v3 files are rejected with
`unsupported_dtype` during eager reads, backed metadata access and backed reads.

| Check | Passed |
|---|---:|
| Debug native and installed consumer | 20/20 |
| ASan/UBSan | 20/20 |
| Core-only | 12/12 |
| HDF5-only | 16/16 |
| Zarr-only | 16/16 |
| Shared installation | 20/20 |
| Full interoperability, including the three FP16 rejection cases | 30/30 |
| HDF5-only / Zarr-only FP16 rejection | 1/1 / 2/2 |

Each backend workflow passed its installed consumer. Fresh full and HDF5-only
installations expose no CUDA/FP16 API in module/config files or library symbols;
a required CUDA component request fails package discovery. The general module
provider and independent Zarr string-mapping guard remain.

Velocity retains FP16 through its own explicit numeric views and Cellerator's
portable binary16 load/round helper. Its debug and ASan/UBSan suites pass 12/12,
including installed FP16 consumers; an actual Cellerator-integrated FP16
consumer also passes. The helper checks all 65,536 encodings and 390,465 NumPy
conversion-oracle cases. The current velocity receipt is
`Cellerator/src/velocity/tests/results/fp16-ownership.json`.

Focused preprocess, Cellerator and nested CellShard CPU consumers pass.
Historical feature sources and receipts remain byte-for-byte in
`Cellerator/quarantine/anndatapp-extensions/`; staging, unrelated source, and
workflow authority were preserved. No GPU execution, benchmarks, leak
qualification, new CellShard format, commits or pushes are claimed.
