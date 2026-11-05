# AnnData++ naming cutover

The clean publication candidate uses `anndatapp` for its C++ modules,
namespace, CMake package and targets, installed libraries, and project-owned
build options. The `AnnData` type and standardized storage encodings retain
their existing names. This is a naming migration, with no numerical or storage
algorithm changes.

Validation used committed source base
`b94bfcc71203c1f14769c4848567f8b328ae191a`, Clang 18.1.3 with the matching
dependency scanner, CMake 3.28.3, and Ninja 1.11.1. Pre-existing uncommitted
feature work is excluded from this publication receipt.

| Configuration or check | Result |
|---|---|
| Debug native suite | 20/20 passed |
| ASan + UBSan native suite | 20/20 passed |
| Core-only package | 12/12 passed |
| HDF5-only package | 16/16 passed |
| Zarr-only package | 16/16 passed |
| Shared package | 20/20 passed |
| Python interoperability | 7/7 passed: semantics, H5AD, storage, streaming and completeness, including threaded gates |
| Installed package identity | New package, feature variables and module sources verified; retired package and targets absent from a fresh prefix |
| Private implementation rebuild | Nine public BMIs and two consumer objects unchanged; restored binary hashes matched |

Every native configuration includes the installed multi-consumer test. Those
consumers import the enabled public modules and perform real file I/O where a
storage backend is enabled. The installed archive was also checked for retired
C++ namespace symbols; none were present.

The Python reference retained the pinned development packages and genuine
Python imports. The optional Scanpy record-interoperability gate was not rerun:
Scanpy was absent from the available reference environment. Native record tests
passed in the configurations above.

Builds used two jobs. The restricted-runner sanitizer settings were
`ASAN_OPTIONS=detect_leaks=0` and `UBSAN_OPTIONS=halt_on_error=1`; Python Zarr
checks used the documented `ANNDATAPP_INTEROP_ZARR_WAKEUP=1` test-only workaround.
The private rebuild probe retained its 30-second per-build cap. These are
correctness and package-boundary checks, not runtime or compilation speedup
measurements.

The historical planning and result files were preserved byte-for-byte, as were
archived patches. Historical reports retain the source names, paths, commands,
hashes, and measured revisions they originally recorded. Reproduce those
campaigns at their cited revisions; their original names do not provide
compatibility imports in the current package. Live setup instructions and
current benchmark source/tooling use the renamed identity.
