# Backend engine correctness probes

[backends.cpp](backends.cpp) is the byte-identical source actually used for the backend review's two bounded engine investigations. Its original local path was `/tmp/anndatapp-backend-review/repro.cpp`. It compiles the private byte engine directly, without public module wrappers, CMake BMIs, Python, HDF5 or a complete AnnData store. The observations therefore apply to the engine paths described here; they do not establish public-wrapper error codes or general upstream format compatibility.

The source creates its own metadata fixtures under the fixed paths `/tmp/anndatapp-backend-review/stores/v2`, `v3` and `nul`. It recreates root metadata at those paths on each run; reserve that directory for this probe. The fixtures have no numerical payload. The probe does not call the export adapter, `prepare_destination`, destination truncation or store deletion. **Destructive destination behavior for an embedded-NUL root is untested.** The review's concern about that path is inferred from frozen source, not from a demonstrated deletion.

Run from the AnnData++ repository root with the frozen source tree available. The reviewed tree is `build/thin-review-20261105/source`, captured from dirty baseline HEAD `d2260c2abb37e9028fa42c95d1712ffcaa282b0e`; HEAD alone is insufficient to identify it. [The backend ledger](../backends-ledger.json) contains SHA-256 hashes of every reviewed backend source file, including `zarr_engine.hpp`, `zarr_engine.cpp`, `zarr_codecs.cpp` and `zarr_sharding.cpp`, which must match for the original-source observation. Do not substitute live production files when reproducing that baseline.

The actual environment used Ubuntu Clang 18.1.3 (`1ubuntu1`), target `x86_64-pc-linux-gnu`, the default system C++ standard library, nlohmann_json 3.11.3, zlib 1.3, Zstd 1.5.5 and Blosc 1.21.5. Headers and link libraries were installed at the system search locations. The dependency versions were checked with:

```sh
clang++-18 --version
pkg-config --modversion nlohmann_json zlib libzstd blosc
```

The original build used `-std=c++20 -O0` and linked `-lzstd -lz -lblosc`. The following uses the preserved source bytes at their repository path, with identical engine files and flags; the executable and generated stores remain outside tracked files:

```sh
mkdir -p /tmp/anndatapp-backend-review
clang++-18 -std=c++20 -O0 -I build/thin-review-20261105/source/src \
  docs/review/probes/backends.cpp \
  build/thin-review-20261105/source/src/zarr_engine.cpp \
  build/thin-review-20261105/source/src/zarr_codecs.cpp \
  build/thin-review-20261105/source/src/zarr_sharding.cpp \
  -lzstd -lz -lblosc -o /tmp/anndatapp-backend-review/repro
/tmp/anndatapp-backend-review/repro
```

The original final probe exited 0 and printed this JSON, preserved locally in `/tmp/anndatapp-backend-review/result.json`:

```json
{
  "consolidated_v2": {
    "after": false,
    "before": true,
    "exception": "Zarr element does not exist"
  },
  "consolidated_v3": {
    "after": false,
    "before": true,
    "exception": "Zarr element does not exist"
  },
  "root_with_embedded_nul_exception": "basic_filebuf::underflow error reading the file: Is a directory"
}
```

For F02, each child `X` exists only in consolidated metadata: the engine's read fallback initially discovers it. `Store::attributes` removes the consolidation before its own child inspection, then throws; the same child is no longer discoverable. `before: true`, the missing-element exception and `after: false` are the reproduced defect expectations. They do not assert that every upstream version supports mutation of consolidated-only child metadata. A fix must either preserve accepted metadata through a successful operation or reject unsupported writing before changing the store.

For F01, the source appends an embedded NUL plus `unrequested` to an existing fixture root, then constructs a read-only `Store` and inspects its root. The captured exception shows that the invalid root reached filesystem/stream work instead of deliberate argument rejection. The exact text is standard-library/platform dependent. The final source catches `std::exception` to report the observation; it does not establish the dynamic exception class or inspect a public `ErrorCode`. A fix should reject the invalid root before I/O with the intended public argument diagnostic, tested separately through public wrappers.

This is an observation program, not a regression test with failure assertions. Exit 0 means the final reporting path completed, **not** that F01/F02 are fixed; inspect the JSON fields. A changed implementation may produce different output, and a new preflight rejection can escape outside the probe's operation-specific catches. No timing, memory-use, sanitizer, concurrency, truncation or source/destination-preservation measurement was performed by this probe.

| Original artifact | SHA-256 |
|---|---|
| `backends.cpp` / original `repro.cpp` source bytes | `eb72728aae623f47a54b281576536163c84cd91cbbd6a68a82b1690b4032085f` |
| Original executable | `e910426961d45c58d8f26f5ec1171a22925fb700c3f9c651b165ff9afb40dde1` |
| Original complete result JSON, including final newline | `040e626913c7f7b0fc4832f917e1c9a8e1004c097b4502af54a4aeac33b23816` |

Executable identity is specific to the original source path, compiler, headers and linked environment; a rebuild at a different path is not promised to reproduce the executable hash. The source, frozen-input manifest, exact command and expected observations provide durable reproduction evidence. The raw executable, result file and generated stores remain local artifacts.
