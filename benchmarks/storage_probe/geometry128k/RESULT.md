# A2 128 KiB geometry probe result

## Outcome

The sole authorized attempt completed successfully. `write_zarr` returned success in 0.092244209 s, below the 30 s limit. The independent Python check found exact float32 X values and exact obs/var indexes. The output occupies 945,144 physical bytes in 78 files; its tree manifest SHA256 is `2225efe8f5468d05714f656f2d788a506b8f29da50e4c2638ca554fd0f3d4d2f`. No retry was made.

## Conditions and identities

- Source: `/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/source-512.zarr`, 512×512 float32; X inner chunks `[32,64]`, shards `[256,512]`; obs and var UTF8 index chunks `[64]`. Source X metadata SHA256 `03811ff9d76656d9ad549544416c78164f42568e2113d033861a93572565208a`; fixture manifest SHA256 `94e8a285fdaa449daf06824a6652f4424951c06c31a822df97386c3b18475723`. All 31 fixture files matched the pre-run source hash record on post-run verification.
- Candidate A2 archive SHA256 `b910afcf8e0f9fc1eca3f18015943117706e639512db5cabc3ff1fd01c8dc9a6`; paired96 profiled archive SHA256 `ccb2343c7226e9ebcfd78b3c3a04b9af530ab40ee3f717e00c233deb1e1f03f8`.
- Source scratch and writer scratch: 131,072 bytes each. Intercepted libc read/write caps: 536,870,912 bytes per direction. Rejections: 0 / 0.
- Destination: `/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k/output-a2-128k.zarr`. Its X chunk grid is `[64,64]`; obs and var index chunk grids are `[512]`. Output X is plain regular chunks with `bytes` and Zstd level 3.
- Driver SHA256 `82df061790644e03f507b432a979bec9833051f848c11ebf0dfe8f4fa2d591cd`; allocator preload SHA256 `740aaf7c85d6089b93210e487c268dbf1cf36930f0f352ff2398bf18834132cb`; capped IO/codec preload SHA256 `45ff53af1e7d5014278965215832113f965348711a94c03a16ce657c40c8305d`.

## Counters for the `write_zarr` ROI

- Logical X backend: 32/32 reads and 32/32 writes completed; 262,144 float32 elements and 1,048,576 bytes in each direction.
- Intercepted IO: read 3,883,289 bytes (`read` 1,739,425 / 526 calls; `pread` 2,143,864 / 512); write 2,367,967 bytes (`write` 726,400 / 158 calls; `writev` 1,641,567 / 128).
- Zstd decode: 478 calls, 3,301,212 input bytes, 5,308,024 output bytes, 5,814,635 ns. Zstd encode: 272 calls, 4,259,392 input bytes, 2,355,564 output bytes, 11,688,706 ns. Blosc calls: zero.
- `malloc`: 227,209 calls; 181,418,738 cumulative requested/successful bytes. Regular C++ `new` is included through `malloc`; it is not added again. No calloc, realloc, or aligned-allocation calls were observed. These are cumulative requested bytes, not peak live memory.
- Perf group: 86,338,796 ns enabled/running; 177,276,641 user cycles and 263,467,899 user instructions.

## Raw evidence and reproduction

The complete record is `record-a2-128k/`: stdout/stderr, exit status, run/build identities, fixture copy, source file hashes, output file inventory, output chunk metadata, and `validation.json`. The build and run scripts are `build_geometry128k.sh` and `run_geometry128k_once.sh`; the run script rejects pre-existing output or record paths.

Driver build:

```bash
/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k/build_geometry128k.sh
```

Single measured run:

```bash
/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k/run_geometry128k_once.sh \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k/build-run-v2 \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/source-512.zarr \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k/output-a2-128k.zarr \
  /tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k/record-a2-128k
```

No repository source or build artifact was changed by this probe.
