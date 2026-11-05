# Architecture and compatibility

The library implements reusable biological data structures and storage conventions. Scientific algorithms, model runtimes, and a parallel execution engine are outside its responsibility.

## Core model

`Array` owns dense multidimensional or compressed sparse storage. Its public typed access descriptors expose direct pointers, shape, and strides. CSR and CSC preserve numerical dtype, sparse orientation, duplicate entries, explicit zeros, and independent index/offset widths. `Column` and `Table` supply ordered heterogeneous metadata, UTF-8 indexes, categorical dictionaries, and nullable values. `Element` carries nested typed metadata without forcing values through JSON or floating point. `AnnData` composes these into two aligned axes and the established slots.

Large owners are move-only. Explicit `copy()` allocates independent data. Dataset slices capture read-only bindings and selectors while retaining the referenced storage. Writes through an existing numerical alias may be visible; replacing an owning slot does not retarget an acquired descriptor. Arbitrary selections can be represented lazily even when no contiguous pointer can describe the result.

Arrays, strings, and metadata rows share `Axis`: inline all/affine mappings and
retained immutable gathers. Regular ranges remain compact through composition;
irregular mappings preserve requested order and repetitions. Direct
`obs_column()`/`var_column()` acquisition avoids building unrelated table view
descriptors, while full table snapshots keep their existing binding semantics.

AnnData keeps observation and variable axes, optional X, layers, obsm/varm, obsp/varp, obs/var, uns, and raw. X and layers are two-dimensional. Multidimensional annotations align their first dimension; pairwise arrays align their first two dimensions. Raw retains a separate variable axis and follows observation slicing. Generic arrays and tables remain useful without AnnData.

References: [AnnData encoding specification](https://anndata.readthedocs.io/en/stable/fileformat-prose.html), [AnnData 0.13.4 behavior](https://github.com/scverse/anndata/blob/0.13.4/src/anndata/_core/anndata.py), [raw semantics](https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.raw.html).

## Module and implementation boundaries

`anndatapp.core` owns the public data API. Its arrays, strings, and data interface partitions organize that module; they are not independently importable modules. `anndatapp.backed` adds backend-neutral lazy descriptors, caller-buffer reads, mappings and structural edits. `anndatapp.h5ad` and `anndatapp.zarr` add format operations and options. `anndatapp` re-exports the core, backed API and enabled format modules as the ordinary user entry point.

The private `anndatapp.backed:storage` implementation partition defines the shared storage contract, session and encoding machinery. It is scanned by CMake but is neither a public import nor an installed interface. Backend implementations depend on this contract; public interfaces do not depend on backend headers. Implementation units can change without invalidating public module artifacts or recompiling consumer objects; compilation reports verify the actual dependency behavior.

The package installs one binary plus interface sources and CMake module metadata. Its configuration creates one consumer-local module-interface provider behind `anndatapp::anndatapp`, so multiple executables share the same interface build. This avoids duplicate synthesized module rules in CMake 3.28.3 while preserving ordinary `find_package()` consumption. Static, shared, and core-only configurations use the same public target. Consumers compile their own BMIs; those artifacts are not portable across arbitrary compilers, flags, or standard libraries. Separate exported format-component targets require additional qualification.

Public owners use erased out-of-line implementation. Small templates provide typed memory views. HDF5, Arrow, JSON, codec, and other backend headers belong in implementation boundaries. HDF5 and Zarr are enabled by default and independently optional; disabling both leaves a dependency-free core and backend-neutral descriptors. Numerical loops on acquired buffers require no backend dispatch or library synchronization.

## Reuse decisions

`anndatapp.streaming` imports the backed API and adds caller-executed block plans
and runtime adaptation. Geometry and worker-local `TransferContext` resources
belong to `anndatapp.backed`. Tuning implementation stays out of the core module;
neither module owns an execution pool. See [the transfer contract](streaming.md).

- [HDF5](https://github.com/HDFGroup/hdf5): its C library sits behind the private H5AD adapter. AnnData element codecs are separate from generic HDF5 mechanics.
- [tatami](https://github.com/tatami-inc/tatami), [tatami_hdf5](https://github.com/tatami-inc/tatami_hdf5): optional matrix-access adapters/reference. Their matrix abstractions do not supply the complete annotated data or serialization model.
- [anndata-rs](https://github.com/scverse/anndata-rs): architectural reference for backed access and collections, not a runtime dependency or proof of complete AnnData compatibility.
- Native Zarr candidates were evaluated before selecting a unified local v2/v3 adapter. The implementation shares AnnData codecs across formats, uses private nlohmann JSON parsing and the established zlib, Zstandard and C-Blosc libraries, and reads supported indexed shards by byte range. Native alternatives remain possible behind the same storage boundary; none defines the public memory model.
- [Arrow/Parquet](https://arrow.apache.org/docs/cpp/parquet.html): optional interchange and later spatial infrastructure. Preserve native AnnData table encodings rather than replacing them with Parquet conventions.

## Compatibility matrix

This matrix separates representable core data from complete file/schema compatibility. H5AD support covers the explicit subset in [h5ad.md](h5ad.md); backed H5AD and Zarr v2/v3 support the bounded contracts and codec subset in [storage.md](storage.md). Composed containers and domain validators remain future work.

The dtype inventory covers numerical/bool arrays, arbitrary-dimensional packed
UTF-8 string arrays, nullable/categorical metadata columns, and native records
with numerical/text members. Records have field-buffer and interleaved
representations, including explicit pointer-and-length text references; see
[records.md](records.md). H5AD supports the declared record subset. Zarr record
codecs, nested/subarray members, complex/half-precision arrays, and ragged/Awkward
values require later work. Native numerical memory and preferred serialization
byte order are separate properties: HDF5 converts at the transfer boundary.

| Ecosystem | Kind | Foundation | Required later work |
|---|---|---|---|
| [AnnData](https://anndata.readthedocs.io/en/stable/fileformat-prose.html) | Encodings in HDF5/Zarr | Standard slots, supported in-memory elements, eager/backed H5AD and local Zarr v2/v3 subset, structural updates | Extend element/codec coverage, remote stores and additional backend qualification |
| [MuData](https://mudata.readthedocs.io/stable/io/spec.html) | Multimodal container | Reusable arrays/tables | Modalities, order, shared axes, observation/variable maps |
| [TreeData](https://treedata.readthedocs.io/en/stable/fileformat.html) | Tree-bearing AnnData extension | Reusable attributes | Trees, node/edge attributes, alignment, versioned HDF5/Zarr representations |
| [AnnDataSet](https://snapatac2.scverse.org/version/2.8/api/_autosummary/snapatac2.read_dataset.html) | SnapATAC2 linked .h5ads collection | Component primitives | Backed linked datasets, relative paths, relocation/missing-component handling |
| [SpatialData](https://spatialdata.scverse.org/en/latest/api/data_formats.html) | Versioned Zarr spatial container | Arrays/tables | Multiscale rasters, labels, points, shapes, Parquet, coordinate systems/transforms |
| [GRnnData](https://cantinilab.github.io/GRnnData/) | AnnData conventions | Generic network-bearing slots | Typed network accessors and validation |
| [scVelo](https://scvelo.readthedocs.io/en/latest/VelocityBasics.html)/[CellRank](https://cellrank.readthedocs.io/en/latest/api/developer.html) | AnnData conventions | Layers, pairwise arrays, probabilities, metadata | Versioned convention adapters |
| [scirpy](https://scirpy.scverse.org/en/stable/data-structure.html)/[AIRR](https://docs.airr-community.org/en/stable/datarep/rearrangements.html) | AIRR standards plus ragged AnnData data | Ordinary tables | Ragged records, Awkward form/buffers, AIRR mappings |
| [Scanpy](https://scanpy.readthedocs.io/en/stable/)/[epiScanpy](https://episcanpy.readthedocs.io/en/latest/basic_usage.html) | AnnData conventions | Supported generic slots | Optional convention validators |
| [Pertpy](https://pertpy.readthedocs.io/en/stable/)/[scvi-tools](https://docs.scvi-tools.org/en/stable/) | AnnData/MuData conventions | Supported generic elements | Schema adapters and multimodal coverage |
| [CELLxGENE Discover](https://cellxgene.cziscience.com/docs/032__Contribute%20and%20Publish%20Data) | Versioned H5AD submission schemas | Metadata primitives | Ontology/schema validation |

CELLxGENE Census uses TileDB-SOMA and is a distinct future integration decision.

## Compatibility hazards and progression

Serialization must preserve dtype, endianness, ordering, string encoding, categorical dictionary/order/missing codes, nullable masks, sparse widths/duplicates, and raw's independent variables. Structured and ragged data require actual typed representations. Unknown extension subtrees may be retained through backend-specific opaque handles where faithful preservation is possible; unsupported cross-backend conversions must report diagnostics. Never silently discard unknown values or stringify them for convenience.

HDF5 variable-length strings need correct reclamation, filters may require plugins, and backend thread-safety does not follow from the in-memory API. Zarr v2/v3, codecs, sharding, and store behavior need separate qualification. TreeData serialization versions, linked collection paths, and SpatialData's multiple versioned components require pinned fixtures.

Delivered progression: in-memory foundation; H5AD codecs and bidirectional independent round trips; backed/block access and local Zarr; native records; reusable transfers and bounded runtime tuning. The completeness pass adds shard-aware gathered writes and allocation-free numerical destinations. The next format milestones are composed MuData/TreeData/AnnDataSet, followed by typed biological schemas and SpatialData. Before each format implementation, follow the [extension contract](extension-contract.md), pin its upstream specifications and reference versions, and expand the explicit tested matrix.
