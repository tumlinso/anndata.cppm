# Architecture and compatibility

The library implements reusable biological data structures and storage conventions. Scientific algorithms, model runtimes, and a parallel execution engine are outside its responsibility.

## Core model

`Array` owns dense multidimensional or compressed sparse storage. Its public typed access descriptors expose direct pointers, shape, and strides. CSR and CSC preserve numerical dtype, sparse orientation, duplicate entries, explicit zeros, and independent index/offset widths. `Column` and `Table` supply ordered heterogeneous metadata, UTF-8 indexes, categorical dictionaries, and nullable values. `Element` carries nested typed metadata without forcing values through JSON or floating point. `AnnData` composes these into two aligned axes and the established slots.

Large owners are move-only. Explicit `copy()` allocates independent data. Dataset slices capture read-only bindings and selectors while retaining the referenced storage. Writes through an existing numerical alias may be visible; replacing an owning slot does not retarget an acquired descriptor. Arbitrary selections can be represented lazily even when no contiguous pointer can describe the result.

AnnData keeps observation and variable axes, optional X, layers, obsm/varm, obsp/varp, obs/var, uns, and raw. X and layers are two-dimensional. Multidimensional annotations align their first dimension; pairwise arrays align their first two dimensions. Raw retains a separate variable axis and follows observation slicing. Generic arrays and tables remain useful without AnnData.

References: [AnnData encoding specification](https://anndata.readthedocs.io/en/stable/fileformat-prose.html), [AnnData 0.13.4 behavior](https://github.com/scverse/anndata/blob/0.13.4/src/anndata/_core/anndata.py), [raw semantics](https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.raw.html).

## Module and implementation boundaries

`anndata.core` owns the public data API. Its arrays and data interface partitions organize that module; they are not independently importable modules. `anndata` re-exports the core as the ordinary user entry point. Future named format modules add specialized controls, while the facade exposes common enabled I/O.

The first package places all interfaces in a single exported CMake target. Consumers install interface sources plus CMake module metadata and compile their own BMIs. A temporary probe with CMake 3.28.3/Clang 18 found installed dependent module targets unreliable when split across exported targets. Separate component targets therefore require qualification before introduction. BMIs are not portable distribution artifacts across arbitrary compilers, flags, or standard libraries.

Public owners use erased out-of-line implementation. Small templates provide typed memory views. HDF5, Arrow, JSON, codec, and other backend headers belong in implementation boundaries. No storage dependency is required by the current core.

## Reuse decisions

- [HDF5](https://github.com/HDFGroup/hdf5): use its C library behind a private adapter in the H5AD milestone. Implement AnnData codecs separately from generic HDF5 mechanics.
- [tatami](https://github.com/tatami-inc/tatami), [tatami_hdf5](https://github.com/tatami-inc/tatami_hdf5): optional matrix-access adapters/reference. Their matrix abstractions do not supply the complete annotated data or serialization model.
- [anndata-rs](https://github.com/scverse/anndata-rs): architectural reference for backed access and collections, not a runtime dependency or proof of complete AnnData compatibility.
- [z5](https://github.com/constantinpape/z5) and other native Zarr implementations: candidates to qualify against versioned AnnData/Zarr fixtures before dependency selection.
- [Arrow/Parquet](https://arrow.apache.org/docs/cpp/parquet.html): optional interchange and later spatial infrastructure. Preserve native AnnData table encodings rather than replacing them with Parquet conventions.

## Compatibility matrix

This matrix separates representable core data from complete file/schema compatibility. Current file I/O support is **not implemented**.

The initial dtype inventory covers numerical/bool arrays and one-dimensional UTF-8 metadata columns. Multidimensional string ndarrays, complex/half-precision arrays, compound records, and ragged/Awkward values require later representation work. A string-valued table is not claimed to substitute for a string ndarray encoding.

| Ecosystem | Kind | Foundation | Required later work |
|---|---|---|---|
| [AnnData](https://anndata.readthedocs.io/en/stable/fileformat-prose.html) | Encodings in HDF5/Zarr | Standard slots and supported in-memory elements | H5AD read/write, backed storage, separate Zarr v2/v3 qualification |
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

Progression: in-memory foundation; H5AD codecs and bidirectional independent round trips; backed/block access and Zarr; composed MuData/TreeData/AnnDataSet; typed biological schemas and SpatialData. Before each format implementation, pin its upstream specifications and reference versions and expand the explicit tested matrix.
