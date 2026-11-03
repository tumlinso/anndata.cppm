# H5AD interoperability

`import anndata;` exposes common H5AD operations in the default build. `import anndata.h5ad;` independently exposes the same operations and the core. The private backend uses the HDF5 C library. Python is only a development reference.

```cpp
import anndata;

int main() {
    auto data = anndata::read_h5ad("input.h5ad");
    auto selected = data.slice(anndata::indices{2, 0}, anndata::all);
    anndata::WriteOptions options;
    options.compression = anndata::Compression::gzip;
    anndata::write_h5ad("selected.h5ad", selected, options);
}
```

`read_h5ad()` is eager: the file closes before it returns. Numerical descriptors retain the resulting native allocations and expose their ordinary pointers and strides. Use `open_h5ad()` for [lazy backed access and updates](storage.md); `anndata.zarr` supplies Zarr v2/v3 operations. External collection links remain subsequent work.

## Encoding boundary

The implementation follows the [AnnData encoding specification](https://anndata.readthedocs.io/en/stable/fileformat-prose.html) and the pinned AnnData 0.13.4 reference codecs. Tests inspect actual on-disk attributes and dtypes independently with h5py; Python AnnData reads C++ output.

| Encoding | Version | Representation |
|---|---|---|
| `anndata`, `raw` | 0.1.0 | Standard slots and raw's independent variable axis; optional X |
| `dict`, `null` | 0.1.0 | Nested mappings and distinct null elements |
| `array`, `numeric-scalar` | 0.2.0 | bool, signed/unsigned 8–64-bit integers, float32/64 |
| `string`, `string-array` | 0.2.0 | UTF-8 scalar text and dense n-dimensional packed strings |
| `rec-array` | 0.2.0 | Ordered primitive numerical and text fields with a common outer shape |
| `csr_matrix`, `csc_matrix` | 0.1.0 | Values, indices, indptr, orientation and independent integer widths |
| `dataframe` | 0.2.0 | Ordered heterogeneous columns and named string row index |
| `categorical` | 0.2.0 | Codes, complete category dictionary, ordering, missing code |
| `nullable-integer`, `nullable-boolean`, `nullable-string-array` | 0.1.0 | Values and missing mask; string NA/NaN discriminator |

The supported scope preserves duplicate row labels, requested slice ordering, sparse duplicates and explicit zeros, nullable distinctions, category dictionaries, nested metadata, aligned arrays/tables, and raw. Array layout is materialized as required for file storage; file chunking and compression are not promises about in-memory strides.

Explicit rank-zero C++ `Array` and `StringArray` owners use the established `array` and `string-array` encodings, retaining their kinds when reread by C++. An actual `Element` scalar uses its scalar encoding. Python AnnData 0.13.4 promotes its zero-dimensional arrays to scalars on write and can also expose these values as scalars on read; Python round trips promise the supported value/dtype semantics, not retention of C++ owner classification.

A plain string `Column` and a one-dimensional `StringArray` use the same established wire encoding. Outside a dataframe's column context, reopening may expose a `StringArray`; values and shape agree. The writer does not invent a format attribute to preserve a distinction the encoding does not carry.

Raw data without X can retain an independent raw variable table in C++ and on disk. AnnData 0.13.4's eager Raw constructor substitutes the parent variable metadata when raw X is None. Independent raw metadata in that case therefore has no Python-semantic round-trip guarantee; the library preserves the stored metadata rather than inventing numerical values or discarding the axis.

Structured Scanpy result arrays use [native records](records.md). Compound
members may be supported primitive numbers or UTF-8 text; nested records,
subarray members, and non-string variable-length data remain unsupported.
Numerical widths, field order, shape, values, and serialization byte order are
preserved. Original member offsets, padding, and fixed-string capacity are
normalized rather than preserved as a binary layout.

Unknown encodings or versions, legacy unversioned data, complex and
half-precision arrays, ragged/Awkward data, and nullable floating-point columns
produce contextual errors. Such data is not silently removed or converted into
a different encoding. Eager loading into the in-memory model does not retain
arbitrary extra HDF5 attributes, matching ordinary AnnData loading behavior.
[Backed updates](storage.md) can preserve untouched extension data and same-store
opaque attributes; export rejects unsupported preservation. This is a
logical-data round trip for the declared encodings, not a byte-identical HDF5
rewrite.

## Memory and validation

Dense numeric reads target their final uninitialized allocation. Compatible writes pass acquired buffers to HDF5; arbitrary strided/gathered dense selections use bounded packing. Selected sparse output is materialized when necessary. Compressed data, strings, and foreign-endian values require conversion; this is not a memory-mapped or universal zero-copy reader.

All numerical memory is native-endian for typed pointer access. Small serialization hints retain the input byte order separately for values, sparse indices/offsets, and scalars. `FileByteOrder::preserve` is the default on write; `native`, `little`, and `big` explicitly override it. Moves, views and copies preserve hints. HDF5 performs byte-order conversion at the transfer boundary without retaining a second numerical buffer.

SciPy 1.17.1 rejects foreign-endian sparse numerical data even when HDF5 stores it correctly. Use `FileByteOrder::native` when converting such files for that Python stack. Preserving the original byte order preserves this upstream limitation; it does not establish Python compatibility for an input Python itself cannot load.

Read and write options default to `Validation::full`. Use `Validation::none` for trusted data to skip structural content scans; transfer sizes, dtypes, and required format metadata still must be usable. Trusted input must also satisfy C++ representation invariants, including canonical boolean bytes. Full loading canonicalizes numerical boolean bytes before typed access. There are no per-element library checks in a pointer loop after loading. Packed string views similarly expose byte and offset buffers; mutable offset access is explicitly unsafe.

## File behavior and diagnostics

Writing defaults to `WriteMode::truncate`, replacing the destination directly. A write failure can leave a partial output file. Use `exclusive` to refuse replacement. No temporary-file transaction is performed implicitly.

Compression defaults to `Compression::none`. `Compression::gzip` uses `gzip_level` (default 4). Readers use filters available to the linked HDF5 installation; missing filters produce an `unavailable_filter` diagnostic.

Errors carry a stable code and file/element path. Unsupported encoding/version/dtype failures are distinct from ordinary I/O failures. Names containing path separators, embedded NUL text, reserved dataframe names, and conflicting index/column names are rejected where the upstream representation cannot preserve them.

The library adds no numerical execution threads or synchronization. Applications control concurrent access and must respect the thread-safety properties of their HDF5 build when making simultaneous I/O calls.
