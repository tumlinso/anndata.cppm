# Native records

`RecordArray` represents an ordered set of named scalar fields with a common
outer shape. It is a move-only core owner; `RecordArrayView` retains its field
bindings. Records can occur in `Element` mappings, including AnnData `uns`.
They do not replace a dataframe: records have no row index, nullable columns,
or category dictionaries.

## Field buffers and interleaved records

`RecordArray::with_shape(shape)` starts a field-buffer owner. `set(name, Array)`
and `set(name, StringArray)` move existing dense numerical or packed UTF-8
owners into it without copying their buffers. Every field must have the same
outer shape. Replacing a field preserves its position; existing views retain
the previous binding. Numerical aliases still observe numerical writes.

An interleaved layout specifies record size/alignment and the ordered members'
names, physical kinds, offsets, and numerical dtypes. Its members may be native
numerical scalars, inline fixed-capacity UTF-8, or `Utf8Ref`, a plain pointer and
length. Numerical types are the canonical fixed-width integer types, `bool`,
`float`, and `double`. The layout is immutable after construction.

```cpp
import anndatapp;

struct Row { float score; anndatapp::Utf8Ref gene; };
```

Use `sizeof(Row)`, `alignof(Row)`, and `offsetof(Row, member)` when describing
live native records. `borrow_interleaved()` retains no allocation;
`share_interleaved()` retains the supplied ownership token;
`adopt_interleaved()` calls the supplied deleter. The retained token or deleter
must cover both the records and their referenced text. Signed strides are in
bytes, and the pointer identifies the first logical record.

The caller supplies live members of the declared canonical types. Checking
offsets and alignment cannot establish an external object's lifetime or prove
that a referenced string is readable. Library-owned interleaved storage
establishes the supported members' lifetimes, but does not create an arbitrary
user-defined `Row` object. Raw bytes do not imply a checked `Row*` conversion.

## Acquire once, then compute

`view.member<T>(name)` and `owner.edit_member<T>(name)` resolve the name, type,
alignment, and retained ownership during acquisition. `RecordMemberView<T>`
exposes the first member pointer, extents, and byte strides. Its one-dimensional
`operator[]`/`ptr(i)` uses byte-stride addressing without bounds checks or
runtime type dispatch. `flat_ptr(i)` explicitly unravels multidimensional
indices; use byte strides directly when controlling a multidimensional loop.

Members in successive structs are distinct scalar subobjects. `data()` points
to the first one; it does not promise a `T[]` spanning later records.
`record_bytes()` exposes retained record bytes and their actual layout.

For field-buffer owners, `array(name)` and `string_array(name)` return the
existing array views. They reject interleaved storage rather than silently
materializing it. `numeric_bytes(name)` exposes numerical members in either
representation. `text(name)` acquires a text field once and exposes its strings.

`unsafe_edit_record_bytes()` and `unsafe_edit_text_refs()` deliberately expose
writable representation data. Rebinding a text pointer transfers no ownership.
The caller then controls its validity, lifetime, aliasing, and synchronization.
`validate()` checks representational invariants when requested; no checks or
synchronization are inserted into acquired numerical loops.

## Explicit copies

- `copy()` preserves the physical representation and makes independent data.
- `copy_fields()` produces independent dense arrays and packed string arrays.
- `copy_interleaved(layout)` packs into the requested layout and owns a copied
  text arena. Copied `Utf8Ref` values point into that arena, not the source.

Text exceeding an inline member's capacity causes an error. Pointer addresses,
padding bytes, and original HDF5 member offsets are not logical data to preserve.
Fixed UTF-8 strips trailing zero padding. Internal NUL bytes remain part of a
native value, but logical text ending in NUL cannot be converted to that fixed
representation without loss and is rejected. HDF5 string writes reject embedded
NUL according to their existing encoding contract.
No arbitrary pointer fields, `std::string` members, nested records, or array-valued
members are inferred from a C++ struct.

## H5AD and backed access

H5AD uses the established `rec-array` / `0.2.0` compound encoding from the pinned
[AnnData implementation](https://raw.githubusercontent.com/scverse/anndata/0.13.4/src/anndata/_io/specs/methods.py).
Numerical widths, field order, outer shape, values, and per-field byte order
are preserved. Text is normalized to logical UTF-8 and written through HDF5
string types. Default reads produce field buffers. Interleaving is an explicit
copy, and compressed or encoded file data is not presented as zero-copy memory.

`BackedElement::field_names()` inspects a record schema;
`read_field(name)` materializes one numerical or string-array element;
`read()` materializes the complete record. Reading a single field does not
materialize other fields, though a compressed compound chunk may still need
to be read and decoded in full. Materialized owners survive session closure.
Existing structural replacement and export operations accept complete records.

Eager record codecs use a normal 1 MiB staging target and a private 64 MiB
single-record cap. Variable-length reads admit pointer metadata, allocation
headers, and live text before allocation, shrink a rejected prefix, and reuse
the successful prefix/budget. Final owned output and HDF5's internal codec and
cache allocations are outside this staging bound. Larger individual values
require backed access with an explicitly larger scratch budget. Failed partial
VLEN reads release their allocations before retry or propagation of an error.

Zarr record codecs remain unsupported. Attempts to write or export records to
Zarr fail before destination replacement. Other supported Zarr encodings retain
their existing behavior.

The supported record subset covers primitive numerical and text members.
Unsupported nested/subarray members, non-string variable-length data, complex
or half-precision fields, and unknown encoding versions produce contextual
diagnostics. HDF5 byte layout and arbitrary NumPy dtype padding are not promised
to round-trip byte-for-byte.
