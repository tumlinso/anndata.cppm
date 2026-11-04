# Extending the package

The qualified foundation is a native data library: arrays, tables, records,
nested elements, AnnData alignment, and explicit backed transfers. New formats
should compose these pieces and retain their ownership and execution contracts.
The [compatibility matrix](architecture.md) distinguishes supported encodings
from future containers and biological conventions.

## Logical data and memory

AnnData keeps two annotated axes; generic dense arrays remain N-dimensional.
CSR/CSC retain orientation, independent integer widths, duplicates and explicit
zeros. Preserve labels and ordering, null distinctions, categories, `.raw` and
unknown-data diagnostics before adding convenience accessors.

Owners are move-only and explicit `copy()` allocates independent data. Retained
views keep their allocation alive; borrowing transfers no lifetime obligation.
Direct acquisition exposes actual pointers and strides. A gather, compressed
chunk, or remote object is not contiguous memory merely because it has a shape.

`ArrayView::copy_into(ArrayEdit, Validation)` copies dense numerical selections
into existing storage without allocating a numerical destination or casting
values. Source and destination must not overlap. Metadata acquisition may
allocate; subsequent numerical loops perform ordinary loads and stores. Use
explicit `copy()` where overlap requires independent data.

## Formats and backends

Keep HDF5, Zarr, JSON and codec headers behind compiled implementation boundaries.
Core types do not depend on Python, analysis frameworks, storage libraries, or
downstream applications. New named format modules may import the core/backed
API; internal partitions remain implementation components, not public modules.

Use established encoding names and versions. Pin the upstream specification and
reference implementation before writing a codec, independently generate input,
and independently reopen C++ output. Extend the existing element codecs where
possible; do not create a second numerical/table model for each container.

The private backed contract handles native-endian numerical transfers and
backend capabilities. File byte order remains separate from native access.
Composed formats will need their own container relationships and dataset-root
handling; those are format decisions, not reasons to generalize every core type
in advance. Domain conventions such as Scanpy results normally need typed
accessors or validation, not another storage engine.

Untouched extension data can remain in its original store. Unsupported encodings
and conversions must produce contextual errors rather than losing fields or
stringifying them. The eager model and cross-backend export do not promise
arbitrary opaque attribute preservation; see [H5AD](h5ad.md) and
[backed storage](storage.md).

## Transfers and execution

Callers own scheduling, synchronization and output storage. One transfer context
belongs to each worker. Structural mutation and explicit close follow existing
descriptor-invalidation rules. Native loops on acquired buffers never acquire
library locks or invoke type-erased accessors per value.

Physical chunks and shards define conflict units independently of logical
selections. Numerical writes complete within their existing calls; there is no
implicit cross-call transaction or background flush. A shard replacement is not
a whole-dataset transaction or a durability guarantee.

Charge library-owned grouping, index, packing and codec staging to the declared
scratch budget. Keep caller buffers and third-party internal allocations
separately identified. Raise an explicit capacity diagnostic when indivisible
work cannot fit; do not silently materialize a full dataset.

## Qualification

New adapters must pass native tests without Python where applicable, independent
format round trips, affected sanitizer/concurrency tests, and installed-consumer
checks. Preserve core-only and optional-backend configurations. CMake installs
module sources and metadata; compatible consumers build their own BMIs.

Keep public interfaces small and compiled implementations private. Record import
costs when changing interfaces and prove that private edits leave public BMIs and
consumer objects untouched. Start performance work with bounded, representative
geometry and call/byte/allocation counts; report exact source and fixture identity.

Current qualification targets Linux with Clang 18 and matching dependency
scanning. Additional compilers and unsupported encodings remain explicit future
coverage. Extension readiness does not imply completion of MuData, TreeData,
linked collections, SpatialData, ragged records, or every AnnData dtype.
