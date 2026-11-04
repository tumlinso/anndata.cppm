# Reusable backed block I/O

`import anndata.streaming;` provides caller-executed block streams and imports
the backed data API. The ordinary `anndata` facade also exports it. The core
module retains its dependency-free memory interface.

`BackedArray::storage_geometry()` describes physical source storage separately
from the selected logical shape. `data`, `indices`, and `offsets` describe the
three sparse components independently. Chunk/shard extents are nominal and edge
regions can be smaller. Affine selection axes expose source start and step;
gathers are explicitly non-affine. Geometry never implies directly addressable
memory or zero-copy decompression.

## Transfer contexts

`prepare_transfer(scratch_bytes)` creates a move-only, retained worker context.
Zero uses the session's available scratch budget. Context preparation inspects
the schema and obtains private backend resources; repeated block transfers do
not repeat ordinary schema discovery. Preparation must be serialized with
structural mutation. Each concurrent worker owns a context and output buffers.

`read_into(start, count, DenseBuffer)` and `write_from(start, count, DenseBuffer)`
operate in captured logical selection coordinates. Dense buffers expose signed
byte strides; their pointer denotes the first logical value. Copies and codec
work occur at transfer boundaries, never in the application's acquired numerical
loops. Existing whole-array APIs keep their behavior.

CSR/CSC transfers read consecutive native major-axis blocks with the complete
minor axis. `requirements(start, count)` returns NNZ and the original numerical,
index, and offset widths before the caller allocates or admits output storage.
`read_into(..., SparseBuffer)` retains duplicates, explicit zeros, and orientation.
Sparse structure is never densified. Gathered sparse axes and parallel sparse
structural writes are outside this interface.

Contexts retain their source session after owner destruction. Explicit close or
affected structural replacement invalidates transfers. Caller synchronization
is required for structural edits and external modifications. Unsafe changes to
sparse indices or offsets also invalidate cached requirements by contract:
discard and recreate the context before reading the modified structure.

For example, a single caller can reuse one float32 buffer with manual settings:

```cpp
import anndata;

int main() {
    auto disk = anndata::open_h5ad("input.h5ad");
    auto io = disk.x().prepare_transfer();
    anndata::StreamOptions options;
    options.mode = anndata::StreamMode::manual;
    anndata::BlockStream blocks(io.geometry(), options);
    auto storage = anndata::Array::empty<float>({options.max_block_bytes/sizeof(float)});
    float* p = storage.edit().dense<float>().data();
    while (auto block = blocks.next()) {
        io.read_into(block->start, block->count,
                     anndata::DenseBuffer::from(p, block->count));
        // Work directly with p and block->count; no transfer machinery runs here.
        blocks.complete(*block, block->estimated_bytes);
    }
}
```

The example assumes float32 input; other dtypes require matching output storage.
An adaptive caller reports each transfer's duration through `BlockCompletion`.
Parallel callers issue several requests, execute them with independent contexts
and buffers, then serialize completion reporting back to the stream.

The scratch cap covers library-owned numerical payload staging, not output
storage, schema metadata, HDF5 caches, or third-party codec internal allocations.
`stats()` separates backend transfer calls and native transfer bytes from owned
scratch capacity. These are not physical device-byte counters. Payload reuse
does not imply that every metadata or third-party allocation disappears.

The initial reusable Zarr pipeline covers unsharded C-order native-endian raw
buffers and the common gzip/zlib, Zstd, and Blosc encodings. Sharded transfers,
foreign-endian conversion, transpose, and more complex codec pipelines use the
existing bounded implementation after releasing reusable pipeline capacity.
They retain format support but do not yet reuse all codec or shard staging.

## Parallel execution

`BlockStream` issues requests; the application executes and completes them.
Issuance, completion reporting, and configuration are caller-serialized. The
library creates no execution threads and adds no synchronization to numerical
pointer loops. Outstanding requests can complete out of order.

Dense Zarr write requests retain exclusive ownership of a physical chunk, or a
whole shard for sharded arrays, until every fragment of that unit completes.
Different streams or external writers must coordinate their own overlapping
units. Payload-only writes preserve metadata; creation, resizing, and structural
updates require workers to drain first. There is no cross-process transaction
or universal filesystem atomicity guarantee.

HDF5 capabilities reflect the linked library. Thread safety does not imply
parallel HDF5 execution. HDF5 write streams are limited to one outstanding
request; numerical work can still run on independent buffers in application
workers. Independently opened serial HDF5 processes are not concurrent writers.

## Adaptation and manual control

New streams default to adaptive mode, within explicitly reserved output,
scratch, and worker limits. Decisions use completed useful transfers, not disk
type labels, mount names, or hidden calibration reads. Logical byte throughput
can reflect compression, cache behavior, and selection costs; it is not a disk
bandwidth claim.

`BlockCompletion` accepts useful bytes and I/O duration. Common monotonic start
and finish timestamps additionally establish overlap for concurrency tuning.
Without interval evidence, only block sizing can adapt. Include transfer time
and exclude application numerical computation from the reported I/O interval.

The controller compares epochs of at least 16 completions, changes one parameter
at a time, retains improvements exceeding 5%, and rolls back degradation over
20%. Stable settings are reviewed every 64 completions; two consecutive shifts
over 25% reopen exploration. Small streams can finish before evidence supports
any tuning. Settings and limit reasons remain inspectable.

Manual mode fixes target block size and in-flight limits, subject to backend
capabilities and physical conflict units. It ignores completion timing unless
statistics are requested. Drain outstanding requests before changing modes or
configuration; reserved capacities cannot grow implicitly. Manual mode provides
deterministic coverage and an escape from unsuitable adaptive choices.

Already-owned dense regions retain their tiling until completion. New target
sizes apply to subsequent regions, preserving exact coverage during mode changes.
Exploration samples nearby settings; it does not guarantee a global optimum.

Sparse output size is unknown until its requirements are read. If capacity is
insufficient, explicitly split an unexecuted request into fewer major-axis rows
or columns. A single over-capacity row/column requires larger caller storage or
the existing lower-level selection API; no partial completion is reported.

See [streaming performance](streaming-performance.md) for measured workloads,
source identities, budgets, and remaining implementation limitations.
