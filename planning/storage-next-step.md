# Dense-transfer performance work and remaining priorities

This is an authored proposal, not Todo authority state. The current Project Control workflow supplied no usable handle; the authority and existing active tasks were preserved.

The focused dense-transfer work is implemented. The original campaign was checkpointed and stopped; its exact report remains locally preserved and ignored. Runtime codec, allocator, backend-call and I/O counters confirmed repeated row-wise chunk work. The implementation now coalesces forward rectangles around destination codec units, reuses bounded staging storage and reserves backend workspace, including sharded writes. No public API, cache, execution pool or acquired numerical loop changed.

The [compact evidence](../benchmarks/results/storage-summary.json) records geometry-faithful small pairs and four single-run 64 MiB confirmations, each with 30-second and explicit I/O caps. The larger runs verified complete values and axes. These observations are bounded evidence, not general speed or memory guarantees. Raw reports, profiles and generated stores are ignored; reproducible measurement sources remain tracked.

Remaining performance priorities are sparse access across the compressed orientation, small-budget metadata chunk selection, and cases that still use the gather/stride fallback. The supplied CSR dataset's column selection remains slow; it was not changed by the dense fix. New work should start with one small faithful probe, operation/byte counts and profiling, then validate any candidate against the existing interoperability and sanitizer gates. Stop repeating an established bottleneck; do not restart the full campaign without an evidence gap.

A small read-only physical chunk/shard geometry descriptor would also help callers partition storage work. Its API should follow the measured transfer needs without exposing HDF5 or Zarr implementation classes.

Composed formats and biological schema adapters remain separate future milestones. None is claimed or implemented by this performance step.
