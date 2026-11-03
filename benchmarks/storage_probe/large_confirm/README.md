# Candidate B 64 MiB storage confirmation

Prepared in an isolated /tmp root. No repository files or source fixtures are edited. The assigned parent GO authorizes exactly these four candidate cases, once each, sequentially:

1. full Zarr2 export
2. full Zarr2 reindex
3. full Zarr3 export
4. full Zarr3 reindex

Each candidate process is wrapped in a 30 second timeout. Its capped ROI is write_zarr for export, or edit.apply() plus flush() for reindex, using 64 MiB source/writer scratch and a zero decoded cache. The interposer enforces independent 256 MiB ceilings for intercepted libc reads and writes. Any nonzero return, timeout, rejected request, cap reached, missing counter, unexpected action/format, or failed full validator ends the sequence. The claim files prevent a second attempt for the same case. Partial outputs and records are kept; nothing is deleted or retried.

The operations mirror benchmarks/storage.cpp: export writes the full backed view; reindex retains observation rows [0, 4095) and all variables; a 32×32 (export) or 31×32 (reindex) reference read occurs before timing/capping. Reindex works on an exact disposable copy. The original comparison fixtures are read-only inputs and are hash-checked before and after each case.

The independent Python/Zarr validator reads and exactly compares all X values plus all observation and variable index labels. It checks expected X shape/dtype and axis-group attributes, and records source/destination X and index-array geometries. It runs after the timed/capped candidate process, has its own 30 second timeout, and its file reads/tree hashing are outside the interposer cap.

## Limits and uncapped work

IO_BUDGET covers only the eight dynamic libc entrypoints wrapped by codec_counts_limited_256.c: read, pread, readv, preadv, write, pwrite, writev, and pwritev while the gate is open. It does not claim total kernel/device traffic and can miss direct syscalls, memory mapping, and non-interposed libc internals. Read/write totals and per-entrypoint calls/bytes are retained in each interposer-counters.json and raw stderr.

Input evidence hashes read every source fixture file outside the capped ROI. Reindex copy I/O and copy hashes are recorded separately in copy-evidence.json; Python validation performs full-array reads and hashes source/target files outside the cap. These file/logical byte estimates are not physical-device measurements, and filesystem cache may serve reads. The baseline-compatible reference tile also warms some input chunk/shard data before the timed call. No cache clearing is performed.

The campaign baseline was three uncapped observations per operation (seconds): Zarr2 export 190.062, 191.115, 190.373; Zarr2 reindex 191.703, 193.046, 191.986; Zarr3 export 186.559, 186.759, 185.503; Zarr3 reindex 191.130, 193.607, 191.698. Those observations are in benchmarks/results/storage-baseline.json. Candidate B is measured once per case with an interposer and hard caps, so timing comparisons are directional, not identical-instrumentation pairs.

## Build and run

The driver build links only the frozen B archive, whose identity is checked against archive SHA256 213270a9e5adfa2565a97b2a761f5573fb7afcd758fc8c8f3b7d47a9f70c8299 and source-manifest SHA256 39d5e66667103eff4de17f8e4c70b2166108a942e71c6304f8b2ad3f2377b89e. Build is isolated under build-v2/ and does not modify the candidate archive.

Run the cases one at a time, inspecting each case-result.json before starting the next:

    ROOT=/tmp/anndata-storage-b-confirm-20261103-mnXjYU
    PYTHON=/home/tumlinson/anndata.cppm/.venv/bin/python
    "$PYTHON" "$ROOT/case_runner.py" export-zarr2 "$ROOT/results/export-zarr2"
    "$PYTHON" "$ROOT/case_runner.py" reindex-zarr2 "$ROOT/results/reindex-zarr2"
    "$PYTHON" "$ROOT/case_runner.py" export-zarr3 "$ROOT/results/export-zarr3"
    "$PYTHON" "$ROOT/case_runner.py" reindex-zarr3 "$ROOT/results/reindex-zarr3"

Do not continue after a failed case and do not rerun a claimed case. A case result with status ok includes candidate return status, operation/open timing, intercepted calls/bytes, independent correctness, physical metadata, geometry, source preservation, and uncapped validation/copy I/O estimates. Raw stdout/stderr and manifests remain beside it.
