# A2 128 KiB geometry probe preparation

This directory contains a one-run candidate-only probe derived from the preserved 64 KiB recipe. It reuses the exact source fixture `/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry64k/source-512.zarr` and its fixture identity. Source and writer scratch are each 131072 bytes. A single attempt is capped at 30 seconds and 536870912 intercepted libc read bytes and write bytes each. The exact A2 candidate archive and paired96 profiled archive plus codec and allocator probes are hash-checked during the driver build.

The failed 64 KiB output remains preserved at `geometry64k/output-a2.zarr`; this recipe writes a separate destination `geometry128k/output-a2-128k.zarr`. The run script refuses any pre-existing destination or record directory. Do not retry a failed/partial attempt without further authorization.
