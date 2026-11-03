# Backed record HDF5 call-count probe

This bounded probe counts calls reaching `H5Dread` and `H5Dwrite` while one
native backed operation is active. For each call it projects the native logical
buffer size as `H5Tget_size(memory_type) * selected_memory_points`. The probe
does not measure compressed bytes, device/filesystem I/O, or how many compressed
chunks the HDF5 filter pipeline decodes.

`make_fixture.py` writes AnnData 0.13.4 H5AD metadata and adds a gzip-compressed
numeric `rec-array` at `/uns/probe`: 8,192 rows, 2 or 8 float64 fields, and
256-record chunks. The complete numerical payload is 128 KiB or 512 KiB. The
native driver opens the H5AD store before resetting/enabling optional hooks by
`dlsym`; only `read_field("field_0")`, `read()`, or backed H5AD export is
counted. It checks every returned field value exactly after disabling counts.
Per-dataset counts include source and destination datasets with the same path
under separate read/write totals.

Build the driver as a private consumer target against the current HDF5-enabled
`anndata::anndata` target, linking `${CMAKE_DL_LIBS}` and adding this directory
to its private include path. Build `interposer.c` independently:

```sh
cc -std=c11 -O2 -fPIC -shared \
  -I/usr/include/hdf5/serial -Ibenchmarks/record_io_probe \
  benchmarks/record_io_probe/interposer.c \
  -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -ldl -pthread \
  -o build/record-io-probe/librecord_io_probe.so
```

Run from the Python reference environment with the exact driver and interposer
paths. The harness applies a 30-second timeout to each native process and runs
each of the six cases once:

```sh
python benchmarks/record_io_probe/run.py \
  --driver build/storage-records/record_io_probe \
  --interposer build/record-io-probe/librecord_io_probe.so
```

The machine-readable raw result, fixture identities, output identities, exact
commands, and unmodified process output are written to `artifacts/record-io-counts.json`.
Repeated execution is unnecessary unless setup fails. No performance timing is
collected.
