#!/usr/bin/env python3
"""Replace one rec-array UTF-8 value with the invalid byte 0xff."""

import sys

import h5py


if len(sys.argv) != 2:
    raise SystemExit("usage: record_validation_corrupt.py SOURCE.h5ad")

with h5py.File(sys.argv[1], "r+") as file:
    records = file["/uns/record_validation"]
    rows = records[()]
    if rows.dtype.fields is None or "label" not in rows.dtype.fields:
        raise RuntimeError(f"expected compound rec-array, got {rows.dtype}")
    rows["label"][0] = b"\xff"
    records[...] = rows
    print(f"corrupted /uns/record_validation[label][0] in {sys.argv[1]}")
