#!/usr/bin/env python3
"""Make a small gzip-compressed H5AD carrying one numeric rec-array."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def make(path: Path, fields: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ad.AnnData(
        obs=pd.DataFrame(index=["obs-0"]),
        var=pd.DataFrame(index=["var-0"]),
    )
    data.write_h5ad(path, compression="gzip", convert_strings_to_categoricals=False)
    dtype = np.dtype([(f"field_{j}", "<f8") for j in range(fields)])
    rows = np.empty(8192, dtype=dtype)
    pattern = np.arange(256, dtype=np.float64)
    for j in range(fields):
        rows[f"field_{j}"] = j * 100000.0 + np.resize(pattern, 8192)
    with h5py.File(path, "r+") as f:
        record = f["uns"].create_dataset(
            "probe", data=rows, chunks=(256,), compression="gzip", compression_opts=4
        )
        record.attrs["encoding-type"] = "rec-array"
        record.attrs["encoding-version"] = "0.2.0"
    with h5py.File(path, "r") as f:
        record = f["uns/probe"]
        assert record.shape == (8192,) and record.chunks == (256,)
        assert record.compression == "gzip" and record.dtype.names == tuple(dtype.names or ())
        assert record.dtype.itemsize * record.shape[0] <= 512 * 1024
        for j, name in enumerate(dtype.names or ()):
            np.testing.assert_array_equal(record[name], j * 100000.0 + np.resize(pattern, 8192))
    print(f"fixture={path}")
    print(f"fields={fields} logical_payload_bytes={rows.nbytes} chunks={8192 // 256}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--fields", type=int, choices=(2, 8), required=True)
    args = parser.parse_args()
    make(args.output, args.fields)


if __name__ == "__main__":
    main()
