#!/usr/bin/env python3
"""Independently reopen benchmark writes after their measurement checkpoint."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from zarr.storage import LocalStore


def equal(a, b, path: str) -> None:
    if sp.issparse(a) or sp.issparse(b):
        assert sp.issparse(a) and sp.issparse(b), f"{path}: sparse representation changed"
        assert a.format == b.format and a.shape == b.shape, f"{path}: sparse layout changed"
        for key in ("data", "indices", "indptr"):
            equal(getattr(a, key), getattr(b, key), f"{path}/{key}")
    elif isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=path)
    elif isinstance(a, pd.Index) or isinstance(b, pd.Index):
        pd.testing.assert_index_equal(a, b, exact=True, obj=path)
    elif isinstance(a, Mapping) or isinstance(b, Mapping):
        assert isinstance(a, Mapping) and isinstance(b, Mapping), f"{path}: mapping changed"
        assert set(a) == set(b), f"{path}: keys changed"
        for key in a:
            equal(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (np.ndarray, np.generic)) or isinstance(b, (np.ndarray, np.generic)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, f"{path}: dtype or shape changed"
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert type(a) is type(b) and a == b, f"{path}: scalar changed"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unchanged_store(source: Path, output: Path, first, count) -> dict:
    metadata = json.loads((source / "X/zarr.json").read_text())
    shards = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    separator = metadata["chunk_key_encoding"]["configuration"].get("separator", "/")
    assert separator == "/", "fixture requires slash-separated default chunk keys"
    touched = {f"X/c/{i}/{j}" for i in range(first[0] // shards[0], (first[0] + count[0] - 1) // shards[0] + 1)
               for j in range(first[1] // shards[1], (first[1] + count[1] - 1) // shards[1] + 1)}
    left = {str(p.relative_to(source)): p for p in source.rglob("*") if p.is_file()}
    right = {str(p.relative_to(output)): p for p in output.rglob("*") if p.is_file()}
    assert (set(left) ^ set(right)) <= touched, "unrelated store files added or removed"
    checked = 0
    for key in left:
        if key not in touched:
            assert key in right and digest(left[key]) == digest(right[key]), f"{key}: unrelated bytes changed"
            checked += 1
    changed = [key for key in sorted(touched) if key not in left or key not in right or digest(left[key]) != digest(right[key])]
    return {"unchanged_files_checked": checked, "touched_shards": sorted(touched), "changed_shards": changed}


def validate(case: str, source: Path, output: Path) -> dict:
    assert source.resolve() != output.resolve(), "source and output must be distinct"
    is_zarr = case.startswith("zarr3_")
    read = (lambda p: ad.read_zarr(LocalStore(str(p)))) if is_zarr else ad.read_h5ad
    before, after = read(source), read(output)
    assert before.shape == after.shape, "AnnData shape changed"
    for slot in ("obs", "var", "obsm", "varm", "obsp", "varp", "uns"):
        equal(getattr(before, slot), getattr(after, slot), slot)
    # AnnData 0.13 exposes X again as layers[None]; compare X separately below.
    equal({k: before.layers[k] for k in before.layers if k is not None},
          {k: after.layers[k] for k in after.layers if k is not None}, "layers")
    assert before.raw is not None and after.raw is not None, "raw missing"
    for key in ("X", "var", "varm", "obs_names"):
        equal(getattr(before.raw, key), getattr(after.raw, key), f"raw/{key}")
    expected = before.X
    physical = {}
    if is_zarr:
        first, count = ((0, 0), (256, 512)) if case == "zarr3_full_write" else ((224, 448), (96, 128))
        assert not sp.issparse(expected), "write fixture requires dense X"
        expected = expected.copy()
        expected[first[0]:first[0] + count[0], first[1]:first[1] + count[1]] = np.float32(7.5)
        physical = unchanged_store(source, output, first, count)
    else:
        with h5py.File(output, "r") as f:
            x = f["X"]
            assert x.chunks is None and x.compression is None, "full write changed contiguous uncompressed X contract"
            assert x.attrs["encoding-type"] == "array" and x.attrs["encoding-version"] == "0.2.0"
            physical = {"x_chunks": x.chunks, "x_compression": x.compression, "file_bytes": output.stat().st_size}
    equal(expected, after.X, "X")
    return {"phase": "independent_write_validation", "case": case, "status": "passed",
            "shape": list(after.shape), "dtype": str(after.X.dtype),
            "x_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(after.X))).hexdigest(),
            "sparse_layer_format": after.layers["sparse_layer"].format,
            "raw_shape": list(after.raw.shape), "physical": physical}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("h5ad_eager_write_dense", "zarr3_full_write", "zarr3_partial_write"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if os.environ.get("ANNDATA_INTEROP_ZARR_WAKEUP") == "1":
        raise RuntimeError("head-to-head validation requires normal asyncio, without polling")
    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
    print(json.dumps(validate(args.case, args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
