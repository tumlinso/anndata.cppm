#!/usr/bin/env python3
"""Create small deterministic AnnData fixtures for the V1 comparison."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import warnings

if os.environ.get("ANNDATAPP_INTEROP_ZARR_WAKEUP") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from python_storage import install_restricted_runner_wakeup
    install_restricted_runner_wakeup()

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from zarr.storage import LocalStore


DENSE_SHAPE = (256, 256)
CONTEXT_SHAPE = (2048, 256)
SPARSE_SHAPE = (512, 512)
Z2_CHUNKS = (64, 64)
Z3_CHUNKS = (64, 64)
Z3_SHARDS = (128, 128)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def dense_x(shape: tuple[int, int]) -> np.ndarray:
    r = np.arange(shape[0], dtype=np.uint64)[:, None]
    c = np.arange(shape[1], dtype=np.uint64)[None, :]
    return (((17 * r + 13 * c + (r * c) % 97) % 1009) / 1009.0).astype(np.float32)


def sparse_x() -> sp.csr_matrix:
    n = SPARSE_SHAPE[0]
    r = np.arange(n, dtype=np.int64)[:, None]
    k = np.arange(8, dtype=np.int64)[None, :]
    cols = np.sort((r + k * 61) % n, axis=1).astype(np.int32)
    rows = np.broadcast_to(r, cols.shape)
    values = (1 + (17 * rows + 13 * cols) % 101).astype(np.float32)
    indptr = (np.arange(n + 1, dtype=np.int32) * 8).astype(np.int32)
    matrix = sp.csr_matrix((values.reshape(-1), cols.reshape(-1), indptr), shape=SPARSE_SHAPE)
    matrix.sort_indices()
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    if not np.all(np.diff(matrix.indptr) == 8):
        raise RuntimeError("sparse fixture must have exactly eight entries in every row")
    return matrix


def axes(n_obs: int, n_vars: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.DataFrame({
        "batch": pd.Categorical([f"batch{i % 4}" for i in range(n_obs)],
                                 categories=["batch0", "batch1", "batch2", "batch3"]),
        "observation_order": np.arange(n_obs, dtype=np.int32),
    }, index=pd.Index([f"cell{i}" for i in range(n_obs)], name="cell_id"))
    var = pd.DataFrame({
        "feature_type": pd.Categorical(
            [("gene", "peak", "protein", "other")[i % 4] for i in range(n_vars)],
            categories=["gene", "peak", "protein", "other"]),
        "feature_order": np.arange(n_vars, dtype=np.int32),
    }, index=pd.Index([f"feature{i}" for i in range(n_vars)], name="feature_id"))
    return obs, var


def adata(x: np.ndarray | sp.spmatrix) -> ad.AnnData:
    obs, var = axes(x.shape[0], x.shape[1])
    out = ad.AnnData(X=x, obs=obs, var=var)
    out.uns["v1_fixture"] = {"generator": "deterministic-formula-v1", "seed": 20261105}
    return out


def check_adata(path: Path, expected: np.ndarray | sp.spmatrix, sparse_format: str | None = None) -> dict:
    saved = ad.read_h5ad(path) if path.suffix == ".h5ad" else ad.read_zarr(LocalStore(str(path)))
    x = saved.X
    if sparse_format is None:
        actual = np.asarray(x)
        if actual.dtype != np.dtype(np.float32) or not np.array_equal(actual, expected):
            raise RuntimeError(f"dense values mismatch after writing {path.name}")
    else:
        if not sp.issparse(x) or x.format != sparse_format:
            raise RuntimeError(f"expected {sparse_format.upper()} X after writing {path.name}")
        actual = x.tocsr()
        want = expected.tocsr()
        if (actual.dtype != np.dtype(np.float32) or actual.nnz != want.nnz or
                not np.array_equal(actual.toarray(), want.toarray())):
            raise RuntimeError(f"sparse values mismatch after writing {path.name}")
        if actual.indices.dtype != np.dtype(np.int32) or actual.indptr.dtype != np.dtype(np.int32):
            raise RuntimeError(f"expected int32 sparse indices and offsets in {path.name}")
    if len(saved.obs) != expected.shape[0] or len(saved.var) != expected.shape[1]:
        raise RuntimeError(f"axis lengths mismatch in {path.name}")
    if saved.obs.index.name != "cell_id" or saved.var.index.name != "feature_id":
        raise RuntimeError(f"AnnData axis index names missing in {path.name}")
    if not isinstance(saved.obs["batch"].dtype, pd.CategoricalDtype):
        raise RuntimeError(f"obs categorical metadata missing in {path.name}")
    if not isinstance(saved.var["feature_type"].dtype, pd.CategoricalDtype):
        raise RuntimeError(f"var categorical metadata missing in {path.name}")
    return {"shape": list(saved.shape), "dtype": "float32",
            "obs_columns": list(saved.obs.columns), "var_columns": list(saved.var.columns),
            "x_format": sparse_format or "dense"}


def write_h5ad_dense(path: Path, x: np.ndarray) -> dict:
    adata(x).write_h5ad(path, compression=None, convert_strings_to_categoricals=False)
    # AnnData's writer does not expose dense HDF5 chunk geometry. Replace X
    # while retaining its standard encoding attributes and all other slots.
    with h5py.File(path, "r+") as f:
        attrs = dict(f["X"].attrs)
        del f["X"]
        dataset = f.create_dataset("X", data=x, chunks=(128, 128), compression=None,
                                   dtype=np.float32)
        for key, value in attrs.items():
            dataset.attrs[key] = value
    result = check_adata(path, x)
    with h5py.File(path, "r") as f:
        ds = f["X"]
        if ds.chunks != (128, 128) or ds.compression is not None or ds.dtype != np.dtype(np.float32):
            raise RuntimeError("dense H5AD X did not retain the requested layout")
        result["chunks"] = list(ds.chunks)
        result["compression"] = ds.compression
    return result


def write_h5ad_sparse(path: Path, x: sp.csr_matrix, fmt: str) -> dict:
    stored = x if fmt == "csr" else x.tocsc()
    stored.sort_indices()
    stored.indices = stored.indices.astype(np.int32, copy=False)
    stored.indptr = stored.indptr.astype(np.int32, copy=False)
    adata(stored).write_h5ad(path, compression=None, convert_strings_to_categoricals=False)
    result = check_adata(path, x, sparse_format=fmt)
    with h5py.File(path, "r") as f:
        group = f["X"]
        stored_format = group.attrs.get("encoding-type", "")
        if stored_format != f"csr_matrix" and fmt == "csr":
            raise RuntimeError(f"H5AD X encoding is {stored_format!r}, expected csr_matrix")
        if stored_format != f"csc_matrix" and fmt == "csc":
            raise RuntimeError(f"H5AD X encoding is {stored_format!r}, expected csc_matrix")
        result["encoding"] = str(stored_format)
        result["nnz"] = int(group["data"].size)
        result["entries_per_row"] = 8
        result["index_dtype"] = str(group["indices"].dtype)
        result["offset_dtype"] = str(group["indptr"].dtype)
    return result


def write_zarr2(path: Path, x: np.ndarray) -> dict:
    with ad.settings.override(zarr_write_format=2), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        adata(x).write_zarr(LocalStore(str(path)), chunks=Z2_CHUNKS,
                            convert_strings_to_categoricals=False, consolidate_metadata=True)
    root = zarr.open_group(LocalStore(str(path)), mode="a", zarr_format=2,
                           use_consolidated=False)
    attrs = root["X"].attrs.asdict()
    del root["X"]
    if (path / "X").exists():
        shutil.rmtree(path / "X")
    array = root.create_array("X", data=x, chunks=Z2_CHUNKS, compressor=None,
                              filters=None, attributes=attrs)
    zarr.consolidate_metadata(root.store)
    codecs = tuple(array.compressors)
    if tuple(array.chunks) != Z2_CHUNKS or codecs:
        raise RuntimeError("Zarr2 X must be uncompressed with 64x64 chunks")
    result = check_adata(path, x)
    result.update({"zarr_format": 2, "chunks": list(array.chunks), "compressors": []})
    return result


def write_zarr3(path: Path, x: np.ndarray) -> dict:
    with ad.settings.override(zarr_write_format=3), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        adata(x).write_zarr(LocalStore(str(path)), chunks=Z3_CHUNKS,
                            convert_strings_to_categoricals=False, consolidate_metadata=False)
    root = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
    attrs = dict(root["X"].attrs)
    del root["X"]
    if (path / "X").exists():
        shutil.rmtree(path / "X")
    array = root.create_array("X", data=x, chunks=Z3_CHUNKS, shards=Z3_SHARDS,
                              compressors=(zarr.codecs.ZstdCodec(level=1),), attributes=attrs)
    zarr.consolidate_metadata(root.store)
    if tuple(array.chunks) != Z3_CHUNKS or tuple(array.shards) != Z3_SHARDS:
        raise RuntimeError("Zarr3 X did not retain the requested chunk and shard geometry")
    result = check_adata(path, x)
    result.update({"zarr_format": 3, "chunks": list(array.chunks),
                   "shards": list(array.shards),
                   "compressors": [type(codec).__name__ for codec in array.compressors]})
    return result


def files(path: Path) -> dict[str, dict[str, int | str]]:
    if path.is_file():
        items = [path]
        return {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}}
    items = sorted(p for p in path.rglob("*") if p.is_file())
    return {p.relative_to(path).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in items}


def package_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name)
                for name in ("anndata", "numpy", "pandas", "scipy", "h5py", "zarr", "numcodecs")}
    versions["hdf5"] = h5py.version.hdf5_version
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="fixture output directory")
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    dense = dense_x(DENSE_SHAPE)
    sparse = sparse_x()
    created = {
        "dense.h5ad": (write_h5ad_dense(out / "dense.h5ad", dense), dense),
        "csr.h5ad": (write_h5ad_sparse(out / "csr.h5ad", sparse, "csr"), sparse),
        "csc.h5ad": (write_h5ad_sparse(out / "csc.h5ad", sparse, "csc"), sparse),
        "z2.zarr": (write_zarr2(out / "z2.zarr", dense), dense),
        "z3.zarr": (write_zarr3(out / "z3.zarr", dense), dense),
        "contexts.zarr": (write_zarr2(out / "contexts.zarr", dense_x(CONTEXT_SHAPE)), dense_x(CONTEXT_SHAPE)),
    }
    fixtures = {}
    for name, (layout, value) in created.items():
        path = out / name
        logical = value.toarray() if sp.issparse(value) else value
        fixtures[name] = {**layout, "logical_x_sha256": array_sha256(logical),
                          "files": files(path)}
    manifest = {
        "schema": "anndatapp.v1_performance.fixtures.v1",
        "generator": {"path": "benchmarks/v1/make_fixtures.py",
                      "sha256": sha256(Path(__file__).resolve())},
        "environment": {"python": sys.version.split()[0], "packages": package_versions()},
        "dense_formula": "float32(((17*r+13*c+(r*c)%97)%1009)/1009.0)",
        "sparse_formula": {
            "shape": list(SPARSE_SHAPE), "entries_per_row": 8,
            "columns": "sorted((r + k*61) % 512 for k in 0..7)",
            "value": "float32(1 + (17*r + 13*c) % 101)",
            "indices_dtype": "int32", "indptr_dtype": "int32"},
        "fixtures": fixtures,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "manifest": str(manifest_path),
                      "manifest_sha256": sha256(manifest_path),
                      "fixtures": list(fixtures), "versions": manifest["environment"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
