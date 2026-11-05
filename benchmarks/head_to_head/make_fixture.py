#!/usr/bin/env python3
"""Create deterministic AnnData H5AD and sharded Zarr3 comparison fixtures."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import warnings


for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")


class PollPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick() -> None:
            loop.call_later(0.010, tick)

        loop.call_later(0.010, tick)
        return loop


if os.environ.get("ANNDATAPP_INTEROP_ZARR_WAKEUP") == "1":
    asyncio.set_event_loop_policy(PollPolicy())

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from numcodecs import blosc
from zarr.storage import LocalStore

SEED = 20261104
DEFAULT_SIZES = (1024, 4096)
ZARR_CHUNKS = (64, 128)
ZARR_SHARDS = (256, 512)


def x_block(rows, cols):
    r = np.asarray(rows, dtype=np.uint64)[:, None]
    c = np.asarray(cols, dtype=np.uint64)[None, :]
    code = ((r * 0x9E3779B1) ^ (c * 0x85EBCA77) ^
            ((r + 1) * (c + 3) * 0xC2B2AE3D)) & 1023
    keep = ((r * 13 + c * 29 + r * c * 7 + 17) % 100) < 9
    return np.where(keep, code.astype(np.float32) / np.float32(64), np.float32(0))


def x_matrix(n: int) -> np.ndarray:
    axis = np.arange(n, dtype=np.uint64)
    return np.ascontiguousarray(x_block(axis, axis), dtype=np.float32)


def sparse_layer(n: int) -> sp.csr_matrix:
    locations = np.random.default_rng(SEED + n)
    values = np.random.default_rng(SEED + n + 1)
    return sp.random(
        n, n, density=0.03, format="csr", dtype=np.float32,
        random_state=locations,
        data_rvs=lambda count: values.integers(1, 32, count, dtype=np.int32).astype(np.float32),
    )


def metadata(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.DataFrame({
        "batch": pd.Categorical([f"batch{i % 4}" for i in range(n)],
                                 categories=[f"batch{i}" for i in range(4)]),
        "condition": pd.Categorical(["treated" if i % 2 else "control" for i in range(n)],
                                     categories=["control", "treated"]),
        "row_order": np.arange(n, dtype=np.int32),
    }, index=pd.Index([f"cell{i}" for i in range(n)], name="cell_id"))
    var = pd.DataFrame({
        "feature_class": pd.Categorical(
            [("enzyme", "receptor", "transport", "other")[i % 4] for i in range(n)],
            categories=["enzyme", "receptor", "transport", "other"]),
        "coordinate": np.arange(n, dtype=np.int32),
    }, index=pd.Index([f"gene{i}" for i in range(n)], name="gene_id"))
    return obs, var


def make_anndata(x, layer: sp.csr_matrix, obs: pd.DataFrame, var: pd.DataFrame) -> ad.AnnData:
    result = ad.AnnData(X=x, obs=obs, var=var, layers={"sparse_layer": layer})
    result.raw = result[:, :min(64, result.n_vars)].copy()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sparse_expectation(matrix, csr: bool) -> dict:
    digest = hashlib.sha256()
    blocks = []
    n = matrix.shape[0]
    for lo in range(0, n, 256):
        block = matrix[lo:lo + 256, :] if csr else matrix[:, lo:lo + 256]
        for part in (block.data, block.indices, block.indptr):
            digest.update(np.ascontiguousarray(part).tobytes())
        blocks.append({
            "major_start": lo, "major_count": 256, "nnz": int(block.nnz),
            "data_dtype": str(block.data.dtype),
            "index_dtype": str(block.indices.dtype),
            "offset_dtype": str(block.indptr.dtype),
            "offset_count": int(block.indptr.size),
        })
    return {"sha256": digest.hexdigest(), "blocks": blocks}


def expected_records(x: np.ndarray, csr, csc) -> dict:
    n = x.shape[0]
    rows = np.asarray([0, 3, 63, 64, 127, 128, 223, 224, 255, 256,
                       511, 512, 767, 768, n - 2, n - 1], dtype=np.int64)
    cols = (np.arange(64, dtype=np.int64) * 37) % n
    gathered = x_block(rows, cols)
    fused = x.copy()
    for _ in range(8):
        np.multiply(fused, np.float32(0.5), out=fused)
        np.add(fused, np.float32(0.125), out=fused)
    return {
        "shape": list(x.shape), "dtype": str(x.dtype),
        "dense_x_sha256": array_sha256(x),
        "gather": {"rows": rows.tolist(), "columns": cols.tolist(),
                   "repeats": 8, "shape": list(gathered.shape),
                   "sha256": array_sha256(gathered)},
        "fused": {"passes": 8, "sha256": array_sha256(fused)},
        "sparse_csr_rows": sparse_expectation(csr, True),
        "sparse_csc_columns": sparse_expectation(csc, False),
    }


def files(path: Path) -> dict[str, dict[str, int | str]]:
    items = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {
        (item.name if path.is_file() else item.relative_to(path).as_posix()): {
            "bytes": item.stat().st_size, "sha256": sha256_file(item),
        }
        for item in items
    }


def validate_adata(path: Path, expected: np.ndarray, layer: sp.csr_matrix) -> dict:
    data = ad.read_h5ad(path) if path.suffix == ".h5ad" else ad.read_zarr(LocalStore(str(path)))
    x = data.X.toarray() if sp.issparse(data.X) else np.asarray(data.X)
    assert x.shape == expected.shape and np.array_equal(x, expected)
    assert data.obs["batch"].dtype.name == "category"
    assert data.obs["condition"].dtype.name == "category"
    assert data.var["feature_class"].dtype.name == "category"
    assert "sparse_layer" in data.layers and sp.issparse(data.layers["sparse_layer"])
    assert np.array_equal(data.layers["sparse_layer"].toarray(), layer.toarray())
    assert data.raw is not None and data.raw.n_vars == min(64, data.n_vars)
    raw = data.raw.X.toarray() if sp.issparse(data.raw.X) else np.asarray(data.raw.X)
    assert np.array_equal(raw, expected[:, :min(64, expected.shape[1])])
    return {
        "shape": list(x.shape), "dtype": str(x.dtype), "nnz": int(np.count_nonzero(x)),
        "layer_format": "csr" if sp.isspmatrix_csr(data.layers["sparse_layer"]) else "other",
        "raw_shape": list(raw.shape),
    }


def write_h5ad(path: Path, x, layer, obs, var, expected) -> dict:
    data = make_anndata(x, layer, obs, var)
    data.write_h5ad(path, compression=None, convert_strings_to_categoricals=False)
    del data
    validation = validate_adata(path, expected, layer)
    with h5py.File(path, "r") as handle:
        dense = handle["X"] if isinstance(handle["X"], h5py.Dataset) else None
        if dense is not None:
            assert dense.chunks is None and dense.compression is None
        validation["x_storage"] = {
            "kind": "dense" if dense is not None else str(handle["X"].attrs.get("encoding-type", "sparse")),
            "chunks": None if dense is None else dense.chunks,
            "compression": None if dense is None else dense.compression,
        }
        validation["hdf5_x_members"] = {
            key: {"shape": list(handle["X"][key].shape),
                  "chunks": handle["X"][key].chunks,
                  "compression": handle["X"][key].compression}
            for key in handle["X"].keys()
        } if dense is None else {}
    return validation


def write_zarr(path: Path, x: np.ndarray, layer, obs, var, expected) -> dict:
    data = make_anndata(x, layer, obs, var)
    with ad.settings.override(zarr_write_format=3), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        data.write_zarr(LocalStore(str(path)), chunks=ZARR_CHUNKS,
                        convert_strings_to_categoricals=False, consolidate_metadata=False)
    del data

    root = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
    attrs = dict(root["X"].attrs)
    del root["X"]
    root.create_array("X", data=x, chunks=ZARR_CHUNKS, shards=ZARR_SHARDS,
                      compressors=(zarr.codecs.ZstdCodec(level=0),), attributes=attrs)
    zarr.consolidate_metadata(root.store)

    check = zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=False)
    actual = check["X"]
    assert tuple(actual.shape) == expected.shape
    assert tuple(actual.chunks) == ZARR_CHUNKS and tuple(actual.shards) == ZARR_SHARDS
    assert np.array_equal(np.asarray(actual[:]), expected)
    return validate_adata(path, expected, layer) | {
        "x_chunks": list(actual.chunks), "x_shards": list(actual.shards),
        "x_compressors": [type(codec).__name__ for codec in actual.compressors],
    }


def package_versions() -> dict:
    versions = {name: importlib.metadata.version(name) for name in
                ("anndata", "numpy", "pandas", "scipy", "h5py", "zarr", "numcodecs")}
    versions["hdf5"] = h5py.version.hdf5_version
    return versions


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_fixture.py ROOT [SIZE ...]")
    root = Path(sys.argv[1]).resolve()
    sizes = tuple(map(int, sys.argv[2:])) if len(sys.argv) > 2 else DEFAULT_SIZES
    if not sizes or any(n < 512 or n % 256 for n in sizes):
        raise SystemExit("sizes must be positive multiples of 256 and at least 512")
    manifest_path = root / "fixture-manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite manifest: {manifest_path}")
    root.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        if (root / str(n)).exists():
            raise SystemExit(f"refusing to overwrite fixture directory: {root / str(n)}")

    zarr.config.set({"async.concurrency": 1, "threading.max_workers": 1})
    blosc.set_nthreads(1)
    manifest = {
        "schema": "anndatapp.head_to_head.fixtures.v1",
        "seed": SEED,
        "sizes": list(sizes),
        "x_formula": "uint64 coordinate mix, 9% deterministic support, 10-bit values scaled by 1/64",
        "sparse_layer": {"format": "csr", "density": 0.03, "value_range": [1, 31]},
        "raw_variables": 64,
        "h5ad_x": {"compression": None, "dense_chunks": None},
        "zarr3_x": {"format": 3, "chunks": list(ZARR_CHUNKS), "shards": list(ZARR_SHARDS),
                     "codec": "ZstdCodec(level=0)", "codec_threads": 1},
        "versions": package_versions(),
        "fixtures": {},
    }
    for n in sizes:
        directory = root / str(n)
        directory.mkdir()
        print(f"generating {n}x{n}", flush=True)
        x = x_matrix(n)
        layer = sparse_layer(n)
        obs, var = metadata(n)
        matrix = {}
        for kind in ("dense", "csr", "csc"):
            path = directory / f"{kind}.h5ad"
            stored = x if kind == "dense" else sp.csr_matrix(x) if kind == "csr" else sp.csc_matrix(x)
            matrix[kind] = write_h5ad(path, stored, layer, obs, var, x) | {
                "files": files(path),
            }
            del stored
        csr = sp.csr_matrix(x)
        csc = sp.csc_matrix(x)
        zpath = directory / "sharded.zarr"
        zarr_record = write_zarr(zpath, x, layer, obs, var, x) | {"files": files(zpath)}
        manifest["fixtures"][str(n)] = {"h5ad": matrix, "zarr3": zarr_record}
        expected = expected_records(x, csr, csc)
        (directory / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        del csr, csc
        del x, layer, obs, var

    body = json.dumps(manifest, indent=2, sort_keys=True).encode()
    manifest["manifest_sha256"] = hashlib.sha256(body).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sizes": list(sizes),
                      "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
