#!/usr/bin/env python3
"""Create deterministic dense Zarr2/Zarr3 and sparse H5AD probe fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import warnings

if os.environ.get("ANNDATAPP_INTEROP_ZARR_WAKEUP") == "1":
    from python_storage import install_restricted_runner_wakeup

    install_restricted_runner_wakeup()

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from zarr.storage import LocalStore

SHAPE = (4096, 1024)
DENSE_CHUNKS_Z2 = (128, 128)
DENSE_CHUNKS_Z3 = (64, 128)
DENSE_SHARDS_Z3 = (256, 512)
SPARSE_PER_ROW = 32


def dense_values() -> np.ndarray:
    r = np.arange(SHAPE[0], dtype=np.int64)[:, None]
    c = np.arange(SHAPE[1], dtype=np.int64)[None, :]
    return (((17 * r + 13 * c + (r * c) % 97) % 1009).astype(np.float32)
            / np.float32(1009))


def sparse_values() -> sp.csr_matrix:
    m, n = SHAPE
    nnz = m * SPARSE_PER_ROW
    indptr = np.arange(0, nnz + 1, SPARSE_PER_ROW, dtype=np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.int32)
    step = np.arange(SPARSE_PER_ROW, dtype=np.int64) * 127
    for r in range(m):
        j = np.sort((37 * r + step) % n).astype(np.int32)
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        indices[lo:hi] = j
        data[lo:hi] = 1 + (11 * r + 7 * j) % 11
    return sp.csr_matrix((data, indices, indptr), shape=SHAPE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_records(path: Path) -> dict[str, dict[str, int | str]]:
    return {
        file.relative_to(path).as_posix(): {
            "bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in sorted(path.rglob("*")) if file.is_file()
    }


def fnv1a64_update(state: int, data: bytes) -> int:
    for byte in data:
        state = ((state ^ byte) * 1099511628211) & 0xffffffffffffffff
    return state


def sparse_block_hashes(matrix: sp.spmatrix, storage_format: str) -> dict[str, object]:
    axis = 0 if storage_format == "csr" else 1
    extent = SHAPE[axis]
    state = 1469598103934665603
    blocks: list[dict[str, object]] = []
    for start in range(0, extent, 64):
        stop = min(start + 64, extent)
        block = matrix[start:stop, :] if axis == 0 else matrix[:, start:stop]
        if block.format != storage_format:
            block = block.asformat(storage_format)
        block_state = 1469598103934665603
        for values in (block.data, block.indices, block.indptr):
            payload = np.ascontiguousarray(values, dtype="<i4").tobytes()
            block_state = fnv1a64_update(block_state, payload)
            state = fnv1a64_update(state, payload)
        blocks.append({
            "start": start,
            "stop": stop,
            "nnz": int(block.nnz),
            "expected_hash": f"0x{block_state:016x}",
        })
    return {
        "algorithm": "FNV-1a-64; initial=1469598103934665603; bytes=data,indices,indptr",
        "axis": "rows" if axis == 0 else "columns",
        "block_size": 64,
        "blocks": blocks,
        "expected_hash": f"0x{state:016x}",
    }


def make_dense_zarr2(path: Path, x: np.ndarray, obs: pd.DataFrame,
                     var: pd.DataFrame) -> dict[str, object]:
    data = ad.AnnData(X=x, obs=obs, var=var)
    with ad.settings.override(zarr_write_format=2), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        data.write_zarr(LocalStore(str(path)), chunks=DENSE_CHUNKS_Z2,
                        convert_strings_to_categoricals=False)

    root = zarr.open_group(LocalStore(str(path)), mode="r")
    actual = np.asarray(root["X"][:])
    assert tuple(root["X"].shape) == SHAPE
    assert tuple(root["X"].chunks) == DENSE_CHUNKS_Z2
    np.testing.assert_array_equal(actual, x)
    return {
        "format": "zarr2",
        "shape": list(SHAPE),
        "dtype": "float32",
        "chunks": list(root["X"].chunks),
        "shards": None,
        "x_sha256": sha256_bytes(x.tobytes(order="C")),
        "files": tree_records(path),
    }


def make_dense_zarr3(path: Path, x: np.ndarray, obs: pd.DataFrame,
                     var: pd.DataFrame) -> dict[str, object]:
    data = ad.AnnData(X=x, obs=obs, var=var)
    store = LocalStore(str(path))
    with ad.settings.override(zarr_write_format=3), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        data.write_zarr(store, chunks=DENSE_CHUNKS_Z3,
                        convert_strings_to_categoricals=False,
                        consolidate_metadata=False)

    root = zarr.open_group(store, mode="a", use_consolidated=False)
    attrs = dict(root["X"].attrs)
    del root["X"]
    X = root.create_array("X", data=x, chunks=DENSE_CHUNKS_Z3,
                          shards=DENSE_SHARDS_Z3)
    X.attrs.update(attrs)
    zarr.consolidate_metadata(root.store)

    reopened = zarr.open_group(store, mode="r", use_consolidated=False)
    actual = np.asarray(reopened["X"][:])
    assert tuple(reopened["X"].shape) == SHAPE
    assert tuple(reopened["X"].chunks) == DENSE_CHUNKS_Z3
    assert tuple(reopened["X"].shards) == DENSE_SHARDS_Z3
    np.testing.assert_array_equal(actual, x)
    return {
        "format": "zarr3",
        "shape": list(SHAPE),
        "dtype": "float32",
        "chunks": list(reopened["X"].chunks),
        "shards": list(reopened["X"].shards),
        "x_metadata_sha256": file_sha256(path / "X" / "zarr.json"),
        "x_sha256": sha256_bytes(x.tobytes(order="C")),
        "files": tree_records(path),
    }


def make_sparse_h5ad(path: Path, matrix: sp.csr_matrix,
                     obs: pd.DataFrame, var: pd.DataFrame,
                     storage_format: str) -> dict[str, object]:
    X = matrix if storage_format == "csr" else matrix.tocsc()
    data = ad.AnnData(X=X, obs=obs, var=var)
    data.write_h5ad(path, compression="gzip", compression_opts=1,
                     convert_strings_to_categoricals=False)

    reopened = ad.read_h5ad(path)
    assert sp.issparse(reopened.X)
    assert reopened.X.format == storage_format
    assert tuple(reopened.shape) == SHAPE
    expected = matrix if storage_format == "csr" else matrix.tocsc()
    np.testing.assert_array_equal(reopened.X.data, expected.data)
    np.testing.assert_array_equal(reopened.X.indices, expected.indices)
    np.testing.assert_array_equal(reopened.X.indptr, expected.indptr)
    with h5py.File(path, "r") as f:
        x_group = f["X"]
        encoding = x_group.attrs["encoding-type"]
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        assert encoding == storage_format + "_matrix"
        assert x_group["data"].compression == "gzip"
    return {
        "format": "h5ad",
        "sparse_format": storage_format,
        "shape": list(SHAPE),
        "dtype": str(reopened.X.dtype),
        "nnz": int(reopened.X.nnz),
        "compression": "gzip",
        "file_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "logical_csr_sha256": sha256_bytes(matrix.data.tobytes() +
                                            matrix.indices.tobytes() +
                                            matrix.indptr.tobytes()),
        "expected_hash": sparse_block_hashes(matrix, storage_format),
    }


def make_fixture(root: Path) -> dict[str, object]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite fixture root: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    paths = {
        "dense_zarr2": root / "dense_zarr2",
        "dense_zarr3": root / "dense_zarr3",
        "sparse_csr": root / "sparse_csr.h5ad",
        "sparse_csc": root / "sparse_csc.h5ad",
    }

    x = dense_values()
    obs = pd.DataFrame(index=[f"cell{i:04d}" for i in range(SHAPE[0])])
    var = pd.DataFrame(index=[f"feature{i:04d}" for i in range(SHAPE[1])])
    csr = sparse_values()
    manifest = {
        "schema": "anndatapp.streaming_probe.fixture.v1",
        "shape": list(SHAPE),
        "dense_formula": "float32(((17*r + 13*c + (r*c)%97)%1009) / 1009)",
        "dense_chunks_zarr2": list(DENSE_CHUNKS_Z2),
        "dense_chunks_zarr3": list(DENSE_CHUNKS_Z3),
        "dense_shards_zarr3": list(DENSE_SHARDS_Z3),
        "sparse_entries_per_row": SPARSE_PER_ROW,
        "sparse_values": "1 + (11*r + 7*c)%11 at sorted c=(37*r + 127*k)%1024, k=0..31",
        "dense_x_sha256": sha256_bytes(x.tobytes(order="C")),
        "fixtures": {
            "dense_zarr2": make_dense_zarr2(paths["dense_zarr2"], x, obs, var),
            "dense_zarr3": make_dense_zarr3(paths["dense_zarr3"], x, obs, var),
            "sparse_csr.h5ad": make_sparse_h5ad(paths["sparse_csr"], csr, obs, var, "csr"),
            "sparse_csc.h5ad": make_sparse_h5ad(paths["sparse_csc"], csr, obs, var, "csc"),
        },
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode()
    manifest["manifest_sha256"] = sha256_bytes(encoded)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="new fixture directory; must not exist")
    args = parser.parse_args()
    manifest = make_fixture(args.output.resolve())
    print(json.dumps({
        "path": str(args.output.resolve()),
        "shape": manifest["shape"],
        "dense_x_sha256": manifest["dense_x_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
