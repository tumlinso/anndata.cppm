"""Independent AnnData fixtures and readers for reusable block transfers.

Run with tests/requirements-interop.txt. Output stores are kept under the
supplied artifact directory so failing runs and physical layouts remain
inspectable. ANNDATA_INTEROP_ZARR_WAKEUP=1 enables the restricted-runner
event-loop workaround shared by the existing storage tests.
"""
from __future__ import annotations

from importlib.metadata import version
import hashlib
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

from python_storage import install_restricted_runner_wakeup

install_restricted_runner_wakeup()

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
import numcodecs
from zarr.codecs import BloscCodec, BytesCodec, GzipCodec, ZstdCodec
from zarr.storage import LocalStore

from python_storage import replace_boundary_zarr_array

N, M = 35, 67


def sparse(layout: str) -> sp.spmatrix:
    major, minor = (N, M) if layout == "csr" else (M, N)
    values = np.empty(3 * major, dtype=np.float32)
    indices = np.empty(3 * major, dtype=np.int32)
    for i in range(major):
        values[3*i:3*i+3] = (i + 1, 0, i + 2)
        indices[3*i:3*i+3] = (i % minor, i % minor, (3*i + 1) % minor)
    offsets = np.arange(0, 3*major + 1, 3, dtype=np.int64)
    ctor = sp.csr_matrix if layout == "csr" else sp.csc_matrix
    result = ctor((values, indices, offsets), shape=(N, M))
    # SciPy may narrow indices on construction; exercise independent widths.
    result.indices = indices
    result.indptr = offsets
    return result


def fixture() -> ad.AnnData:
    x = np.arange(N*M, dtype=np.float32).reshape(N, M)
    obs = pd.DataFrame({"kind": pd.Categorical(
        ["B" if i % 3 == 0 else "T" for i in range(N)],
        categories=["T", "B", "unused"], ordered=True,
    )}, index=pd.Index([f"cell-{i}" for i in range(N)], name="cell_id"))
    var = pd.DataFrame({"score": np.arange(M, dtype=np.int16)},
        index=pd.Index([f"gene-{i}" for i in range(M)], name="gene_id"))
    a = ad.AnnData(x, obs=obs, var=var)
    a.layers["csr"] = sparse("csr")
    a.layers["csc"] = sparse("csc")
    a.uns["description"] = "untouched λ"
    a.obsm["embedding"] = np.arange(N*2, dtype=np.float64).reshape(N, 2)
    raw_var = pd.DataFrame(index=pd.Index([f"raw-{i}" for i in range(M+1)], name="raw_id"))
    a.raw = ad.AnnData(np.arange(N*(M+1), dtype=np.float32).reshape(N, M+1),
                       obs=obs.copy(), var=raw_var)
    return a


def write_fixture(path: Path, kind: str, a: ad.AnnData) -> None:
    if kind.startswith("hdf5"):
        a.write_h5ad(path, convert_strings_to_categoricals=False)
        if kind == "hdf5-chunked":
            with h5py.File(path, "r+") as f:
                values = f["X"][:]
                attrs = dict(f["X"].attrs)
                del f["X"]
                x = f.create_dataset("X", data=values, chunks=(8, 16), compression="gzip")
                for key, value in attrs.items():
                    x.attrs[key] = value
        with h5py.File(path) as f:
            assert f["X"].chunks == ((8, 16) if kind == "hdf5-chunked" else None)
            assert f["layers/csr/indices"].dtype == np.dtype("int32")
            assert f["layers/csr/indptr"].dtype == np.dtype("int64")
        return
    fmt = 2 if kind == "zarr2" else 3
    with ad.settings.override(zarr_write_format=fmt), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        a.write_zarr(LocalStore(str(path)), convert_strings_to_categoricals=False,
                     consolidate_metadata=True)
    replace_boundary_zarr_array(path, "X", fmt, np.asarray(a.X))
    group = zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=False)
    assert tuple(group["X"].chunks) == (8, 16)
    if fmt == 3:
        assert tuple(group["X"].shards) == (16, 32)
    assert group["layers/csr/indices"].dtype == np.dtype("int32")
    assert group["layers/csr/indptr"].dtype == np.dtype("int64")


def equal_sparse(actual: sp.spmatrix, expected: sp.spmatrix) -> None:
    assert actual.format == expected.format
    assert actual.indices.dtype == expected.indices.dtype
    assert actual.indptr.dtype == expected.indptr.dtype
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)


def check_output(path: Path, kind: str, expected: ad.AnnData) -> None:
    actual = ad.read_h5ad(path) if kind.startswith("hdf5") else ad.read_zarr(LocalStore(str(path)))
    np.testing.assert_array_equal(actual.X, np.asarray(expected.X) + np.float32(1000))
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    np.testing.assert_array_equal(actual.obsm["embedding"], expected.obsm["embedding"])
    assert actual.uns["description"] == expected.uns["description"]
    for key in ("csr", "csc"):
        equal_sparse(actual.layers[key], expected.layers[key])
    assert actual.raw is not None and expected.raw is not None
    pd.testing.assert_frame_equal(actual.raw.var, expected.raw.var)
    np.testing.assert_array_equal(actual.raw.X, expected.raw.X)


def metadata_hashes(path: Path) -> dict[str, str]:
    names = {".zmetadata", ".zgroup", ".zarray", ".zattrs", "zarr.json"}
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file() and p.name in names}


def replace_codec_array(path: Path, fmt: int, codec: str, values: np.ndarray) -> None:
    group = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
    attrs = dict(group["X"].attrs)
    del group["X"]
    if fmt == 2:
        codecs = {"raw": None, "gzip": numcodecs.GZip(level=2), "zlib": numcodecs.Zlib(level=2),
                  "zstd": numcodecs.Zstd(level=2)}
        x = group.create_array("X", data=values, chunks=(8, 16), compressor=codecs[codec])
    else:
        codecs = {"raw": [], "gzip": [GzipCodec(level=2)], "zstd": [ZstdCodec(level=2)],
                  "blosc": [BloscCodec(cname="lz4", clevel=2, typesize=4, shuffle="shuffle")]}
        x = group.create_array("X", data=values, chunks=(8, 16), serializer=BytesCodec(endian="little"),
                               compressors=codecs[codec])
    x.attrs.update(attrs)
    zarr.consolidate_metadata(group.store)
    assert tuple(x.chunks) == (8, 16) and x.shards is None


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python_streaming.py DRIVER ARTIFACT_DIRECTORY")
    for name, expected in (("anndata", "0.13.4"), ("zarr", "3.4.2")):
        if version(name) != expected:
            raise SystemExit(f"expected {name} {expected}, got {version(name)}")
    driver, artifacts = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="streaming-", dir=artifacts))
    a = fixture()
    seeds = {}
    for kind in ("hdf5-contiguous", "hdf5-chunked", "zarr2", "zarr3"):
        path = root / (kind + (".h5ad" if kind.startswith("hdf5") else ".zarr"))
        write_fixture(path, kind, a)
        if kind.startswith("zarr"):
            seeds[kind] = root / (kind + "-seed.zarr")
            shutil.copytree(path, seeds[kind])
        before = {} if kind.startswith("hdf5") else metadata_hashes(path)
        subprocess.run([str(driver), kind, str(path)], check=True, timeout=30)
        if before:
            assert metadata_hashes(path) == before, "payload writes changed Zarr metadata"
        check_output(path, kind, a)
        print(f"{kind}: exact values, sparse buffers, annotations and raw passed", flush=True)
    codecs = ((2, "raw"), (2, "gzip"), (2, "zlib"), (2, "zstd"),
              (3, "raw"), (3, "gzip"), (3, "zstd"), (3, "blosc"))
    for fmt, codec in codecs:
        kind = f"zarr{fmt}-{codec}"
        path = root / (kind + ".zarr")
        shutil.copytree(seeds[f"zarr{fmt}"], path)
        replace_codec_array(path, fmt, codec, np.asarray(a.X))
        before = metadata_hashes(path)
        subprocess.run([str(driver), kind, str(path)], check=True, timeout=30)
        assert metadata_hashes(path) == before, "prepared codec writes changed Zarr metadata"
        check_output(path, kind, a)
        print(f"{kind}: repeated chunk reads and writes passed", flush=True)
    print(f"retained fixtures: {root}")


if __name__ == "__main__":
    main()
