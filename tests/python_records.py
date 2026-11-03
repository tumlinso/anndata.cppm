"""Independent Scanpy/HDF5 interoperability checks for native records."""
from __future__ import annotations

from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def run(driver: str, *args: object, fail: str = "", path: str = "") -> None:
    cmd = [driver, *(str(a) for a in args)]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    if fail:
        assert p.returncode != 0, (cmd, p.stdout, p.stderr)
        assert f"anndata error[{fail}]" in p.stderr, (cmd, p.stderr)
        assert path in p.stderr, (cmd, p.stderr)
    elif p.returncode:
        raise AssertionError(f"{cmd} failed ({p.returncode})\n{p.stdout}\n{p.stderr}")


def fixture() -> ad.AnnData:
    counts = np.array([
        [3, 1, 8, 2, 1], [7, 0, 2, 3, 2], [5, 2, 5, 0, 3], [4, 3, 9, 1, 4],
        [1, 6, 2, 9, 2], [0, 3, 4, 7, 3], [2, 5, 1, 8, 1], [3, 4, 3, 5, 5],
    ], dtype=np.float32)
    obs = pd.DataFrame({
        "group": pd.Categorical(["A"] * 4 + ["B"] * 4, categories=["A", "B"]),
        "kind": pd.Categorical(["T", "B"] * 4, categories=["T", "B", "unused"], ordered=True),
    }, index=pd.Index([f"cell-{i}" for i in range(8)], name="cell_id"))
    var = pd.DataFrame({"selected": [True, False, True, True, False]},
                       index=pd.Index(["gene-α", "gene-β", "gene-c", "gene-d", "gene-e"], name="gene_id"))
    a = ad.AnnData(sp.csr_matrix(np.log1p(counts)), obs=obs, var=var)
    a.layers["counts"] = sp.csr_matrix(counts)
    raw_var = pd.DataFrame(index=pd.Index([*var.index, "raw-only"], name="gene_id"))
    raw = np.column_stack((counts, np.arange(8, dtype=np.float32)))
    a.raw = ad.AnnData(sp.csr_matrix(raw), obs=obs.copy(), var=raw_var)
    sc.tl.rank_genes_groups(a, "group", method="t-test", use_raw=False, pts=True)

    dtype = [("signed", ">i2"), ("unsigned", "<u8"), ("score", ">f4"),
             ("flag", "?"), ("label", object)]
    mixed = np.empty((2, 3), dtype=dtype)
    mixed["signed"] = np.arange(6).reshape(2, 3) - 3
    mixed["unsigned"] = np.arange(6).reshape(2, 3) + 2**40
    mixed["score"] = np.array([-0.0, 0.25, np.nan, np.inf, -np.inf, 1.25]).reshape(2, 3)
    mixed["flag"] = np.array([False, True, False, True, True, False]).reshape(2, 3)
    mixed["label"] = np.array(["α", "ß", "δ", "", "gene", "other"], dtype=object).reshape(2, 3)
    a.uns["mixed"] = mixed
    a.uns["empty"] = np.empty((0, 2), dtype=[("n", "<i4"), ("text", object)])
    return a


def write_fixture(path: Path) -> None:
    fixture().write_h5ad(path, compression="gzip", convert_strings_to_categoricals=False)
    with h5py.File(path, "r+") as f:
        # HDF5 scalar datasets cannot carry the gzip option applied by write_h5ad.
        scalar = np.array((3, 1.5), dtype=[("a", "<i4"), ("b", "<f8")])
        ad.io.write_elem(f["uns"], "scalar", scalar)
        fixed = np.array([(b"AAAA", 4), (b"bb", 5)], dtype=[("id", "S4"), ("n", "<i4")])
        d = f["uns"].create_dataset("fixed", data=fixed)
        d.attrs["encoding-type"] = "rec-array"
        d.attrs["encoding-version"] = "0.2.0"


def equal_sparse(a: sp.spmatrix, b: sp.spmatrix) -> None:
    assert sp.isspmatrix_csr(a) and sp.isspmatrix_csr(b)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.indptr, b.indptr)


def equal_record(a: np.ndarray, b: np.ndarray) -> None:
    assert a.shape == b.shape and a.dtype.names == b.dtype.names
    for name in a.dtype.names or ():
        x, y = a[name], b[name]
        if x.dtype.kind in "biuf":
            assert x.dtype.kind == y.dtype.kind and x.dtype.itemsize == y.dtype.itemsize
            np.testing.assert_array_equal(x, y)
            if x.dtype.kind == "f":
                np.testing.assert_array_equal(np.signbit(x[x == 0]), np.signbit(y[x == 0]))
        else:
            np.testing.assert_array_equal(x.astype(str), y.astype(str))


def compare(source: Path, result: Path, mode: str) -> None:
    a, b = ad.read_h5ad(source), ad.read_h5ad(result)
    pd.testing.assert_frame_equal(a.obs, b.obs)
    pd.testing.assert_frame_equal(a.var, b.var)
    equal_sparse(a.X, b.X)
    equal_sparse(a.layers["counts"], b.layers["counts"])
    assert a.raw is not None and b.raw is not None
    pd.testing.assert_frame_equal(a.raw.var, b.raw.var)
    equal_sparse(a.raw.X, b.raw.X)

    x, y = a.uns["rank_genes_groups"], b.uns["rank_genes_groups"]
    assert x["params"] == y["params"]
    for key in ("pts", "pts_rest"):
        pd.testing.assert_frame_equal(x[key], y[key])
    for key in ("names", "scores", "logfoldchanges", "pvals", "pvals_adj"):
        expected = x[key].copy()
        if mode == "eager" and key == "scores": expected["A"][0] += np.float32(10)
        if mode == "eager" and key == "names": expected["A"][0] = "gene-β"
        if mode == "backed" and key == "scores": expected["B"][1] -= np.float32(5)
        equal_record(expected, y[key])
    scanpy_table = sc.get.rank_genes_groups_df(b, group="A")
    assert scanpy_table["names"].tolist() == y["names"]["A"].tolist()

    for key in ("mixed", "empty", "scalar", "fixed"):
        equal_record(a.uns[key], b.uns[key])
    made = b.uns["native_records"]
    assert made.shape == (2,) and made.dtype.names == ("value", "gene")
    np.testing.assert_array_equal(made["value"], np.array([2.5, -3.0], dtype=np.float32))
    np.testing.assert_array_equal(made["gene"].astype(str), ["μ", "gene-β"])

    with h5py.File(result, "r") as f:
        for key in ("rank_genes_groups/names", "rank_genes_groups/scores", "mixed",
                    "empty", "scalar", "fixed", "native_records"):
            d = f["uns/" + key]
            assert d.attrs["encoding-type"] == "rec-array"
            assert d.attrs["encoding-version"] == "0.2.0"
        d = f["uns/mixed"]
        assert d.dtype.names == ("signed", "unsigned", "score", "flag", "label")
        assert d.dtype.fields["signed"][0].byteorder == ">"
        assert d.dtype.fields["score"][0].byteorder == ">"
        assert h5py.check_string_dtype(d.dtype.fields["label"][0]).encoding == "utf-8"


def add_bad(path: Path, name: str, dtype: np.dtype) -> None:
    with h5py.File(path, "r+") as f:
        d = f["uns"].create_dataset(name, data=np.zeros(2, dtype=dtype))
        d.attrs["encoding-type"] = "rec-array"
        d.attrs["encoding-version"] = "0.2.0"


def directory_bytes(root: Path) -> list[tuple[str, bytes | None]]:
    return [(str(p.relative_to(root)), None if p.is_dir() else p.read_bytes())
            for p in sorted(root.rglob("*"))]


def main() -> None:
    if len(sys.argv) != 2: raise SystemExit("usage: python_records.py DRIVER")
    driver = sys.argv[1]
    saved = os.environ.get("ANNDATA_RECORD_FIXTURE_DIR")
    directory = nullcontext(saved) if saved else tempfile.TemporaryDirectory(prefix="anndata-record-interop-")
    with directory as temp:
        root = Path(temp)
        if saved: root.mkdir(parents=True, exist_ok=False)
        source, eager, backed, exported = (root / n for n in
                                            ("input.h5ad", "eager.h5ad", "backed.h5ad", "export.h5ad"))
        write_fixture(source)
        run(driver, "eager", source, eager)
        compare(source, eager, "eager")

        shutil.copyfile(source, backed)
        run(driver, "backed", backed, exported)
        compare(source, backed, "backed")
        compare(source, exported, "backed")

        # AnnData 0.13.4 reads fixed-byte compound members with ASCII decoding.
        # Valid non-ASCII fixed UTF-8 input must be normalized before its reader
        # can consume the rewritten file.
        fixed_utf8 = root / "fixed-utf8.h5ad"
        shutil.copyfile(source, fixed_utf8)
        with h5py.File(fixed_utf8, "r+") as f:
            dtype = np.dtype([("id", h5py.string_dtype("utf-8", length=4)), ("n", "<i4")])
            values = np.array([("λ".encode(), 4), ("😀".encode(), 5)], dtype=dtype)
            d = f["uns"].create_dataset("fixed_utf8", data=values)
            d.attrs["encoding-type"] = "rec-array"
            d.attrs["encoding-version"] = "0.2.0"
        for mode in ("fixed-eager", "fixed-backed"):
            result = root / f"{mode}.h5ad"
            run(driver, mode, fixed_utf8, result)
            a = ad.read_h5ad(result)
            np.testing.assert_array_equal(a.uns["fixed_utf8"]["id"].astype(str), ["λ", "😀"])
            np.testing.assert_array_equal(a.uns["fixed_utf8"]["n"], [4, 5])
            with h5py.File(result, "r") as f:
                dt = f["uns/fixed_utf8"].dtype.fields["id"][0]
                assert h5py.check_string_dtype(dt).encoding == "utf-8"
                assert h5py.check_string_dtype(dt).length is None

        for name, dtype in (
            ("nested", np.dtype([("bad", [("inner", "i4")])])),
            ("subarray", np.dtype([("bad", "i4", (2,))])),
        ):
            bad = root / f"{name}.h5ad"
            shutil.copyfile(source, bad)
            add_bad(bad, name, dtype)
            for cmd in ("read", "backed-read"):
                run(driver, cmd, bad, name, fail="unsupported_dtype", path="/uns/" + name)

        bad_version = root / "bad-version.h5ad"
        shutil.copyfile(source, bad_version)
        with h5py.File(bad_version, "r+") as f:
            f["uns/mixed"].attrs["encoding-version"] = "9.9.9"
        run(driver, "read", bad_version, "mixed", fail="unsupported_version", path="/uns/mixed")
        run(driver, "backed-read", bad_version, "mixed", fail="unsupported_version", path="/uns/mixed")

        budget = root / "budget.h5ad"
        shutil.copyfile(source, budget)
        long_text = np.array([(37, "x" * 16_384)], dtype=[("value", "<i4"), ("text", object)])
        with h5py.File(budget, "r+") as f: ad.io.write_elem(f["uns"], "oversized", long_text)
        run(driver, "budget", budget, fail="invalid_argument", path="/uns/oversized")

        features = subprocess.run([driver, "features"], check=True, text=True,
                                  capture_output=True, timeout=10).stdout
        if "zarr=1" in features:
            for mode in ("zarr-eager", "zarr-backed"):
                for fmt in (2, 3):
                    dest = root / f"{mode}-{fmt}.zarr"
                    dest.mkdir()
                    (dest / "sentinel").write_bytes(b"original contents\x00\xff")
                    (dest / "nested").mkdir()
                    (dest / "nested" / "data").write_bytes(b"existing bytes")
                    before = directory_bytes(dest)
                    run(driver, mode, source, dest, fmt, fail="unsupported_encoding", path="/uns/")
                    assert directory_bytes(dest) == before
                    missing = root / f"missing-{mode}-{fmt}.zarr"
                    run(driver, mode, source, missing, fmt, fail="unsupported_encoding", path="/uns/")
                    assert not missing.exists()

    print("record interoperability passed:", {name: version(name) for name in
          ("anndata", "scanpy", "h5py", "numpy", "pandas", "scipy")})


if __name__ == "__main__": main()
