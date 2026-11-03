"""Independent file interoperability checks against pinned AnnData versions.

This test exercises on-disk H5AD codecs. The older python_semantics.py remains
the separate in-memory slicing oracle. The default mode is the full 0.13.4
matrix; --compat-0.12 runs only the shared format subset against 0.12.10.
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def make_fixture():
    obs = pd.DataFrame(
        {
            "kind": pd.Categorical(
                ["T", "B", None], categories=["T", "B", "unused"], ordered=True
            ),
            "kind_unsigned": pd.Categorical(
                ["T", "B", "T"], categories=["T", "B", "unused"], ordered=False
            ),
            "count": pd.array([7, None, -2], dtype="Int16"),
            "enabled": pd.array([True, None, False], dtype="boolean"),
            "label": pd.array(["α", None, "cell γ"], dtype="string"),
            "label_nan": pd.array(
                ["left", np.nan, "right"], dtype=pd.StringDtype(na_value=np.nan)
            ),
            "unsigned": np.array([1, 255, 7], dtype=np.uint8),
        },
        index=pd.Index(["cell-a", "cell-a", "cell-γ"], name="cell_id"),
    )
    var = pd.DataFrame(
        {"active": np.array([True, False], dtype=np.bool_)},
        index=pd.Index(["gene-1", "gene-2"], name="gene_id"),
    )
    a = ad.AnnData(np.arange(6, dtype=np.float32).reshape(3, 2), obs=obs, var=var)

    # Keep the repeated entries and explicit zero. This is a storage fixture,
    # so construct the buffers directly instead of canonicalizing via COO.
    a.layers["counts"] = sp.csr_matrix(
        (
            np.array([1, 0, 2, 3, 4], dtype=np.int64),
            np.array([1, 0, 0, 0, 1], dtype=np.int32),
            np.array([0, 2, 4, 5], dtype=np.int32),
        ),
        shape=(3, 2),
    )
    a.layers["dense_layer"] = np.arange(6, dtype=np.uint16).reshape(3, 2)
    a.obsm["embedding"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    a.obsm["labels"] = np.array([["a", "β"], ["", "d"], ["e", "f"]], dtype="U")
    a.obsm["table"] = pd.DataFrame(
        {"score": np.array([0.25, 0.5, 0.75], dtype=np.float64)},
        index=a.obs_names,
    )
    a.varm["embedding"] = np.array([[1, 2], [3, 4]], dtype=np.int16)
    a.obsp["connectivities"] = sp.csr_matrix(
        np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]], dtype=np.float32)
    )
    a.varp["similarity"] = sp.csc_matrix(
        np.array([[1, 0], [0, 1]], dtype=np.float64)
    )
    a.uns = {
        "nested": {
            "description": "nested α",
            "null": None,
            "threshold": np.float32(0.5),
        },
        # AnnData's upstream writer intentionally scalarizes these 0-D arrays.
        "zero_numeric_array": np.array(23, dtype=np.int16),
        "zero_string_array": np.array("zero λ", dtype="U"),
        "matrix_text": np.array([["μ", ""], ["g2", "gene β"]], dtype="U"),
        "dtypes": {
            "bool": np.array([True, False], dtype=np.bool_),
            "i8": np.array([-2, 3], dtype=np.int8),
            "u8": np.array([2, 250], dtype=np.uint8),
            "i16": np.array([-300, 1000], dtype=np.int16),
            "u16": np.array([1000, 60000], dtype=np.uint16),
            "i32": np.array([-100000, 200000], dtype=np.int32),
            "u32": np.array([100000, 4000000000], dtype=np.uint32),
            "i64": np.array([-(2**40), 2**40], dtype=np.int64),
            "u64": np.array([2**40, 2**63 + 1], dtype=np.uint64),
            "f32": np.array([0.5, 1.25], dtype=np.float32),
            "f64": np.array([np.pi, -1.5], dtype=np.float64),
        },
    }
    a.raw = a.copy()
    a.raw.varm["label_table"] = pd.DataFrame(
        {"label": ["raw α", "raw β"]}, index=a.raw.var_names
    )
    return a


def assert_python_file(path: Path, *, cpp_rank0_arrays: bool = False) -> None:
    a = ad.read_h5ad(path)
    assert a.shape == (3, 2)
    np.testing.assert_array_equal(a.X, np.arange(6, dtype=np.float32).reshape(3, 2))
    assert a.obs_names.tolist() == ["cell-a", "cell-a", "cell-γ"]
    assert a.obs_names.name == "cell_id"
    assert a.var_names.tolist() == ["gene-1", "gene-2"]
    assert a.obs["kind"].cat.categories.tolist() == ["T", "B", "unused"]
    assert a.obs["kind"].cat.ordered
    assert a.obs["kind"].cat.codes.tolist() == [0, 1, -1]
    assert a.obs["kind_unsigned"].cat.categories.tolist() == ["T", "B", "unused"]
    assert not a.obs["kind_unsigned"].cat.ordered
    assert a.obs["kind_unsigned"].cat.codes.tolist() == [0, 1, 0]
    assert a.obs["count"].iloc[0] == 7 and pd.isna(a.obs["count"].iloc[1])
    assert a.obs["count"].iloc[2] == -2
    assert a.obs["enabled"].iloc[0] and pd.isna(a.obs["enabled"].iloc[1])
    assert not a.obs["enabled"].iloc[2]
    assert a.obs["label"].iloc[0] == "α" and pd.isna(a.obs["label"].iloc[1])
    assert a.obs["label"].iloc[2] == "cell γ"
    assert a.obs["label_nan"].isna().tolist() == [False, True, False]
    counts = a.layers["counts"]
    assert counts.format == "csr"
    np.testing.assert_array_equal(counts.data, [1, 0, 2, 3, 4])
    np.testing.assert_array_equal(counts.indices, [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(counts.indptr, [0, 2, 4, 5])
    assert a.layers["dense_layer"].dtype == np.dtype("uint16")
    np.testing.assert_array_equal(a.obsm["labels"], [["a", "β"], ["", "d"], ["e", "f"]])
    assert a.obsm["table"]["score"].tolist() == [0.25, 0.5, 0.75]
    assert sp.isspmatrix_csr(a.obsp["connectivities"])
    assert sp.isspmatrix_csc(a.varp["similarity"])
    assert a.uns["nested"]["description"] == "nested α"
    assert a.uns["nested"]["null"] is None
    if cpp_rank0_arrays:
        numeric = np.asarray(a.uns["zero_numeric_array"])
        strings = np.asarray(a.uns["zero_string_array"])
        assert numeric.shape == () and numeric.dtype == np.dtype("int16")
        assert int(numeric) == 23
        assert strings.shape == () and str(strings.item()) == "zero λ"
    else:
        assert not isinstance(a.uns["zero_numeric_array"], np.ndarray)
        assert int(a.uns["zero_numeric_array"]) == 23
        assert isinstance(a.uns["zero_string_array"], str)
        assert a.uns["zero_string_array"] == "zero λ"
    np.testing.assert_array_equal(
        a.uns["matrix_text"], [["μ", ""], ["g2", "gene β"]]
    )
    for key, value in make_fixture().uns["dtypes"].items():
        np.testing.assert_array_equal(a.uns["dtypes"][key], value)
        assert a.uns["dtypes"][key].dtype == value.dtype
    assert a.raw is not None
    np.testing.assert_array_equal(a.raw.X, a.X)
    assert a.raw.varm["label_table"]["label"].tolist() == ["raw α", "raw β"]


def check_hdf5_encodings(
    path: Path, *, require_big_endian: bool, cpp_rank0_arrays: bool = False
) -> None:
    with h5py.File(path, "r") as f:
        assert f.attrs["encoding-type"] == "anndata"
        assert f.attrs["encoding-version"] == "0.1.0"
        assert f["X"].attrs["encoding-type"] == "array"
        assert f["layers/counts"].attrs["encoding-type"] == "csr_matrix"
        assert f["obs"].attrs["encoding-type"] == "dataframe"
        assert f["obs"].attrs["column-order"].tolist() == [
            "kind", "kind_unsigned", "count", "enabled", "label", "label_nan", "unsigned"
        ]
        assert f["obs/kind"].attrs["encoding-type"] == "categorical"
        assert bool(f["obs/kind"].attrs["ordered"])
        assert f["obs/kind_unsigned/codes"].dtype == np.dtype("uint8")
        assert f["obs/count"].attrs["encoding-type"] == "nullable-integer"
        assert f["obs/enabled"].attrs["encoding-type"] == "nullable-boolean"
        bool_type = f["obs/enabled/values"].id.get_type()
        assert bool_type.get_class() == h5py.h5t.ENUM
        assert {
            bool_type.get_member_name(i).decode("ascii"):
            bool_type.get_member_value(i)
            for i in range(bool_type.get_nmembers())
        } == {"FALSE": 0, "TRUE": 1}
        np.testing.assert_array_equal(f["obs/enabled/mask"], [False, True, False])
        np.testing.assert_array_equal(f["obs/count/mask"], [False, True, False])
        assert f["obs/label"].attrs["encoding-type"] == "nullable-string-array"
        assert f["obs/label_nan"].attrs["na-value"] == "NaN"
        assert f["obsm/labels"].attrs["encoding-type"] == "string-array"
        assert f["uns/matrix_text"].attrs["encoding-type"] == "string-array"
        assert f["raw"].attrs["encoding-type"] == "raw"
        label = ad.io.read_elem(f["obs/label"])
        label_nan = ad.io.read_elem(f["obs/label_nan"])
        assert label.dtype.na_value is pd.NA
        assert np.isnan(label_nan.dtype.na_value)
        if require_big_endian:
            assert f["X"].dtype.byteorder == ">", f["X"].dtype
        else:
            assert f["X"].dtype == np.dtype("float32")
        # The ordinary interoperability fixture stores sparse components and
        # scalars in host order. A separate adversarial file checks all-big
        # inputs without asking SciPy to read its unsupported sparse dtype.
        for dataset in (
            f["layers/counts/data"], f["layers/counts/indices"],
            f["layers/counts/indptr"], f["uns/nested/threshold"],
        ):
            assert dataset.dtype.isnative, (dataset.name, dataset.dtype)
        assert f["layers/counts/data"].dtype == np.dtype("int64")
        if Path(f.filename).name in {"cpp.h5ad", "cpp-gzip.h5ad"}:
            assert f["layers/counts/indices"].dtype == np.dtype("int32")
            assert f["layers/counts/indptr"].dtype == np.dtype("int64")
        else:
            assert f["layers/counts/indices"].dtype == np.dtype("int32")
            assert f["layers/counts/indptr"].dtype == np.dtype("int32")
        assert f["uns/nested/threshold"].dtype == np.dtype("float32")
        if cpp_rank0_arrays:
            assert f["uns/zero_numeric_array"].attrs["encoding-type"] == "array"
            assert f["uns/zero_numeric_array"].attrs["encoding-version"] == "0.2.0"
            assert f["uns/zero_numeric_array"].shape == ()
            assert f["uns/zero_string_array"].attrs["encoding-type"] == "string-array"
            assert f["uns/zero_string_array"].attrs["encoding-version"] == "0.2.0"
        else:
            assert f["uns/zero_numeric_array"].attrs["encoding-type"] == "numeric-scalar"
            assert f["uns/zero_numeric_array"].attrs["encoding-version"] == "0.2.0"
            assert f["uns/zero_string_array"].attrs["encoding-type"] == "string"
            assert f["uns/zero_string_array"].attrs["encoding-version"] == "0.2.0"
        if Path(f.filename).name in {"cpp.h5ad", "cpp-gzip.h5ad"}:
            assert f["layers/counts/indices"].dtype == np.dtype("int32")
            assert f["layers/counts/indptr"].dtype == np.dtype("int64")


def set_big_endian(path: Path, dataset_path: str, dtype: str) -> None:
    with h5py.File(path, "r+") as f:
        if "/" in dataset_path:
            parent_path, name = dataset_path.rsplit("/", 1)
            parent = f[parent_path]
        else:
            parent, name = f, dataset_path
        old = parent[name]
        values = old[()]
        attrs = dict(old.attrs)
        del parent[name]
        new = parent.create_dataset(name, data=np.asarray(values, dtype=np.dtype(dtype)))
        for key, value in attrs.items():
            new.attrs[key] = value


def set_foreign_endian_fixture(source: Path, target: Path) -> None:
    shutil.copyfile(source, target)
    set_big_endian(target, "X", ">f4")
    set_big_endian(target, "layers/counts/data", ">i8")
    set_big_endian(target, "layers/counts/indices", ">i4")
    set_big_endian(target, "layers/counts/indptr", ">i4")
    set_big_endian(target, "uns/nested/threshold", ">f4")


def write_compact_sparse_fixture(path: Path) -> None:
    a = ad.AnnData(X=None, obs=pd.DataFrame(index=["c0", "c1", "c2"]),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(300)]))
    a.uns["compact8"] = sp.csr_matrix(
        (np.array([3.0]), np.array([1], dtype=np.int32),
         np.array([0, 0, 1, 1], dtype=np.int32)), shape=(3, 2)
    )
    a.uns["compact16"] = sp.csr_matrix(
        (np.array([5.0]), np.array([299], dtype=np.int32),
         np.array([0, 1, 1], dtype=np.int32)), shape=(2, 300)
    )
    with ad.settings.override(write_csr_csc_indices_with_min_possible_dtype=True):
        a.write_h5ad(path)


def write_filter_plugin_fixture(source: Path, target: Path) -> None:
    shutil.copyfile(source, target)
    with h5py.File(target, "r+") as f:
        old = f["X"]
        attrs = dict(old.attrs)
        del f["X"]
        space = h5py.h5s.create_simple((3, 2))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((3, 2))
        # A synthetic filter id with no registered decoder. The dataset need
        # not contain chunks: the C++ reader must reject the declared filter
        # before attempting to read values.
        dcpl.set_filter(32015, h5py.h5z.FLAG_OPTIONAL)
        did = h5py.h5d.create(f.id, b"X", h5py.h5t.IEEE_F32LE, space, dcpl=dcpl)
        d = h5py.Dataset(did)
        for key, value in attrs.items():
            d.attrs[key] = value


def write_empty_axis_fixtures(driver: str, root: Path) -> None:
    cases = (
        ("empty-no-x.h5ad", [], []),
        ("rows-no-vars-no-x.h5ad", ["c0", "c1"], []),
    )
    for name, obs_names, var_names in cases:
        source = root / name
        output = root / f"cpp-{name}"
        a = ad.AnnData(
            X=None,
            obs=pd.DataFrame(index=pd.Index(obs_names, name="cell_id")),
            var=pd.DataFrame(index=pd.Index(var_names, name="gene_id")),
        )
        a.write_h5ad(source)
        with h5py.File(source, "r") as f:
            for axis in ("obs", "var"):
                order = f[axis].attrs["column-order"]
                assert order.shape == (0,)
                assert order.dtype.kind == "f"  # AnnData 0.13.4 writes float64 here.
            assert "X" not in f
        subprocess.run([driver, "copy", str(source), str(output)], check=True)
        reopened = ad.read_h5ad(output)
        assert reopened.X is None
        assert reopened.shape == (len(obs_names), len(var_names))
        assert reopened.obs_names.name == "cell_id"
        assert reopened.var_names.name == "gene_id"


def patch_raw_enum_byte(path: Path, dataset_path: str, element: int, value: int) -> None:
    with h5py.File(path, "r") as f:
        dataset = f[dataset_path]
        assert dataset.dtype.kind in ("b", "u") and dataset.dtype.itemsize == 1
        assert dataset.id.get_type().get_class() == h5py.h5t.ENUM
        offset = dataset.id.get_offset()
        assert offset is not None
    # This deliberately edits the raw enum byte. h5py's high-level enum write
    # casts through bool and cannot express a non-canonical enum representation.
    with path.open("r+b") as stream:
        stream.seek(offset + element)
        stream.write(bytes([value]))


def expect_read_error(driver: str, path: Path, code: str, path_fragment: str) -> None:
    result = subprocess.run(
        [driver, "expect-read-error", str(path), code, path_fragment],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"expected {code} at {path_fragment} for {path}: {result.stderr.strip()}"
        )


def shared_fixture_012() -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "kind": pd.Categorical(["T", "B", None], categories=["T", "B", "unused"], ordered=True),
            "count": pd.array([7, None, -2], dtype="Int16"),
            "enabled": pd.array([True, None, False], dtype="boolean"),
            "label": pd.Categorical(["α", None, "γ"], categories=["α", "γ", "unused"]),
        },
        index=pd.Index(["cell-a", "cell-a", "cell-γ"], name="cell_id"),
    )
    var = pd.DataFrame(index=pd.Index(["gene-1", "gene-2"], name="gene_id"))
    a = ad.AnnData(np.arange(6, dtype=np.float32).reshape(3, 2), obs=obs, var=var)
    a.layers["counts"] = sp.csr_matrix(
        (
            np.array([1, 0, 2, 3, 4], dtype=np.int64),
            np.array([1, 0, 0, 0, 1], dtype=np.int32),
            np.array([0, 2, 4, 5], dtype=np.int32),
        ),
        shape=(3, 2),
    )
    a.uns = {"nested": {"description": "shared subset α", "null": None, "number": np.int16(17)}}
    a.raw = a.copy()
    return a


def run_compat_012(driver: str, root: Path) -> None:
    source = root / "python-0.12.10.h5ad"
    output = root / "cpp-0.12.10.h5ad"
    a = shared_fixture_012()
    a.write_h5ad(source)
    subprocess.run([driver, "copy", str(source), str(output)], check=True)
    b = ad.read_h5ad(output)
    assert b.shape == (3, 2)
    np.testing.assert_array_equal(b.X, np.arange(6, dtype=np.float32).reshape(3, 2))
    assert b.obs_names.tolist() == ["cell-a", "cell-a", "cell-γ"]
    assert b.obs_names.name == "cell_id"
    assert b.obs["kind"].cat.categories.tolist() == ["T", "B", "unused"]
    assert b.obs["kind"].cat.ordered
    assert b.obs["kind"].cat.codes.tolist() == [0, 1, -1]
    assert b.obs["count"].tolist()[0] == 7 and pd.isna(b.obs["count"].tolist()[1])
    assert b.obs["enabled"].iloc[0] and pd.isna(b.obs["enabled"].iloc[1])
    assert b.obs["label"].cat.categories.tolist() == ["α", "γ", "unused"]
    counts = b.layers["counts"]
    assert counts.format == "csr"
    np.testing.assert_array_equal(counts.data, [1, 0, 2, 3, 4])
    np.testing.assert_array_equal(counts.indices, [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(counts.indptr, [0, 2, 4, 5])
    assert b.uns["nested"]["description"] == "shared subset α"
    assert b.uns["nested"]["null"] is None
    assert int(b.uns["nested"]["number"]) == 17
    assert b.raw is not None
    np.testing.assert_array_equal(b.raw.X, b.X)
    print("H5AD shared subset passed: AnnData 0.12.10 -> C++ -> Python")


def write_python_fixture(path: Path) -> None:
    with ad.settings.override(allow_write_nullable_strings=True):
        make_fixture().write_h5ad(path)
        # AnnData.write_h5ad converts low-cardinality string dataframe columns
        # into categoricals by default. Replace these two entries with the
        # explicit nullable-string encodings so the reader tests both sentinels.
        with h5py.File(path, "r+") as f:
            del f["obs/label"]
            del f["obs/label_nan"]
            ad.io.write_elem(f["obs"], "label", pd.array(["α", pd.NA, "cell γ"], dtype="string"))
            ad.io.write_elem(
                f["obs"], "label_nan",
                pd.array(["left", np.nan, "right"], dtype=pd.StringDtype(na_value=np.nan)),
            )
            old = f["obs/kind_unsigned/codes"]
            attrs = dict(old.attrs)
            codes = old[:].astype(np.uint8)
            del f["obs/kind_unsigned/codes"]
            new = f["obs/kind_unsigned"].create_dataset("codes", data=codes, dtype=np.uint8)
            for key, value in attrs.items():
                new.attrs[key] = value


def must_fail(
    driver: str, *args: str, expected_code: str | None = None,
    path_fragment: str | None = None,
) -> None:
    result = subprocess.run([driver, *args], text=True, capture_output=True)
    if result.returncode == 0:
        raise AssertionError(f"expected command to fail: {args}")
    if not result.stderr.strip():
        raise AssertionError(f"failure had no diagnostic: {args}")
    if expected_code and f"anndata error[{expected_code}]" not in result.stderr:
        raise AssertionError(f"expected {expected_code}, got: {result.stderr.strip()}")
    if path_fragment and path_fragment not in result.stderr:
        raise AssertionError(f"expected path {path_fragment}, got: {result.stderr.strip()}")


def assert_big_physical_order(path: Path) -> None:
    with h5py.File(path, "r") as f:
        for name in (
            "X", "layers/counts/data", "layers/counts/indices",
            "layers/counts/indptr", "uns/nested/threshold",
        ):
            assert f[name].dtype.byteorder == ">", (name, f[name].dtype)


def assert_native_physical_order(path: Path) -> None:
    with h5py.File(path, "r") as f:
        for name in (
            "X", "layers/counts/data", "layers/counts/indices",
            "layers/counts/indptr", "uns/nested/threshold",
        ):
            assert f[name].dtype.isnative, (name, f[name].dtype)


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: python_h5ad.py /path/to/h5ad_interop_driver [--compat-0.12]"
        )
    driver = str(Path(sys.argv[1]).resolve())
    installed = version("anndata")
    compat = len(sys.argv) == 3 and sys.argv[2] == "--compat-0.12"
    if len(sys.argv) == 3 and not compat:
        raise SystemExit(f"unknown option: {sys.argv[2]}")
    expected = "0.12.10" if compat else "0.13.4"
    if installed != expected:
        raise SystemExit(f"expected AnnData {expected}, got {installed}")

    with tempfile.TemporaryDirectory(prefix="anndata-cppm-h5ad-") as temp:
        root = Path(temp)
        if compat:
            run_compat_012(driver, root)
            return

        source, reread, cpp_output, compressed = (root / name for name in
            ("python.h5ad", "cpp-reread.h5ad", "cpp.h5ad", "cpp-gzip.h5ad"))
        write_python_fixture(source)
        set_big_endian(source, "X", ">f4")

        # C++ validates directly loaded values and serializes the same owner.
        subprocess.run([driver, "verify", str(source), str(reread)], check=True)
        assert_python_file(reread)
        check_hdf5_encodings(reread, require_big_endian=True)

        # C++ independently creates a complete object; Python reopens the file.
        subprocess.run([driver, "write", str(cpp_output)], check=True)
        assert_python_file(cpp_output, cpp_rank0_arrays=True)
        check_hdf5_encodings(cpp_output, require_big_endian=False, cpp_rank0_arrays=True)

        subprocess.run([driver, "write-gzip", str(compressed)], check=True)
        assert_python_file(compressed, cpp_rank0_arrays=True)
        check_hdf5_encodings(compressed, require_big_endian=True, cpp_rank0_arrays=True)
        with h5py.File(compressed, "r") as f:
            assert f["X"].compression == "gzip"
            assert f["layers/counts/data"].compression == "gzip"
            assert f["X"].dtype.byteorder == ">"
            assert f["layers/counts/data"].dtype == np.dtype("int64")

        # Exclusive creation must not replace an existing file.
        before = cpp_output.read_bytes()
        must_fail(driver, "exclusive", str(cpp_output), expected_code="io_error")
        assert cpp_output.read_bytes() == before

        # Encoding/version corruption must carry stable error codes and paths.
        missing = root / "missing-encoding.h5ad"
        shutil.copyfile(source, missing)
        with h5py.File(missing, "r+") as f:
            del f.attrs["encoding-type"]
        expect_read_error(driver, missing, "unsupported_encoding", "/")

        unsupported = root / "unsupported-encoding.h5ad"
        shutil.copyfile(source, unsupported)
        with h5py.File(unsupported, "r+") as f:
            ds = f["uns/nested/description"]
            ds.attrs["encoding-type"] = "unknown-special-value"
            ds.attrs["encoding-version"] = "0.1.0"
        expect_read_error(driver, unsupported, "unsupported_encoding", "/uns/nested/description")

        old_version = root / "unsupported-version.h5ad"
        shutil.copyfile(source, old_version)
        with h5py.File(old_version, "r+") as f:
            f.attrs["encoding-version"] = "99.0.0"
        expect_read_error(driver, old_version, "unsupported_version", "/")

        filter_plugin = root / "missing-filter-plugin.h5ad"
        write_filter_plugin_fixture(source, filter_plugin)
        expect_read_error(driver, filter_plugin, "unavailable_filter", "/X")

        malformed = root / "bad-sparse.h5ad"
        shutil.copyfile(source, malformed)
        with h5py.File(malformed, "r+") as f:
            f["layers/counts/indices"][0] = 99
        expect_read_error(driver, malformed, "invalid_sparse_structure", "/layers/counts")
        # Trusted callers can explicitly skip the content scan. This test only
        # checks acquisition; it never dereferences or writes the bad indices.
        subprocess.run([driver, "read-none", str(malformed)], check=True)

        bad_category = root / "bad-category.h5ad"
        shutil.copyfile(source, bad_category)
        with h5py.File(bad_category, "r+") as f:
            f["obs/kind/codes"][0] = 99
        expect_read_error(driver, bad_category, "out_of_bounds", "/obs/kind")

        bad_column = root / "bad-obs-column-length.h5ad"
        shutil.copyfile(source, bad_column)
        with h5py.File(bad_column, "r+") as f:
            old = f["obs/unsigned"]
            values = old[:2]
            attrs = dict(old.attrs)
            del f["obs/unsigned"]
            new = f["obs"].create_dataset("unsigned", data=values, dtype=values.dtype)
            for key, value in attrs.items():
                new.attrs[key] = value
        expect_read_error(driver, bad_column, "shape_mismatch", "/obs/unsigned")

        for filename, parent_path, member, path_fragment in (
            ("extra-sparse-child.h5ad", "layers/counts", "unexpected", "/layers/counts/unexpected"),
            ("extra-category-child.h5ad", "obs/kind", "unexpected", "/obs/kind/unexpected"),
            ("extra-nullable-child.h5ad", "obs/count", "unexpected", "/obs/count/unexpected"),
        ):
            path = root / filename
            shutil.copyfile(source, path)
            with h5py.File(path, "r+") as f:
                f[parent_path].create_dataset(member, data=np.array([1], dtype=np.int8))
            expect_read_error(driver, path, "unsupported_encoding", path_fragment)

        # AnnData 0.13.4 writes bool values with an enum HDF5 type. A raw
        # non-canonical enum byte must be handled without typed-bool UB and
        # normalized to the upstream truthiness behavior on a full read.
        malformed_bool = root / "bad-bool-enum-byte.h5ad"
        shutil.copyfile(source, malformed_bool)
        patch_raw_enum_byte(malformed_bool, "obs/enabled/values", 0, 2)
        normalized_bool = root / "normalized-bool-enum.h5ad"
        subprocess.run(
            [driver, "normalize-bool", str(malformed_bool), str(normalized_bool)],
            check=True,
        )
        with h5py.File(normalized_bool, "r") as f:
            values = ad.io.read_elem(f["obs/enabled/values"])
            assert values.tolist() == [True, False, False]

        # A valid but non-canonical unsigned index representation is written
        # by AnnData when its explicit compact-index setting is enabled.
        compact = root / "compact-unsigned-sparse.h5ad"
        compact_cpp = root / "compact-unsigned-sparse-cpp.h5ad"
        write_compact_sparse_fixture(compact)
        with h5py.File(compact, "r") as f:
            assert f["uns/compact8/indices"].dtype == np.dtype("uint8")
            assert f["uns/compact16/indices"].dtype == np.dtype("uint16")
        subprocess.run([driver, "compact-indices", str(compact), str(compact_cpp)], check=True)
        compact_roundtrip = ad.read_h5ad(compact_cpp)
        assert compact_roundtrip.uns["compact8"].indices.dtype == np.dtype("uint8")
        assert compact_roundtrip.uns["compact16"].indices.dtype == np.dtype("uint16")
        np.testing.assert_array_equal(compact_roundtrip.uns["compact8"].toarray(),
                                      [[0, 0], [0, 3], [0, 0]])
        assert compact_roundtrip.uns["compact16"][0, 299] == 5.0

        # AnnData's own 0-D ndarray writer converts to scalar encodings. C++
        # preserves explicit Array/StringArray ownership and writes the valid
        # array encodings, which Python also reads as zero-D NumPy arrays.
        with h5py.File(source, "r") as f:
            assert f["uns/zero_numeric_array"].attrs["encoding-type"] == "numeric-scalar"
            assert f["uns/zero_string_array"].attrs["encoding-type"] == "string"
        write_empty_axis_fixtures(driver, root)

        # The C++ writer preserves the raw-without-X state. Python's AnnData
        # Raw constructor substitutes parent X/var state for this case, so
        # check the actual stored slots without claiming Python semantic parity.
        no_x_raw = root / "no-x-raw.h5ad"
        subprocess.run([driver, "write-no-x-raw", str(no_x_raw)], check=True)
        with h5py.File(no_x_raw, "r") as f:
            assert "X" not in f
            assert f["raw"].attrs["encoding-type"] == "raw"
            assert f["raw/X"].attrs["encoding-type"] == "null"

        # The preserve write carries original byte-order hints on dense data,
        # each sparse component, and scalar metadata. This representation is
        # h5py-readable but SciPy rejects its big-endian sparse values; an
        # explicit native-order write remains fully readable by AnnData.
        foreign = root / "foreign-endian.h5ad"
        preserve = root / "foreign-endian-preserved.h5ad"
        native = root / "foreign-endian-native.h5ad"
        set_foreign_endian_fixture(source, foreign)
        subprocess.run([driver, "byte-order", str(foreign), str(preserve), str(native)], check=True)
        assert_big_physical_order(preserve)
        assert_native_physical_order(native)
        assert_python_file(native)
        check_hdf5_encodings(native, require_big_endian=False)

        # AnnData 0.13.4 emits an empty column-order attribute with float64
        # dtype for zero-column obs/var tables. It is a real upstream fixture.
        # Setter and index-name collisions remain explicit lossy-write errors.
        must_fail(driver, "write-bad-key", str(root / "bad-key.h5ad"),
                  expected_code="invalid_argument", path_fragment="/uns")
        must_fail(driver, "write-nul-key", str(root / "nul-key.h5ad"),
                  expected_code="invalid_argument", path_fragment="/uns")
        must_fail(driver, "write-nul-string", str(root / "nul-string.h5ad"),
                  expected_code="invalid_argument", path_fragment="/uns")
        must_fail(driver, "write-bad-index", str(root / "bad-index.h5ad"),
                  expected_code="invalid_argument", path_fragment="/obs")
        must_fail(driver, "write-bad-index-name", str(root / "bad-index-name.h5ad"),
                  expected_code="invalid_argument", path_fragment="/obs")

        invalid_sparse_none = root / "trusted-invalid-sparse.h5ad"
        subprocess.run([driver, "write-invalid-sparse", str(invalid_sparse_none), "none"], check=True)
        expect_read_error(driver, invalid_sparse_none, "invalid_sparse_structure", "/X")
        subprocess.run([driver, "read-none", str(invalid_sparse_none)], check=True)
        must_fail(driver, "write-invalid-sparse", str(root / "invalid-sparse-full-write.h5ad"), "full",
                  expected_code="invalid_sparse_structure", path_fragment="/X")

        print(
            "H5AD interop passed: AnnData 0.13.4 -> C++ -> Python, "
            "C++ -> Python, 0-D arrays, both byte orders, compact sparse indexes, "
            "empty axes, compression, and contextual diagnostics"
        )


if __name__ == "__main__":
    main()
