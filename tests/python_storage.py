"""Independent Python interoperability checks for backed H5AD and Zarr stores.

The fixture writers and file readers are the pinned upstream implementations
from ``tests/requirements-interop.txt``.  Zarr event-loop polling is available
only through ``ANNDATA_INTEROP_ZARR_WAKEUP=1`` for syscall-restricted test
runners; ordinary CI keeps asyncio's default policy.
"""
from __future__ import annotations

import asyncio
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Optional
import warnings


def install_restricted_runner_wakeup() -> None:
    if os.environ.get("ANNDATA_INTEROP_ZARR_WAKEUP") != "1":
        return

    class PollPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            loop = super().new_event_loop()

            def tick() -> None:
                loop.call_later(0.010, tick)

            loop.call_later(0.010, tick)
            return loop

    asyncio.set_event_loop_policy(PollPolicy())


# Install before AnnData or Zarr can create their process-local async loop.
install_restricted_runner_wakeup()

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, VLenUTF8Codec, ZstdCodec
from zarr.storage import LocalStore

from python_h5ad import make_fixture, patch_raw_enum_byte, write_python_fixture


EXPECTED_X = np.arange(6, dtype=np.float32).reshape(3, 2)
MUTATED_X = np.array([[5.0], [11.0]], dtype=np.float32)


def run(driver: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([driver, *args], text=True, capture_output=True)
    if result.returncode != 0:
        raise AssertionError(
            f"storage driver failed ({args}): {result.stderr.strip()}\n{result.stdout}"
        )
    return result


def expect_failure(driver: str, *args: str, code: str) -> str:
    result = subprocess.run([driver, *args], text=True, capture_output=True)
    if result.returncode == 0:
        raise AssertionError(f"expected storage driver failure: {args}")
    if f"anndata error[{code}]" not in result.stderr:
        raise AssertionError(f"expected {code}, got: {result.stderr.strip()}")
    return result.stderr


def assert_rich_object(a: ad.AnnData, *, x: np.ndarray) -> None:
    assert a.shape == x.shape
    np.testing.assert_array_equal(np.asarray(a.X), x)
    assert a.obs_names.tolist() == ["cell-a", "cell-a", "cell-γ"]
    assert a.obs_names.name == "cell_id"
    assert a.var_names.tolist() == ["gene-1", "gene-2"]
    assert a.var_names.name == "gene_id"

    kind = a.obs["kind"]
    assert kind.cat.categories.tolist() == ["T", "B", "unused"]
    assert kind.cat.ordered
    assert kind.cat.codes.tolist() == [0, 1, -1]
    unsigned_kind = a.obs["kind_unsigned"]
    assert unsigned_kind.cat.categories.tolist() == ["T", "B", "unused"]
    assert not unsigned_kind.cat.ordered
    assert unsigned_kind.cat.codes.tolist() == [0, 1, 0]

    count = a.obs["count"]
    assert str(count.dtype) == "Int16"
    assert count.iloc[0] == 7 and pd.isna(count.iloc[1]) and count.iloc[2] == -2
    enabled = a.obs["enabled"]
    assert str(enabled.dtype) == "boolean"
    assert bool(enabled.iloc[0]) and pd.isna(enabled.iloc[1]) and not bool(enabled.iloc[2])
    label = a.obs["label"]
    label_nan = a.obs["label_nan"]
    assert label.iloc[0] == "α" and pd.isna(label.iloc[1]) and label.iloc[2] == "cell γ"
    assert label_nan.iloc[0] == "left" and pd.isna(label_nan.iloc[1]) and label_nan.iloc[2] == "right"
    assert getattr(label.dtype, "na_value", None) is pd.NA
    assert isinstance(label_nan.dtype, pd.StringDtype)
    assert np.isnan(label_nan.dtype.na_value)
    assert a.obs["unsigned"].dtype == np.dtype("uint8")
    np.testing.assert_array_equal(a.obs["unsigned"].to_numpy(), [1, 255, 7])

    counts = a.layers["counts"]
    assert sp.isspmatrix_csr(counts)
    np.testing.assert_array_equal(counts.data, [1, 0, 2, 3, 4])
    np.testing.assert_array_equal(counts.indices, [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(counts.indptr, [0, 2, 4, 5])
    assert a.layers["dense_layer"].dtype == np.dtype("uint16")
    np.testing.assert_array_equal(a.layers["dense_layer"], np.arange(6, dtype=np.uint16).reshape(3, 2))

    np.testing.assert_array_equal(a.obsm["embedding"], np.arange(6, dtype=np.float64).reshape(3, 2))
    np.testing.assert_array_equal(a.obsm["labels"], [["a", "β"], ["", "d"], ["e", "f"]])
    assert a.obsm["table"]["score"].tolist() == [0.25, 0.5, 0.75]
    np.testing.assert_array_equal(a.varm["embedding"], [[1, 2], [3, 4]])
    assert sp.isspmatrix_csr(a.obsp["connectivities"])
    assert sp.isspmatrix_csc(a.varp["similarity"])

    assert a.uns["nested"]["description"] == "nested α"
    assert a.uns["nested"]["null"] is None
    assert float(a.uns["nested"]["threshold"]) == 0.5
    assert int(np.asarray(a.uns["zero_numeric_array"])) == 23
    assert str(np.asarray(a.uns["zero_string_array"]).item()) == "zero λ"
    np.testing.assert_array_equal(a.uns["matrix_text"], [["μ", ""], ["g2", "gene β"]])
    assert a.uns["dtypes"]["u64"].dtype == np.dtype("uint64")
    assert int(a.uns["dtypes"]["u64"][1]) == 2**63 + 1
    assert a.raw is not None
    np.testing.assert_array_equal(a.raw.X, x)
    assert a.raw.varm["label_table"]["label"].tolist() == ["raw α", "raw β"]


def add_unknown_h5ad(path: Path) -> None:
    with h5py.File(path, "r+") as f:
        f.attrs["vendor-marker"] = "keep-root-attribute"
        f["X"].attrs["vendor-marker"] = "keep-dataset-attribute"
        vendor = f.create_group("vendor_extension")
        vendor.attrs["format"] = "opaque-private-extension"
        vendor.create_dataset("payload", data=np.array([17, 29], dtype=np.uint16))


def add_unknown_zarr(path: Path) -> None:
    group = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
    group.attrs["vendor-marker"] = "keep-root-attribute"
    group["X"].attrs["vendor-marker"] = "keep-dataset-attribute"
    vendor = group.create_group("vendor_extension")
    vendor.attrs["format"] = "opaque-private-extension"
    vendor.create_array("payload", data=np.array([17, 29], dtype=np.uint16), chunks=(2,))
    zarr.consolidate_metadata(group.store)


def write_python_zarr(path: Path, zarr_format: int) -> None:
    # AnnData 0.13.4 supplies the v3 default shard layout and Zarr v2's default
    # Blosc-LZ4 plus VLenUTF8 codecs.  Disable string-to-categorical conversion
    # so nullable strings and their distinct NA sentinels remain in the fixture.
    with ad.settings.override(
        allow_write_nullable_strings=True,
        zarr_write_format=zarr_format,
    ), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        make_fixture().write_zarr(
            LocalStore(str(path)),
            convert_strings_to_categoricals=False,
            consolidate_metadata=True,
        )
    patch_unsigned_category_codes(path, zarr_format)


BOUNDARY_OBS = 35
BOUNDARY_VARS = 67
BOUNDARY_RAW_VARS = 71


def make_boundary_fixture() -> ad.AnnData:
    """A small two-axis rectangle whose extents cross several storage chunks."""
    obs_names = [f"boundary-cell-{i:03d}" for i in range(BOUNDARY_OBS)]
    var_names = [f"boundary-gene-{i:03d}" for i in range(BOUNDARY_VARS)]
    obs = pd.DataFrame(
        {
            "kind": pd.Categorical(
                ["Other" if i % 5 == 0 else "T" if i % 2 == 0 else "B"
                 for i in range(BOUNDARY_OBS)],
                categories=["T", "B", "Other"],
                ordered=True,
            ),
            "count": pd.array(
                [None if i % 7 == 0 else i * 3 for i in range(BOUNDARY_OBS)],
                dtype="Int16",
            ),
            "label": pd.array(
                [None if i % 9 == 0 else f"label-{i}" for i in range(BOUNDARY_OBS)],
                dtype="string",
            ),
        },
        index=pd.Index(obs_names, name="cell_id"),
    )
    var = pd.DataFrame(
        {"active": np.asarray([i % 3 != 0 for i in range(BOUNDARY_VARS)], dtype=np.bool_)},
        index=pd.Index(var_names, name="gene_id"),
    )
    x = np.arange(BOUNDARY_OBS * BOUNDARY_VARS, dtype=np.float32).reshape(
        BOUNDARY_OBS, BOUNDARY_VARS
    )
    a = ad.AnnData(x, obs=obs, var=var)
    sparse = (np.arange(x.size, dtype=np.int32).reshape(x.shape) + 1) % 11
    sparse[sparse % 3 != 0] = 0
    a.layers["counts"] = sp.csr_matrix(sparse)
    a.obsm["embedding"] = np.column_stack(
        (np.arange(BOUNDARY_OBS), np.arange(BOUNDARY_OBS) * 2)
    ).astype(np.float32)
    a.varm["weights"] = np.column_stack(
        (np.arange(BOUNDARY_VARS), np.arange(BOUNDARY_VARS) * 3)
    ).astype(np.float32)
    a.obsp["connectivities"] = sp.csr_matrix(sp.eye(BOUNDARY_OBS, dtype=np.float32))
    a.varp["similarity"] = sp.csc_matrix(sp.eye(BOUNDARY_VARS, dtype=np.float64))
    a.uns["kind_colors"] = np.asarray(["#df4a4a", "#3977bb", "#888888"], dtype="U")

    raw_var = pd.DataFrame(
        {"source": np.arange(BOUNDARY_RAW_VARS, dtype=np.int16)},
        index=pd.Index([f"raw-gene-{i:03d}" for i in range(BOUNDARY_RAW_VARS)], name="gene_id"),
    )
    raw_x = (
        np.arange(BOUNDARY_OBS * BOUNDARY_RAW_VARS, dtype=np.float32).reshape(
            BOUNDARY_OBS, BOUNDARY_RAW_VARS
        )
        + np.float32(100_000)
    )
    a.raw = ad.AnnData(raw_x, obs=obs.copy(), var=raw_var)
    return a


def replace_boundary_zarr_array(
    path: Path, array_path: str, zarr_format: int, values: np.ndarray
) -> None:
    """Apply explicit chunk/shard geometry while retaining AnnData attributes."""
    group = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
    assert group.metadata.zarr_format == zarr_format
    attrs = dict(group[array_path].attrs)
    parent_path, separator, name = array_path.rpartition("/")
    parent = group[parent_path] if separator else group
    del group[array_path]
    if zarr_format == 2:
        replacement = parent.create_array(name or array_path, data=values, chunks=(8, 16))
    else:
        replacement = parent.create_array(
            name or array_path, data=values, chunks=(8, 16), shards=(16, 32)
        )
    replacement.attrs.update(attrs)
    zarr.consolidate_metadata(group.store)
    assert replacement.attrs["encoding-type"] == "array"
    assert replacement.attrs["encoding-version"] == "0.2.0"
    if zarr_format == 2:
        assert tuple(replacement.chunks) == (8, 16)
    else:
        sharding = replacement.metadata.codecs[0]
        assert isinstance(sharding, ShardingCodec)
        assert tuple(sharding.chunk_shape) == (8, 16)
        assert tuple(replacement.shards) == (16, 32)


def write_boundary_fixture(path: Path, backend: str, zarr_format: Optional[int] = None) -> ad.AnnData:
    a = make_boundary_fixture()
    with ad.settings.override(allow_write_nullable_strings=True), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        if backend == "h5ad":
            a.write_h5ad(path, convert_strings_to_categoricals=False)
        else:
            assert zarr_format in (2, 3)
            with ad.settings.override(zarr_write_format=zarr_format):
                a.write_zarr(
                    LocalStore(str(path)),
                    convert_strings_to_categoricals=False,
                    consolidate_metadata=True,
                )
            replace_boundary_zarr_array(path, "X", zarr_format, np.asarray(a.X))
            replace_boundary_zarr_array(path, "raw/X", zarr_format, np.asarray(a.raw.X))
    return a


def read_boundary_fixture(path: Path, backend: str) -> ad.AnnData:
    if backend == "h5ad":
        return ad.read_h5ad(path)
    return ad.read_zarr(LocalStore(str(path)))


def assert_boundary_object(actual: ad.AnnData, expected: ad.AnnData) -> None:
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(actual.X), np.asarray(expected.X))
    assert actual.obs_names.equals(expected.obs_names)
    assert actual.var_names.equals(expected.var_names)
    assert actual.obs_names.name == expected.obs_names.name == "cell_id"
    assert actual.var_names.name == expected.var_names.name == "gene_id"
    assert actual.obs["kind"].cat.categories.tolist() == ["T", "B", "Other"]
    assert actual.obs["kind"].cat.ordered
    pd.testing.assert_series_equal(actual.obs["kind"], expected.obs["kind"])
    assert str(actual.obs["count"].dtype) == "Int16"
    pd.testing.assert_series_equal(actual.obs["count"], expected.obs["count"])
    assert str(actual.obs["label"].dtype) == "string"
    pd.testing.assert_series_equal(actual.obs["label"], expected.obs["label"])
    np.testing.assert_array_equal(actual.uns["kind_colors"], expected.uns["kind_colors"])
    assert sp.isspmatrix_csr(actual.layers["counts"])
    np.testing.assert_array_equal(actual.layers["counts"].toarray(), expected.layers["counts"].toarray())
    np.testing.assert_array_equal(actual.obsm["embedding"], expected.obsm["embedding"])
    np.testing.assert_array_equal(actual.varm["weights"], expected.varm["weights"])
    assert sp.isspmatrix_csr(actual.obsp["connectivities"])
    np.testing.assert_array_equal(
        actual.obsp["connectivities"].toarray(), expected.obsp["connectivities"].toarray()
    )
    assert sp.isspmatrix_csc(actual.varp["similarity"])
    np.testing.assert_array_equal(
        actual.varp["similarity"].toarray(), expected.varp["similarity"].toarray()
    )
    assert actual.raw is not None and expected.raw is not None
    assert actual.raw.X.shape == expected.raw.X.shape
    assert actual.raw.X.shape[1] == BOUNDARY_RAW_VARS
    assert actual.raw.var_names.equals(expected.raw.var_names)
    np.testing.assert_array_equal(actual.raw.X, expected.raw.X)


def patch_unsigned_category_codes(path: Path, zarr_format: int) -> None:
    # Pandas stores category codes as signed integers even when every code is
    # nonnegative. Reuse the payload and change only the valid Zarr dtype
    # metadata to exercise unsigned-code decoding from an independent store.
    if zarr_format == 2:
        node_path = path / "obs" / "kind_unsigned" / "codes" / ".zarray"
        node = json.loads(node_path.read_text(encoding="utf-8"))
        assert node["dtype"] in ("|i1", "<i1")
        node["dtype"] = "|u1"
        node_path.write_text(json.dumps(node), encoding="utf-8")

        consolidated_path = path / ".zmetadata"
        consolidated = json.loads(consolidated_path.read_text(encoding="utf-8"))
        key = "obs/kind_unsigned/codes/.zarray"
        assert consolidated["metadata"][key]["dtype"] in ("|i1", "<i1")
        consolidated["metadata"][key]["dtype"] = "|u1"
        consolidated_path.write_text(json.dumps(consolidated), encoding="utf-8")
    else:
        node_path = path / "obs" / "kind_unsigned" / "codes" / "zarr.json"
        node = json.loads(node_path.read_text(encoding="utf-8"))
        assert node["data_type"] == "int8"
        node["data_type"] = "uint8"
        node_path.write_text(json.dumps(node), encoding="utf-8")

        root_path = path / "zarr.json"
        root = json.loads(root_path.read_text(encoding="utf-8"))
        key = "obs/kind_unsigned/codes"
        consolidated = root["consolidated_metadata"]["metadata"][key]
        assert consolidated["data_type"] == "int8"
        consolidated["data_type"] = "uint8"
        root_path.write_text(json.dumps(root), encoding="utf-8")


def assert_zarr_fixture_storage(path: Path, zarr_format: int, *, cpp_output: bool = False) -> None:
    group = zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=True)
    assert group.metadata.zarr_format == zarr_format
    assert group.metadata.consolidated_metadata is not None
    assert group.attrs["encoding-type"] == "anndata"
    assert group["X"].attrs["encoding-type"] == "array"
    if cpp_output:
        assert group["obs/kind_unsigned/codes"].dtype == np.dtype("uint8")

    if zarr_format == 3 and not cpp_output:
        numeric = group["X"]
        assert numeric.shards == (3, 2)
        sharding = numeric.metadata.codecs[0]
        assert isinstance(sharding, ShardingCodec)
        assert isinstance(sharding.codecs[0], BytesCodec)
        assert isinstance(sharding.codecs[1], ZstdCodec)
        assert isinstance(sharding.index_codecs[-1], Crc32cCodec)
        strings = group["obsm/labels"]
        string_sharding = strings.metadata.codecs[0]
        assert isinstance(string_sharding, ShardingCodec)
        assert isinstance(string_sharding.codecs[0], VLenUTF8Codec)
        assert isinstance(string_sharding.codecs[1], ZstdCodec)
        assert isinstance(string_sharding.index_codecs[-1], Crc32cCodec)
    elif zarr_format == 2 and not cpp_output:
        numeric = group["X"]
        assert numeric.compressor is not None
        assert numeric.compressor.cname == "lz4"
        # Zarr-Python exposes v2 object strings as NumPy StringDType in this
        # NumPy 2 environment; the persisted v2 metadata is the authoritative
        # object/VLenUTF8 contract.
        strings = json.loads((path / "obsm" / "labels" / ".zarray").read_text(encoding="utf-8"))
        assert strings["dtype"] == "|O"
        assert [item["id"] for item in strings["filters"]] == ["vlen-utf8"]
        assert strings["compressor"]["id"] == "blosc"
        assert strings["compressor"]["cname"] == "lz4"


def assert_mutated_h5ad(path: Path) -> None:
    # AnnData's H5AD reader turns every root child into an AnnData constructor
    # argument, so it cannot reopen a vendor-defined root group. Verify that
    # extension in the original file, then strip only that unsupported child
    # from a temporary semantic-validation copy.
    validation = path.with_name(path.stem + "-validation.h5ad")
    shutil.copyfile(path, validation)
    with h5py.File(path, "r") as f:
        assert f.attrs["vendor-marker"] == "keep-root-attribute"
        assert f["X"].attrs["vendor-marker"] == "keep-dataset-attribute"
        np.testing.assert_array_equal(f["vendor_extension/payload"], [17, 29])
    with h5py.File(validation, "r+") as f:
        del f["vendor_extension"]

    a = ad.read_h5ad(validation)
    assert a.shape == (2, 1)
    np.testing.assert_array_equal(a.X, MUTATED_X)
    assert a.obs_names.tolist() == ["cell-γ", "cell-a"]
    assert a.var_names.tolist() == ["gene-2"]
    np.testing.assert_array_equal(a.obsm["embedding"], [[4.0, 5.0], [0.0, 1.0]])
    np.testing.assert_array_equal(a.obsm["labels"], [["e", "f"], ["a", "β"]])
    assert a.obsm["table"]["score"].tolist() == [0.75, 0.25]
    np.testing.assert_array_equal(a.varm["embedding"], [[3, 4]])
    np.testing.assert_array_equal(a.obsp["connectivities"].toarray(), [[1.0, 0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(a.varp["similarity"].toarray(), [[1.0]])
    np.testing.assert_array_equal(a.raw.X, [[4.0, 5.0], [0.0, 1.0]])
    with h5py.File(path, "r") as f:
        assert f["obs"].attrs["_index"] == "cell_id"
        assert f["var"].attrs["_index"] == "gene_id"


def assert_mutated_zarr(path: Path, zarr_format: int) -> None:
    group = zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=True)
    assert group.metadata.zarr_format == zarr_format
    assert group.metadata.consolidated_metadata is not None
    assert group.attrs["vendor-marker"] == "keep-root-attribute"
    assert group["X"].attrs["vendor-marker"] == "keep-dataset-attribute"
    np.testing.assert_array_equal(group["vendor_extension/payload"][:], [17, 29])

    validation = path.with_name(path.stem + "-validation.zarr")
    shutil.copytree(path, validation)
    validation_group = zarr.open_group(
        LocalStore(str(validation)), mode="a", use_consolidated=False
    )
    del validation_group["vendor_extension"]
    zarr.consolidate_metadata(validation_group.store)
    a = ad.read_zarr(LocalStore(str(validation)))
    assert a.shape == (2, 1)
    np.testing.assert_array_equal(np.asarray(a.X), MUTATED_X)
    assert a.obs_names.tolist() == ["cell-γ", "cell-a"]
    assert a.var_names.tolist() == ["gene-2"]
    np.testing.assert_array_equal(a.obsm["embedding"], [[4.0, 5.0], [0.0, 1.0]])
    np.testing.assert_array_equal(a.obsm["labels"], [["e", "f"], ["a", "β"]])
    assert a.obsm["table"]["score"].tolist() == [0.75, 0.25]
    np.testing.assert_array_equal(a.varm["embedding"], [[3, 4]])
    np.testing.assert_array_equal(a.obsp["connectivities"].toarray(), [[1.0, 0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(a.varp["similarity"].toarray(), [[1.0]])
    np.testing.assert_array_equal(a.raw.X, [[4.0, 5.0], [0.0, 1.0]])


def check_upstream_lazy(path: Path, backend: str) -> None:
    if backend == "h5ad":
        with h5py.File(path, "r") as f:
            # The pinned interop environment does not install Dask. Verify the
            # upstream HDF5 representation remains metadata-backed until an
            # explicit dataset read instead of making a Dask dependency part
            # of this storage interoperability test.
            for key in ("X", "layers/dense_layer", "obsm/embedding"):
                dataset = f[key]
                assert isinstance(dataset, h5py.Dataset)
                assert dataset.shape == (3, 2)
    else:
        group = zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=True)
        assert tuple(group["X"].shape) == (3, 2)
        assert tuple(group["layers/dense_layer"].shape) == (3, 2)
        # Zarr array handles expose shape/chunk metadata without materializing
        # their payload; explicit indexing is exercised by the Python reopen
        # checks below.
        assert isinstance(group["X"], zarr.Array)
        assert isinstance(group["layers/dense_layer"], zarr.Array)


def inject_unknown_uns_encoding(path: Path) -> None:
    with h5py.File(path, "r+") as f:
        opaque = f["uns"].create_group("opaque_extension")
        opaque.attrs["encoding-type"] = "vendor-special-object"
        opaque.attrs["encoding-version"] = "99.0.0"
        opaque.create_dataset("payload", data=np.array([1], dtype=np.uint8))


def make_bad_zarr_format_copy(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)
    metadata = destination / "zarr.json"
    root_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    assert root_metadata["zarr_format"] == 3
    root_metadata["zarr_format"] = 99
    metadata.write_text(json.dumps(root_metadata), encoding="utf-8")


def mutate_zarr_v3_x_metadata(path: Path, update) -> None:
    root_metadata_path = path / "zarr.json"
    x_metadata_path = path / "X" / "zarr.json"
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    x_metadata = json.loads(x_metadata_path.read_text(encoding="utf-8"))
    update(root_metadata["consolidated_metadata"]["metadata"]["X"])
    update(x_metadata)
    root_metadata_path.write_text(json.dumps(root_metadata), encoding="utf-8")
    x_metadata_path.write_text(json.dumps(x_metadata), encoding="utf-8")


def make_bad_zarr_shard_copy(source: Path, destination: Path, *, bad_crc: bool) -> None:
    shutil.copytree(source, destination)
    shard_candidates = sorted((destination / "X" / "c").glob("**/*"))
    shard = next(path for path in shard_candidates if path.is_file())
    payload = bytearray(shard.read_bytes())
    # This v3 fixture has a single inner chunk.  Its sharding index is the
    # final 16-byte entry plus the 4-byte CRC32C value.
    assert len(payload) >= 20
    if bad_crc:
        payload[-1] ^= 0x01
    else:
        payload[-20] ^= 0x7F
    shard.write_bytes(payload)


def assert_cpp_output(driver: str, root: Path) -> None:
    for target, backend, zformat in (
        (root / "cpp.h5ad", "h5ad", None),
        (root / "cpp-v2.zarr", "zarr2", 2),
        (root / "cpp-v3.zarr", "zarr3", 3),
    ):
        run(driver, "write", backend, str(target))
        if backend == "h5ad":
            assert_rich_object(ad.read_h5ad(target), x=EXPECTED_X)
        else:
            assert_zarr_fixture_storage(target, zformat, cpp_output=True)
            assert_rich_object(ad.read_zarr(LocalStore(str(target))), x=EXPECTED_X)

    # Cross-backend writes read from backed views and are reopened independently
    # through the corresponding AnnData reader.
    from_h5ad = root / "export-h5ad-to-v3.zarr"
    from_v3 = root / "export-v3-to-h5ad.h5ad"
    run(driver, "export", "h5ad", str(root / "cpp.h5ad"), "zarr3", str(from_h5ad))
    run(driver, "export", "zarr", str(root / "cpp-v3.zarr"), "h5ad", str(from_v3))
    assert_zarr_fixture_storage(from_h5ad, 3, cpp_output=True)
    assert_rich_object(ad.read_zarr(LocalStore(str(from_h5ad))), x=EXPECTED_X)
    assert_rich_object(ad.read_h5ad(from_v3), x=EXPECTED_X)


def check_rectangular_boundary_transfers(driver: str, root: Path, only: Optional[str] = None) -> None:
    sources = (
        ("h5ad", "h5ad", "boundary-source.h5ad", None),
        ("zarr", "zarr2", "boundary-source-v2.zarr", 2),
        ("zarr", "zarr3", "boundary-source-v3.zarr", 3),
    )
    targets = (("h5ad", "h5ad"), ("zarr2", "zarr"), ("zarr3", "zarr"))

    for source_backend, source_format, filename, zarr_format in sources:
        if only is not None and source_format != only:
            continue
        source_path = root / filename
        original = write_boundary_fixture(source_path, source_format, zarr_format)
        expected = original[1:-1, 2:-2].copy()

        for target_format, target_reader in targets:
            extension = ".h5ad" if target_reader == "h5ad" else ".zarr"
            target_path = root / f"boundary-export-{source_format}-to-{target_format}{extension}"
            run(
                driver,
                "export-rectangle",
                source_backend,
                str(source_path),
                target_format,
                str(target_path),
            )
            actual = read_boundary_fixture(target_path, target_reader)
            assert_boundary_object(actual, expected)

        if source_format == "h5ad":
            reindexed_path = root / "boundary-reindexed.h5ad"
            shutil.copyfile(source_path, reindexed_path)
        else:
            reindexed_path = root / f"boundary-reindexed-{source_format}.zarr"
            shutil.copytree(source_path, reindexed_path)
        run(driver, "reindex-rectangle", source_backend, str(reindexed_path))
        reindexed = read_boundary_fixture(reindexed_path, source_backend)
        assert_boundary_object(reindexed, expected)


def assert_zarr_export_path_guards(driver: str, pristine: Path, root: Path) -> None:
    guard_root = root / "export-path-guards"
    guard_root.mkdir()
    source = guard_root / "source.zarr"
    shutil.copytree(pristine, source)

    def snapshot() -> dict[str, str]:
        return {
            item.relative_to(source).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted(source.rglob("*"))
            if item.is_file()
        }

    original = snapshot()
    targets = (
        source,
        guard_root,
        source / "nested-target.zarr",
    )
    for target in targets:
        expect_failure(
            driver,
            "export",
            "zarr",
            str(source),
            "zarr3",
            str(target),
            code="invalid_argument",
        )
        assert snapshot() == original, f"overlapping Zarr export mutated source for {target}"
    assert not (source / "nested-target.zarr").exists()

    alias = guard_root / "source-alias.zarr"
    alias.symlink_to(source, target_is_directory=True)
    expect_failure(
        driver,
        "export",
        "zarr",
        str(source),
        "zarr3",
        str(alias),
        code="invalid_argument",
    )
    assert snapshot() == original, "symlink-alias export mutated the Zarr source"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python_storage.py /path/to/storage_interop_driver")
    driver = str(Path(sys.argv[1]).resolve())
    if version("anndata") != "0.13.4":
        raise SystemExit(f"expected AnnData 0.13.4, got {version('anndata')}")
    if version("zarr") != "3.4.2":
        raise SystemExit(f"expected Zarr 3.4.2, got {version('zarr')}")

    with tempfile.TemporaryDirectory(prefix="anndata-cppm-backed-") as temp:
        root = Path(temp)

        # Python-generated H5AD is read and modified by C++; unknown HDF5
        # metadata remains byte-for-byte present on untouched storage nodes.
        python_h5ad = root / "python.h5ad"
        with ad.settings.override(allow_write_nullable_strings=True):
            write_python_fixture(python_h5ad)
        add_unknown_h5ad(python_h5ad)
        check_upstream_lazy(python_h5ad, "h5ad")
        run(driver, "inspect", "h5ad", str(python_h5ad))
        run(driver, "mutate", "h5ad", str(python_h5ad))
        assert_mutated_h5ad(python_h5ad)

        # AnnData 0.13.4 defaults to sharded Zarr v3.  Exercise both its v3
        # codecs and the explicit v2 compatibility writer from the same object.
        pristine_v3 = root / "python-v3-pristine.zarr"
        for zformat, filename in ((3, "python-v3.zarr"), (2, "python-v2.zarr")):
            path = root / filename
            write_python_zarr(path, zformat)
            assert_zarr_fixture_storage(path, zformat)
            if zformat == 3:
                shutil.copytree(path, pristine_v3)
            add_unknown_zarr(path)
            check_upstream_lazy(path, "zarr")
            run(driver, "inspect", "zarr", str(path))
            run(driver, "mutate", "zarr", str(path))
            assert_mutated_zarr(path, zformat)

        assert_cpp_output(driver, root)
        check_rectangular_boundary_transfers(driver, root)

        # Unknown opaque AnnData slot encodings may be retained in their source
        # store, but a cross-format writer must report that it cannot transcode
        # the value.  The normal source-store updates above prove preservation.
        opaque = root / "opaque-uns.h5ad"
        with ad.settings.override(allow_write_nullable_strings=True):
            write_python_fixture(opaque)
        inject_unknown_uns_encoding(opaque)
        expect_failure(
            driver,
            "export",
            "h5ad",
            str(opaque),
            "zarr3",
            str(root / "unsupported-export.zarr"),
            code="unsupported_encoding",
        )

        opaque_zarr = root / "opaque-uns.zarr"
        shutil.copytree(pristine_v3, opaque_zarr)
        group = zarr.open_group(LocalStore(str(opaque_zarr)), mode="a", use_consolidated=False)
        opaque_group = group["uns"].create_group("opaque_extension")
        opaque_group.attrs["encoding-type"] = "vendor-special-object"
        opaque_group.attrs["encoding-version"] = "99.0.0"
        opaque_group.create_array("payload", data=np.array([1], dtype=np.uint8), chunks=(1,))
        zarr.consolidate_metadata(group.store)
        expect_failure(
            driver,
            "export",
            "zarr",
            str(opaque_zarr),
            "h5ad",
            str(root / "unsupported-export.h5ad"),
            code="unsupported_encoding",
        )

        # Unsupported Zarr format and corrupt v3 shard index/checksum are
        # diagnosed only when a numerical block is requested; open remains
        # metadata-only.
        pristine = pristine_v3
        bad_format = root / "bad-format.zarr"
        bad_crc = root / "bad-shard-crc.zarr"
        bad_index = root / "bad-shard-index.zarr"
        bad_dtype = root / "bad-dtype.zarr"
        make_bad_zarr_format_copy(pristine, bad_format)
        make_bad_zarr_shard_copy(pristine, bad_crc, bad_crc=True)
        make_bad_zarr_shard_copy(pristine, bad_index, bad_crc=False)
        shutil.copytree(pristine, bad_dtype)
        mutate_zarr_v3_x_metadata(
            bad_dtype,
            lambda metadata: metadata.__setitem__("data_type", "complex64"),
        )
        bad_codec = root / "bad-codec.zarr"
        shutil.copytree(pristine, bad_codec)

        def inject_unknown_codec(metadata: dict) -> None:
            metadata["codecs"][0]["configuration"]["codecs"][1]["name"] = "vendor_codec"

        mutate_zarr_v3_x_metadata(bad_codec, inject_unknown_codec)
        expect_failure(driver, "read", "zarr", str(bad_format), code="unsupported_version")
        expect_failure(driver, "read", "zarr", str(bad_crc), code="unsupported_encoding")
        expect_failure(driver, "read", "zarr", str(bad_index), code="unsupported_encoding")
        expect_failure(driver, "read", "zarr", str(bad_dtype), code="unsupported_dtype")
        expect_failure(driver, "read", "zarr", str(bad_codec), code="unavailable_filter")

        # HDF5 bool enums can contain noncanonical nonzero bytes.  Backed reads
        # normalize them to true before exposing the native bool buffer.
        bad_bool = root / "bad-bool.h5ad"
        with ad.settings.override(allow_write_nullable_strings=True):
            write_python_fixture(bad_bool)
        patch_raw_enum_byte(bad_bool, "obs/enabled/values", 0, 2)
        run(driver, "read", "h5ad", str(bad_bool))
        reopened_bool = ad.read_h5ad(bad_bool)
        assert bool(reopened_bool.obs["enabled"].iloc[0])

        assert_zarr_export_path_guards(driver, pristine_v3, root)

    print(
        "backed storage interop passed: AnnData 0.13.4 / Zarr 3.4.2, H5AD, "
        "Zarr v2/v3, lazy regions, direct mutation, structural reindex, "
        "unknown metadata preservation, streamed cross-format export, and "
        "small rectangular chunk/shard boundaries"
    )


if __name__ == "__main__":
    main()
