"""Independent file fixtures for shard-batched and irregular numerical writes.

Generated inputs and outputs remain below ARTIFACT_DIRECTORY. The focused
--threaded mode runs real concurrent gathered writes through independent C++
contexts; it is also the ThreadSanitizer gate.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import shutil
import stat
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
import zarr
from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec, ZstdCodec
from zarr.storage import LocalStore

from python_storage import replace_boundary_zarr_array
from python_streaming import equal_sparse, fixture, metadata_hashes

N, M = 35, 67
INNER, SHARD = (8, 16), (16, 32)
FILL = np.float32(-7)
MISSING = (1 << 64) - 1
TYPES = {"b": np.bool_, "i1": np.int8, "u1": np.uint8, "i2": np.int16,
         "u2": np.uint16, "i4": np.int32, "u4": np.uint32, "i8": np.int64,
         "u8": np.uint64, "f4": np.float32, "f8": np.float64}


def type_initial(dtype: type) -> np.ndarray:
    raw = np.arange(9*13, dtype=np.uint8).reshape(9, 13)
    return (raw % 2 != 0) if dtype == np.bool_ else raw.astype(dtype)


def type_expected(dtype: type, original: np.ndarray) -> np.ndarray:
    out = original.copy()
    if dtype == np.bool_:
        values = np.asarray([bool(i % 2) for i in range(20)], dtype=dtype).reshape(4, 5)
    elif np.issubdtype(dtype, np.signedinteger):
        values = np.asarray([int(np.iinfo(dtype).min)+i for i in range(20)], dtype=dtype).reshape(4, 5)
    elif np.issubdtype(dtype, np.unsignedinteger):
        values = np.asarray([int(np.iinfo(dtype).max)-i for i in range(20)], dtype=dtype).reshape(4, 5)
    else:
        values = np.asarray([0.125*i+0.25 for i in range(20)], dtype=dtype).reshape(4, 5)
    for i, row in enumerate((8, 1, 8, 0)):
        for j, col in enumerate((12, 2, 12, 0, 5)):
            out[row, col] = values[i, j]
    return out


def make_seed(path: Path, kind: str, *, large: bool = False, huge: bool = False,
              budget: bool = False, nan_fill: bool = False, typed: bool = False) -> ad.AnnData:
    a = fixture()
    fill = np.float32(np.nan) if nan_fill else FILL
    if kind.startswith("zarr3"):
        a.X[16:32, 0:32] = fill       # A missing complete shard.
        a.X[0:8, 32:48] = fill        # A missing inner chunk in a present shard.
    if budget:
        a.X[:] = FILL
    if large:
        a.uns["large"] = np.zeros((1, 1), dtype=np.float32)
    if huge:
        a.uns["huge"] = np.zeros((1, 1), dtype=np.float32)
    if typed:
        for name, dtype in TYPES.items(): a.uns["typed_"+name] = type_initial(dtype)
    if kind == "hdf5":
        a.write_h5ad(path, convert_strings_to_categoricals=False)
        with h5py.File(path, "r+") as f:
            attrs = dict(f["X"].attrs)
            del f["X"]
            x = f.create_dataset("X", data=np.asarray(a.X), chunks=INNER, compression="gzip")
            for key, value in attrs.items():
                x.attrs[key] = value
            if large:
                attrs = dict(f["uns/large"].attrs)
                del f["uns/large"]
                x = f["uns"].create_dataset("large", shape=(2048, 1024), dtype=np.float32,
                                             chunks=(64, 128), fillvalue=0)
                for key, value in attrs.items():
                    x.attrs[key] = value
        return a

    fmt = 2 if kind == "zarr2" else 3
    with ad.settings.override(zarr_write_format=fmt), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        a.write_zarr(LocalStore(str(path)), convert_strings_to_categoricals=False,
                     consolidate_metadata=True)
    if fmt == 2:
        replace_boundary_zarr_array(path, "X", 2, np.asarray(a.X))
    else:
        group = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
        attrs = dict(group["X"].attrs)
        del group["X"]
        if budget:
            codec = ShardingCodec(chunk_shape=(1, 1),
                                  codecs=(BytesCodec(endian="little"), ZstdCodec(level=1)),
                                  index_location="start")
            x = group.create_array("X", shape=(N, M), dtype=np.float32, chunks=SHARD,
                                   serializer=codec, compressors=[], fill_value=float(FILL))
        else:
            transposed = kind == "zarr3-start"
            inner = ([TransposeCodec(order=(1, 0))] if transposed else [])
            inner += [BytesCodec(endian="little" if transposed else "big"), ZstdCodec(level=1)]
            codec = ShardingCodec(chunk_shape=INNER, codecs=inner,
                                  index_location="start" if transposed else "end")
            x = group.create_array("X", data=np.asarray(a.X), chunks=SHARD,
                                   serializer=codec, compressors=[], fill_value=float(fill),
                                   config={"write_empty_chunks": False})
        x.attrs.update(attrs)
        zarr.consolidate_metadata(group.store)
        assert tuple(x.shards) == SHARD
        assert tuple(x.chunks) == ((1, 1) if budget else INNER)

    if typed:
        assert fmt == 3
        group = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
        uns = group["uns"]
        for name, dtype in TYPES.items():
            key = "typed_"+name
            attrs = dict(uns[key].attrs)
            del uns[key]
            endian = "little" if np.dtype(dtype).itemsize == 1 else "big"
            codec = ShardingCodec(chunk_shape=(4, 4),
                                  codecs=(BytesCodec(endian=endian), ZstdCodec(level=1)))
            fill = False if dtype == np.bool_ else 0
            target = uns.create_array(key, data=np.asarray(a.uns[key]), chunks=(8, 8),
                                      serializer=codec, compressors=[], fill_value=fill)
            target.attrs.update(attrs)
        zarr.consolidate_metadata(group.store)

    if large or huge:
        group = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
        uns = group["uns"]
        key = "large" if large else "huge"
        attrs = dict(uns[key].attrs)
        del uns[key]
        shape = (2048, 1024) if large else (1 << 30, 1 << 20)
        if fmt == 2:
            target = uns.create_array(key, shape=shape, dtype=np.float32, chunks=(64, 128), fill_value=0)
        else:
            codec = ShardingCodec(chunk_shape=(16, 32),
                                  codecs=(BytesCodec(endian="little"), ZstdCodec(level=1)))
            target = uns.create_array(key, shape=shape, dtype=np.float32, chunks=(64, 128),
                                      serializer=codec, compressors=[], fill_value=0)
        target.attrs.update(attrs)
        zarr.consolidate_metadata(group.store)
    return a


def files(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def shard_modes(path: Path) -> dict[str, int]:
    base = path / "X" / "c"
    return {str(p.relative_to(base)): stat.S_IMODE(p.stat().st_mode)
            for p in base.rglob("*") if p.is_file()}


def shard_payloads(path: Path, array_path: str = "X") -> dict[tuple[int, int], bytes | None]:
    """Read little-endian index entries; offsets can change, untouched payloads cannot."""
    array = path / array_path
    meta = json.loads((array / "zarr.json").read_text())
    config = meta["codecs"][0]["configuration"]
    inner = tuple(config["chunk_shape"])
    outer = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    per = (outer[0]//inner[0], outer[1]//inner[1])
    assert all(x > 0 for x in per) and all(outer[i] % inner[i] == 0 for i in range(2))
    assert [c["name"] for c in config["index_codecs"]] == ["bytes", "crc32c"]
    assert config["index_codecs"][0]["configuration"]["endian"] == "little"
    start = config["index_location"] == "start"
    size = per[0]*per[1]*16 + 4
    found: dict[tuple[int, int], bytes | None] = {}
    base = array / "c"
    for p in base.rglob("*") if base.exists() else ():
        if not p.is_file():
            continue
        shard = tuple(int(x) for x in p.relative_to(base).parts)
        assert len(shard) == 2
        data = p.read_bytes()
        index = data[:size] if start else data[-size:]
        entries = np.frombuffer(index[:-4], dtype="<u8").reshape(per[0], per[1], 2)
        for i in range(per[0]):
            for j in range(per[1]):
                offset, length = (int(x) for x in entries[i, j])
                if offset == MISSING or length == MISSING:
                    assert offset == length == MISSING
                    payload = None
                else:
                    assert 0 <= offset <= len(data) and 0 < length <= len(data)-offset
                    payload = data[offset:offset+length]
                found[(shard[0]*per[0]+i, shard[1]*per[1]+j)] = payload
    return found


def expected_x(mode: str, original: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = original.copy()
    changed: set[tuple[int, int]] = set()

    def assign(rows: list[int], cols: list[int], values: np.ndarray) -> None:
        # Explicit traversal establishes the baseline C-order last-write result.
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                out[row, col] = values[i, j]
                changed.add((row//INNER[0], col//INNER[1]))

    if mode == "gather-negative":
        assign([34, 1, 17, 1, 0, 16, 34, 8], [66, 2, 33, 2, 0, 31, 66, 16],
               np.arange(10000, 10064, dtype=np.float32)[::-1].reshape(8, 8))
    elif mode == "broadcast":
        assign([2, 19, 2, 32], [1, 17, 33, 49, 65],
               np.broadcast_to(np.arange(20200, 20205, dtype=np.float32), (4, 5)))
    elif mode == "gathered-source":
        values = np.arange(20000, 20090, dtype=np.float32).reshape(9, 10)
        assign([1, 18, 1, 34, 16, 0], [2, 34, 2, 66, 32, 1, 65],
               values[np.ix_([8, 0, 8, 2, 5, 7], [9, 1, 9, 0, 2, 8, 3])])
    elif mode == "reverse":
        assign(list(range(34, 3, -3)), list(range(66, 0, -5)),
               np.arange(11000, 11154, dtype=np.float32).reshape(11, 14))
    elif mode == "rectangle":
        values = 13000 + np.arange(27, dtype=np.float32)[:, None]*100 + np.arange(51, dtype=np.float32)[None, :]
        assign(list(range(4, 31)), list(range(9, 60)), values)
    elif mode == "parallel":
        for i in range(4):
            r, c = (i//2)*16, (i%2)*32
            values = np.asarray([[32000+100*i+10*j+k for k in range(3)] for j in range(3)], dtype=np.float32)
            assign([r+1, r, r+1], [c+2, c+1, c+2], values)
    elif mode == "special":
        assign([34, 0], [66, 0], np.asarray([[-0.0, np.inf], [-np.inf, np.nan]], dtype=np.float32))
    else:
        assert mode in ("empty", "budget", "bounded-source", "huge", "all-types")
    return out, changed


def check_ann_data(path: Path, kind: str, expected: ad.AnnData, x: np.ndarray,
                   *, large: bool = False, typed: bool = False) -> None:
    actual = ad.read_h5ad(path) if kind == "hdf5" else ad.read_zarr(LocalStore(str(path)))
    np.testing.assert_array_equal(actual.X, x)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    np.testing.assert_array_equal(actual.obsm["embedding"], expected.obsm["embedding"])
    assert actual.uns["description"] == expected.uns["description"]
    for key in ("csr", "csc"):
        equal_sparse(actual.layers[key], expected.layers[key])
    assert actual.raw is not None and expected.raw is not None
    pd.testing.assert_frame_equal(actual.raw.var, expected.raw.var)
    np.testing.assert_array_equal(actual.raw.X, expected.raw.X)
    if large:
        tile = np.asarray([[51000, 51001], [51002, 51003]], dtype=np.float32)
        np.testing.assert_array_equal(actual.uns["large"], np.tile(tile, (1024, 512)))
    if typed:
        for name, dtype in TYPES.items():
            values = actual.uns["typed_"+name]
            assert values.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(values, type_expected(dtype, type_initial(dtype)))


def check_huge(path: Path, expected: ad.AnnData) -> None:
    group = zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=False)
    array = group["uns/huge"]
    n, m = 1 << 30, 1 << 20
    assert array.shape == (n, m)
    rows, cols = [n-1, 0, n-1, n//2], [m-1, 0, m-1, m//2]
    values: dict[tuple[int, int], float] = {}
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            values[(row, col)] = float(43000+i*4+j)
    for key, value in values.items():
        assert float(array[key]) == value
    assert float(array[1, 1]) == 0
    chunks = path / "uns" / "huge" / "c"
    stored = [p for p in chunks.rglob("*") if p.is_file()]
    assert 0 < len(stored) <= len(values), "huge gather materialized extra shard files"
    np.testing.assert_array_equal(np.asarray(group["X"]), np.asarray(expected.X))
    assert group["uns"].attrs["encoding-type"] == "dict"


def run(driver: Path, mode: str, kind: str, path: Path, *, bounded: bool = False) -> None:
    def limit() -> None:
        cap = 512*1024*1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    subprocess.run([str(driver), mode, "hdf5" if kind == "hdf5" else "zarr", str(path)],
                   check=True, timeout=30,
                   preexec_fn=limit if bounded and os.environ.get("ANNDATA_COMPLETENESS_NO_AS_LIMIT") != "1" else None)


def main() -> None:
    threaded = len(sys.argv) == 4 and sys.argv[3] == "--threaded"
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and not threaded):
        raise SystemExit("usage: python_completeness.py DRIVER ARTIFACT_DIRECTORY [--threaded]")
    for name, expected in (("anndata", "0.13.4"), ("zarr", "3.4.2")):
        if version(name) != expected:
            raise SystemExit(f"expected {name} {expected}, got {version(name)}")
    driver, artifacts = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="completeness-", dir=artifacts))
    kinds = ("zarr3-start", "zarr3-end") if threaded else ("hdf5", "zarr2", "zarr3-start", "zarr3-end")
    for kind in kinds:
        suffix = ".h5ad" if kind == "hdf5" else ".zarr"
        seed = root / (kind + "-seed" + suffix)
        original = make_seed(seed, kind)
        modes = ("parallel",) if threaded else ("gather-negative", "broadcast", "gathered-source", "reverse", "rectangle", "empty", "special")
        if kind.startswith("zarr3") and not threaded:
            old = shard_payloads(seed)
            assert old.get((0, 2)) is None, "fixture must contain a missing inner chunk"
            assert not (seed / "X/c/1/0").exists(), "fixture must contain a missing shard"
        for mode in modes:
            target = root / (kind + "-" + mode + suffix)
            if kind == "hdf5": shutil.copy2(seed, target)
            else: shutil.copytree(seed, target)
            if kind.startswith("zarr3") and mode == "gather-negative":
                (target / "X/c/0/0").chmod(0o640)
            before_meta = {} if kind == "hdf5" else metadata_hashes(target)
            before_all = files(target) if kind != "hdf5" and mode == "empty" else None
            before_payload = shard_payloads(target) if kind.startswith("zarr3") else {}
            before_modes = shard_modes(target) if kind.startswith("zarr3") else {}
            run(driver, mode, kind, target)
            expected, changed = expected_x(mode, np.asarray(original.X))
            if before_meta:
                assert metadata_hashes(target) == before_meta, "payload write changed Zarr metadata"
            if before_all is not None:
                assert files(target) == before_all, "empty write changed the store"
            if before_payload:
                after = shard_payloads(target)
                for key in before_payload.keys() | after.keys():
                    if key not in changed:
                        assert before_payload.get(key) == after.get(key), f"untouched encoded chunk changed: {key}"
            if before_modes:
                after_modes = shard_modes(target)
                for key, value in before_modes.items():
                    assert after_modes.get(key) == value, f"shard permissions changed: {key}"
            check_ann_data(target, kind, original, expected)
            if mode == "special":
                if kind == "hdf5":
                    with h5py.File(target) as f: zero = f["X"][34, 66]
                else:
                    zero = zarr.open_group(LocalStore(str(target)), mode="r", use_consolidated=False)["X"][34, 66]
                assert zero == 0 and np.signbit(zero), "negative zero sign was lost"
            print(f"{kind} {mode}: exact logical values and independent readback passed", flush=True)

    if not threaded:
        target = root / "zarr3-all-types.zarr"
        original = make_seed(target, "zarr3-end", typed=True)
        before = metadata_hashes(target)
        payload = {name: shard_payloads(target, "uns/typed_"+name) for name in TYPES}
        run(driver, "all-types", "zarr3-end", target)
        assert metadata_hashes(target) == before
        changed = {(r//4, c//4) for r in (8, 1, 8, 0) for c in (12, 2, 12, 0, 5)}
        for name in TYPES:
            after = shard_payloads(target, "uns/typed_"+name)
            for key in payload[name].keys() | after.keys():
                if key not in changed: assert payload[name].get(key) == after.get(key)
        check_ann_data(target, "zarr3-end", original, np.asarray(original.X), typed=True)
        print("all supported numeric dtypes and native widths preserved", flush=True)
        target = root / "zarr3-nan-fill.zarr"
        original = make_seed(target, "zarr3-end", nan_fill=True)
        before, payload = metadata_hashes(target), shard_payloads(target)
        run(driver, "gather-negative", "zarr3-end", target)
        expected, changed = expected_x("gather-negative", np.asarray(original.X))
        assert metadata_hashes(target) == before
        after = shard_payloads(target)
        for key in payload.keys() | after.keys():
            if key not in changed: assert payload.get(key) == after.get(key)
        check_ann_data(target, "zarr3-end", original, expected)
        print("NaN fill and missing chunks preserved", flush=True)
        for kind in ("hdf5", "zarr2", "zarr3-end"):
            suffix = ".h5ad" if kind == "hdf5" else ".zarr"
            target = root / (kind + "-bounded-source" + suffix)
            original = make_seed(target, kind, large=True)
            before = {} if kind == "hdf5" else metadata_hashes(target)
            run(driver, "bounded-source", kind, target, bounded=True)
            if before: assert metadata_hashes(target) == before
            check_ann_data(target, kind, original, np.asarray(original.X), large=True)
            print(f"{kind} gathered source: bounded numeric staging passed", flush=True)
        target = root / "zarr3-budget.zarr"
        original = make_seed(target, "zarr3-start", budget=True)
        before = files(target)
        run(driver, "budget", "zarr3-start", target)
        assert files(target) == before, "scratch-budget rejection modified the destination"
        check_ann_data(target, "zarr3-start", original, np.asarray(original.X))
        print("scratch-budget rejection leaves destination unchanged", flush=True)
        target = root / "zarr3-huge.zarr"
        original = make_seed(target, "zarr3-start", huge=True)
        before = metadata_hashes(target)
        run(driver, "huge", "zarr3-start", target, bounded=True)
        assert metadata_hashes(target) == before
        check_huge(target, original)
        print("metadata-only 2^30 × 2^20 gathered write passed under 512 MiB address limit", flush=True)
    print(f"retained fixtures: {root}")


if __name__ == "__main__":
    main()
