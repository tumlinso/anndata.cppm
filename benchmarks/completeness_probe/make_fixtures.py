#!/usr/bin/env python3
"""Create bounded AnnData fixtures varying inner chunks per identical shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

if os.environ.get("ANNDATA_INTEROP_ZARR_WAKEUP") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from python_storage import install_restricted_runner_wakeup
    install_restricted_runner_wakeup()

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from zarr.storage import LocalStore


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def files(path: Path) -> dict[str, dict[str, int | str]]:
    return {p.relative_to(path).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(path.rglob("*")) if p.is_file()}


def check_seed(seed: Path) -> dict:
    manifest = json.loads((seed / "manifest.json").read_text())
    expected = manifest["fixtures"]["dense_zarr3"]["files"]
    actual = files(seed / "dense_zarr3")
    if actual != expected: raise RuntimeError("preserved seed fixture does not match its manifest")
    return manifest


def data(values: np.ndarray) -> ad.AnnData:
    rows, cols = values.shape
    obs = pd.DataFrame({"group": pd.Categorical(["B" if i % 3 == 0 else "T" for i in range(rows)],
                                                categories=["T", "B", "unused"], ordered=True)},
                       index=pd.Index([f"cell{i}" for i in range(rows)], name="cell_id"))
    var = pd.DataFrame({"score": np.arange(cols, dtype=np.int32)},
                       index=pd.Index([f"feature{i}" for i in range(cols)], name="feature_id"))
    a = ad.AnnData(X=values.copy(), obs=obs, var=var)
    indices = np.empty(3*rows, dtype=np.int32)
    for i in range(rows): indices[3*i:3*i+3] = (i % cols, i % cols, (37*i+19) % cols)
    offsets = np.arange(0, 3*rows+1, 3, dtype=np.int64)
    values_sparse = np.tile(np.array([1,0,3], dtype=np.float32), rows)
    counts = sp.csr_matrix((values_sparse, indices, offsets), shape=(rows, cols))
    counts.indices, counts.indptr = indices, offsets
    a.layers["counts"] = counts
    a.uns["note"] = "unchanged μ"
    a.raw = ad.AnnData(X=values[:, :128].copy(), obs=obs.copy(), var=var.iloc[:128].copy())
    return a


def create(path: Path, a: ad.AnnData, inner: tuple[int, int], shard: tuple[int, int]) -> None:
    store = LocalStore(str(path))
    with ad.settings.override(zarr_write_format=3), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        a.write_zarr(store, chunks=inner, convert_strings_to_categoricals=False,
                     consolidate_metadata=False)
    root = zarr.open_group(store, mode="a", use_consolidated=False)
    attrs = dict(root["X"].attrs)
    del root["X"]
    x = root.create_array("X", data=np.asarray(a.X), chunks=inner, shards=shard)
    x.attrs.update(attrs)
    zarr.consolidate_metadata(root.store)
    check = ad.read_zarr(store)
    np.testing.assert_array_equal(check.X, a.X)
    np.testing.assert_array_equal(check.layers["counts"].data, a.layers["counts"].data)
    assert check.obs["group"].cat.ordered and list(check.obs["group"].cat.categories) == ["T", "B", "unused"]
    assert check.raw is not None and check.raw.shape == a.raw.shape


def create_huge(path: Path) -> None:
    # Only metadata is initialized: four points will be written into an otherwise
    # fill-valued array. Never ask Python AnnData to materialize the full object.
    store = LocalStore(str(path))
    small = ad.AnnData(X=np.zeros((2,2), dtype=np.float32))
    with ad.settings.override(zarr_write_format=3), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        small.write_zarr(store, convert_strings_to_categoricals=False, consolidate_metadata=False)
    root = zarr.open_group(store, mode="a", use_consolidated=False)
    attrs = dict(root["X"].attrs)
    del root["X"]
    x = root.create_array("X", shape=(1<<30,1<<20), dtype="float32", chunks=(64,128),
                          shards=(256,512), fill_value=0)
    x.attrs.update(attrs)
    for axis, length in (("obs",1<<30),("var",1<<20)):
        group = root[axis]; name = group.attrs["_index"]
        previous = group[name]; dtype, saved = previous.dtype, dict(previous.attrs)
        del group[name]
        index = group.create_array(name, shape=(length,), dtype=dtype, chunks=(1024,), fill_value="")
        index.attrs.update(saved)
    zarr.consolidate_metadata(root.store)
    check = zarr.open_group(store, mode="r", use_consolidated=False)
    assert check["X"].shape == (1<<30,1<<20)
    assert float(check["X"][1,1]) == 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("seed", type=Path, help="preserved prior streaming fixture directory")
    p.add_argument("output", type=Path, help="new output directory; must not exist")
    args = p.parse_args()
    seed, root = args.seed.resolve(), args.output.resolve()
    check_seed(seed)
    root.mkdir(parents=True, exist_ok=False)
    source = zarr.open_group(LocalStore(str(seed/"dense_zarr3")), mode="r", use_consolidated=False)
    values = np.asarray(source["X"][:512,:1024]); a = data(values)
    layouts = {"zarr3_4": (128,256), "zarr3_16": (64,128), "zarr3_64": (32,64)}
    for name, inner in layouts.items(): create(root/name, a, inner, (256,512))
    create_huge(root/"huge_gap")
    manifest = {"schema": "anndata.completeness.fixture.v1", "seed_manifest_sha256": digest(seed/"manifest.json"),
                "generator_sha256": digest(Path(__file__).resolve()), "shape": list(values.shape),
                "layouts": {k: {"inner": list(v), "shard": [256,512], "index_bytes": 4+16*(256//v[0])*(512//v[1]),
                                "files": files(root/k)} for k,v in layouts.items()},
                "huge_gap": {"shape": [1<<30,1<<20], "inner": [64,128], "shard": [256,512],
                             "index_bytes": 260, "files": files(root/"huge_gap")}}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"root":str(root),"manifest_sha256":digest(root/"manifest.json"),
                      "seed_manifest_sha256":manifest["seed_manifest_sha256"],"layouts":list(layouts)}))


if __name__ == "__main__": main()
