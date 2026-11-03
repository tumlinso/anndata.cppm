#!/usr/bin/env python3
"""Generate one valid AnnData Zarr3 fixture with controlled X and index geometry."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import warnings


class PollPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick() -> None:
            loop.call_later(0.010, tick)

        loop.call_later(0.010, tick)
        return loop


asyncio.set_event_loop_policy(PollPolicy())

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from zarr.storage import LocalStore

SHAPE = (512, 512)
SOURCE_INNER_CHUNKS = (32, 64)
SOURCE_SHARDS = (256, 512)
INDEX_CHUNK = (64,)
SEED = 20261103


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_fixture.py SOURCE_ZARR3_PATH")
    path = Path(sys.argv[1]).resolve()
    if path.exists():
        raise SystemExit(f"refusing to overwrite fixture path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    x = rng.random(SHAPE, dtype=np.float32)
    obs = pd.DataFrame(index=[f"cell{i:03d}" for i in range(SHAPE[0])])
    var = pd.DataFrame(index=[f"gene{i:03d}" for i in range(SHAPE[1])])
    data = ad.AnnData(X=x, obs=obs, var=var)

    with ad.settings.override(zarr_write_format=3), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        data.write_zarr(
            LocalStore(str(path)),
            chunks=SOURCE_INNER_CHUNKS,
            convert_strings_to_categoricals=False,
            consolidate_metadata=False,
        )

    root = zarr.open_group(LocalStore(str(path)), mode="a", use_consolidated=False)
    x_attrs = dict(root["X"].attrs)
    del root["X"]
    replacement = root.create_array(
        "X", data=x, chunks=SOURCE_INNER_CHUNKS, shards=SOURCE_SHARDS
    )
    replacement.attrs.update(x_attrs)

    for axis in ("obs", "var"):
        axis_group = root[axis]
        old_index = axis_group["_index"]
        labels = np.asarray(old_index[:])
        index_attrs = dict(old_index.attrs)
        del axis_group["_index"]
        new_index = axis_group.create_array(
            "_index", data=labels, chunks=INDEX_CHUNK
        )
        new_index.attrs.update(index_attrs)

    zarr.consolidate_metadata(root.store)

    reopened = zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=False)
    x_metadata = json.loads((path / "X" / "zarr.json").read_text())
    index_metadata = {
        axis: json.loads((path / axis / "_index" / "zarr.json").read_text())
        for axis in ("obs", "var")
    }
    assert tuple(reopened["X"].shape) == SHAPE
    assert tuple(reopened["X"].chunks) == SOURCE_INNER_CHUNKS
    assert tuple(reopened["X"].shards) == SOURCE_SHARDS
    assert np.array_equal(np.asarray(reopened["X"][:]), x)
    for axis in ("obs", "var"):
        assert tuple(reopened[axis]["_index"].chunks) == INDEX_CHUNK
        assert len(reopened[axis]["_index"][:]) == SHAPE[0]

    records = {
        "path": str(path),
        "shape": list(SHAPE),
        "dtype": "float32",
        "seed": SEED,
        "useful_x_payload_bytes": int(x.nbytes),
        "zarr_format": 3,
        "x_metadata": x_metadata,
        "index_metadata": index_metadata,
        "x_metadata_sha256": file_sha256(path / "X" / "zarr.json"),
        "x_files": {
            file.relative_to(path).as_posix(): {
                "bytes": file.stat().st_size,
                "sha256": file_sha256(file),
            }
            for file in sorted((path / "X").rglob("*")) if file.is_file()
        },
        "index_chunk_size_labels": 64,
        "event_loop_policy": "PollPolicy installed before importing AnnData or Zarr",
    }
    manifest = json.dumps(records, indent=2, sort_keys=True).encode()
    records["fixture_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    evidence = path.parent / "fixture-evidence.json"
    evidence.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "path": str(path),
        "x_shape": list(SHAPE),
        "x_inner_chunks": list(SOURCE_INNER_CHUNKS),
        "x_shards": list(SOURCE_SHARDS),
        "index_chunk_size_labels": 64,
        "x_metadata_sha256": records["x_metadata_sha256"],
        "fixture_manifest_sha256": records["fixture_manifest_sha256"],
        "fixture_evidence": str(evidence),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
