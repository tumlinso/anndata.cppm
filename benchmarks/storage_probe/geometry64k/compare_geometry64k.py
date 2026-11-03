#!/usr/bin/env python3
"""Independently compare a completed 512x512 Zarr3 export to its source."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys


class PollPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick() -> None:
            loop.call_later(0.010, tick)

        loop.call_later(0.010, tick)

        return loop


asyncio.set_event_loop_policy(PollPolicy())

import numpy as np
import zarr


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tree_manifest(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            records[item.relative_to(path).as_posix()] = {
                "bytes": item.stat().st_size,
                "sha256": digest,
            }
    return records


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_geometry64k.py SOURCE DESTINATION JSON_RESULT")
    source_path, destination_path, result_path = map(Path, sys.argv[1:])
    source_path = source_path.resolve()
    destination_path = destination_path.resolve()
    source_meta = json.loads((source_path / "X" / "zarr.json").read_text())
    destination_meta = json.loads((destination_path / "X" / "zarr.json").read_text())
    source = zarr.open_group(str(source_path), mode="r")
    destination = zarr.open_group(str(destination_path), mode="r")
    expected = np.asarray(source["X"][:, :])
    actual = np.asarray(destination["X"][:, :])
    assert expected.shape == (512, 512), expected.shape
    assert actual.shape == expected.shape, actual.shape
    assert expected.dtype == np.dtype("float32"), expected.dtype
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert np.array_equal(actual, expected, equal_nan=True), "destination X differs from source"
    assert source_meta["shape"] == [512, 512], source_meta["shape"]
    assert destination_meta["shape"] == [512, 512], destination_meta["shape"]
    manifest = tree_manifest(destination_path)
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    record = {
        "status": "ok",
        "source": str(source_path),
        "destination": str(destination_path),
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "exact_values_equal": True,
        "useful_payload_bytes": int(expected.nbytes),
        "destination_physical_bytes": tree_bytes(destination_path),
        "source_x_metadata": source_meta,
        "destination_x_metadata": destination_meta,
        "destination_tree_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "destination_file_count": len(manifest),
        "event_loop_policy": "PollPolicy installed before importing Zarr",
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: record[key] for key in (
        "status", "shape", "dtype", "exact_values_equal", "useful_payload_bytes",
        "destination_physical_bytes", "destination_file_count",
        "destination_tree_manifest_sha256", "event_loop_policy"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
