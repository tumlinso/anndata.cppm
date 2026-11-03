#!/usr/bin/env python3
"""Independently compare X and obs/var indexes for the 96-row Zarr3 export."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys


class PollPolicy(asyncio.DefaultEventLoopPolicy):
    """Keep Zarr's async local store polling under the restricted runner."""

    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick() -> None:
            loop.call_later(0.010, tick)

        loop.call_later(0.010, tick)
        return loop


asyncio.set_event_loop_policy(PollPolicy())

import numpy as np
import zarr

ROW_START = 2000
ROW_STOP = 2096
N_COLUMNS = 4096


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tree_manifest(path: Path) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            entries[item.relative_to(path).as_posix()] = {
                "bytes": item.stat().st_size,
                "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
            }
    return entries


def metadata(path: Path, relative: str) -> dict[str, object]:
    return json.loads((path / relative / "zarr.json").read_text())


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_96.py SOURCE DESTINATION JSON_RESULT")
    source_path, destination_path, result_path = map(Path, sys.argv[1:])
    source_path = source_path.resolve()
    destination_path = destination_path.resolve()

    source = zarr.open_group(str(source_path), mode="r")
    destination = zarr.open_group(str(destination_path), mode="r")
    expected_x = np.asarray(source["X"][ROW_START:ROW_STOP, :])
    actual_x = np.asarray(destination["X"][:])
    assert list(expected_x.shape) == [96, N_COLUMNS], expected_x.shape
    assert list(actual_x.shape) == [96, N_COLUMNS], actual_x.shape
    assert expected_x.dtype == np.dtype("float32"), expected_x.dtype
    assert actual_x.dtype == expected_x.dtype, (actual_x.dtype, expected_x.dtype)
    exact_x_equal = bool(np.array_equal(actual_x, expected_x, equal_nan=True))
    assert exact_x_equal, "destination X differs from source rows [2000,2096)"

    index_results: dict[str, dict[str, object]] = {}
    for axis in ("obs", "var"):
        expected_index = np.asarray(source[axis]["_index"][:])
        actual_index = np.asarray(destination[axis]["_index"][:])
        equal = bool(np.array_equal(actual_index, expected_index))
        assert equal, f"destination {axis} index differs from source"
        index_results[axis] = {
            "exact_values_equal": equal,
            "shape": list(actual_index.shape),
            "dtype": str(actual_index.dtype),
            "source_metadata": metadata(source_path, f"{axis}/_index"),
            "destination_metadata": metadata(destination_path, f"{axis}/_index"),
        }

    source_x_metadata = metadata(source_path, "X")
    destination_x_metadata = metadata(destination_path, "X")
    assert source_x_metadata["shape"] == [4096, N_COLUMNS], source_x_metadata["shape"]
    assert destination_x_metadata["shape"] == [96, N_COLUMNS], destination_x_metadata["shape"]
    manifest = tree_manifest(destination_path)
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "status": "ok",
        "source": str(source_path),
        "destination": str(destination_path),
        "slice": {"row_start": ROW_START, "row_stop": ROW_STOP, "columns": N_COLUMNS},
        "shape": list(actual_x.shape),
        "dtype": str(actual_x.dtype),
        "exact_x_values_equal": exact_x_equal,
        "indexes": index_results,
        "useful_payload_bytes": int(expected_x.nbytes),
        "destination_physical_bytes": tree_bytes(destination_path),
        "destination_file_count": len(manifest),
        "destination_tree_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "source_x_metadata": source_x_metadata,
        "destination_x_metadata": destination_x_metadata,
        "event_loop_policy": "PollPolicy installed before importing Zarr",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "shape", "dtype", "exact_x_values_equal", "indexes",
        "useful_payload_bytes", "destination_physical_bytes", "destination_file_count",
        "destination_tree_manifest_sha256"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
