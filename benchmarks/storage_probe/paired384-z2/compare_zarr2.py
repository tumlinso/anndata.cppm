#!/usr/bin/env python3
"""Compare the candidate Zarr2 export with the source slice using Python Zarr."""
from __future__ import annotations

import asyncio
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

ROW_START = 1792
ROW_STOP = 2176
N_COLUMNS = 4096


def file_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_zarr2.py SOURCE DESTINATION JSON_RESULT")
    source_path, destination_path, result_path = map(Path, sys.argv[1:])
    source_path = source_path.resolve()
    destination_path = destination_path.resolve()

    source_meta = json.loads((source_path / "X" / ".zarray").read_text())
    destination_meta = json.loads((destination_path / "X" / ".zarray").read_text())
    source = zarr.open_group(str(source_path), mode="r")
    destination = zarr.open_group(str(destination_path), mode="r")
    expected = np.asarray(source["X"][ROW_START:ROW_STOP, :])
    actual = np.asarray(destination["X"][:])

    assert list(expected.shape) == [384, N_COLUMNS], expected.shape
    assert list(actual.shape) == [384, N_COLUMNS], actual.shape
    assert source_meta["shape"] == [4096, N_COLUMNS], source_meta["shape"]
    assert destination_meta["shape"] == [384, N_COLUMNS], destination_meta["shape"]
    assert expected.dtype == np.dtype("float32"), expected.dtype
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    equal = bool(np.array_equal(actual, expected, equal_nan=True))
    assert equal, "destination X differs from source rows [1792,2176)"

    result = {
        "status": "ok",
        "source": str(source_path),
        "destination": str(destination_path),
        "slice": {"row_start": ROW_START, "row_stop": ROW_STOP, "columns": N_COLUMNS},
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "exact_values_equal": equal,
        "useful_payload_bytes": int(expected.nbytes),
        "destination_physical_bytes": file_bytes(destination_path),
        "source_x_metadata": source_meta,
        "destination_x_metadata": destination_meta,
        "event_loop_policy": "PollPolicy installed before importing Zarr",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "status", "shape", "dtype", "exact_values_equal", "useful_payload_bytes",
        "destination_physical_bytes", "event_loop_policy"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
