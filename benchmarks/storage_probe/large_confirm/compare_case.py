#!/usr/bin/env python3
"""Independent full-value and axis check for one B confirmation case."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys


class PollPolicy(asyncio.DefaultEventLoopPolicy):
    """Keep Zarr local-store polling compatible with the restricted runner."""

    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick() -> None:
            loop.call_later(0.010, tick)

        loop.call_later(0.010, tick)
        return loop


asyncio.set_event_loop_policy(PollPolicy())

import numpy as np
import zarr


def metadata_path(root: Path, fmt: str, relative: str) -> Path:
    node = root / relative
    return node / (".zarray" if fmt == "zarr2" else "zarr.json")


def read_array_metadata(root: Path, fmt: str, relative: str) -> dict:
    return json.loads(metadata_path(root, fmt, relative).read_text())


def read_group_metadata(root: Path, fmt: str, relative: str = "") -> dict:
    node = root / relative if relative else root
    name = ".zattrs" if fmt == "zarr2" else "zarr.json"
    result = json.loads((node / name).read_text())
    return result if fmt == "zarr2" else result.get("attributes", {})


def array_geometry(metadata: dict, fmt: str) -> dict:
    if fmt == "zarr2":
        return {
            "shape": metadata.get("shape"),
            "dtype": metadata.get("dtype"),
            "chunk_geometry": metadata.get("chunks"),
            "shard_geometry": None,
            "compression": metadata.get("compressor"),
            "filters": metadata.get("filters"),
        }
    grid = metadata.get("chunk_grid", {}).get("configuration", {})
    codecs = metadata.get("codecs") or []
    shard = next((item.get("configuration", {}) for item in codecs
                  if item.get("name") in {"sharding_indexed", "sharding"}), None)
    return {
        "shape": metadata.get("shape"),
        "dtype": metadata.get("data_type"),
        "chunk_geometry": shard.get("chunk_shape") if shard else grid.get("chunk_shape"),
        "shard_geometry": grid.get("chunk_shape") if shard else None,
        "compression": codecs,
        "filters": None,
    }


def tree_evidence(root: Path) -> dict:
    files = {}
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        files[rel] = {"bytes": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(root.resolve()),
        "kind": "directory",
        "physical_bytes": total,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_hashes": files,
    }


def exact_array(actual, expected, label: str) -> dict:
    if actual.shape != expected.shape:
        raise AssertionError(f"{label} shape differs: {actual.shape} != {expected.shape}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{label} dtype differs: {actual.dtype} != {expected.dtype}")
    if np.issubdtype(actual.dtype, np.inexact):
        equal = bool(np.array_equal(actual, expected, equal_nan=True))
    else:
        equal = bool(np.array_equal(actual, expected))
    if not equal:
        raise AssertionError(f"{label} values differ")
    return {
        "shape": [int(value) for value in actual.shape],
        "dtype": str(actual.dtype),
        "exact_values_equal": equal,
        "logical_bytes_each": int(actual.nbytes),
    }


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: compare_case.py export|reindex zarr2|zarr3 SOURCE TARGET RESULT_JSON"
        )
    action, fmt = sys.argv[1:3]
    source_path, target_path, result_path = map(Path, sys.argv[3:])
    if action not in {"export", "reindex"} or fmt not in {"zarr2", "zarr3"}:
        raise SystemExit("unsupported action or format")
    source_path = source_path.resolve()
    target_path = target_path.resolve()

    source = zarr.open_group(str(source_path), mode="r")
    target = zarr.open_group(str(target_path), mode="r")
    source_x_meta = read_array_metadata(source_path, fmt, "X")
    target_x_meta = read_array_metadata(target_path, fmt, "X")
    source_x = np.asarray(source["X"][:])
    target_x = np.asarray(target["X"][:])
    if source_x.shape != (4096, 4096) or source_x.dtype != np.dtype("float32"):
        raise AssertionError(f"frozen source X differs: shape={source_x.shape}, dtype={source_x.dtype}")
    expected_x = source_x if action == "export" else source_x[:-1, :]
    x_check = exact_array(target_x, expected_x, "X")
    expected_shape = [4096, 4096] if action == "export" else [4095, 4096]
    if target_x_meta.get("shape") != expected_shape:
        raise AssertionError(f"target X metadata shape differs: {target_x_meta.get('shape')}")
    target_dtype = target_x_meta.get("dtype" if fmt == "zarr2" else "data_type")
    if target_dtype not in {"<f4", ">f4", "float32"}:
        raise AssertionError(f"target X metadata dtype differs: {target_dtype!r}")

    axis_checks = {}
    axis_metadata = {}
    for axis in ("obs", "var"):
        source_index_meta = read_array_metadata(source_path, fmt, f"{axis}/_index")
        target_index_meta = read_array_metadata(target_path, fmt, f"{axis}/_index")
        source_index = np.asarray(source[axis]["_index"][:])
        target_index = np.asarray(target[axis]["_index"][:])
        if action == "reindex" and axis == "obs":
            expected_index = source_index[:-1]
        else:
            expected_index = source_index
        axis_checks[axis] = exact_array(target_index, expected_index, f"{axis}/_index")
        source_attrs = read_group_metadata(source_path, fmt, axis)
        target_attrs = read_group_metadata(target_path, fmt, axis)
        for key in ("_index", "encoding-type", "encoding-version", "column-order"):
            if source_attrs.get(key) != target_attrs.get(key):
                raise AssertionError(f"{axis} metadata attribute {key!r} differs")
        axis_metadata[axis] = {
            "source_index": array_geometry(source_index_meta, fmt),
            "target_index": array_geometry(target_index_meta, fmt),
            "source_group_attributes": source_attrs,
            "target_group_attributes": target_attrs,
        }

    expected_rows = 4096 if action == "export" else 4095
    if int(target["X"].shape[0]) != expected_rows or int(target["X"].shape[1]) != 4096:
        raise AssertionError("target X axes differ")
    if target["obs"]["_index"].shape[0] != expected_rows or target["var"]["_index"].shape[0] != 4096:
        raise AssertionError("target observation/variable axes differ")

    source_evidence = tree_evidence(source_path)
    target_evidence = tree_evidence(target_path)
    result = {
        "status": "ok",
        "action": action,
        "format": fmt,
        "source": str(source_path),
        "target": str(target_path),
        "source_tree": source_evidence,
        "target_tree": target_evidence,
        "source_x_metadata": source_x_meta,
        "target_x_metadata": target_x_meta,
        "source_x_geometry": array_geometry(source_x_meta, fmt),
        "target_x_geometry": array_geometry(target_x_meta, fmt),
        "x_full_exact_check": x_check,
        "axes_exact_checks": axis_checks,
        "axis_metadata": axis_metadata,
        "validation_outside_timed_capped_operation": True,
        "validation_logical_x_read_bytes": int(source_x.nbytes + target_x.nbytes),
        "validation_note": (
            "Full X and full observation/variable index arrays were compared exactly. "
            "Validation runs without the candidate I/O interposer and may warm OS caches."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "ok", "action": action, "format": fmt,
        "x_shape": x_check["shape"], "x_dtype": x_check["dtype"],
        "x_exact": x_check["exact_values_equal"],
        "obs_index_exact": axis_checks["obs"]["exact_values_equal"],
        "var_index_exact": axis_checks["var"]["exact_values_equal"],
        "target_tree_sha256": target_evidence["sha256"],
        "target_physical_bytes": target_evidence["physical_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
