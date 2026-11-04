#!/usr/bin/env python3
"""Run one bounded observation per streaming workload and checkpoint results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


CASES = {
    "dense_sequential": ("dense", "zarr2", "dense_zarr2", 128, 128),
    "dense_boundary": ("dense", "zarr3", "dense_zarr3", 64, 96),
    "sparse_csr": ("sparse", "h5ad", "sparse_csr.h5ad", 64, 64),
    "sparse_csc": ("sparse", "h5ad", "sparse_csc.h5ad", 64, 64),
    "adaptive_boundary": ("adaptive", "zarr3", "dense_zarr3", 64, 96),
    "parallel_read": ("parallel_read", "zarr3", "dense_zarr3", 256, 512),
    "parallel_write": ("parallel_write", "zarr3", "dense_zarr3", 256, 512),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--interposer", type=Path, required=True)
    p.add_argument("--parallel-driver", type=Path)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--write-root", type=Path)
    p.add_argument("--cases", nargs="+", choices=list(CASES), default=["dense_sequential", "dense_boundary", "sparse_csr", "sparse_csc"])
    args = p.parse_args()
    fixture_manifest = args.fixtures / "manifest.json"
    if not fixture_manifest.is_file(): raise SystemExit("fixture manifest is required")
    fixtures = json.loads(fixture_manifest.read_text())["fixtures"]
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(args.interposer.resolve())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    record = {"schema": 1, "driver_sha256": digest(args.driver), "interposer_sha256": digest(args.interposer),
              "fixture_manifest_sha256": digest(fixture_manifest), "machine": platform.platform(),
              "budget": {"seconds_per_case": 30, "intercepted_bytes_each_direction": 512*1024*1024},
              "scope": "probe-armed calls only; setup, fixture creation and validation are outside cap",
              "cases": {}}
    if args.parallel_driver: record["parallel_driver_sha256"] = digest(args.parallel_driver)
    if baseline and (baseline["fixture_manifest_sha256"] != record["fixture_manifest_sha256"] or
                     baseline["interposer_sha256"] != record["interposer_sha256"]):
        raise SystemExit("fixture manifest or interposer differs from baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.output)

    checkpoint()
    for name in args.cases:
        action, fmt, fixture, rows, columns = CASES[name]
        source = args.fixtures / fixture
        if action.startswith("parallel_"):
            if not args.parallel_driver: raise SystemExit("parallel case requires --parallel-driver")
            target = source
            if action == "parallel_write":
                if not args.write_root: raise SystemExit("write case requires --write-root")
                target = args.write_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
            command = [str(args.parallel_driver.resolve()), action.removeprefix("parallel_"), str(target.resolve()),
                       str(rows), str(columns), "4"]
        else:
            command = [str(args.driver.resolve()), action, fmt, str(source.resolve()), str(rows), str(columns)]
        entry = {"command": command}
        started = time.monotonic()
        try:
            run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30, check=False)
            entry = {"command": command, "returncode": run.returncode,
                     "wall_seconds": time.monotonic() - started, "stdout": run.stdout, "stderr": run.stderr}
            if run.returncode == 0:
                entry["result"] = json.loads(run.stdout)
                counter_lines = [line.removeprefix("ANNDATA_PROBE ") for line in run.stderr.splitlines()
                                 if line.startswith("ANNDATA_PROBE ")]
                if len(counter_lines) != 1: raise ValueError("expected exactly one counter record")
                entry["counters"] = json.loads(counter_lines[0])
                if entry["counters"]["read_rejections"] or entry["counters"]["write_rejections"]:
                    entry["status"] = "budget_exceeded"
                if action == "sparse":
                    expected = int(fixtures[fixture]["expected_hash"]["expected_hash"], 16)
                    entry["matches_independent_scipy"] = entry["result"]["hash"] == expected
                    if not entry["matches_independent_scipy"]: entry["status"] = "mismatch"
                if baseline and name not in baseline["cases"] and action != "adaptive":
                    entry["status"] = "baseline_missing"
                elif baseline and name in baseline["cases"]:
                    old = baseline["cases"][name]["result"]
                    new = entry["result"]
                    entry["matches_baseline"] = all(old[key] == new[key] for key in ("calls", "payload_bytes", "hash"))
                    if not entry["matches_baseline"]: entry["status"] = "mismatch"
                entry.setdefault("status", "ok")
                if action == "parallel_write" and entry["status"] == "ok":
                    entry["status"] = "validation_pending"
                    record["cases"][name] = entry
                    checkpoint()
                    # Independent post-run readback is deliberately outside
                    # both the timed operation and intercepted-I/O caps.
                    try:
                        if os.environ.get("ANNDATA_INTEROP_ZARR_WAKEUP") == "1":
                            from python_storage import install_restricted_runner_wakeup
                            install_restricted_runner_wakeup()
                        import numpy as np
                        import zarr
                        before = np.asarray(zarr.open_group(str(source), mode="r", use_consolidated=False)["X"][:])
                        after = np.asarray(zarr.open_group(str(target), mode="r", use_consolidated=False)["X"][:])
                        entry["matches_independent_zarr"] = bool(np.array_equal(after, before + np.float32(0.5)))
                        entry["status"] = "ok" if entry["matches_independent_zarr"] else "mismatch"
                    except Exception as error:
                        entry["status"] = "validation_failed"
                        entry["validation_error"] = str(error)
            else: entry["status"] = "failed"
        except subprocess.TimeoutExpired as error:
            entry = {"command": command, "status": "timeout", "wall_seconds": time.monotonic() - started,
                     "stdout": str(error.stdout or ""), "stderr": str(error.stderr or "")}
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            entry.update({"status": "invalid_result", "error": str(error)})
        record["cases"][name] = entry
        checkpoint()
        if entry["status"] != "ok": break
    print(args.output)
    if any(entry["status"] != "ok" for entry in record["cases"].values()): raise SystemExit(1)


if __name__ == "__main__":
    main()
