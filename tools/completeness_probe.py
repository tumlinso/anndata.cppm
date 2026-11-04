#!/usr/bin/env python3
"""Run bounded completeness observations, preserving each result before validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if os.environ.get("ANNDATA_INTEROP_ZARR_WAKEUP") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from python_storage import install_restricted_runner_wakeup
    install_restricted_runner_wakeup()

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from zarr.storage import LocalStore

WRITES = {
    "full_4": ("full", "zarr3_4"),
    "full_16": ("full", "zarr3_16"),
    "full_64": ("full", "zarr3_64"),
    "partial_16": ("partial", "zarr3_16"),
    "gather_stride": ("gather_stride", "zarr3_16"),
    "gather_view": ("gather_view", "zarr3_16"),
    "huge_gap": ("huge_gap", "huge_gap"),
}
ADAPTIVE = {"fixed_1": "1", "fixed_2": "2", "fixed_4": "4", "fixed_8": "8", "adaptive": "adaptive"}
ROWS = (270,64,270,9,256,511,257,3,95,10,448,1,384,65,258,0)
COLS = (520,4,520,129,513,255,127,6)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def verify_files(root: Path, expected: dict) -> None:
    actual = {p.relative_to(root).as_posix(): {"bytes":p.stat().st_size,"sha256":digest(p)}
              for p in sorted(root.rglob("*")) if p.is_file()}
    if actual != expected: raise RuntimeError(f"fixture integrity changed: {root}")


def metadata(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("zarr.json")):
        value = json.loads(path.read_text())
        if path == root / "zarr.json": value.pop("consolidated_metadata", None)
        result[path.relative_to(root).as_posix()] = value
    return result


def value(i: int,j: int) -> np.float32: return np.float32(10000 + 131*i + j)


def validate_write(case: str, source: Path, target: Path) -> dict:
    source_store, target_store = LocalStore(str(source)), LocalStore(str(target))
    if metadata(source) != metadata(target): raise AssertionError("semantic Zarr metadata changed")
    group = zarr.open_group(target_store, mode="r", use_consolidated=False)
    x = group["X"]
    if case == "huge_gap":
        ri, ci = (0,(1<<30)-1), (0,(1<<20)-1)
        actual = x.oindex[list(ri),list(ci)]
        expected = np.array([[value(i,j) for j in range(2)] for i in range(2)], dtype=np.float32)
        np.testing.assert_array_equal(actual,expected)
        assert float(x[1,1]) == 0 and float(x[1<<29,1<<19]) == 0
        return {"independent": "zarr sparse-point and fill checks", "x_points_sha256":hashlib.sha256(actual.tobytes()).hexdigest()}
    before, after = ad.read_zarr(source_store), ad.read_zarr(target_store)
    expected = np.asarray(before.X).copy()
    if case == "full":
        for i in range(256):
            for j in range(512): expected[i,j] = value(i,j)
    elif case == "partial":
        for i in range(200):
            for j in range(400): expected[150+i,300+j] = value(i,j)
    else:
        for i,r in enumerate(ROWS):
            for j,c in enumerate(COLS):
                if case == "gather_stride": expected[r,c] = value(len(ROWS)-1-i,len(COLS)-1-j)
                else: expected[r,c] = value(len(ROWS)-1-i,(3*j)%len(COLS))
    np.testing.assert_array_equal(after.X,expected)
    pd.testing.assert_frame_equal(after.obs,before.obs)
    pd.testing.assert_frame_equal(after.var,before.var)
    assert after.uns["note"] == before.uns["note"]
    a,b = after.layers["counts"],before.layers["counts"]
    assert sp.isspmatrix_csr(a) and a.shape == b.shape
    for name in ("data","indices","indptr"):
        av,bv = getattr(a,name),getattr(b,name)
        assert av.dtype == bv.dtype
        np.testing.assert_array_equal(av,bv)
    assert after.raw is not None and before.raw is not None
    np.testing.assert_array_equal(after.raw.X,before.raw.X)
    pd.testing.assert_frame_equal(after.raw.var,before.raw.var)
    return {"independent":"AnnData full-array and metadata checks", "x_sha256":hashlib.sha256(np.asarray(after.X).tobytes()).hexdigest()}


def verify_adaptive_seed(path: Path, manifest: dict) -> str:
    x = np.asarray(zarr.open_group(LocalStore(str(path)), mode="r", use_consolidated=False)["X"][:])
    rows = np.arange(x.shape[0], dtype=np.int64)[:,None]
    cols = np.arange(x.shape[1], dtype=np.int64)[None,:]
    expected = ((17*rows+13*cols+(rows*cols)%97)%1009).astype(np.float32)/np.float32(1009)
    np.testing.assert_array_equal(x,expected)
    h = hashlib.sha256(x.tobytes()).hexdigest()
    if h != manifest["dense_x_sha256"]: raise AssertionError("adaptive seed checksum differs")
    return h


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--adaptive-driver", type=Path, required=True)
    p.add_argument("--interposer", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--cases", nargs="+", choices=list(WRITES)+list(ADAPTIVE), required=True)
    args = p.parse_args()
    if len(args.cases) != len(set(args.cases)): raise SystemExit("duplicate cases are not permitted")
    root, seed = args.fixtures.resolve(),args.seed.resolve()
    generated = json.loads((root/"manifest.json").read_text())
    old = json.loads((seed/"manifest.json").read_text())
    if digest(seed/"manifest.json") != generated["seed_manifest_sha256"]: raise SystemExit("seed manifest differs")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    reference = {"fixtures_sha256":digest(root/"manifest.json"),"seed_sha256":digest(seed/"manifest.json"),
                 "interposer_sha256":digest(args.interposer)}
    if baseline and any(baseline[k]!=v for k,v in reference.items()): raise SystemExit("baseline fixture or interposer differs")
    if args.output.exists() or args.work_root.exists(): raise SystemExit("output and work root must not already exist")
    args.output.parent.mkdir(parents=True,exist_ok=True); args.work_root.mkdir(parents=True,exist_ok=False)
    record = {"schema":"anndata.completeness.probe.v1", **reference, "driver_sha256":digest(args.driver),
              "adaptive_driver_sha256":digest(args.adaptive_driver),"runner_sha256":digest(Path(__file__).resolve()),
              "machine":platform.platform(),"budget":{"seconds_per_case":30,"intercepted_bytes_each_direction":512*1024*1024},
              "scope":"armed libc calls only; setup, copying, fixture checks and independent validation are outside caps",
              "cases":{}}
    def checkpoint() -> None:
        path = args.output.with_suffix(args.output.suffix+".tmp")
        path.write_text(json.dumps(record,indent=2)+"\n"); path.replace(args.output)
    checkpoint()
    env = os.environ.copy(); env["LD_PRELOAD"] = str(args.interposer.resolve())
    slow = {"ANNDATA_COMPLETENESS_DELAY_FAST_US":"500","ANNDATA_COMPLETENESS_DELAY_SLOW_US":"2000",
            "ANNDATA_COMPLETENESS_DELAY_SLOW_FROM_SHARD_ROW":"8"}
    adaptive_seed_checked = False
    for name in args.cases:
        target = None
        if name in WRITES:
            action,fixture = WRITES[name]
            spec = generated["huge_gap"] if fixture == "huge_gap" else generated["layouts"][fixture]
            source = root/fixture; verify_files(source,spec["files"])
            target = (args.work_root/name).resolve(); shutil.copytree(source,target)
            command = [str(args.driver.resolve()),action,str(target),str(spec["index_bytes"]),"0"]
        else:
            source = seed/"dense_zarr3"
            if not adaptive_seed_checked:
                verify_files(source,old["fixtures"]["dense_zarr3"]["files"])
                record["adaptive_seed_x_sha256"] = verify_adaptive_seed(source,old)
                adaptive_seed_checked = True
            command = [str(args.adaptive_driver.resolve()),ADAPTIVE[name],str(source)]
        case_env = env.copy()
        for key in slow: case_env.pop(key,None)
        if name in ADAPTIVE: case_env.update(slow)
        entry = {"command":command,"status":"running","delay":slow if name in ADAPTIVE else None}
        record["cases"][name] = entry; checkpoint()
        start = time.monotonic()
        try:
            proc = subprocess.run(command,env=case_env,text=True,capture_output=True,timeout=30,check=False)
            entry.update(returncode=proc.returncode,wall_seconds=time.monotonic()-start,stdout=proc.stdout,stderr=proc.stderr)
            entry["status"] = "completed_unvalidated" if proc.returncode == 0 else "failed"
            checkpoint()
            if proc.returncode != 0: return 1
            entry["result"] = json.loads(proc.stdout)
            lines = [line.removeprefix("ANNDATA_COMPLETENESS_PROBE ") for line in proc.stderr.splitlines()
                     if line.startswith("ANNDATA_COMPLETENESS_PROBE ")]
            if len(lines)!=1: raise ValueError("expected exactly one counter record")
            entry["counters"] = json.loads(lines[0])
            checkpoint()
            if target is not None: entry["validation"] = validate_write(WRITES[name][0],source,target)
            else:
                if entry["result"]["calls"] != 512 or entry["result"]["payload_bytes"] != 4096*1024*4:
                    raise AssertionError("adaptive stream coverage differs")
                entry["validation"] = {"independent_seed_x_sha256":record["adaptive_seed_x_sha256"],"coverage":"each source cell checked against independent formula"}
            if baseline:
                if name not in baseline["cases"]: raise ValueError("matching baseline case is absent")
                oldcase = baseline["cases"][name]
                if oldcase.get("status")!="ok": raise ValueError("matching baseline case is not valid")
                if name in WRITES and entry["validation"] != oldcase["validation"]:
                    raise AssertionError("output differs from baseline")
                if name in ADAPTIVE and entry["validation"] != oldcase["validation"]:
                    raise AssertionError("adaptive coverage differs from baseline")
                entry["matches_baseline"] = True
            entry["status"] = "ok"; checkpoint()
        except subprocess.TimeoutExpired as e:
            entry.update(status="timeout",wall_seconds=time.monotonic()-start,stdout=e.stdout.decode() if isinstance(e.stdout,bytes) else e.stdout,
                         stderr=e.stderr.decode() if isinstance(e.stderr,bytes) else e.stderr)
            checkpoint(); return 1
        except Exception as e:
            entry.update(status="validation_failed",error=repr(e)); checkpoint(); return 1
    if adaptive_seed_checked: verify_files(seed/"dense_zarr3",old["fixtures"]["dense_zarr3"]["files"])
    return 0


if __name__ == "__main__": raise SystemExit(main())
