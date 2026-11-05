#!/usr/bin/env python3
"""Bounded independent HDF5 link/cycle probe against existing release drivers."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, default=ROOT / "build/hdf5-links")
parser.add_argument("--eager", type=Path, default=ROOT / "build/interop/h5ad_interop_driver")
parser.add_argument("--backed", type=Path, default=ROOT / "build/interop/storage_interop_driver")
args = parser.parse_args()
OUT, EAGER, BACKED = args.work.resolve(), args.eager.resolve(), args.backed.resolve()
OUT.mkdir(parents=True, exist_ok=True)
BUDGET = 29.0
started = time.monotonic()

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(argv: list[str]) -> dict:
    remaining = BUDGET - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("whole probe exceeded 29 second budget")
    t0 = time.monotonic()
    p = subprocess.run(argv, capture_output=True, text=True, timeout=remaining)
    return {"argv": [str(x) for x in argv], "returncode": p.returncode,
            "stdout": p.stdout, "stderr": p.stderr,
            "seconds": time.monotonic() - t0}

def make_base(path: Path) -> None:
    a = ad.AnnData(
        X=np.asarray([[1.25, 2.5], [3.75, 5.0]], dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(["cell-a", "cell-b"], name="cell_id")),
        var=pd.DataFrame(index=pd.Index(["gene-a", "gene-b"], name="gene_id")),
    )
    a.uns = {"nested": {"description": "hard-link-target", "value": np.int32(17)}}
    a.write_h5ad(path, convert_strings_to_categoricals=False)

def inject(base: Path, name: str) -> Path:
    path = OUT / f"{name}.h5ad"
    shutil.copy2(base, path)
    with h5py.File(path, "r+") as f:
        uns = f["uns"]
        if name == "cycle":
            group = uns.create_group("cycle")
            group.attrs["encoding-type"] = "dict"
            group.attrs["encoding-version"] = "0.1.0"
            group["self"] = group
        elif name == "soft-link":
            group = uns.create_group("cycle")
            group.attrs["encoding-type"] = "dict"
            group.attrs["encoding-version"] = "0.1.0"
            group["nested"] = h5py.SoftLink("/uns/nested")
        elif name == "external-link":
            group = uns.create_group("cycle")
            group.attrs["encoding-type"] = "dict"
            group.attrs["encoding-version"] = "0.1.0"
            group["nested"] = h5py.ExternalLink("missing-target.h5", "/uns/nested")
        elif name == "hard-alias":
            group = uns.create_group("cycle")
            group.attrs["encoding-type"] = "dict"
            group.attrs["encoding-version"] = "0.1.0"
            group["nested_alias"] = uns["nested"]
        else:
            raise ValueError(name)
        f.flush()
    return path

def expect_error(case: str, path: Path, fragment: str, reason: str) -> dict:
    eager = run([str(EAGER), "read", str(path)])
    assert eager["returncode"] != 0, (case, "eager unexpectedly succeeded")
    for text in ("anndatapp error[unsupported_encoding]", fragment, reason):
        assert text in eager["stderr"], (case, "eager missing expected error context", text, eager)
    helper = run([str(EAGER), "expect-read-error", str(path), "unsupported_encoding", fragment])
    assert helper["returncode"] == 0, (case, "existing eager error assertion rejected result", helper)
    backed = run([str(BACKED), "read", "h5ad", str(path)])
    assert backed["returncode"] != 0, (case, "backed materialization unexpectedly succeeded")
    for text in ("anndatapp error[unsupported_encoding]", fragment, reason):
        assert text in backed["stderr"], (case, "backed missing expected error context", text, backed)
    return {"case": case, "path_fragment": fragment, "reason": reason,
            "eager": eager, "eager_expect_read_error": helper, "backed_materialize": backed}

def expect_success(case: str, path: Path) -> dict:
    eager = run([str(EAGER), "read", str(path)])
    assert eager["returncode"] == 0, (case, "eager valid file failed", eager)
    backed = run([str(BACKED), "read", "h5ad", str(path)])
    assert backed["returncode"] == 0, (case, "backed valid file failed", backed)
    result = {"case": case, "eager": eager, "backed_materialize": backed}
    if case == "hard-alias":
        with h5py.File(path, "r") as f:
            assert isinstance(f["uns/cycle"].get("nested_alias", getlink=True), h5py.HardLink)
            assert f["uns/nested"].id == f["uns/cycle/nested_alias"].id
        copied = OUT / "hard-alias-roundtrip.h5ad"
        cp = run([str(EAGER), "copy", str(path), str(copied)])
        assert cp["returncode"] == 0, (case, "eager copy failed", cp)
        with h5py.File(copied, "r") as f:
            assert f["uns/nested/description"][()].decode() == "hard-link-target"
            assert f["uns/cycle/nested_alias/description"][()].decode() == "hard-link-target"
            assert np.asarray(f["X"]).tolist() == [[1.25, 2.5], [3.75, 5.0]]
        result["eager_copy"] = cp
        result["roundtrip_values_verified"] = True
        result["roundtrip_sha256"] = sha(copied)
    return result

def main() -> None:
    before = {"eager": sha(EAGER), "backed": sha(BACKED)}
    pkg = subprocess.run(["pkg-config", "--modversion", "hdf5"], capture_output=True, text=True)
    ldd = subprocess.run(["ldd", str(EAGER)], capture_output=True, text=True)
    linked_hdf5 = [line.strip() for line in ldd.stdout.splitlines() if "libhdf5" in line.lower()]
    base = OUT / "base.h5ad"
    make_base(base)
    with h5py.File(base, "r") as f:
        assert f.attrs["encoding-type"] == "anndata"
        assert f["uns/nested/description"][()].decode() == "hard-link-target"
        assert np.asarray(f["X"]).tolist() == [[1.25, 2.5], [3.75, 5.0]]
    results = [expect_success("baseline", base)]
    results.append(expect_error("hardlink-cycle", inject(base, "cycle"),
                                "/uns/cycle/self", "cyclic HDF5"))
    results.append(expect_error("nested-soft-link", inject(base, "soft-link"),
                                "/uns/cycle/nested", "soft and external HDF5 links"))
    results.append(expect_error("external-link", inject(base, "external-link"),
                                "/uns/cycle/nested", "soft and external HDF5 links"))
    results.append(expect_success("hard-alias", inject(base, "hard-alias")))
    after = {"eager": sha(EAGER), "backed": sha(BACKED)}
    assert before == after, "driver binaries changed during probe"
    receipt = {
        "schema": "anndatapp-hdf5-link-probe/1",
        "status": "passed",
        "scope": "independent h5py HDF5 fixtures; eager error code/path/reason and backed materialization; valid sibling hard alias round trip",
        "budget_seconds": BUDGET,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {"python": sys.version, "anndata": version("anndata"), "h5py": h5py.__version__,
                        "python_hdf5": h5py.version.hdf5_version,
                        "driver_hdf5_pkg_config": pkg.stdout.strip() if pkg.returncode == 0 else None,
                        "eager_driver_hdf5_linkage": linked_hdf5,
                        "platform": sys.platform},
        "drivers": {"eager": str(EAGER), "eager_sha256_before_after": [before["eager"], after["eager"]],
                    "backed": str(BACKED), "backed_sha256_before_after": [before["backed"], after["backed"]]},
        "fixtures": {str(p.name): sha(p) for p in sorted(OUT.glob("*.h5ad"))},
        "results": results,
    }
    out = OUT / "receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "elapsed_seconds": receipt["elapsed_seconds"],
                      "receipt": str(out), "cases": [r["case"] for r in results]}, sort_keys=True))

if __name__ == "__main__":
    main()
