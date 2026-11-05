"""Upstream-generated FP16 storage fixtures must be rejected by AnnData++."""
from __future__ import annotations

import asyncio
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def install_restricted_runner_wakeup() -> None:
    if os.environ.get("ANNDATAPP_INTEROP_ZARR_WAKEUP") != "1":
        return

    class PollPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            loop = super().new_event_loop()

            def tick() -> None:
                loop.call_later(0.010, tick)

            loop.call_later(0.010, tick)
            return loop

    asyncio.set_event_loop_policy(PollPolicy())


install_restricted_runner_wakeup()

import anndata as ad
import h5py
import numpy as np
from zarr.storage import LocalStore
import zarr


def make_upstream_float16(path: Path, backend: str) -> None:
    expected = np.array([[0.5, -2.0], [3.25, 8.0]], dtype=np.float16)
    data = ad.AnnData(expected.copy())

    if backend == "h5ad":
        data.write_h5ad(path)
        with h5py.File(path, "r") as handle:
            physical = handle["X"]
            if physical.shape != expected.shape or physical.dtype != np.dtype("float16"):
                raise AssertionError(
                    f"upstream H5AD X is {physical.shape}/{physical.dtype}, "
                    "expected 2x2 float16"
                )
        reopened = ad.read_h5ad(path)
    else:
        zarr_format = int(backend[-1])
        with ad.settings.override(zarr_write_format=zarr_format):
            data.write_zarr(path)
        group = zarr.open_group(
            LocalStore(str(path)), mode="r", use_consolidated=False
        )
        physical = group["X"]
        if physical.shape != expected.shape or physical.dtype != np.dtype("float16"):
            raise AssertionError(
                f"upstream Zarr v{zarr_format} X is {physical.shape}/{physical.dtype}, "
                "expected 2x2 float16"
            )
        reopened = ad.read_zarr(LocalStore(str(path)))

    if reopened.X.dtype != np.dtype("float16"):
        raise AssertionError(f"upstream AnnData reopened X as {reopened.X.dtype}, not float16")
    np.testing.assert_array_equal(reopened.X, expected)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python_unsupported_dtype.py DRIVER h5ad|zarr2|zarr3 ARTIFACT_DIR"
        )
    driver = str(Path(sys.argv[1]).resolve())
    backend = sys.argv[2]
    artifact_root = os.environ.get("ANNDATAPP_UNSUPPORTED_DTYPE_ARTIFACT_DIR")
    artifact_dir = Path(sys.argv[3]) if artifact_root is None else Path(artifact_root) / backend
    if version("anndata") != "0.13.4":
        raise SystemExit(f"expected AnnData 0.13.4, got {version('anndata')}")
    if version("h5py") != "3.16.0":
        raise SystemExit(f"expected h5py 3.16.0, got {version('h5py')}")
    if version("zarr") != "3.4.2":
        raise SystemExit(f"expected Zarr 3.4.2, got {version('zarr')}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".h5ad" if backend == "h5ad" else ".zarr"
    store = artifact_dir / f"float16-{uuid4().hex}{suffix}"
    make_upstream_float16(store, backend)

    result = subprocess.run(
        [driver, backend, str(store)], text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise AssertionError(
            f"C++ rejection driver failed for {backend} ({result.returncode}):\n"
            f"{result.stdout}{result.stderr}"
        )
    print(result.stdout, end="")
    print(f"upstream {backend} float16 fixture rejected at eager, backed metadata, and read gates")


if __name__ == "__main__":
    main()
