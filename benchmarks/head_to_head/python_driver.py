#!/usr/bin/env python3
"""Run one prepared Python-side head-to-head operation in a fresh process."""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import warnings


PROCESS_START = time.perf_counter()
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")


class PollPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick() -> None:
            loop.call_later(0.010, tick)

        loop.call_later(0.010, tick)
        return loop


if os.environ.get("ANNDATA_INTEROP_ZARR_WAKEUP") == "1":
    asyncio.set_event_loop_policy(PollPolicy())

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import zarr
from numcodecs import blosc
from zarr.core.buffer.cpu import NDBuffer
from zarr.storage import LocalStore

IMPORT_SECONDS = time.perf_counter() - PROCESS_START
ZARR_CONFIG = {"async.concurrency": 1, "threading.max_workers": 1}
ZARR_CHUNKS = (64, 128)
ZARR_SHARDS = (256, 512)
H5PY_CACHE = {"rdcc_nbytes": 1 << 20, "rdcc_nslots": 521, "rdcc_w0": 0.75}
VERSIONS = {
    "anndata": importlib.metadata.version("anndata"),
    "numpy": importlib.metadata.version("numpy"),
    "pandas": importlib.metadata.version("pandas"),
    "scipy": importlib.metadata.version("scipy"),
    "h5py": importlib.metadata.version("h5py"),
    "hdf5": h5py.version.hdf5_version,
    "zarr": importlib.metadata.version("zarr"),
    "numcodecs": importlib.metadata.version("numcodecs"),
}


def rss_status() -> dict[str, int]:
    values = {}
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _unit = line.split()
                values["rss_before_kib" if key == "VmRSS:" else "lifetime_hwm_kib"] = int(value)
            elif line.startswith("Threads:"):
                values["threads"] = int(line.split()[1])
    return values


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check_adata(data: ad.AnnData, n: int) -> dict:
    assert data.shape == (n, n)
    assert isinstance(data.obs["batch"].dtype, pd.CategoricalDtype)
    assert isinstance(data.obs["condition"].dtype, pd.CategoricalDtype)
    assert isinstance(data.var["feature_class"].dtype, pd.CategoricalDtype)
    assert "sparse_layer" in data.layers and sp.issparse(data.layers["sparse_layer"])
    assert data.raw is not None and data.raw.n_vars == min(64, n)
    assert data.raw.X.shape == (n, min(64, n))
    return {"shape": [n, n], "obs_categories": ["batch", "condition"],
            "var_categories": ["feature_class"], "sparse_layer": True,
            "raw_shape": [n, min(64, n)]}


def fixture_expected(path: Path) -> dict:
    return json.loads((path.parent / "expected.json").read_text())


def check_expected(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")


def start_probe(x_dir: Path):
    if not os.environ.get("HEAD_TO_HEAD_PROBE"):
        return None
    library = ctypes.CDLL(None)
    begin = library.anndata_completeness_probe_begin
    end = library.anndata_completeness_probe_end
    begin.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p,
                      ctypes.c_size_t, ctypes.c_int]
    begin.restype = ctypes.c_int
    end.argtypes = []
    end.restype = None
    resolved = x_dir.resolve(strict=True)
    if begin(512 << 20, 512 << 20, os.fsencode(resolved), 0, 0):
        raise RuntimeError(f"could not arm bounded probe for {resolved}")
    return end


def arm_zarr() -> None:
    zarr.config.set(ZARR_CONFIG)
    blosc.set_nthreads(1)


def check_zarr_x(array, shape: tuple[int, int]) -> None:
    if tuple(array.shape) != shape or np.dtype(array.dtype) != np.dtype(np.float32):
        raise RuntimeError(f"unexpected Zarr X array: shape={array.shape} dtype={array.dtype}")
    if tuple(array.chunks) != ZARR_CHUNKS or tuple(array.shards) != ZARR_SHARDS:
        raise RuntimeError(f"unexpected Zarr3 geometry: chunks={array.chunks} shards={array.shards}")


def sparse_digest(blocks: list) -> dict:
    digest = hashlib.sha256()
    nnz = 0
    for block in blocks:
        for part in (block.data, block.indices, block.indptr):
            digest.update(np.ascontiguousarray(part).tobytes())
        nnz += int(block.nnz)
    first = blocks[0]
    return {"sha256": digest.hexdigest(), "elements": nnz, "blocks": len(blocks),
            "value_dtype": str(first.data.dtype), "index_dtype": str(first.indices.dtype),
            "offset_dtype": str(first.indptr.dtype),
            "block_nnz": [int(block.nnz) for block in blocks]}


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python_driver.py CASE FIXTURE OUTPUT")
    case = sys.argv[1]
    fixture = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve()
    if not fixture.exists():
        raise SystemExit(f"fixture does not exist: {fixture}")
    if not output.parent.exists():
        raise SystemExit(f"output parent does not exist: {output.parent}")

    valid_cases = {
        "h5ad_eager_read_dense", "h5ad_eager_write_dense", "h5py_dense_read_direct",
        "h5ad_sparse_rows_csr", "h5ad_sparse_cols_csc", "zarr3_sharded_read",
        "zarr3_gather_read", "zarr3_full_write", "zarr3_partial_write",
        "numpy_fused_loop", "numba_fused_loop",
    }
    if case not in valid_cases:
        raise SystemExit(f"unknown case: {case}")
    is_zarr_case = case.startswith("zarr3_")
    if is_zarr_case:
        arm_zarr()

    prepare_start = time.perf_counter()
    n = 0
    result = None
    op = None
    validate = None
    probe_dir = fixture.parent
    timing = {}
    cleanup = []

    if case in {"h5ad_eager_read_dense", "h5ad_eager_write_dense"}:
        if fixture.suffix != ".h5ad":
            raise SystemExit("eager AnnData cases require an H5AD fixture")
        n = int(fixture.parent.name)
        if case == "h5ad_eager_read_dense":
            op = lambda: ad.read_h5ad(fixture)
            validate = lambda value: verify_adata(value, n, fixture, "dense_x_sha256")
        else:
            if output.exists():
                raise SystemExit(f"refusing to overwrite output: {output}")
            data = ad.read_h5ad(fixture)

            def write_h5ad() -> str:
                data.write_h5ad(output, compression=None, convert_strings_to_categoricals=False)
                return "written"

            op = write_h5ad

            def validate_write(_value):
                saved = ad.read_h5ad(output)
                info = check_adata(saved, n)
                with h5py.File(output, "r") as handle:
                    x = handle["X"]
                    if not isinstance(x, h5py.Dataset) or x.chunks is not None or x.compression is not None:
                        raise RuntimeError("H5AD write X is not an uncompressed contiguous dataset")
                    info["x_chunks"] = x.chunks
                    info["x_compression"] = x.compression
                checksum = array_sha256(np.asarray(saved.X))
                expected = fixture_expected(fixture)["dense_x_sha256"]
                check_expected(checksum, expected, "H5AD write X")
                return {"sha256": checksum, "elements": n * n, "metadata": info,
                        "x_storage": "uncompressed contiguous"}

            validate = validate_write

    elif case == "h5py_dense_read_direct":
        if fixture.suffix != ".h5ad":
            raise SystemExit("h5py read_direct requires a dense H5AD fixture")
        n = int(fixture.parent.name)
        handle = h5py.File(fixture, "r", **H5PY_CACHE)
        cleanup.append(handle)
        dataset = handle["X"]
        if not isinstance(dataset, h5py.Dataset) or dataset.chunks is not None or dataset.compression is not None:
            raise SystemExit("read_direct requires uncompressed contiguous dense X")
        if dataset.shape != (n, n) or dataset.dtype != np.dtype(np.float32):
            raise SystemExit("unexpected HDF5 X shape or dtype")
        dense = np.empty((n, n), dtype=np.float32, order="C")
        selections = [(lo, min(lo + 128, n)) for lo in range(0, n, 128)]

        def read_direct() -> np.ndarray:
            for lo, hi in selections:
                dataset.read_direct(dense, source_sel=np.s_[lo:hi, :], dest_sel=np.s_[lo:hi, :])
            return dense

        op = read_direct
        validate = lambda value: verify_array(value, fixture, "dense_x_sha256")

    elif case in {"h5ad_sparse_rows_csr", "h5ad_sparse_cols_csc"}:
        if fixture.suffix != ".h5ad":
            raise SystemExit("sparse_dataset cases require a CSR or CSC H5AD fixture")
        n = int(fixture.parent.name)
        handle = h5py.File(fixture, "r", **H5PY_CACHE)
        cleanup.append(handle)
        sds = ad.io.sparse_dataset(handle["X"], should_cache_indptr=True)
        if tuple(sds.shape) != (n, n):
            raise SystemExit("unexpected sparse X shape")
        expected_format = "csr" if case.endswith("csr") else "csc"
        if sds.format != expected_format:
            raise SystemExit(f"expected {expected_format.upper()} fixture, got {sds.format}")
        _indptr = sds._indptr  # force the default cached compressed offsets before timing
        starts = list(range(0, n, 256))
        blocks = []

        def read_sparse() -> list:
            for lo in starts:
                hi = min(lo + 256, n)
                blocks.append(sds[lo:hi, :] if case == "h5ad_sparse_rows_csr" else sds[:, lo:hi])
            return blocks

        op = read_sparse

        def verify_sparse(_value):
            summary = sparse_digest(blocks)
            expected = fixture_expected(fixture)["sparse_csr_rows" if case.endswith("csr") else "sparse_csc_columns"]
            check_expected(summary["sha256"], expected["sha256"], "sparse block triplets")
            if [b["nnz"] for b in expected["blocks"]] != summary["block_nnz"]:
                raise RuntimeError("sparse block nnz differs from independent expectation")
            summary["selected_shape"] = [256, n] if case.endswith("csr") else [n, 256]
            summary["sparse_indptr_warmup"] = "private cached sds._indptr read outside timing"
            return summary

        validate = verify_sparse

    elif case in {"zarr3_sharded_read", "zarr3_gather_read"}:
        if not fixture.is_dir():
            raise SystemExit("Zarr3 read cases require a Zarr store directory")
        n = int(fixture.parent.name)
        source = zarr.open_group(LocalStore(str(fixture)), mode="r", use_consolidated=False)
        array = source["X"]
        check_zarr_x(array, (n, n))
        probe_dir = fixture / "X"
        if case == "zarr3_sharded_read":
            dense = np.empty((n, n), dtype=np.float32, order="C")
            rows = [(lo, min(lo + 128, n)) for lo in range(0, n, 128)]
            buffers = [NDBuffer.from_numpy_array(dense[lo:hi, :]) for lo, hi in rows]

            def read_zarr() -> np.ndarray:
                for (lo, hi), buffer in zip(rows, buffers):
                    array.get_basic_selection((slice(lo, hi), slice(0, n)), out=buffer)
                return dense

            op = read_zarr
            validate = lambda value: verify_array(value, fixture, "dense_x_sha256")
            timing["zarr_output_buffers"] = "one full C-order float32 matrix with prebuilt NDBuffer row views"
        else:
            rows = np.asarray([0, 3, 63, 64, 127, 128, 223, 224, 255, 256,
                               511, 512, 767, 768, n - 2, n - 1], dtype=np.int64)
            cols = (np.arange(64, dtype=np.int64) * 37) % n
            dense = np.empty((16, 64), dtype=np.float32, order="C")
            buffer = NDBuffer.from_numpy_array(dense)

            def read_gather() -> np.ndarray:
                for _ in range(8):
                    array.get_orthogonal_selection((rows, cols), out=buffer)
                return dense

            op = read_gather

            def verify_gather(value):
                expected = fixture_expected(fixture)["gather"]
                checksum = array_sha256(value)
                check_expected(checksum, expected["sha256"], "Zarr gather")
                return {"sha256": checksum, "elements": value.size, "shape": list(value.shape),
                        "rows": rows.tolist(), "columns": cols.tolist(), "repeats": 8}

            validate = verify_gather
            timing["zarr_output_buffers"] = "one C-order float32 16x64 matrix and one prebuilt NDBuffer"

    elif case in {"zarr3_full_write", "zarr3_partial_write"}:
        if not fixture.is_dir() or not output.is_dir():
            raise SystemExit("Zarr3 writes require an input store and an existing disposable output copy")
        if fixture == output:
            raise SystemExit("refusing to write into the input fixture")
        n = int(fixture.parent.name)
        target = zarr.open_group(LocalStore(str(output)), mode="r+", use_consolidated=False)
        array = target["X"]
        check_zarr_x(array, (n, n))
        probe_dir = output / "X"
        full = case == "zarr3_full_write"
        selection = (slice(0, 256), slice(0, 512)) if full else (slice(224, 320), slice(448, 576))
        shape = (256, 512) if full else (96, 128)
        payload = np.full(shape, np.float32(7.5), dtype=np.float32, order="C")

        def write_zarr() -> str:
            array[selection] = payload
            return "written"

        op = write_zarr

        def verify_zarr_write(_value):
            saved = zarr.open_group(LocalStore(str(output)), mode="r", use_consolidated=False)
            written = np.asarray(saved["X"][selection])
            if written.shape != shape or not np.array_equal(written, payload):
                raise RuntimeError("Zarr output selection differs from the written payload")
            metadata = ad.read_zarr(LocalStore(str(output)))
            check_adata(metadata, n)
            checksum = array_sha256(written)
            return {"sha256": checksum, "elements": written.size, "shape": list(written.shape),
                    "selection": [[selection[0].start, selection[0].stop],
                                  [selection[1].start, selection[1].stop]],
                    "metadata": "AnnData categories, sparse layer, and raw verified"}

        validate = verify_zarr_write
        timing["write_contract"] = "synchronous LocalStore assignment from preallocated contiguous float32 payload; no explicit flush API"

    elif case in {"numpy_fused_loop", "numba_fused_loop"}:
        if fixture.suffix != ".h5ad":
            raise SystemExit("fused-loop cases require a dense H5AD fixture")
        n = int(fixture.parent.name)
        data = ad.read_h5ad(fixture)
        if not isinstance(data.X, np.ndarray) or data.X.dtype != np.dtype(np.float32):
            raise SystemExit("fused-loop input X must be dense float32")
        dense = data.X
        if not dense.flags.c_contiguous:
            raise SystemExit("fused-loop X must be C-contiguous")
        scale, shift = np.float32(0.5), np.float32(0.125)
        if case == "numpy_fused_loop":
            def fused_numpy() -> np.ndarray:
                for _ in range(8):
                    np.multiply(dense, scale, out=dense)
                    np.add(dense, shift, out=dense)
                return dense

            op = fused_numpy
        else:
            if n != 4096:
                raise SystemExit("numba_fused_loop is qualified only for the 4096x4096 case")
            case_import_start = time.perf_counter()
            import numba
            from llvmlite import binding as llvm

            numba.set_num_threads(1)

            @numba.njit(cache=False, fastmath=False)
            def recurrence(values, rounds, factor, bias):
                for _round in range(rounds):
                    for i in range(values.size):
                        product = values[i] * factor
                        values[i] = product + bias

            warm = np.arange(32, dtype=np.float32)
            jit_start = time.perf_counter()
            recurrence(warm, 8, scale, shift)
            timing["jit_compile_seconds"] = time.perf_counter() - jit_start
            timing["numba_import_seconds"] = jit_start - case_import_start
            timing["numba_target_cpu"] = llvm.get_host_cpu_name()
            timing["numba_host_features"] = sorted(
                name for name, enabled in llvm.get_host_cpu_features().items() if enabled
            )
            timing["numba_isa_policy"] = "Numba default host CPU specialization; fastmath=False"
            VERSIONS["numba"] = numba.__version__
            flat = dense.reshape(-1)

            def fused_numba() -> np.ndarray:
                recurrence(flat, 8, scale, shift)
                return dense

            op = fused_numba

        def verify_fused(value):
            metadata = check_adata(data, n)
            answer = verify_array(value, fixture, "fused.sha256")
            answer["metadata"] = metadata
            return answer

        validate = verify_fused

    if op is None or validate is None or n < 1:
        raise RuntimeError(f"case preparation failed: {case}")
    prepare_seconds = time.perf_counter() - prepare_start
    before = rss_status()
    probe = start_probe(probe_dir)
    operation_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        result = op()
        operation_seconds = time.perf_counter() - operation_start
        cpu_seconds = time.process_time() - cpu_start
    finally:
        if probe is not None:
            probe()
    after = rss_status()
    measurement = {
        "phase": "measurement", "case": case, "size": n,
        "startup_import_seconds": IMPORT_SECONDS, "prepare_seconds": prepare_seconds,
        "seconds": operation_seconds, "operation_wall_seconds": operation_seconds,
        "cpu_seconds": cpu_seconds,
        "rss_before_kib": before["rss_before_kib"], "rss_after_kib": after["rss_before_kib"],
        "lifetime_hwm_kib": after["lifetime_hwm_kib"],
        "threads_before": before["threads"], "threads_after": after["threads"],
        "shape": [n, n], "dtype": "float32", "backend_versions": dict(VERSIONS),
        "zarr_config": dict(ZARR_CONFIG) if is_zarr_case else "not used",
        "blosc_threads": 1 if is_zarr_case else "not used",
        "h5py_cache": dict(H5PY_CACHE) if case in {"h5py_dense_read_direct", "h5ad_sparse_rows_csr", "h5ad_sparse_cols_csc"} else "AnnData defaults",
        "buffer_contract": timing,
    }
    sys.stdout.write(json.dumps(measurement, sort_keys=True) + "\n")
    sys.stdout.flush()
    if not sys.stdin.isatty():
        response = sys.stdin.readline().strip()
        if response != "continue":
            raise RuntimeError(f"unexpected checkpoint response: {response}")

    verification = validate(result)
    for resource in cleanup:
        resource.close()
    verification["phase"] = "verification"
    verification["case"] = case
    sys.stdout.write(json.dumps(verification, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


def verify_array(array: np.ndarray, fixture: Path, expected_key: str) -> dict:
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise RuntimeError(f"expected C-order float32 result, got {array.dtype} {array.shape}")
    checksum = array_sha256(array)
    expected = fixture_expected(fixture)
    for part in expected_key.split("."):
        expected = expected[part]
    check_expected(checksum, expected, expected_key)
    return {"sha256": checksum, "elements": int(array.size), "shape": list(array.shape)}


def verify_adata(data: ad.AnnData, n: int, fixture: Path, expected_key: str) -> dict:
    metadata = check_adata(data, n)
    array = np.asarray(data.X)
    result = verify_array(array, fixture, expected_key)
    result["metadata"] = metadata
    return result


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="anndata")
        raise SystemExit(main())
