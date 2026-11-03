"""Differential in-memory semantics; Python owns H5AD I/O in this test.

This is deliberately not an assertion that the C++ library reads or writes H5AD.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from importlib.metadata import version

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py


def fixture():
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["T", "B", "T"], categories=["T", "B", "unused"], ordered=True
            ),
            "count": pd.array([1, None, 3], dtype="Int32"),
            "flag": pd.array([True, None, False], dtype="boolean"),
            "note": pd.array(["α", None, "γ"], dtype="string"),
        },
        index=pd.Index(["cell-a", "cell-b", "cell-c"], name="cell_id"),
    )
    a = ad.AnnData(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(["g1", "g2"], name="gene_id")),
    )
    counts = sp.csr_matrix([[1, 0], [0, 2], [3, 0]], dtype=np.int32)
    # Equal sparse index widths keep SciPy reverse slicing usable; mixed-width
    # buffers are checked independently in the native array tests.
    a.layers["counts"] = counts
    a.obsm["embedding"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    a.obsp["connectivities"] = np.eye(3, dtype=np.float32)
    a.uns = {
        "nested": {"kind": "fixture", "threshold": np.float32(0.5)},
        "cell_type_colors": np.array(["red", "blue", "green"]),
    }
    a.varm["table"] = pd.DataFrame({"label": ["gene A", "gene B"]}, index=a.var_names)
    a.raw = a.copy()
    return a


def normalized(values):
    return [None if pd.isna(value) else value for value in values.tolist()]


def describe(a, threshold_dtype):
    counts = a.layers["counts"].copy()
    return {
        "shape": list(a.shape),
        "X": a.X.tolist(),
        "obs_names": a.obs_names.tolist(),
        "var_names": a.var_names.tolist(),
        "categories": a.obs.cell_type.cat.categories.tolist(),
        "codes": a.obs.cell_type.cat.codes.tolist(),
        "ordered": bool(a.obs.cell_type.cat.ordered),
        "count": normalized(a.obs["count"]),
        "flag": normalized(a.obs["flag"]),
        "note": normalized(a.obs["note"]),
        "colors": a.uns["cell_type_colors"].tolist(),
        # raw.shape in some Python versions reports the parent count even when
        # raw.X has been correctly sliced; compare the actual matrix values.
        "raw_X": a.raw.X.tolist(),
        "raw_var_names": a.raw.var_names.tolist(),
        "varm_labels": a.varm["table"]["label"].tolist(),
        "raw_varm_labels": a.raw.varm["table"]["label"].tolist(),
        "embedding": a.obsm["embedding"].tolist(),
        "connectivities": a.obsp["connectivities"].tolist(),
        "counts_data": counts.data.tolist(),
        "counts_indices": counts.indices.tolist(),
        "counts_indptr": counts.indptr.tolist(),
        "index_name": a.obs_names.name,
        "column_names": a.obs.columns.tolist(),
        # Python's scalar reader may return a Python float (losing the NumPy
        # width); C++ preserves the physical dtype stored by the reference.
        "threshold_dtype": threshold_dtype,
        "threshold": float(a.uns["nested"]["threshold"]),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python_semantics.py /path/to/interop_driver")
    driver = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="anndata-cppm-semantics-") as directory:
        path = Path(directory) / "reference.h5ad"
        with ad.settings.override(allow_write_nullable_strings=True):
            fixture().write_h5ad(path)
        a = ad.read_h5ad(path)
        with h5py.File(path, "r") as store:
            threshold_dtype = str(store["uns/nested/threshold"].dtype)
        scalar_colors = a.copy()
        scalar_colors.obs["cell_type"] = pd.Categorical(["T", "T", "T"], categories=["T"], ordered=True)
        scalar_colors.uns["cell_type_colors"] = "red"
        scenarios = {
            "base": a,
            "all": a[:, :],
            "subset": a[[2, 0], [1]],
            "repeat": a[[1, 1, 0], :],
            "reverse": a[::-1, ::-1],
            "range_nested": a[0:3:2, ::-1][::-1, 0:2],
            # The pinned AnnData/SciPy pair cannot reshape the sparse result
            # of this nested repeated-row/reversed-column view. Materialize
            # only the reference intermediate; the C++ path remains nested.
            "gather_range": a[[2, 0, 2], :].copy()[0:3:2, ::-1],
            "range_gather": a[::-1, :][[2, 0, 2], [1, 0]],
            "range_empty_nested": a[::-1, :][1:1, :],
            "empty": a[[], :],
            "empty_vars": a[:, []],
            "mask": a[np.array([True, False, True]), ["g2"]],
            "labels": a[["cell-c", "cell-a"], ["g2", "g1"]],
            "singleton": a[-1, 0],
            "nested": a[[2, 0], [1]][[1], :],
            "scalar_colors": scalar_colors[[2, 0], :],
            "scalar_colors_empty": scalar_colors[[], :],
        }
        for name, selected in scenarios.items():
            got = json.loads(subprocess.check_output([driver, name], text=True))
            expected = describe(selected, threshold_dtype)
            if got != expected:
                differences = {key: {"cpp": got.get(key), "python": expected.get(key)}
                               for key in got.keys() | expected.keys()
                               if got.get(key) != expected.get(key)}
                raise AssertionError(f"scenario {name}: {json.dumps(differences, ensure_ascii=False, indent=2)}")
        print(json.dumps({"status": "passed", "scenarios": len(scenarios),
                          "reference": {name: version(name) for name in
                                        ("anndata", "numpy", "scipy", "pandas", "h5py")},
                          "coverage": "C++ in-memory behavior against Python-written/reopened H5AD"}))


if __name__ == "__main__":
    main()
