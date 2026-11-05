"""Generate an independent H5AD fixed-string padding fixture and byte report.

The file starts with AnnData's writer, then h5py low-level APIs install fixed
UTF-8 datasets, compound members, and encoding attributes with each HDF5
string-padding mode.  The report records exact logical field bytes read back
with the file datatype, plus h5py's ordinary high-level results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd


WIDTH = 16
SAMPLES = (b"A\x00B", b"tail\x00", b"mid\x00x")
PADDING = {
    "STR_NULLTERM": h5py.h5t.STR_NULLTERM,
    "STR_NULLPAD": h5py.h5t.STR_NULLPAD,
    "STR_SPACEPAD": h5py.h5t.STR_SPACEPAD,
}


def fixed_type(mode: str, width: int) -> h5py.h5t.TypeID:
    typ = h5py.h5t.C_S1.copy()
    typ.set_size(width)
    typ.set_cset(h5py.h5t.CSET_UTF8)
    typ.set_strpad(PADDING[mode])
    return typ


def padded(value: bytes, mode: str, width: int = WIDTH) -> bytes:
    if len(value) > width:
        raise ValueError(f"value of {len(value)} bytes exceeds fixed width {width}")
    fill = b" " if mode == "STR_SPACEPAD" else b"\x00"
    return value + fill * (width - len(value))


def fixed_attribute(obj: h5py.Group | h5py.Dataset, name: str, value: bytes,
                    mode: str, width: int = WIDTH) -> None:
    if name in obj.attrs:
        del obj.attrs[name]
    typ = fixed_type(mode, width)
    space = h5py.h5s.create(h5py.h5s.SCALAR)
    attr = h5py.h5a.create(obj.id, name.encode("utf-8"), typ, space)
    buf = np.empty((), dtype=f"V{width}")
    buf[()] = np.void(padded(value, mode, width))
    attr.write(buf, mtype=typ)


def fixed_dataset(group: h5py.Group, name: str, values: tuple[bytes, ...],
                  mode: str, width: int = WIDTH,
                  shape: tuple[int, ...] | None = None) -> h5py.Dataset:
    shape = (len(values),) if shape is None else shape
    count = int(np.prod(shape)) if shape else 1
    if count != len(values):
        raise ValueError(f"shape {shape} requires {count} values, got {len(values)}")
    typ = fixed_type(mode, width)
    space = h5py.h5s.create_simple(shape) if shape else h5py.h5s.create(h5py.h5s.SCALAR)
    did = h5py.h5d.create(group.id, name.encode("utf-8"), typ, space)
    buf = np.empty(shape, dtype=f"V{width}") if shape else np.empty((), dtype=f"V{width}")
    for i, value in enumerate(values):
        buf.reshape(-1)[i] = np.void(padded(value, mode, width))
    if values:
        did.write(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=typ)
    return h5py.Dataset(did)


def raw_values(dataset: h5py.Dataset) -> list[str]:
    typ = dataset.id.get_type()
    size = typ.get_size()
    buf = np.empty(dataset.shape, dtype=f"V{size}")
    dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=typ)
    return [bytes(value).hex() for value in buf.reshape(-1)]


def high_level_values(dataset: h5py.Dataset) -> list[str]:
    values = dataset.asstr()[()]
    return [str(value) for value in np.asarray(values).reshape(-1)]


def high_level_bytes(dataset: h5py.Dataset) -> list[str]:
    values = np.asarray(dataset[()]).reshape(-1)
    return [bytes(value).hex() for value in values]


def native_decode(value: bytes, mode: str) -> str:
    if mode == "STR_NULLTERM": value = value.split(b"\x00", 1)[0]
    elif mode == "STR_NULLPAD": value = value.rstrip(b"\x00")
    else: value = value.rstrip(b" ")
    return value.decode("utf-8")


def padding_name(typ: h5py.h5t.TypeID) -> str:
    return next((key for key, enum in PADDING.items()
                 if enum == typ.get_strpad()), "unknown")


def add_plain_string_dataset(group: h5py.Group, name: str, path: str,
                             values: tuple[bytes, ...], mode: str, encoding: str,
                             shape: tuple[int, ...] | None = None) -> dict[str, object]:
    dataset = fixed_dataset(group, name, values, mode, shape=shape)
    fixed_attribute(dataset, "encoding-type", encoding.encode("ascii"), mode)
    fixed_attribute(dataset, "encoding-version", b"0.2.0", mode)
    if values:
        fixed_attribute(dataset, "padding-probe", SAMPLES[0], mode)
        fixed_attribute(dataset, "padding-trailing-probe", SAMPLES[1], mode)
    return {
        "path": path,
        "padding": mode,
        "encoding": encoding,
        "shape": list(dataset.shape),
        "width": dataset.id.get_type().get_size(),
        "raw_values_hex": raw_values(dataset),
        "h5py_bytes_values_hex": high_level_bytes(dataset),
        "h5py_values": high_level_values(dataset),
        "native_expected_values": [native_decode(value, mode) for value in values],
        "attributes": {
            key: attr_observation(dataset, key)
            for key in ("encoding-type", "encoding-version", "padding-probe",
                        "padding-trailing-probe") if key in dataset.attrs
        },
    }


def attr_observation(obj: h5py.Group | h5py.Dataset, name: str) -> dict[str, object]:
    attr = h5py.h5a.open(obj.id, name.encode("utf-8"))
    typ = attr.get_type()
    width = typ.get_size()
    buf = np.empty((), dtype=f"V{width}")
    attr.read(buf, mtype=typ)
    value = obj.attrs[name]
    high_level_hex = bytes(value).hex() if isinstance(value, (bytes, np.bytes_)) else None
    mode = padding_name(typ)
    raw = bytes(buf[()])
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    return {
        "padding": mode,
        "width": width,
        "raw_hex": raw.hex(),
        "h5py_value_hex": high_level_hex,
        "h5py_value": str(value),
        "native_expected_value": native_decode(raw, mode),
    }


def replace_obs_columns(path: Path) -> dict[str, object]:
    report: dict[str, object] = {}
    with h5py.File(path, "r+") as f:
        obs = f["obs"]
        for mode in PADDING:
            name = mode.lower()
            del obs[name]
            report[name] = add_plain_string_dataset(
                obs, name, f"/obs/{name}", SAMPLES, mode, "string-array"
            )
    return report


def add_uns_strings(path: Path) -> dict[str, object]:
    report: dict[str, object] = {"arrays": {}, "scalars": {}, "empty_arrays": {}}
    with h5py.File(path, "r+") as f:
        uns = f["uns"]
        for mode in PADDING:
            key = mode.lower()
            array_name = f"padding_{key}_array"
            report["arrays"][array_name] = add_plain_string_dataset(
                uns, array_name, f"/uns/{array_name}", SAMPLES, mode, "string-array"
            )
            scalar_name = f"padding_{key}_scalar"
            report["scalars"][scalar_name] = add_plain_string_dataset(
                uns, scalar_name, f"/uns/{scalar_name}", (SAMPLES[0],), mode,
                "string", shape=()
            )
            empty_name = f"padding_{key}_empty"
            report["empty_arrays"][empty_name] = add_plain_string_dataset(
                uns, empty_name, f"/uns/{empty_name}", (), mode,
                "string-array", shape=(0,)
            )
    return report


def add_record(path: Path) -> dict[str, object]:
    modes = tuple(PADDING)
    offsets = {mode.lower(): i * WIDTH for i, mode in enumerate(modes)}
    size = len(modes) * WIDTH
    compound = h5py.h5t.create(h5py.h5t.COMPOUND, size)
    for mode in modes:
        name = mode.lower().encode("ascii")
        compound.insert(name, offsets[name.decode("ascii")], fixed_type(mode, WIDTH))

    with h5py.File(path, "r+") as f:
        uns = f["uns"]
        space = h5py.h5s.create_simple((len(SAMPLES),))
        did = h5py.h5d.create(uns.id, b"padding_record", compound, space)
        rows = np.empty((len(SAMPLES),), dtype=f"V{size}")
        for i in range(len(SAMPLES)):
            row = bytearray(size)
            for j, mode in enumerate(modes):
                start = j * WIDTH
                row[start:start + WIDTH] = padded(SAMPLES[i], mode)
            rows[i] = np.void(bytes(row))
        did.write(h5py.h5s.ALL, h5py.h5s.ALL, rows, mtype=compound)
        dataset = h5py.Dataset(did)
        fixed_attribute(dataset, "encoding-type", b"rec-array", "STR_SPACEPAD")
        fixed_attribute(dataset, "encoding-version", b"0.2.0", "STR_SPACEPAD")
        fixed_attribute(dataset, "padding-probe", SAMPLES[0], "STR_SPACEPAD")
        fixed_attribute(dataset, "padding-trailing-probe", SAMPLES[1], "STR_SPACEPAD")

        exact = np.empty(dataset.shape, dtype=f"V{size}")
        dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, exact, mtype=compound)
        raw_records = [bytes(value) for value in exact.reshape(-1)]
        field_report = {}
        for mode in modes:
            name = mode.lower()
            start = offsets[name]
            field_type = compound.get_member_type(modes.index(mode))
            high = dataset[name][()]
            field_report[name] = {
                "padding": mode,
                "width": field_type.get_size(),
                "raw_values_hex": [
                    record[start:start + WIDTH].hex() for record in raw_records
                ],
                "h5py_bytes_values_hex": [bytes(value).hex() for value in high],
                "h5py_values": [
                    bytes(value).decode("utf-8") for value in high
                ],
                "native_expected_values": [native_decode(value, mode) for value in SAMPLES],
            }
        return {
            "path": "/uns/padding_record",
            "fields": field_report,
            "attributes": {
                key: attr_observation(dataset, key)
                for key in ("encoding-type", "encoding-version", "padding-probe",
                            "padding-trailing-probe")
            },
        }


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def create_fixture(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fixed-string-padding.h5ad"
    obs = pd.DataFrame(
        {mode.lower(): ["placeholder-0", "placeholder-1", "placeholder-2"]
         for mode in PADDING},
        index=pd.Index(["cell-0", "cell-1", "cell-2"], name="cell_id"),
    )
    var = pd.DataFrame(index=pd.Index(["gene-0"], name="gene_id"))
    data = ad.AnnData(
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32), obs=obs, var=var
    )
    data.write_h5ad(path, convert_strings_to_categoricals=False)

    ordinary = replace_obs_columns(path)
    uns = add_uns_strings(path)
    record = add_record(path)
    try:
        python_result = ad.read_h5ad(path)
        anndata_readback: dict[str, object] = {
            "status": "read",
            "obs_values": {
                mode.lower(): [str(value) for value in python_result.obs[mode.lower()].tolist()]
                for mode in PADDING
            },
            "record_values": {
                name: [str(value) for value in python_result.uns["padding_record"][name].tolist()]
                for name in python_result.uns["padding_record"].dtype.names or ()
            },
        }
    except Exception as error:
        anndata_readback = {
            "status": "rejected",
            "exception": type(error).__name__,
            "message": str(error),
        }

    payload: dict[str, object] = {
        "fixture": str(path.resolve()),
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "anndata": version("anndata"),
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "padding_enums": {name: int(value) for name, value in PADDING.items()},
        "samples_input_hex": [value.hex() for value in SAMPLES],
        "ordinary": ordinary,
        "uns": uns,
        "record": record,
        "anndata_readback": anndata_readback,
        "native_policy": {
            "description": "Parent-selected physical fixed-string padding rule: NULLTERM stops at first NUL; NULLPAD removes trailing NUL bytes only; SPACEPAD removes trailing spaces only.",
            "obs": {name: item["native_expected_values"]
                    for name, item in ordinary.items()},
            "uns_arrays": {name: item["native_expected_values"]
                           for name, item in uns["arrays"].items()},
            "uns_scalars": {name: item["native_expected_values"][0]
                            for name, item in uns["scalars"].items()},
            "uns_empty_arrays": {name: item["native_expected_values"]
                                 for name, item in uns["empty_arrays"].items()},
            "record_fields": {name: item["native_expected_values"]
                              for name, item in record["fields"].items()},
        },
        "notes": [
            "raw_values_hex are HDF5 logical fixed-width bytes read with the same file datatype; they are not offsets from the HDF5 file image.",
            "encoding-type and encoding-version are semantic fixed-string attributes consumed by eager and backed readers.",
            "padding-probe and padding-trailing-probe are ignored extension attributes; they record raw bytes, h5py's high-level value, and the parent-selected native decode.",
            "h5py observations describe this installed build, not a claim about every HDF5 binding or an external text standard.",
        ],
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", nargs="?",
        default="build/v1-cleanup/padding-fixture",
        help="directory for ignored H5AD and evidence.json artifacts",
    )
    args = parser.parse_args()
    payload = create_fixture(Path(args.output_dir))
    print(json.dumps({
        "fixture": payload["fixture"],
        "fixture_sha256": payload["fixture_sha256"],
        "evidence": str((Path(args.output_dir) / "evidence.json").resolve()),
        "environment": payload["environment"],
    }, indent=2))


if __name__ == "__main__":
    main()
