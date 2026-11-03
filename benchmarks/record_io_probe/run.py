#!/usr/bin/env python3
"""Run the six bounded record-call-count cases once each."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=30, check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--interposer", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    fixture_script = here / "make_fixture.py"
    env = os.environ.copy()
    old_preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = str(args.interposer.resolve()) + (f":{old_preload}" if old_preload else "")

    record = {
        "probe": "H5Dread/H5Dwrite call counts and sum(H5Tget_size(memory_type)*selected_points)",
        "byte_semantics": "native logical memory projection; not compressed bytes, physical I/O, or decoded filter bytes",
        "hdf5_version": h5py.version.hdf5_version,
        "driver_sha256": sha256(args.driver.resolve()),
        "interposer_sha256": sha256(args.interposer.resolve()),
        "fixtures": {},
        "runs": [],
    }
    for fields in (2, 8):
        source = artifacts / f"record-{fields}-fields.h5ad"
        fixture_run = run(
            [sys.executable, str(fixture_script), str(source), "--fields", str(fields)], os.environ.copy()
        )
        if fixture_run.returncode:
            raise RuntimeError(f"fixture setup failed: {fixture_run.stdout}\n{fixture_run.stderr}")
        record["fixtures"][str(fields)] = {
            "path": str(source),
            "sha256": sha256(source),
            "file_bytes": source.stat().st_size,
            "logical_payload_bytes": 8192 * fields * 8,
            "chunk_records": 256,
            "chunk_count": 32,
            "compression": "gzip level 4",
            "fixture_stdout": fixture_run.stdout.strip(),
        }
        for operation in ("field", "full", "export"):
            output = artifacts / f"export-{fields}-fields.h5ad"
            cmd = [str(args.driver.resolve()), operation, str(source), str(fields)]
            if operation == "export": cmd.append(str(output))
            result = run(cmd, env)
            entry = {
                "fields": fields,
                "operation": operation,
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
            if result.returncode:
                record["runs"].append(entry)
                path = artifacts / "record-io-counts.json"
                path.write_text(json.dumps(record, indent=2) + "\n")
                raise RuntimeError(f"probe failed; details retained in {path}: {result.stderr}")
            count_lines = [line.partition("=")[2] for line in result.stdout.splitlines()
                           if line.startswith("hdf5_counts=")]
            if len(count_lines) != 1:
                raise RuntimeError(f"expected one hdf5_counts line, got {len(count_lines)}: {result.stdout}")
            entry["counts"] = json.loads(count_lines[0])
            if operation == "export":
                entry["output_path"] = str(output)
                entry["output_sha256"] = sha256(output)
                entry["output_file_bytes"] = output.stat().st_size
            record["runs"].append(entry)
    path = artifacts / "record-io-counts.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
