#!/usr/bin/env python3
"""Run one uniquely claimed Candidate B full-size storage case."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path("/tmp/anndatapp-storage-b-confirm-20261103-mnXjYU")
CAMPAIGN = Path("/tmp/anndatapp-storage-campaign-20261103/comparison-4096")
BUILD = ROOT / "build-v2"
PYTHON = Path("/home/tumlinson/anndatapp/.venv/bin/python")
LIMIT = 256 * 1024 * 1024
TIMEOUT_SECONDS = 30
EXPECTED = {
    "zarr2": {
        "tree_sha256": "bbf41f74bb63084beb22221828db1718f51f47968f5d743294d158a8dd8bdd7e",
        "physical_bytes": 59367950,
        "x_metadata_sha256": "afffd55495c363ff9cfe58f2704b762982bf1e31897bda6a9fbab3a26dfdaeb9",
        "x_chunk_geometry": [512, 512],
        "x_shard_geometry": None,
    },
    "zarr3": {
        "tree_sha256": "23db0bea7a153c189974c63e9a873722a9ec93470069b8262de24233bdc19822",
        "physical_bytes": 59803222,
        "x_metadata_sha256": "e6be8a846e58b2e005aa57f85fc8be74f6af6d67af17f77ac72ef5f68bd7b5bc",
        "x_chunk_geometry": [256, 512],
        "x_shard_geometry": [2048, 4096],
    },
}
CASES = {
    "export-zarr2": ("export", "zarr2"),
    "reindex-zarr2": ("reindex", "zarr2"),
    "export-zarr3": ("export", "zarr3"),
    "reindex-zarr3": ("reindex", "zarr3"),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def evidence(root: Path) -> dict:
    records = {}
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        records[rel] = {"bytes": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(root.resolve()),
        "kind": "directory",
        "physical_bytes": total,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_hashes": records,
    }


def check_fixture(source: Path, fmt: str, ev: dict) -> None:
    expect = EXPECTED[fmt]
    if ev["sha256"] != expect["tree_sha256"] or ev["physical_bytes"] != expect["physical_bytes"]:
        raise RuntimeError(
            f"frozen {fmt} source fixture tree identity mismatch: "
            f"{ev['sha256']} / {ev['physical_bytes']}"
        )
    meta_relative = "X/.zarray" if fmt == "zarr2" else "X/zarr.json"
    actual_meta = ev["file_hashes"].get(meta_relative, {}).get("sha256")
    if actual_meta != expect["x_metadata_sha256"]:
        raise RuntimeError(f"frozen {fmt} X metadata hash mismatch: {actual_meta}")


def parse_probe(stderr: str) -> dict:
    budget_pattern = re.compile(
        r"IO_BUDGET read_bytes=(\d+) write_bytes=(\d+) cap_each=(\d+) "
        r"rejected_read=(\d+) rejected_write=(\d+)"
    )
    budget_match = budget_pattern.search(stderr)
    if not budget_match:
        raise RuntimeError("interposer did not emit its IO_BUDGET line")
    read_bytes, write_bytes, cap, rejected_read, rejected_write = map(int, budget_match.groups())
    probe_line = next((line for line in stderr.splitlines() if line.startswith("PROBE_COUNTS ")), None)
    codec_line = next((line for line in stderr.splitlines() if line.startswith("CODEC_COUNTS ")), None)
    if probe_line is None or codec_line is None:
        raise RuntimeError("interposer omitted I/O call or codec counters")
    calls_by_api = {}
    for name, byte_count, call_count in re.findall(
        r"(read|pread|readv|preadv|write|pwrite|writev|pwritev)=(\d+)/(\d+)", probe_line
    ):
        calls_by_api[name] = {"bytes": int(byte_count), "calls": int(call_count)}
    if len(calls_by_api) != 8:
        raise RuntimeError("interposer call-counter line is incomplete")
    return {
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "read_budget_used_bytes": read_bytes,
        "write_budget_used_bytes": write_bytes,
        "cap_each_bytes": cap,
        "rejected_read": rejected_read,
        "rejected_write": rejected_write,
        "calls_by_api": calls_by_api,
        "aggregate_read_calls": sum(v["calls"] for k, v in calls_by_api.items() if k in {"read", "pread", "readv", "preadv"}),
        "aggregate_write_calls": sum(v["calls"] for k, v in calls_by_api.items() if k in {"write", "pwrite", "writev", "pwritev"}),
        "probe_counts_line": probe_line,
        "codec_counts_line": codec_line,
        "raw_io_budget_line": budget_match.group(0),
    }


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in CASES:
        raise SystemExit("usage: case_runner.py export-zarr2|reindex-zarr2|export-zarr3|reindex-zarr3 RECORD_DIR")
    case, record_arg = sys.argv[1:]
    action, fmt = CASES[case]
    record = Path(record_arg).resolve()
    claim = ROOT / f"{case}.claimed"
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit(f"refusing repeat attempt; claim already exists: {claim}")
    with os.fdopen(fd, "w") as stream:
        stream.write(f"case={case}\nclaimed_unix={time.time():.6f}\n")
    if record.exists():
        raise SystemExit(f"refusing existing record directory: {record}")
    record.mkdir(parents=True)
    for name in ("build-context.txt", "build-hashes.sha256", "candidate-b-identity.json", "candidate-b-source.sha256"):
        shutil.copy2(BUILD / name, record / name)

    source = CAMPAIGN / f"dense.{fmt}"
    expected = EXPECTED[fmt]
    working = source
    target = record / "export.zarr" if action == "export" else record / "working-fixture"
    context = {
        "case": case,
        "action": action,
        "format": fmt,
        "candidate_label": "B_destination_codec_units",
        "candidate_archive": "/tmp/anndatapp-dense-transfer-candidate/libanndatapp.a",
        "candidate_archive_sha256": "213270a9e5adfa2565a97b2a761f5573fb7afcd758fc8c8f3b7d47a9f70c8299",
        "candidate_source_manifest_sha256": "39d5e66667103eff4de17f8e4c70b2166108a942e71c6304f8b2ad3f2377b89e",
        "source_fixture": str(source),
        "source_expected_tree_sha256": expected["tree_sha256"],
        "source_expected_physical_bytes": expected["physical_bytes"],
        "scratch_bytes": 67108864,
        "decoded_cache_bytes": 0,
        "timeout_seconds": TIMEOUT_SECONDS,
        "interposed_libc_read_cap_bytes": LIMIT,
        "interposed_libc_write_cap_bytes": LIMIT,
        "cap_scope": "only intercepted dynamic libc read/pread/readv/preadv and write/pwrite/writev/pwritev calls between bulk_probe_begin/end around write_zarr or reindex apply+flush",
        "cap_is_not_entire_process_total": True,
        "unaccounted_by_cap": [
            "source tree hashing before and after the target process",
            "Zarr open and the baseline-compatible pre-operation reference tile",
            "reindex fixture copy and its hash checks",
            "independent full-value/index validation and its output tree hashing",
            "direct syscalls, mmap I/O, and non-interposed libc internals",
        ],
        "cache_policy": "no cache clearing; source manifest hashing and the baseline-compatible reference tile may warm OS/backend caches; copy/hash/validation I/O is outside the measured cap",
        "baseline_campaign_operation_seconds_n3_uncapped": {
            "export_zarr2": [190.062, 191.115, 190.373],
            "reindex_zarr2": [191.703, 193.046, 191.986],
            "export_zarr3": [186.559, 186.759, 185.503],
            "reindex_zarr3": [191.130, 193.607, 191.698],
        },
        "baseline_timing_note": "from benchmarks/results/storage-baseline.json; three uninstrumented trials, so one capped/interposed Candidate B observation is directional rather than identical-instrumentation timing",
        "baseline_operation_semantics": (
            "benchmarks/storage.cpp: export writes backed.view(); reindex retains obs [0, rows-1) "
            "and all vars, then edit.apply()+flush; pre-operation reference tile is 32x32 for export "
            "and 31x32 for reindex"
        ),
        "source_x_chunk_geometry": expected["x_chunk_geometry"],
        "source_x_shard_geometry": expected["x_shard_geometry"],
        "overlap_check": "parent reported the profiling worker quiescent; this runner starts only this case process",
    }
    write_json(record / "run-context.json", context)

    source_before = evidence(source)
    check_fixture(source, fmt, source_before)
    write_json(record / "source-before.json", source_before)

    copy_evidence = None
    if action == "reindex":
        copy_start = time.monotonic()
        shutil.copytree(source, target)
        copy_seconds = time.monotonic() - copy_start
        copied = evidence(target)
        source_after_copy = evidence(source)
        if copied["sha256"] != source_before["sha256"] or copied["file_hashes"] != source_before["file_hashes"]:
            write_json(record / "copy-evidence.json", {"status": "copy_mismatch", "copy": copied})
            raise RuntimeError("disposable reindex copy does not match the original source")
        if source_after_copy["sha256"] != source_before["sha256"]:
            raise RuntimeError("original source fixture changed while making reindex copy")
        copy_bytes = source_before["physical_bytes"]
        copy_evidence = {
            "status": "ok",
            "source_sha256": source_before["sha256"],
            "copy_sha256_before_mutation": copied["sha256"],
            "source_physical_bytes": copy_bytes,
            "estimated_copy_read_bytes": copy_bytes,
            "estimated_copy_write_bytes": copy_bytes,
            "copy_wall_seconds": copy_seconds,
            "hash_io_bytes_outside_cap_estimate": 3 * copy_bytes,
            "hash_io_note": "source pre-copy hash, copy hash, and source post-copy hash; excludes metadata traversal",
            "source_after_copy": source_after_copy,
            "copy_before_mutation": copied,
        }
        write_json(record / "copy-evidence.json", copy_evidence)
        working = target

    command = [str(BUILD / "confirm_driver"), action, fmt, str(working)]
    if action == "export":
        command.append(str(target))
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env["LD_PRELOAD"] = str(BUILD / "libcodec_counts_limited_256.so")
    timed_command = [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=1s",
        f"{TIMEOUT_SECONDS}s", *command,
    ]
    start = time.monotonic()
    process = None
    timed_out = False
    try:
        process = subprocess.run(
            timed_command, capture_output=True, text=True, env=env,
            cwd=str(record), timeout=TIMEOUT_SECONDS + 2,
        )
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        returncode = 124
    wall_seconds = time.monotonic() - start
    (record / "run.stdout").write_text(stdout)
    (record / "run.stderr").write_text(stderr)
    write_json(record / "process-result.json", {
        "command": timed_command,
        "returncode": returncode,
        "timed_out_by_outer_guard": timed_out,
        "wall_seconds_including_timeout_wrapper": wall_seconds,
        "timeout_seconds_target_process": TIMEOUT_SECONDS,
        "cwd": str(record),
    })
    (record / "exit-code.txt").write_text(f"{returncode}\n")

    target_json = None
    if stdout.strip():
        try:
            target_json = json.loads(stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            pass
    try:
        counters = parse_probe(stderr)
        write_json(record / "interposer-counters.json", counters)
    except Exception as error:
        counters = None
        (record / "counter-parse-failure.txt").write_text(f"{type(error).__name__}: {error}\n")

    failure_reasons = []
    if returncode != 0:
        failure_reasons.append(f"candidate process returned {returncode}")
    if target_json is None or target_json.get("status") != "ok":
        failure_reasons.append("candidate success JSON missing or malformed")
    elif target_json.get("action") != action or target_json.get("format") != fmt:
        failure_reasons.append("candidate reported an unexpected action or format")
    if counters is None:
        failure_reasons.append("required capped I/O counters missing")
    else:
        if counters["cap_each_bytes"] != LIMIT:
            failure_reasons.append("interposer cap differs from 256 MiB")
        if counters["rejected_read"] or counters["rejected_write"]:
            failure_reasons.append("interposed I/O cap rejected a request")
        if counters["read_bytes"] >= LIMIT or counters["write_bytes"] >= LIMIT:
            failure_reasons.append("interposed I/O reached the 256 MiB case cap")
        if counters["aggregate_read_calls"] == 0 or counters["aggregate_write_calls"] == 0:
            failure_reasons.append("target operation reported unexpected zero-direction I/O")

    validation = None
    if not failure_reasons:
        validator = [str(PYTHON), str(ROOT / "compare_case.py"), action, fmt,
                     str(source), str(target), str(record / "validation.json")]
        try:
            checked = subprocess.run(
                validator, capture_output=True, text=True, cwd=str(record), timeout=TIMEOUT_SECONDS,
            )
            (record / "validation.stdout").write_text(checked.stdout)
            (record / "validation.stderr").write_text(checked.stderr)
            if checked.returncode != 0:
                failure_reasons.append(f"independent full validation returned {checked.returncode}")
            else:
                validation = json.loads((record / "validation.json").read_text())
        except subprocess.TimeoutExpired as error:
            (record / "validation.stdout").write_text(
                error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            )
            (record / "validation.stderr").write_text(
                error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            )
            failure_reasons.append("independent full validation exceeded 30 seconds")
        except Exception as error:
            failure_reasons.append(f"independent full validation failed: {type(error).__name__}: {error}")

    source_after = evidence(source)
    if source_after["sha256"] != source_before["sha256"] or source_after["file_hashes"] != source_before["file_hashes"]:
        failure_reasons.append("original source fixture changed during case")
    write_json(record / "source-after.json", source_after)

    if failure_reasons and target.exists():
        try:
            write_json(record / "partial-target-evidence.json", evidence(target))
        except Exception as error:
            (record / "partial-target-evidence-failure.txt").write_text(
                f"{type(error).__name__}: {error}\n"
            )

    uncapped_io = {
        "source_manifest_hashing_before_bytes": source_before["physical_bytes"],
        "source_manifest_hashing_after_bytes": source_after["physical_bytes"],
        "pre_operation_reference_tile_logical_bytes": (32 if action == "export" else 31) * 32 * 4,
        "pre_operation_reference_note": "logical tile payload only; backend may decode its containing chunk/shard",
        "reindex_copy": copy_evidence,
        "full_validation_logical_x_read_bytes": validation.get("validation_logical_x_read_bytes") if validation else None,
        "full_validation_tree_hash_file_bytes": (
            validation["source_tree"]["physical_bytes"] + validation["target_tree"]["physical_bytes"]
            if validation else None
        ),
        "full_validation_note": "Python Zarr reads full source and target X and both complete axis indexes, then hashes every source and target file; all outside the candidate interposer ROI",
        "os_cache_and_physical_io_limit": "reported byte estimates are file/logical bytes; the OS may serve reads from cache and no physical-device traffic was measured",
    }
    write_json(record / "uncapped-io-overhead.json", uncapped_io)

    final = {
        "status": "ok" if not failure_reasons else "failed",
        "case": case,
        "failure_reasons": failure_reasons,
        "operation_seconds": target_json.get("operation_seconds") if target_json else None,
        "open_seconds": target_json.get("open_seconds") if target_json else None,
        "close_seconds_outside_roi": target_json.get("close_seconds_outside_roi") if target_json else None,
        "interposer_counters": counters,
        "copy_evidence": copy_evidence,
        "validation": validation,
        "uncapped_io_overhead": uncapped_io,
        "source_fixture_unchanged": source_after["sha256"] == source_before["sha256"],
        "candidate_wall_seconds": wall_seconds,
        "candidate_process_returncode": returncode,
        "candidate_archive_sha256": context["candidate_archive_sha256"],
    }
    write_json(record / "case-result.json", final)
    print(json.dumps({
        "status": final["status"], "case": case,
        "operation_seconds": final["operation_seconds"],
        "read_bytes": counters.get("read_bytes") if counters else None,
        "write_bytes": counters.get("write_bytes") if counters else None,
        "read_calls": counters.get("aggregate_read_calls") if counters else None,
        "write_calls": counters.get("aggregate_write_calls") if counters else None,
        "failure_reasons": failure_reasons,
        "record": str(record),
    }, sort_keys=True))
    return 0 if not failure_reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
