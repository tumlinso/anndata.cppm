"""Check one private implementation edit without changing the final source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    return {str(p): {"sha256": digest(p), "mtime_ns": p.stat().st_mtime_ns} for p in paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, build, output = args.source.resolve(), args.build.resolve(), args.output.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n" not in cache:
        raise RuntimeError("build directory does not belong to the supplied source")

    unit = source / "src/records.cpp"
    before = unit.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.cpp.original").write_bytes(before)
    bmis = sorted(p for p in (build / "CMakeFiles/anndata.dir").glob("*.pcm")
                  if p.name != "anndata.backed-storage.pcm")
    if len(bmis) != 9:
        raise RuntimeError(f"expected nine public BMIs in the full build, found {len(bmis)}")
    consumers = [build / "CMakeFiles/anndata_example.dir/examples/foundation.cpp.o",
                 build / "CMakeFiles/test_records.dir/tests/test_records.cpp.o"]
    binaries = [build / "CMakeFiles/anndata.dir/src/records.cpp.o",
                build / "libanndata.a", build / "anndata_example", build / "test_records"]
    public_before, binaries_before = state(bmis + consumers), state(binaries)
    command = ["cmake", "--build", str(build), "--target", "anndata_example", "test_records",
               "--parallel", "2", "--verbose"]

    def rebuild(label: str) -> str:
        process = subprocess.Popen(command, cwd=source, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            data, _ = process.communicate(timeout=30)
        except BaseException as error:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try:
                tail, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                tail, _ = process.communicate()
            prefix = error.output if isinstance(error, subprocess.TimeoutExpired) else b""
            (output / f"{label}.log").write_bytes(tail or prefix or b"")
            raise
        result = data.decode(errors="replace")
        (output / f"{label}.log").write_text(result)
        if process.returncode:
            raise RuntimeError(f"{label} rebuild failed; see {output / (label + '.log')}")
        return result

    private_error = None
    edited_output = ""
    def stop(_signal: int, _frame: object) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        unit.write_bytes(before + b"\nstatic_assert(true); // private rebuild probe\n")
        edited_output = rebuild("edit")
        after_edit = state(bmis + consumers)
        if after_edit != public_before:
            raise RuntimeError("private edit changed a public BMI or consumer object")
        compiles = [line for line in edited_output.splitlines()
                    if "clang++" in line and "clang-scan-deps" not in line
                    and " -c " in line and " -o CMakeFiles/" in line]
        if len(compiles) != 1 or "src/records.cpp" not in compiles[0]:
            raise RuntimeError(f"expected one records.cpp compilation, observed {compiles}")
    except BaseException as error:
        private_error = error
    finally:
        unit.write_bytes(before)
        rebuild("restore")

    if private_error:
        raise private_error
    if unit.read_bytes() != before or state(bmis + consumers) != public_before:
        raise RuntimeError("source or public artifacts differ after restoration")
    restored = state(binaries)
    if any(restored[str(p)]["sha256"] != binaries_before[str(p)]["sha256"] for p in binaries):
        raise RuntimeError("a restored object, library or executable differs from its original bytes")
    result = {
        "source": str(source), "build": str(build),
        "source_sha256": hashlib.sha256(before).hexdigest(),
        "public_artifacts": public_before, "binaries_before": binaries_before,
        "restored_binaries": restored, "command": command,
        "edit_log_sha256": digest(output / "edit.log"),
        "restore_log_sha256": digest(output / "restore.log"),
        "public_bmis_unchanged": len(bmis), "consumer_objects_unchanged": len(consumers),
        "restored_binary_hashes_match": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_sha256": result["source_sha256"], "public_bmis_unchanged": len(bmis),
                      "consumer_objects_unchanged": len(consumers), "restored_binary_hashes_match": True}))


if __name__ == "__main__":
    main()
