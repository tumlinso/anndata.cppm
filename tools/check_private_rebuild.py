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
    parser.add_argument("--unit", default="src/records.cpp", help="private implementation unit relative to source")
    args = parser.parse_args()
    source, build, output = args.source.resolve(), args.build.resolve(), args.output.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n" not in cache:
        raise RuntimeError("build directory does not belong to the supplied source")

    unit = source / args.unit
    if not unit.is_file() or unit.suffix != ".cpp" or source not in unit.resolve().parents:
        raise RuntimeError("private probe unit must be a .cpp inside the source tree")
    before = unit.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    (output / (unit.name + ".original")).write_bytes(before)
    component_dirs = [build / "CMakeFiles/anndatapp_core.dir", build / "CMakeFiles/anndatapp.dir"]
    bmis = sorted(Path(module["bmi"]) for directory in component_dirs
                  for module in json.loads((directory / "CXXModules.json").read_text())["modules"].values()
                  if not module["is-private"])
    if not bmis:
        raise RuntimeError("no public BMIs found")
    consumers = [build / "CMakeFiles/anndatapp_example.dir/examples/foundation.cpp.o",
                 build / "CMakeFiles/test_records.dir/tests/test_records.cpp.o"]
    objects = [directory / (args.unit + ".o") for directory in component_dirs]
    # A reused build tree may retain an object from before a component split.
    # Only count objects with an active Ninja build edge for this source.
    def active_object(path: Path) -> bool:
        if not path.is_file(): return False
        query = subprocess.run(["ninja", "-C", str(build), "-t", "query", str(path.relative_to(build))],
                               text=True, capture_output=True, check=True).stdout
        return "\n  input:" in query and f"\n    {unit}" in query
    objects = [path for path in objects if active_object(path)]
    if len(objects) != 1:
        raise RuntimeError(f"expected one component object for {args.unit}, found {objects}")
    component = objects[0].parents[len(Path(args.unit).parts) - 1].name.removesuffix(".dir")
    libraries = [build / ("lib" + component + suffix) for suffix in (".a", ".so", ".dylib")]
    libraries = [path for path in libraries if path.is_file()]
    if len(libraries) != 1:
        raise RuntimeError(f"expected one component library for {component}, found {libraries}")
    binaries = objects + libraries + [build / "anndatapp_example", build / "test_records"]
    public_before, binaries_before = state(bmis + consumers), state(binaries)
    command = ["cmake", "--build", str(build), "--target", "anndatapp_example", "test_records",
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
        if len(compiles) != 1 or args.unit not in compiles[0]:
            raise RuntimeError(f"expected one {args.unit} compilation, observed {compiles}")
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
        "source": str(source), "build": str(build), "component": component,
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
