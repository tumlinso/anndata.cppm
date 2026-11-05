#!/usr/bin/env python3
"""Measure isolated C++20 module builds and native H5AD I/O."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time

EXCLUDED = {".git", ".todo-orchestrator", ".agents", ".codex", ".aws",
            ".venv", "build", "out", "__pycache__", "results"}
BUILD_SUFFIXES = {".cpp", ".cppm", ".c", ".h", ".hh", ".hpp", ".ixx",
                  ".cc", ".cxx", ".cmake"}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def manifest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()
            and not any(part in EXCLUDED for part in p.relative_to(root).parts)
            and (p.suffix in BUILD_SUFFIXES or p.name.endswith(".cmake.in")
                 or p.name in {"CMakeLists.txt", "CMakePresets.json"})}

def copy_snapshot(source: Path, target: Path) -> dict[str, str]:
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        if any(part in EXCLUDED for part in rel.parts):
            continue
        destination = target / rel
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return manifest(target)

def run(cmd: list[str], cwd: Path, timeout: int = 900) -> tuple[float, str]:
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, check=False,
                            timeout=timeout)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"command failed: {cmd!r}\n{result.stdout[-12000:]}")
    return elapsed, result.stdout

def timed(cmd: list[str], cwd: Path, log: Path) -> dict:
    rss_file = log.with_suffix(".rss")
    start = time.perf_counter()
    result = subprocess.run(["/usr/bin/time", "-f", "%M", "-o", str(rss_file), *cmd],
                            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False, timeout=900)
    elapsed = time.perf_counter() - start
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed: {cmd!r}\n{result.stdout[-12000:]}")
    raw_rss = rss_file.read_text().strip()
    actions = [line.strip() for line in result.stdout.splitlines()
               if "clang++" in line or "clang-scan-deps" in line]
    return {"command": cmd, "elapsed_seconds": elapsed,
            "max_rss_kib": int(raw_rss) if raw_rss.isdigit() else None,
            "compiler_actions": actions, "compiler_action_count": len(actions)}

def artifacts(root: Path, suffixes: set[str]) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256(path)}
    return result

def object_groups(build: Path) -> dict[str, dict]:
    objects = artifacts(build, {".o"})
    return {
        "all_objects": objects,
        "consumer_objects": {
            path: record for path, record in objects.items()
            if path.startswith(("CMakeFiles/h5ad_import_", "CMakeFiles/h5ad_benchmark.dir/"))},
        "library_objects": {
            path: record for path, record in objects.items()
            if path.startswith("CMakeFiles/anndatapp.dir/")},
    }

def build_cmd(build: Path, *targets: str) -> list[str]:
    return ["cmake", "--build", str(build), "--parallel", "2", "--verbose",
            "--target", *targets]

def add_import_probes(source: Path, enabled: bool) -> None:
    probe = source / "benchmarks" / "imports"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "core.cpp").write_text("import anndatapp.core;\nint main() { return 0; }\n")
    (probe / "facade.cpp").write_text("import anndatapp;\nint main() { return 0; }\n")
    addition = "\n# Temporary module import probes.\n"
    for name in ("core", "facade"):
        addition += (f"add_executable(h5ad_import_{name} benchmarks/imports/{name}.cpp)\n"
                     f"target_link_libraries(h5ad_import_{name} PRIVATE anndatapp::anndatapp)\n")
    if enabled:
        (probe / "h5ad.cpp").write_text("import anndatapp.h5ad;\nint main() { return 0; }\n")
        addition += ("add_executable(h5ad_import_module benchmarks/imports/h5ad.cpp)\n"
                     "target_link_libraries(h5ad_import_module PRIVATE anndatapp::anndatapp)\n")
    cmake = source / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + addition)

def compile_trial(root: Path, temp: Path, enabled: bool, trial: int) -> dict:
    source = temp / f"src-{int(enabled)}-{trial}"
    source.mkdir()
    source_hashes = copy_snapshot(root, source)
    add_import_probes(source, enabled)
    build = temp / f"build-{int(enabled)}-{trial}"
    work = temp / f"work-{int(enabled)}-{trial}"
    work.mkdir()
    configure = ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                 "-DCMAKE_CXX_COMPILER=clang++-18", "-DCMAKE_BUILD_TYPE=Release",
                 "-DANNDATAPP_BUILD_TESTS=ON", "-DANNDATAPP_BUILD_EXAMPLES=ON",
                 "-DANNDATAPP_BUILD_BENCHMARKS=ON",
                 f"-DANNDATAPP_ENABLE_HDF5={'ON' if enabled else 'OFF'}",
                 "-DANNDATAPP_ENABLE_INTEROP_TESTS=OFF"]
    configure_seconds, output = run(configure, source)
    (work / "configure.log").write_text(output)
    library = timed(build_cmd(build, "anndatapp"),
                    source, work / "library.log")
    imports = {}
    targets = ["h5ad_import_core", "h5ad_import_facade"]
    if enabled:
        targets.append("h5ad_import_module")
    for target in targets:
        imports[target] = timed(build_cmd(build, target), source, work / f"{target}.log")
    outcome = {"hdf5_enabled": enabled, "trial": trial,
               "configure_command": configure,
               "configure_seconds": configure_seconds, "cold_library_build": library,
               "first_import_builds": imports,
               "bmis": artifacts(build, {".pcm", ".gcm"}),
               **object_groups(build),
               "source_sha256": source_hashes}
    if enabled:
        outcome["benchmark_driver_build"] = timed(
            build_cmd(build, "h5ad_benchmark"), source, work / "h5ad_benchmark.log")
        outcome["io_executable"] = str(build / "h5ad_benchmark")
    return outcome

def invalidation_probe(source: Path, build: Path, work: Path) -> dict:
    """Confirm a private H5AD implementation edit keeps public BMIs/objects."""
    implementation = source / "src" / "h5ad.cpp"
    interface = source / "src" / "anndatapp.h5ad.cppm"
    if not implementation.is_file() or not interface.is_file():
        raise RuntimeError("H5AD implementation/interface paths are missing from the snapshot")
    targets = ["h5ad_import_core", "h5ad_import_facade", "h5ad_import_module",
               "h5ad_benchmark"]
    before_bmis = artifacts(build, {".pcm", ".gcm"})
    before_objects = object_groups(build)

    implementation_bytes = implementation.read_bytes()
    implementation.write_bytes(implementation_bytes + b"\n")
    impl_seconds, impl_log = run(
        build_cmd(build, *targets), source)
    (work / "implementation-edit-rebuild.log").write_text(impl_log)
    impl_actions = [line.strip() for line in impl_log.splitlines()
                    if "clang++" in line or "clang-scan-deps" in line]
    after_impl_bmis = artifacts(build, {".pcm", ".gcm"})
    after_impl_objects = object_groups(build)

    implementation.write_bytes(implementation_bytes)
    interface_bytes = interface.read_bytes()
    interface.write_bytes(interface_bytes + b"\n")
    interface_seconds, interface_log = run(
        build_cmd(build, *targets), source)
    (work / "interface-edit-rebuild.log").write_text(interface_log)
    interface_actions = [line.strip() for line in interface_log.splitlines()
                         if "clang++" in line or "clang-scan-deps" in line]
    after_interface_bmis = artifacts(build, {".pcm", ".gcm"})
    after_interface_objects = object_groups(build)

    def changed(old: dict, new: dict) -> list[str]:
        return sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))

    return {
        "commands": {
            "implementation": build_cmd(build, *targets),
            "interface": build_cmd(build, *targets),
        },
        "implementation_edit": {
            "elapsed_seconds": impl_seconds,
            "compiler_actions": impl_actions,
            "changed_bmis": changed(before_bmis, after_impl_bmis),
            "changed_consumer_objects": changed(before_objects["consumer_objects"],
                                                  after_impl_objects["consumer_objects"]),
            "changed_library_objects": changed(before_objects["library_objects"],
                                                after_impl_objects["library_objects"]),
            "public_bmis_preserved": before_bmis == after_impl_bmis,
            "consumer_objects_preserved": before_objects["consumer_objects"]
                                           == after_impl_objects["consumer_objects"],
        },
        "interface_edit": {
            "elapsed_seconds": interface_seconds,
            "compiler_actions": interface_actions,
            "changed_bmis": changed(after_impl_bmis, after_interface_bmis),
            "changed_consumer_objects": changed(after_impl_objects["consumer_objects"],
                                                  after_interface_objects["consumer_objects"]),
            "changed_library_objects": changed(after_impl_objects["library_objects"],
                                                after_interface_objects["library_objects"]),
            "at_least_one_bmi_rebuilt": bool(changed(after_impl_bmis, after_interface_bmis)),
            "at_least_one_consumer_object_rebuilt": bool(changed(
                after_impl_objects["consumer_objects"], after_interface_objects["consumer_objects"])),
        },
    }

def io_trial(exe: Path, work: Path, kind: str, compression: str, byte_order: str,
             rows: int, cols: int, per_row: int, repetitions: int) -> list[dict]:
    path = work / f"{kind}-{compression}-{byte_order}.h5ad"
    dimensions = [str(rows), str(cols), str(per_row), byte_order]
    base = [str(exe), "seed", kind, compression, str(path), *dimensions]
    run(base, work)
    observations = []
    for action in ("write", "read"):
        for repeat in range(repetitions):
            if action == "read":
                run(base, work)
            log = work / f"{kind}-{compression}-{byte_order}-{action}-{repeat}.log"
            start = time.perf_counter()
            result = subprocess.run(["/usr/bin/time", "-f", "%M", "-o", str(log) + ".rss",
                                     str(exe), action, kind, compression, str(path), *dimensions],
                                    cwd=work, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, check=False)
            wall = time.perf_counter() - start
            if result.returncode:
                raise RuntimeError(f"I/O benchmark failed: {result.stdout[-8000:]}")
            item = json.loads(result.stdout.strip().splitlines()[-1])
            rss = Path(str(log) + ".rss").read_text().strip()
            item.update({"trial": repeat + 1, "command_seconds": wall,
                         "max_rss_kib": int(rss) if rss.isdigit() else None})
            observations.append(item)
    return observations

def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for item in records:
        groups.setdefault((item["kind"], item["compression"], item["byte_order"],
                           item["action"]), []).append(item)
    result = []
    for (kind, compression, byte_order, action), group in groups.items():
        times = [item["seconds"] for item in group]
        rss = [item["max_rss_kib"] for item in group if item["max_rss_kib"] is not None]
        result.append({"kind": kind, "compression": compression,
                       "byte_order": byte_order, "action": action,
                       "median_seconds": statistics.median(times),
                       "min_seconds": min(times), "max_seconds": max(times),
                       "median_max_rss_kib": statistics.median(rss) if rss else None,
                       "payload_bytes": group[0]["payload_bytes"],
                       "median_file_bytes": statistics.median(i["file_bytes"] for i in group),
                       "checksums_match": all(i["checksum"] == group[0]["checksum"] for i in group)})
    return result

def roundtrip_checksums(records: list[dict]) -> list[dict]:
    pairs = {}
    for item in records:
        pairs.setdefault((item["kind"], item["compression"], item["byte_order"],
                          item["action"]), []).append(item["checksum"])
    result = []
    for kind in ("dense", "csr"):
        for compression in ("none", "gzip"):
            for byte_order in ("native", "big"):
                writes = pairs[(kind, compression, byte_order, "write")]
                reads = pairs[(kind, compression, byte_order, "read")]
                result.append({"kind": kind, "compression": compression,
                               "byte_order": byte_order,
                               "write_checksum": writes[0], "read_checksum": reads[0],
                               "all_write_checksums_equal": all(x == writes[0] for x in writes),
                               "all_read_checksums_equal": all(x == reads[0] for x in reads),
                               "write_read_match": writes[0] == reads[0]})
    return result

def reclassify_saved_report(root: Path, temporary: Path, output: Path) -> None:
    """Keep measured trials and rerun only object invalidation in a retained build."""
    report = json.loads(output.read_text())
    enabled_trials = [trial for trial in report["build_trials"] if trial["hdf5_enabled"]]
    if not enabled_trials:
        raise RuntimeError("the saved report has no HDF5-enabled build")
    trial_number = enabled_trials[-1]["trial"]
    source = temporary / f"src-1-{trial_number}"
    build = temporary / f"build-1-{trial_number}"
    work = temporary / f"work-1-{trial_number}"
    expected = enabled_trials[-1]["source_sha256"]
    for relative in ("src/h5ad.cpp", "src/anndatapp.h5ad.cppm"):
        if sha256(root / relative) != expected.get(relative):
            raise RuntimeError(f"the measured source identity changed: {relative}")
    (source / "src/anndatapp.h5ad.cppm").write_bytes(
        (root / "src/anndatapp.h5ad.cppm").read_bytes())
    reset_command = build_cmd(
        build, "h5ad_import_core", "h5ad_import_facade", "h5ad_import_module", "h5ad_benchmark")
    reset_seconds, reset_log = run(reset_command, source)
    (work / "restore-measured-interface.log").write_text(reset_log)
    (work / "restore-measured-interface-seconds").write_text(str(reset_seconds))
    report["rebuild_invalidation"] = invalidation_probe(source, build, work)
    for record in report["build_trials"]:
        objects = record.pop("all_objects", record.pop("consumer_objects", {}))
        record.pop("library_objects", None)
        record["all_objects"] = objects
        record["consumer_objects"] = {
            path: artifact for path, artifact in objects.items()
            if path.startswith(("CMakeFiles/h5ad_import_", "CMakeFiles/h5ad_benchmark.dir/"))}
        record["library_objects"] = {
            path: artifact for path, artifact in objects.items()
            if path.startswith("CMakeFiles/anndatapp.dir/")}
    impl = report["rebuild_invalidation"]["implementation_edit"]
    api = report["rebuild_invalidation"]["interface_edit"]
    report["rebuild_invalidation"]["dependency_assertions_passed"] = bool(
        impl["public_bmis_preserved"] and impl["consumer_objects_preserved"]
        and api["at_least_one_bmi_rebuilt"]
        and api["at_least_one_consumer_object_rebuilt"])
    report["measurement_correction"] = (
        "Cold build and H5AD I/O samples were retained unchanged. The first invalidation "
        "snapshot counted CMakeFiles/anndatapp.dir/src/h5ad.cpp.o as a consumer object; "
        "this was a classification error. Restored the measured H5AD interface source, "
        "rebuilt the same imported consumers, then reran only the implementation/interface "
        "invalidation checks with module-provider objects separated from library objects. "
        "Source snapshots exclude documentation, benchmark reports, and measurement tooling.")
    output.write_text(json.dumps(report, indent=2) + "\n")

def first_line(cmd: list[str]) -> str:
    if not shutil.which(cmd[0]) and not Path(cmd[0]).exists():
        return "unavailable"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unavailable"

def host_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return platform.processor() or "unavailable"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--per-row", type=int, default=128)
    parser.add_argument("--io-executable", type=Path,
                        help="skip compile trials and run I/O with this built executable")
    parser.add_argument("--reclassify-saved-report", type=Path,
                        help="rerun only invalidation checks in a retained measurement temp directory")
    parser.add_argument("--keep-builds", action="store_true")
    args = parser.parse_args()
    if min(args.repetitions, args.rows, args.columns, args.per_row) < 1:
        parser.error("repetitions and workload dimensions must be positive")
    root = Path(__file__).resolve().parents[1]
    if args.reclassify_saved_report:
        reclassify_saved_report(root, args.reclassify_saved_report, args.output)
        print(f"updated invalidation classification in {args.output}; build and I/O trials retained")
        return 0
    git_status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=root, stdout=subprocess.PIPE, text=True, check=False)
    report = {"format": "anndatapp.h5ad-benchmark", "version": 1,
              "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                         stdout=subprocess.PIPE, text=True, check=False).stdout.strip(),
              "measurement_tool_sha256": sha256(Path(__file__).resolve()),
              "working_tree_dirty": bool(git_status.stdout.strip()),
              "working_tree_paths": [line[3:] for line in git_status.stdout.splitlines()],
              "source_sha256": manifest(root),
              "environment": {"platform": platform.platform(), "machine": platform.machine(),
                              "host_cpu_model": host_cpu_model(), "logical_cpus": os.cpu_count(),
                              "compiler": first_line(["clang++-18", "--version"]),
                              "module_scanner": first_line(["clang-scan-deps-18", "--version"]),
                              "module_scanner_package": first_line(
                                  ["dpkg-query", "-W", "-f=Version", "clang-tools-18"]),
                              "cmake": first_line(["cmake", "--version"]),
                              "ninja": first_line(["ninja", "--version"]),
                              "hdf5": first_line(["pkg-config", "--modversion", "hdf5"]),
                              "page_cache": "not cleared; repeated input reads use host cache"},
              "conditions": {"parallel_build_jobs": 2, "repetitions": args.repetitions,
                             "rss": "GNU time maximum resident set for the waited process",
                             "input": f"float32 dense {args.rows}x{args.columns}; float32 CSR "
                                      f"{args.rows}x{args.columns} with {args.per_row} entries per row"},
              "build_trials": [], "rebuild_invalidation": None, "io_raw_trials": []}
    context = None
    if args.keep_builds:
        temp = Path(tempfile.mkdtemp(prefix="anndatapp-h5ad-bench-"))
    else:
        context = tempfile.TemporaryDirectory(prefix="anndatapp-h5ad-bench-")
        temp = Path(context.name)
    try:
        if args.io_executable:
            exe = args.io_executable.resolve()
        else:
            for enabled in (False, True):
                for trial in range(1, args.repetitions + 1):
                    report["build_trials"].append(compile_trial(root, temp, enabled, trial))
            exe = Path(report["build_trials"][-1]["io_executable"])
            if not exe.exists():
                raise RuntimeError("expected h5ad_benchmark in the HDF5-enabled configuration")
            report["rebuild_invalidation"] = invalidation_probe(
                temp / f"src-1-{args.repetitions}", temp / f"build-1-{args.repetitions}",
                temp / f"work-1-{args.repetitions}")
        work = temp / "io"
        work.mkdir(exist_ok=True)
        for kind in ("dense", "csr"):
            for compression in ("none", "gzip"):
                for byte_order in ("native", "big"):
                    report["io_raw_trials"].extend(io_trial(
                        exe, work, kind, compression, byte_order, args.rows, args.columns,
                        args.per_row, args.repetitions))
        report["io_summary"] = summarize(report["io_raw_trials"])
        report["roundtrip_checksums"] = roundtrip_checksums(report["io_raw_trials"])
        if report["rebuild_invalidation"]:
            impl = report["rebuild_invalidation"]["implementation_edit"]
            api = report["rebuild_invalidation"]["interface_edit"]
            report["rebuild_invalidation"]["dependency_assertions_passed"] = bool(
                impl["public_bmis_preserved"] and impl["consumer_objects_preserved"]
                and api["at_least_one_bmi_rebuilt"]
                and api["at_least_one_consumer_object_rebuilt"])
        report["limitations"] = [
            "One Linux host and installed HDF5; page cache was not cleared.",
            "Maximum RSS is GNU time %M for each waited process, not a sum across processes.",
            "Dense and CSR inputs are deterministic synthetic float32 data.",
            "Cold build trials use isolated source snapshots and two Ninja jobs.",
            "The measurements establish workload-specific observations, not general speedups."]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
        if args.keep_builds:
            print(f"temporary snapshots retained at {temp}")
    finally:
        if context:
            context.cleanup()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
