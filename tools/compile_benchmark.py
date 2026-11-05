#!/usr/bin/env python3
"""Measure CMake/Ninja module build dependencies without touching the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


OMIT_PARTS = {".git", ".todo-orchestrator", ".agents", ".codex", ".aws", ".venv",
              "build", "out", "__pycache__"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(source: Path, target: Path) -> None:
    """Copy tracked/untracked source inputs while excluding runtime/build state."""
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {n for n in names if n in OMIT_PARTS}
        # Prior benchmark reports are evidence outputs, not inputs to this
        # source/build experiment; excluding them keeps report identity stable.
        if Path(directory).name == "benchmarks":
            ignored.add("results")
        return ignored

    shutil.copytree(source, target, ignore=ignore)


def checked(cmd: list[str], *, cwd: Path | None = None) -> tuple[float, int, str]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if proc.returncode:
        raise RuntimeError(f"command failed: {cmd!r}\n{proc.stdout[-12000:]}")
    return elapsed, proc.returncode, proc.stdout


def timed(cmd: list[str], log_path: Path, *, cwd: Path | None = None) -> dict:
    """Run a command under GNU time to capture the measured command's maximum RSS."""
    clock = shutil.which("/usr/bin/time")
    if not clock:
        wall, code, output = checked(cmd, cwd=cwd)
        log_path.write_text(output)
        return {"wall_seconds": wall, "max_rss_kib": None, "returncode": code}
    format_path = log_path.with_suffix(".time")
    started = time.perf_counter()
    proc = subprocess.run([clock, "-f", "PEAK_RSS_KIB=%M", "-o", str(format_path), *cmd],
                          cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    wall = time.perf_counter() - started
    log_path.write_text(proc.stdout)
    metrics = format_path.read_text(errors="replace") if format_path.exists() else ""
    match = re.search(r"PEAK_RSS_KIB=(\d+)", metrics)
    if proc.returncode:
        raise RuntimeError(f"command failed: {cmd!r}\n{proc.stdout[-12000:]}\n{metrics}")
    return {"wall_seconds": wall,
            "max_rss_kib": int(match.group(1)) if match else None,
            "returncode": proc.returncode}


def bmi_files(build: Path) -> dict[str, dict]:
    found = {}
    for path in build.rglob("*"):
        # CXXModules.json is CMake's generated dependency-scan metadata, not a
        # compiler BMI. Track actual module artifacts for invalidation claims.
        if path.is_file() and path.suffix in {".pcm", ".gcm"}:
            stat = path.stat()
            found[str(path.relative_to(build))] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest(path),
            }
    return found


def artifact_records(build: Path, pattern: str) -> dict[str, dict]:
    result = {}
    for path in sorted(build.rglob(pattern)):
        if not path.is_file():
            continue
        stat = path.stat()
        result[str(path.relative_to(build))] = {
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}
    return result


def object_records(build: Path) -> dict[str, dict]:
    return artifact_records(build, "*.o")


def matching_artifacts(records: dict[str, dict], fragment: str) -> dict[str, dict]:
    """Select one attributable build product, keeping hash and timestamp evidence."""
    return {path: value for path, value in records.items() if fragment in path}


def compiler_actions(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    return [line.strip() for line in log_path.read_text(errors="replace").splitlines()
            if "clang++" in line or "clang-scan-deps" in line]


def source_manifest_digest(root: Path) -> str:
    # Identify the sources and CMake metadata that drive the builds. Narrative
    # documentation and benchmark outputs are deliberately independent of this
    # digest, so an unrelated docs edit does not invalidate a measurements run.
    manifest = sorted(build_inputs(root).items())
    return hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()


def build_inputs(root: Path) -> dict[str, str]:
    inputs = {}
    suffixes = {".cpp", ".cppm", ".hpp", ".h", ".ixx", ".cc", ".cxx", ".cmake"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in OMIT_PARTS for part in relative.parts):
            continue
        if (path.name in {"CMakeLists.txt", "CMakePresets.json"}
                or path.name.endswith(".cmake.in") or path.suffix in suffixes):
            inputs[relative.as_posix()] = digest(path)
    return inputs


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"samples_seconds": values,
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "p25_seconds": ordered[(len(ordered) - 1) // 4],
            "p75_seconds": ordered[(3 * (len(ordered) - 1)) // 4]}


def get_command_output(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        proc = subprocess.run([exe, *cmd[1:]], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=10, check=False)
        return proc.stdout.splitlines()[0] if proc.stdout else f"exit {proc.returncode}"
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def configure(snapshot_dir: Path, build: Path, compiler: str, config: str, hdf5: bool, zarr: bool,
              dependency_prefix: Path) -> dict:
    build.mkdir(parents=True)
    cmd = ["cmake", "-S", str(snapshot_dir), "-B", str(build), "-G", "Ninja",
           f"-DCMAKE_CXX_COMPILER={compiler}", f"-DCMAKE_BUILD_TYPE={config}",
           "-DANNDATAPP_BUILD_TESTS=OFF", "-DANNDATAPP_BUILD_EXAMPLES=ON",
           "-DANNDATAPP_BUILD_BENCHMARKS=OFF", "-DANNDATAPP_ENABLE_INTEROP_TESTS=OFF",
           f"-DANNDATAPP_ENABLE_HDF5={'ON' if hdf5 else 'OFF'}",
           f"-DANNDATAPP_ENABLE_ZARR={'ON' if zarr else 'OFF'}",
           f"-DANNDATAPP_DEPENDENCY_PREFIX={dependency_prefix}"]
    wall, code, output = checked(cmd)
    return {"command": cmd, "wall_seconds": wall, "returncode": code,
            "output_tail": output[-2000:]}


def module_probe_sources(snapshot_dir: Path) -> None:
    (snapshot_dir / "bench-probes").mkdir(exist_ok=True)
    (snapshot_dir / "bench-probes" / "core.cpp").write_text(
        "import anndatapp.core;\n"
        "int main() { auto a=anndatapp::AnnData::empty(2,2); "
        "a.set_x(anndatapp::Array::zeros<float>({2,2})); "
        "return a.x().extent(0)==2 ? 0 : 1; }\n")
    (snapshot_dir / "bench-probes" / "facade.cpp").write_text(
        "import anndatapp;\n"
        "int main() { auto a=anndatapp::AnnData::empty(2,2); "
        "a.set_x(anndatapp::Array::zeros<float>({2,2})); "
        "return a.x().extent(0)==2 ? 0 : 1; }\n")
    with (snapshot_dir / "CMakeLists.txt").open("a") as f:
        f.write("\nadd_executable(bench_core bench-probes/core.cpp)\n"
                "target_link_libraries(bench_core PRIVATE anndatapp::core)\n"
                "add_executable(bench_facade bench-probes/facade.cpp)\n"
                "target_link_libraries(bench_facade PRIVATE anndatapp::anndatapp)\n")


def measure(args: argparse.Namespace) -> dict:
    source = Path(args.source).resolve()
    compiler = shutil.which(args.compiler) or args.compiler
    dependency_prefix = (Path(args.dependency_prefix).resolve() if args.dependency_prefix
                         else source / "build/dependencies/usr")
    compiler_major_match = re.search(r"clang\+\+-(\d+)", compiler)
    scanner_name = (f"clang-scan-deps-{compiler_major_match.group(1)}"
                    if compiler_major_match else "clang-scan-deps")
    scanner_version = (get_command_output([scanner_name, "--version"])
                       or get_command_output(["clang-scan-deps", "--version"]))
    tmp = Path(tempfile.mkdtemp(prefix="anndatapp-module-benchmark-"))
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    results: dict = {"schema": "anndatapp.module-compile-benchmark.v1",
                     "source_root": str(source),
                     "snapshot_root": "temporary generated sources and builds removed after successful run",
                     "source_manifest_sha256": None,
                     "environment": {
                         "platform": platform.platform(), "machine": platform.machine(),
                         "processor": platform.processor(), "cpu_model": cpu_model,
                         "logical_cpu_count": os.cpu_count(), "load_average_at_start": load_average,
                         "python": sys.version.split()[0],
                         "compiler": get_command_output([compiler, "--version"]),
                         "module_dependency_scanner": scanner_version,
                         "cmake": get_command_output(["cmake", "--version"]),
                         "ninja": get_command_output(["ninja", "--version"]),
                         "gnu_time": get_command_output(["/usr/bin/time", "--version"]),
                         "configuration": args.configuration,
                         "dependency_prefix": str(dependency_prefix),
                         "generator": "Ninja", "parallel_jobs": args.jobs,
                         "cache_policy": "fresh build trees; operating-system page cache was not cleared",
                         "repetitions": args.runs}}
    snap = tmp / "source"
    snapshot(source, snap)
    module_probe_sources(snap)
    results["source_manifest_sha256"] = source_manifest_digest(snap)
    results["build_input_hashes"] = build_inputs(snap)
    results["module_layout"] = {"core_import": "anndatapp.core", "facade_import": "anndatapp",
                                "enabled_format_modules": [name for name, enabled in
                                    (("anndatapp.h5ad", args.hdf5), ("anndatapp.zarr", args.zarr)) if enabled]}
    configure_rows: list[float] = []
    cold_rows: list[float] = []
    no_op_rows: list[float] = []
    no_op_library_rows: list[float] = []
    consumer_rows: list[float] = []
    implementation_rows: list[float] = []
    interface_rows: list[float] = []
    import_rows: dict[str, list[float]] = {"core": [], "facade": []}
    experiments = []
    build_cmd: list[str] | None = None
    for run in range(args.runs):
        run_root = tmp / f"run-{run + 1}"
        run_source = run_root / "source"
        snapshot(source, run_source)
        module_probe_sources(run_source)
        run_manifest = source_manifest_digest(run_source)
        if run_manifest != results["source_manifest_sha256"]:
            raise RuntimeError("source changed during measurement; rerun with a stable checkout")
        build = run_root / "build"
        conf = configure(run_source, build, compiler, args.configuration, args.hdf5, args.zarr, dependency_prefix)
        cache = (build / "CMakeCache.txt").read_text()
        for option, enabled in (("HDF5", args.hdf5), ("ZARR", args.zarr)):
            expected = f"ANNDATAPP_ENABLE_{option}:BOOL={'ON' if enabled else 'OFF'}"
            if expected not in cache.splitlines():
                raise RuntimeError(f"configured backend does not match report: {expected}")
        configure_rows.append(conf["wall_seconds"])
        build_cmd = ["cmake", "--build", str(build), "--target", "anndatapp", "anndatapp_example",
                     "--parallel", str(args.jobs), "--verbose"]
        cold = timed(build_cmd, run_root / "cold.log")
        cold_rows.append(cold["wall_seconds"])
        bmi_before = bmi_files(build)
        object_before = object_records(build)
        # Match the no-op and consumer-edit target sets exactly.
        consumer_build_cmd = ["cmake", "--build", str(build), "--target", "anndatapp_example",
                              "--parallel", str(args.jobs), "--verbose"]
        baseline = timed(consumer_build_cmd, run_root / "noop-consumer.log")
        no_op_rows.append(baseline["wall_seconds"])
        bmi_after_noop = bmi_files(build)
        object_after_noop = object_records(build)

        consumer_source = run_source / "examples" / "foundation.cpp"
        os.utime(consumer_source, ns=(time.time_ns(), time.time_ns()))
        consumer = timed(consumer_build_cmd, run_root / "consumer.log")
        consumer_rows.append(consumer["wall_seconds"])
        consumer_after_bmi = bmi_files(build)
        consumer_after_object = object_records(build)

        library_build_cmd = ["cmake", "--build", str(build), "--target", "anndatapp",
                             "--parallel", str(args.jobs), "--verbose"]
        no_op_library = timed(library_build_cmd, run_root / "noop-library.log")
        no_op_library_rows.append(no_op_library["wall_seconds"])
        impl_file = run_source / "src" / "arrays.cpp"
        with impl_file.open("a") as stream:
            stream.write("\nnamespace { [[maybe_unused]] constexpr int compile_benchmark_impl_revision = 1; }\n")
        impl = timed(library_build_cmd, run_root / "implementation.log")
        implementation_rows.append(impl["wall_seconds"])
        impl_bmi = bmi_files(build)
        impl_object = object_records(build)

        iface_file = run_source / "src" / "anndatapp.arrays.cppm"
        content = iface_file.read_text()
        marker = "} // namespace anndatapp"
        if marker not in content:
            raise RuntimeError(f"benchmark could not find public namespace boundary in {iface_file}")
        content = content.replace(marker,
            f"inline constexpr int compile_benchmark_revision_{run + 1} = {run + 1};\n\n" + marker)
        iface_file.write_text(content)
        interface = timed(consumer_build_cmd, run_root / "interface.log")
        interface_rows.append(interface["wall_seconds"])
        iface_bmi = bmi_files(build)
        iface_object = object_records(build)

        import_run = {"run": run + 1, "source_manifest_sha256": run_manifest,
                      "configure": conf, "cold": cold, "no_op": baseline,
                      "no_op_library": no_op_library,
                      "consumer_edit": consumer, "implementation_edit": impl,
                      "interface_edit": interface,
                      "build_commands": {"cold": build_cmd,
                                         "no_op_consumer": consumer_build_cmd,
                                         "consumer_edit": consumer_build_cmd,
                                         "no_op_library": library_build_cmd,
                                         "implementation_edit": library_build_cmd,
                                         "interface_edit": consumer_build_cmd},
                      "artifacts": {"bmi_before": bmi_before,
                                    "bmi_after_noop": bmi_after_noop,
                                    "bmi_after_consumer_edit": consumer_after_bmi,
                                    "bmi_after_implementation_edit": impl_bmi,
                                    "bmi_after_interface_edit": iface_bmi,
                                    "objects_before": object_before,
                                    "objects_after_noop": object_after_noop,
                                    "objects_after_consumer_edit": consumer_after_object,
                                    "objects_after_implementation_edit": impl_object,
                                    "objects_after_interface_edit": iface_object,
                                    "commands": {kind: compiler_actions(run_root / f"{kind}.log")
                                                 for kind in ("cold", "noop-consumer", "consumer",
                                                              "noop-library", "implementation", "interface")}},
                      "log_files": {kind: str(run_root / f"{kind}.log") for kind in
                                    ("cold", "noop-consumer", "consumer", "noop-library",
                                     "implementation", "interface")}}
        for kind in ("core", "facade"):
            target = f"bench_{kind}"
            row = timed(["cmake", "--build", str(build), "--target", target,
                         "--parallel", str(args.jobs), "--verbose"], run_root / f"import-{kind}.log")
            import_rows[kind].append(row["wall_seconds"])
            import_run[f"import_{kind}"] = row
            import_run["build_commands"][f"import_{kind}"] = [
                "cmake", "--build", str(build), "--target", target,
                "--parallel", str(args.jobs), "--verbose"]
            import_run["artifacts"]["commands"][f"import_{kind}"] = compiler_actions(
                run_root / f"import-{kind}.log")
        # Store commands in the durable report. Keep temporary build trees out of
        # the checkout and make no references to their generated log files.
        import_run.pop("log_files", None)
        experiments.append(import_run)

    results["summary_seconds"] = {
        "configure": summarize(configure_rows), "cold_library_and_facade_example": summarize(cold_rows),
        "no_op_consumer": summarize(no_op_rows), "consumer_source_edit": summarize(consumer_rows),
        "no_op_library": summarize(no_op_library_rows),
        "implementation_source_edit": summarize(implementation_rows),
        "public_interface_edit": summarize(interface_rows),
        "fresh_consumer_import_core": summarize(import_rows["core"]),
        "fresh_consumer_import_facade": summarize(import_rows["facade"])}
    results["runs"] = experiments
    results["dependency_checks"] = []
    for row in experiments:
        a = row["artifacts"]
        results["dependency_checks"].append({
            "run": row["run"],
            "no_op_kept_bmi_metadata": a["bmi_before"] == a["bmi_after_noop"],
            "no_op_kept_objects": a["objects_before"] == a["objects_after_noop"],
            "consumer_edit_kept_bmis": a["bmi_after_noop"] == a["bmi_after_consumer_edit"],
            "consumer_edit_rebuilt_consumer": matching_artifacts(
                a["objects_after_noop"], "examples/foundation.cpp.o") != matching_artifacts(
                a["objects_after_consumer_edit"], "examples/foundation.cpp.o"),
            "implementation_edit_kept_bmis": a["bmi_after_consumer_edit"] == a["bmi_after_implementation_edit"],
            "implementation_edit_kept_consumer": matching_artifacts(
                a["objects_after_consumer_edit"], "examples/foundation.cpp.o") == matching_artifacts(
                a["objects_after_implementation_edit"], "examples/foundation.cpp.o"),
            "implementation_edit_rebuilt_implementation": matching_artifacts(
                a["objects_after_consumer_edit"], "src/arrays.cpp.o") != matching_artifacts(
                a["objects_after_implementation_edit"], "src/arrays.cpp.o"),
            "interface_edit_changed_bmi": a["bmi_after_implementation_edit"] != a["bmi_after_interface_edit"],
            "interface_edit_rebuilt_consumer": matching_artifacts(
                a["objects_after_implementation_edit"], "examples/foundation.cpp.o") != matching_artifacts(
                a["objects_after_interface_edit"], "examples/foundation.cpp.o")})
    results["notes"] = [
        "Cold means a fresh configure/build directory; the OS page cache was not cleared.",
        "Core and facade probes link their respective build-tree targets.",
        "Consumer import measurements reuse the built package module artifacts.",
        "Interface edits add an inline public constant to the arrays partition in a disposable source snapshot.",
        "Implementation edits append an unused private declaration to arrays.cpp in a disposable source snapshot.",
        "Enabled format modules reflect the checked CMake cache; other optional dependencies are outside this probe.",
        "Wall time includes the complete requested CMake/Ninja operation; build-command RSS is reported where GNU time is available."]
    results["temporary_build_root"] = "removed after collecting command and artifact evidence"
    shutil.rmtree(tmp)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=".", help="repository source root")
    parser.add_argument("--output", required=True, help="JSON report path; parent must exist")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--compiler", default="clang++-18")
    parser.add_argument("--configuration", default="Release", choices=("Release", "Debug", "RelWithDebInfo"))
    parser.add_argument("--dependency-prefix", help="private storage dependency prefix (default: source/build/dependencies/usr)")
    parser.add_argument("--hdf5", action=argparse.BooleanOptionalAction, default=True, help="enable H5AD backend (default: enabled)")
    parser.add_argument("--zarr", action=argparse.BooleanOptionalAction, default=True, help="enable Zarr backend (default: enabled)")
    args = parser.parse_args()
    if args.runs < 5:
        parser.error("at least five paired repetitions are required")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = measure(args)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination.resolve()),
                      "source_manifest_sha256": report["source_manifest_sha256"],
                      "summary_seconds": report["summary_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
