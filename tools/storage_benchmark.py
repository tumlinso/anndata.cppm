#!/usr/bin/env python3
"""Reproducible native storage and C++20 module-build benchmark driver.

Build experiments run from an immutable source snapshot in fresh build trees.
Storage actions run as independent processes so failures, elapsed time, and peak
RSS remain attributable to one fixture/action/trial. No benchmark is started by
importing this module.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


EXCLUDED_DIRS = {
    ".git", ".todo", ".todo-orchestrator", ".agents", ".codex", ".aws", ".venv",
    "build", "out", "__pycache__", "results",
}
FORMATS = ("h5ad", "zarr2", "zarr3")
ACTIONS = (
    "open", "dense", "sparse-row", "sparse-column", "map", "parallel",
    "huge", "write", "export", "reindex", "structure",
)
MUTATING_ACTIONS = {"write", "reindex", "structure"}
IMPORTS = {
    "core": "anndata.core",
    "backed": "anndata.backed",
    "facade": "anndata",
    "h5ad": "anndata.h5ad",
    "zarr": "anndata.zarr",
}
CONFIGURATIONS = {
    "core": (False, False),
    "hdf5_only": (True, False),
    "zarr_only": (False, True),
    "both": (True, True),
}
DEFAULT_GENERATED_ACTIONS = {
    "dense": ("open", "dense", "map", "parallel", "write", "export",
              "reindex", "structure"),
    "csr": ("open", "sparse-row", "sparse-column", "export", "reindex", "structure"),
    "csc": ("open", "sparse-row", "sparse-column", "export", "reindex", "structure"),
    "huge": ("open", "huge"),
}


_STOP_REQUESTED = False
_STOP_REASON: str | None = None
_STOP_SIGNAL: int | None = None
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def request_stop(signum: int, _frame: Any) -> None:
    """Defer signal cleanup to run_process, which owns the child process group."""
    global _STOP_REQUESTED, _STOP_REASON, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    _STOP_REASON = f"received {name}"


def stop_reason(stop_file: str | Path | None = None) -> str | None:
    if _STOP_REQUESTED:
        return _STOP_REASON or "stop requested"
    if stop_file is not None and Path(stop_file).exists():
        return f"stop file exists: {Path(stop_file).resolve()}"
    return None


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON file without exposing a partially written report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                          dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def write_checkpoint(path: Path, report: dict[str, Any]) -> None:
    report["checkpoint"] = {
        "path": str(path),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "state": "in_progress",
    }
    atomic_write_json(path, report)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path, *, exclude_dirs: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Hash every file and symlink in a frozen tree, in stable path order."""
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if exclude_dirs and any(part in exclude_dirs for part in relative.parts):
            continue
        rel = relative.as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            records[rel] = {
                "kind": "symlink",
                "target": target,
                "sha256": hashlib.sha256(("symlink:" + target).encode()).hexdigest(),
            }
        elif path.is_file():
            records[rel] = {
                "kind": "file", "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def manifest_digest(records: dict[str, Any]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def snapshot_source(source: Path, target: Path) -> dict[str, dict[str, Any]]:
    before = tree_manifest(source, exclude_dirs=EXCLUDED_DIRS)
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in EXCLUDED_DIRS}

    shutil.copytree(source, target, ignore=ignore, symlinks=True)
    copied = tree_manifest(target)
    after = tree_manifest(source, exclude_dirs=EXCLUDED_DIRS)
    if before != after:
        raise RuntimeError("source tree changed while the benchmark snapshot was copied")
    if before != copied:
        raise RuntimeError("frozen source snapshot differs from the stable source tree")
    return copied


def tool_output(command: list[str], *, timeout: int = 10) -> str | None:
    executable = shutil.which(command[0]) or (command[0] if Path(command[0]).exists() else None)
    if executable is None:
        return None
    try:
        result = subprocess.run([executable, *command[1:]], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                check=False, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return f"{type(error).__name__}: {error}"
    text = result.stdout.strip()
    return text[:4000] if text else f"exit {result.returncode}"


def host_evidence() -> dict[str, Any]:
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key] = value.strip()
    except OSError:
        pass
    cgroup = {}
    for key, path in (("memory_max", Path("/sys/fs/cgroup/memory.max")),
                      ("memory_current", Path("/sys/fs/cgroup/memory.current"))):
        try:
            cgroup[key] = path.read_text().strip()
        except OSError:
            cgroup[key] = None
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(), "load_average_at_start": load_average,
        "memory_at_start": memory, "cgroup_memory_at_start": cgroup,
        "python": sys.version, "python_executable": sys.executable,
    }


def python_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("anndata", "numpy", "scipy", "pandas", "h5py", "zarr", "numcodecs"):
        try:
            versions[package] = package_version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def relevant_environment() -> dict[str, str | None]:
    names = (
        "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
        "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR", "LD_LIBRARY_PATH", "PATH",
    )
    return {name: os.environ.get(name) for name in names}


def cache_values(path: Path) -> dict[str, str]:
    wanted = re.compile(
        r"^(CMAKE_(CXX_COMPILER|CXX_COMPILER_VERSION|CXX_FLAGS(?:_RELEASE)?|BUILD_TYPE|GENERATOR|PREFIX_PATH)|"
        r"ANNDATA_(ENABLE_HDF5|ENABLE_ZARR|DEPENDENCY_PREFIX|BUILD_BENCHMARKS)|"
        r"(ANNDATA_HDF5|zstd|nlohmann_json|Blosc|ZLIB)_.*)"
    )
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if wanted.match(name):
            values[name] = value
    return values


def compiler_actions(output: str) -> dict[str, list[str]]:
    compiler: list[str] = []
    scanner: list[str] = []
    other: list[str] = []
    for line in output.splitlines():
        item = line.strip()
        if "clang-scan-deps" in item:
            scanner.append(item)
        elif re.search(r"(?:^|[/\s])clang\+\+(?:-[0-9]+)?(?:\s|$)", item):
            compiler.append(item)
        elif "clang++" in item:
            compiler.append(item)
        elif item and ("[" in item and "]" in item):
            other.append(item)
    return {"compiler": compiler, "dependency_scanner": scanner, "build_steps": other}


def artifact_records(root: Path, suffixes: set[str]) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            stat = path.stat()
            records[path.relative_to(root).as_posix()] = {
                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
    return records


def build_artifacts(build: Path) -> dict[str, dict[str, Any]]:
    objects = artifact_records(build, {".o"})
    return {
        "bmis": artifact_records(build, {".pcm", ".gcm"}),
        "all_objects": objects,
        "consumer_objects": {
            path: item for path, item in objects.items()
            if path.startswith("CMakeFiles/storage_import_")
        },
        "library_objects": {
            path: item for path, item in objects.items()
            if path.startswith("CMakeFiles/anndata.dir/")
        },
        "other_objects": {
            path: item for path, item in objects.items()
            if not path.startswith(("CMakeFiles/storage_import_", "CMakeFiles/anndata.dir/"))
        },
    }


def changed_keys(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(key for key in before.keys() | after.keys()
                  if before.get(key) != after.get(key))


def artifact_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": sorted(key for key in before.keys() & after.keys()
                           if before[key] != after[key]),
    }


def limit_memory(mib: int):
    if mib <= 0:
        return None
    limit = mib * 1024 * 1024

    def apply_limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return apply_limit


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # The group leader may exit while a descendant ignores SIGTERM and keeps
    # stdout/stderr open. Reap the entire owned session's process group even
    # when wait() has already observed the leader's exit.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_process(
    command: list[str], *, cwd: Path, timeout_seconds: int,
    limit_mib: int | None = None, capture_json: bool = False,
) -> dict[str, Any]:
    global _ACTIVE_PROCESS
    time_exe = "/usr/bin/time"
    time_file: Path | None = None
    wrapped = command
    if Path(time_exe).is_file():
        fd, name = tempfile.mkstemp(prefix="storage-benchmark-time-")
        os.close(fd)
        time_file = Path(name)
        wrapped = [time_exe, "-f", "PEAK_RSS_KIB=%M", "-o", str(time_file), *command]
    started = time.perf_counter()
    try:
        parent_as_limit = resource.getrlimit(resource.RLIMIT_AS)
    except (AttributeError, OSError, ValueError):
        parent_as_limit = None
    cancelled = False
    stopped_before_launch = _STOP_REQUESTED
    process: subprocess.Popen[str] | None = None
    try:
        if stopped_before_launch:
            stdout, stderr, returncode, timed_out = "", "", None, False
        else:
            process = subprocess.Popen(
                wrapped, cwd=cwd, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if capture_json else subprocess.STDOUT,
                start_new_session=True, preexec_fn=limit_memory(limit_mib or 0),
            )
            _ACTIVE_PROCESS = process
            deadline = time.monotonic() + timeout_seconds
            timed_out = False
            while True:
                if _STOP_REQUESTED:
                    cancelled = True
                    terminate_process_group(process)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    terminate_process_group(process)
                    break
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if timed_out or cancelled:
                stdout, stderr = process.communicate()
            returncode = process.returncode
    except OSError as error:
        stdout, stderr, returncode, timed_out = "", str(error), None, False
    finally:
        if process is not None and _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None
    elapsed = time.perf_counter() - started
    peak_rss_kib = None
    timing_output = ""
    if time_file is not None:
        try:
            timing_output = time_file.read_text(errors="replace").strip()
            match = re.search(r"PEAK_RSS_KIB=(\d+)", timing_output)
            if match:
                peak_rss_kib = int(match.group(1))
        except OSError:
            pass
        finally:
            time_file.unlink(missing_ok=True)
    stdout = stdout or ""
    stderr = stderr or ""
    actions = compiler_actions(stdout) if not capture_json else None
    application_success: bool | None = None
    application_error = None
    if capture_json:
        try:
            parsed = json.loads(stdout.strip())
            if isinstance(parsed, dict):
                if isinstance(parsed.get("success"), bool):
                    application_success = parsed["success"]
                elif isinstance(parsed.get("ok"), bool):
                    application_success = parsed["ok"]
                elif str(parsed.get("status", "")).lower() in {"error", "failed", "unsupported"}:
                    application_success = False
                if parsed.get("error") not in (None, ""):
                    application_error = parsed["error"]
                    application_success = False
        except json.JSONDecodeError:
            pass
    result: dict[str, Any] = {
        "command": command, "executed_command": wrapped,
        "working_directory": str(cwd),
        "process_group_pgid": process.pid if process is not None else None,
        "elapsed_seconds": elapsed, "max_rss_kib": peak_rss_kib,
        "returncode": returncode, "timed_out": timed_out,
        "cancelled": cancelled, "stopped_before_launch": stopped_before_launch,
        "memory_limit": ({"kind": "RLIMIT_AS", "mib": limit_mib}
                         if limit_mib else None),
        "parent_rlimit_as_bytes": list(parent_as_limit) if parent_as_limit else None,
        "resource_usage": "GNU time %M when available; otherwise unavailable",
        "stdout": stdout[-12000:], "stderr": stderr[-12000:],
        "time_tool_output": timing_output,
    }
    if capture_json:
        try:
            value = json.loads(stdout.strip())
            result["json"] = value
            result["json_valid"] = isinstance(value, dict)
            if not isinstance(value, dict):
                result["json_error"] = "executable output was JSON but not an object"
        except json.JSONDecodeError as error:
            result["json_valid"] = False
            result["json_error"] = f"{error.msg} at line {error.lineno} column {error.colno}"
    result["success"] = returncode == 0 and not timed_out and not cancelled \
        and not stopped_before_launch
    if capture_json:
        result["success"] = result["success"] and result.get("json_valid", False)
        if application_success is False:
            result["success"] = False
        if application_error is not None:
            result["application_error"] = application_error
    else:
        result["compiler_actions"] = actions
    return result


def add_import_probes(source: Path) -> dict[str, str]:
    probe_root = source / "benchmarks" / "storage_imports"
    probe_root.mkdir(parents=True, exist_ok=True)
    for name, module in IMPORTS.items():
        (probe_root / f"{name}.cpp").write_text(
            f"import {module};\nint main() {{ return 0; }}\n")
    addition = "\n# Disposable storage build benchmark import probes.\n"
    for name in IMPORTS:
        target = f"storage_import_{name}"
        addition += (f"add_executable({target} benchmarks/storage_imports/{name}.cpp)\n"
                     f"target_link_libraries({target} PRIVATE anndata::anndata)\n")
    cmake = source / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + addition)
    return {name: sha256_file(probe_root / f"{name}.cpp") for name in IMPORTS}


def build_command(build: Path, targets: list[str], jobs: int) -> list[str]:
    return ["cmake", "--build", str(build), "--target", *targets,
            "--parallel", str(jobs), "--verbose"]


def configure_command(
    source: Path, build: Path, *, compiler: str, configuration: str,
    hdf5: bool, zarr: bool, dependency_prefix: Path,
) -> list[str]:
    return [
        "cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
        f"-DCMAKE_CXX_COMPILER={compiler}", f"-DCMAKE_BUILD_TYPE={configuration}",
        "-DANNDATA_BUILD_TESTS=OFF", "-DANNDATA_BUILD_EXAMPLES=OFF",
        "-DANNDATA_BUILD_BENCHMARKS=ON", "-DANNDATA_ENABLE_INTEROP_TESTS=OFF",
        f"-DANNDATA_ENABLE_HDF5={'ON' if hdf5 else 'OFF'}",
        f"-DANNDATA_ENABLE_ZARR={'ON' if zarr else 'OFF'}",
        f"-DANNDATA_DEPENDENCY_PREFIX={dependency_prefix}",
    ]


def summarize_seconds(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [record["elapsed_seconds"] for record in records
              if record.get("success") and isinstance(record.get("elapsed_seconds"), (int, float))]
    if not values:
        return {"successful_trials": 0, "samples_seconds": []}
    return {"successful_trials": len(values), "samples_seconds": values,
            "median_seconds": statistics.median(values), "min_seconds": min(values),
            "max_seconds": max(values)}


def run_build_trial(
    frozen_source: Path, work_root: Path, *, name: str, trial: int,
    hdf5: bool, zarr: bool, compiler: str, configuration: str,
    dependency_prefix: Path, jobs: int,
) -> dict[str, Any]:
    source = work_root / f"source-{name}-{trial}"
    shutil.copytree(frozen_source, source, symlinks=True)
    build = work_root / f"build-{name}-{trial}"
    configure = configure_command(
        source, build, compiler=compiler, configuration=configuration,
        hdf5=hdf5, zarr=zarr, dependency_prefix=dependency_prefix,
    )
    conf = run_process(configure, cwd=source, timeout_seconds=900)
    trial_record: dict[str, Any] = {
        "configuration": name, "hdf5_enabled": hdf5, "zarr_enabled": zarr,
        "trial": trial, "configure": conf, "errors": [],
        "configure_command": configure,
    }
    trial_record["configure"]["cmake_cache_values"] = cache_values(build / "CMakeCache.txt")
    if not conf["success"]:
        if conf.get("cancelled") or conf.get("stopped_before_launch"):
            trial_record["cancelled"] = True
        else:
            trial_record["errors"].append("configure failed; build stages skipped")
        trial_record["temporary_source_restored"] = tree_manifest(source) == tree_manifest(frozen_source)
        return trial_record
    cold = run_process(build_command(build, ["anndata"], jobs), cwd=source,
                       timeout_seconds=1800)
    trial_record["cold_library_build"] = cold
    if not cold["success"]:
        if cold.get("cancelled") or cold.get("stopped_before_launch"):
            trial_record["cancelled"] = True
        else:
            trial_record["errors"].append("cold library build failed; import and edit stages skipped")
        trial_record["temporary_source_restored"] = tree_manifest(source) == tree_manifest(frozen_source)
        return trial_record

    import_builds: dict[str, dict[str, Any]] = {}
    successful_targets: list[str] = []
    for key in ("core", "backed", "facade", "h5ad", "zarr"):
        if (key == "h5ad" and not hdf5) or (key == "zarr" and not zarr):
            continue
        target = f"storage_import_{key}"
        row = run_process(build_command(build, [target], jobs), cwd=source,
                          timeout_seconds=1800)
        import_builds[key] = row
        if row["success"]:
            successful_targets.append(target)
        else:
            if row.get("cancelled") or row.get("stopped_before_launch"):
                trial_record["cancelled"] = True
                break
            row.setdefault("stage", f"{key} consumer import")
            trial_record["errors"].append(f"{key} consumer import build failed")

    if _STOP_REQUESTED:
        trial_record.update({
            "first_import_builds": import_builds,
            "cancelled": True,
            "storage_driver_build_tree": str(build),
            "temporary_source_restored": tree_manifest(source) == tree_manifest(frozen_source),
        })
        return trial_record

    before_private = build_artifacts(build)
    private_file = source / "src" / "backed_array.cpp"
    if not private_file.is_file():
        raise RuntimeError("private storage implementation probe src/backed_array.cpp is missing")
    interface_file = source / "src" / "anndata.arrays.cppm"
    if not interface_file.is_file():
        raise RuntimeError("public interface probe source src/anndata.arrays.cppm is missing")
    private_bytes = private_file.read_bytes()
    interface_bytes = interface_file.read_bytes()
    private_file.write_bytes(private_bytes + (
        f"\nnamespace {{ [[maybe_unused]] constexpr int storage_private_revision_{trial} = {trial}; }}\n"
    ).encode())
    private = None
    interface = None
    after_private = before_private
    after_interface = None
    try:
        private_targets = ["anndata", *successful_targets]
        private = run_process(build_command(build, private_targets, jobs),
                              cwd=source, timeout_seconds=1800)
        after_private = build_artifacts(build)
        if not private["success"]:
            if private.get("cancelled") or private.get("stopped_before_launch"):
                trial_record["cancelled"] = True
            else:
                trial_record["errors"].append(
                    "private implementation edit build failed; interface stage skipped")
        else:
            marker = b"} // namespace anndata"
            location = interface_bytes.rfind(marker)
            if location < 0:
                raise RuntimeError("could not locate exported anndata namespace boundary for interface probe")
            edit = (f"inline constexpr int storage_interface_revision_{trial} = {trial};\n\n").encode()
            interface_file.write_bytes(interface_bytes[:location] + edit + interface_bytes[location:])
            # Keep the private edit installed during this build. This leaves its
            # object already current, so the interface measurement does not
            # include the timestamp-triggered rebuild needed to restore it.
            interface = run_process(build_command(build, successful_targets, jobs),
                                    cwd=source, timeout_seconds=1800)
            after_interface = build_artifacts(build)
            if not interface["success"]:
                if interface.get("cancelled") or interface.get("stopped_before_launch"):
                    trial_record["cancelled"] = True
                else:
                    trial_record["errors"].append("public interface edit build failed")
    finally:
        interface_file.write_bytes(interface_bytes)
        private_file.write_bytes(private_bytes)

    for item in (cold, private, interface, *import_builds.values()):
        if item is not None:
            if "compiler_actions" not in item:
                item["compiler_actions"] = compiler_actions(item.get("stdout", ""))

    before_interface = after_private
    if after_interface is None:
        before_interface = after_private
        after_interface = after_private
    trial_record.update({
        "first_import_builds": import_builds,
        "private_edit_build": private,
        "interface_edit_build": interface,
        "artifact_states": {
            "before_private_edit": before_private,
            "after_private_edit": after_private,
            "before_interface_edit": before_interface,
            "after_interface_edit": after_interface,
        },
        "artifact_changes": {
            "private_edit": {
                "bmis": artifact_changes(before_private["bmis"], after_private["bmis"]),
                "consumer_objects": artifact_changes(before_private["consumer_objects"],
                                                       after_private["consumer_objects"]),
                "library_objects": artifact_changes(before_private["library_objects"],
                                                     after_private["library_objects"]),
            },
            "interface_edit": {
                "bmis": artifact_changes(before_interface["bmis"], after_interface["bmis"]),
                "consumer_objects": artifact_changes(before_interface["consumer_objects"],
                                                       after_interface["consumer_objects"]),
                "library_objects": artifact_changes(before_interface["library_objects"],
                                                     after_interface["library_objects"]),
            },
        },
        "storage_driver_build_tree": str(build),
        "temporary_source_restored": tree_manifest(source) == tree_manifest(frozen_source),
    })
    if private and private["success"] and interface and not interface["success"]:
        trial_record["artifact_changes"]["interface_edit"]["measurement_valid"] = False
    elif interface:
        trial_record["artifact_changes"]["interface_edit"]["measurement_valid"] = interface["success"]
    else:
        trial_record["artifact_changes"]["interface_edit"]["measurement_valid"] = False
        trial_record["artifact_changes"]["interface_edit"]["skipped"] = True
    return trial_record


class PollPolicy(asyncio.DefaultEventLoopPolicy):
    """Use short poll intervals for Zarr's synchronous async bridge."""

    def new_event_loop(self):
        loop = super().new_event_loop()

        def tick() -> None:
            loop.call_later(0.01, tick)

        loop.call_later(0.01, tick)
        return loop


def _write_large_axes_h5ad(path: Path, size: int, chunk: int) -> None:
    import h5py
    import numpy as np

    with h5py.File(path, "w") as store:
        store.attrs["encoding-type"] = "anndata"
        store.attrs["encoding-version"] = "0.1.0"
        x = store.create_dataset("X", shape=(size, size), dtype="<f4",
                                 chunks=(min(chunk, size), min(chunk, size)),
                                 fillvalue=0.0)
        x.attrs["encoding-type"] = "array"
        x.attrs["encoding-version"] = "0.2.0"
        for axis, prefix in (("obs", "cell"), ("var", "gene")):
            group = store.create_group(axis)
            group.attrs["encoding-type"] = "dataframe"
            group.attrs["encoding-version"] = "0.2.0"
            group.attrs["_index"] = "_index"
            group.attrs["column-order"] = np.asarray([], dtype=h5py.string_dtype("utf-8"))
            index = group.create_dataset(
                "_index", shape=(size,), dtype=h5py.string_dtype("utf-8"),
                chunks=(min(chunk, size),),
            )
            index.attrs["encoding-type"] = "string-array"
            index.attrs["encoding-version"] = "0.2.0"
            for start in range(0, size, chunk):
                stop = min(size, start + chunk)
                index[start:stop] = np.asarray(
                    [f"{prefix}{i:09d}" for i in range(start, stop)], dtype=object)


def _write_large_axes_zarr(path: Path, size: int, chunk: int, zarr_format: int) -> None:
    import numpy as np
    import zarr
    from numcodecs import VLenUTF8
    from zarr.core.dtype import VariableLengthUTF8

    root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    root.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
    x = root.create_array(
        "X", shape=(size, size), chunks=(min(chunk, size), min(chunk, size)),
        dtype=np.float32, fill_value=0.0, write_data=False,
    )
    x.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    for axis, prefix in (("obs", "cell"), ("var", "gene")):
        group = root.create_group(axis)
        group.attrs.update({
            "encoding-type": "dataframe", "encoding-version": "0.2.0",
            "_index": "_index", "column-order": [],
        })
        if zarr_format == 2:
            options: dict[str, Any] = {
                "dtype": VariableLengthUTF8(), "fill_value": "",
                "filters": [VLenUTF8()],
            }
        else:
            options = {"dtype": np.dtypes.StringDType(), "fill_value": ""}
        index = group.create_array(
            "_index", shape=(size,), chunks=(min(chunk, size),), write_data=False,
            **options,
        )
        index.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})
        for start in range(0, size, chunk):
            stop = min(size, start + chunk)
            index[start:stop] = np.asarray(
                [f"{prefix}{i:09d}" for i in range(start, stop)], dtype=object)


def generate_fixtures(
    directory: Path, *, size: int, include_huge: bool,
    huge_size: int, chunk: int,
) -> list[dict[str, Any]]:
    """Generate small dense/CSR/CSC fixtures and optional unallocated huge X."""
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy import sparse
    except ImportError as error:
        raise RuntimeError(
            "fixture generation requires the active Python to import anndata, "
            "numpy, pandas, scipy, h5py, and zarr"
        ) from error

    directory.mkdir(parents=True, exist_ok=True)
    expected = [directory / f"{layout}.{format_name}"
                for format_name in FORMATS for layout in ("dense", "csr", "csc")]
    if include_huge:
        expected.extend(directory / f"huge-{format_name}" for format_name in FORMATS)
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing fixture paths: " + ", ".join(existing))
    stage = Path(tempfile.mkdtemp(prefix=".storage-fixture-staging-", dir=directory))
    records: list[dict[str, Any]] = []
    old_policy = asyncio.get_event_loop_policy()
    try:
        rng = np.random.default_rng(20261103)
        obs = pd.DataFrame(index=[f"cell{i:06d}" for i in range(size)])
        var = pd.DataFrame(index=[f"gene{i:06d}" for i in range(size)])
        dense = rng.random((size, size), dtype=np.float32)
        sparse_csr = sparse.random(
            size, size, density=min(0.025, 8 / max(1, size)), format="csr",
            random_state=20261103, dtype=np.float32,
            data_rvs=lambda count: rng.random(count, dtype=np.float32),
        )
        matrices = {"dense": dense, "csr": sparse_csr, "csc": sparse_csr.tocsc()}
        asyncio.set_event_loop_policy(PollPolicy())
        try:
            for format_name in FORMATS:
                for layout, matrix in matrices.items():
                    path = stage / f"{layout}.{format_name}"
                    adata = ad.AnnData(X=matrix, obs=obs.copy(), var=var.copy())
                    if format_name == "h5ad":
                        adata.write_h5ad(path, compression=None)
                    else:
                        with ad.settings.override(zarr_write_format=2 if format_name == "zarr2" else 3):
                            adata.write_zarr(path, convert_strings_to_categoricals=False,
                                             consolidate_metadata=False)
                    records.append({
                        "format": format_name, "layout": layout, "path": str(path.resolve()),
                        "generated": True, "kind": layout, "shape": [size, size],
                        "dtype": "float32", "chunk_geometry": None,
                        "shard_geometry": None, "compression": "unknown until metadata inspection",
                        "default_actions": list(DEFAULT_GENERATED_ACTIONS[layout]),
                    })
            if include_huge:
                for format_name in FORMATS:
                    path = stage / f"huge-{format_name}"
                    if format_name == "h5ad":
                        _write_large_axes_h5ad(path, huge_size, chunk)
                    else:
                        _write_large_axes_zarr(path, huge_size, chunk,
                                              2 if format_name == "zarr2" else 3)
                    records.append({
                        "format": format_name, "layout": "dense", "path": str(path.resolve()),
                        "generated": True, "kind": "huge", "shape": [huge_size, huge_size],
                        "dtype": "float32", "chunk_geometry": [min(chunk, huge_size)] * 2,
                        "shard_geometry": None, "compression": "unknown until metadata inspection",
                        "fill_value": 0, "index_chunk_geometry": [min(chunk, huge_size)],
                        "default_actions": list(DEFAULT_GENERATED_ACTIONS["huge"]),
                    })
        finally:
            asyncio.set_event_loop_policy(old_policy)

        for record in records:
            metadata = inspect_fixture_metadata(record)
            if metadata.get("metadata_error"):
                raise RuntimeError(
                    f"generated {record['format']} fixture metadata could not be inspected: "
                    f"{metadata['metadata_error']}"
                )
            record["actual_metadata"] = metadata
            record["shape"] = metadata.get("shape", record.get("shape"))
            record["dtype"] = metadata.get("dtype", record.get("dtype"))
            record["layout"] = metadata.get("layout", record.get("layout"))
            record["chunk_geometry"] = metadata.get("chunk_geometry")
            record["shard_geometry"] = metadata.get("shard_geometry")
            record["index_chunk_geometry"] = {
                axis: geometry.get("chunk_geometry")
                for axis, geometry in metadata.get("index_geometry", {}).items()
            }
            record["index_shard_geometry"] = {
                axis: geometry.get("shard_geometry")
                for axis, geometry in metadata.get("index_geometry", {}).items()
            }
            record["compression"] = metadata.get("compression", "unknown")
            record["physical"] = metadata.get("physical")
        promoted: list[Path] = []
        try:
            targets = [(Path(record["path"]), directory / Path(record["path"]).name)
                       for record in records]
            conflicts = [str(target) for _, target in targets if target.exists()]
            if conflicts:
                raise FileExistsError("refusing to overwrite fixture paths: " + ", ".join(conflicts))
            for staged_path, final_path in targets:
                staged_path.rename(final_path)
                promoted.append(final_path)
            for record, (_, final_path) in zip(records, targets, strict=True):
                record["path"] = str(final_path.resolve())
                record["actual_metadata"]["path"] = str(final_path.resolve())
                record["actual_metadata"]["physical"]["path"] = str(final_path.resolve())
        except Exception:
            for path in promoted:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            raise
        return records
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def fixture_evidence(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    total_bytes = 0
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for item in paths:
        stat = item.stat()
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        manifest[relative] = {"bytes": stat.st_size, "sha256": sha256_file(item)}
        total_bytes += stat.st_size
    return {"path": str(path.resolve()), "kind": "file" if path.is_file() else "directory",
            "physical_bytes": total_bytes, "sha256": manifest_digest(manifest),
            "file_hashes": manifest}


def parse_fixture(value: str) -> dict[str, Any]:
    format_name, separator, raw_path = value.partition(":")
    path = Path(raw_path).expanduser().resolve()
    if not separator or format_name not in FORMATS:
        raise argparse.ArgumentTypeError("fixture must be FORMAT:PATH where FORMAT is h5ad, zarr2, or zarr3")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"fixture path does not exist: {path}")
    return {"format": format_name, "path": str(path), "generated": False,
            "kind": "custom", "default_actions": ["open"]}


def parse_readonly_sparse_fixture(value: str) -> dict[str, Any]:
    record = parse_fixture(value)
    record.update({"kind": "readonly_sparse", "readonly": True,
                   "default_actions": ["open", "sparse-row", "sparse-column"]})
    return record


def zarr_array_geometry(metadata: dict[str, Any]) -> dict[str, Any]:
    if "chunk_geometry" in metadata:
        return {
            "shape": metadata.get("shape"), "dtype": metadata.get("dtype"),
            "chunk_geometry": metadata.get("chunk_geometry"),
            "shard_geometry": metadata.get("shard_geometry"),
            "compression": metadata.get("compression"),
            "filters": metadata.get("filters"),
        }
    if "chunks" in metadata:
        return {
            "shape": metadata.get("shape"), "dtype": metadata.get("dtype"),
            "chunk_geometry": metadata.get("chunks"), "shard_geometry": None,
            "compression": metadata.get("compressor"),
            "filters": metadata.get("filters"),
        }
    grid = metadata.get("chunk_grid", {}).get("configuration", {})
    codecs = metadata.get("codecs") or []
    shard = next((codec.get("configuration", {}) for codec in codecs
                  if codec.get("name") in {"sharding_indexed", "sharding"}), None)
    return {
        "shape": metadata.get("shape"), "dtype": metadata.get("data_type"),
        "chunk_geometry": shard.get("chunk_shape") if shard else grid.get("chunk_shape"),
        "shard_geometry": grid.get("chunk_shape") if shard else None,
        "compression": codecs,
    }


def inspect_fixture_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Read storage metadata only; this never indexes an X payload."""
    path = Path(record["path"])
    result: dict[str, Any] = {"format": record["format"], "path": str(path.resolve())}
    if record["format"] == "h5ad":
        try:
            import h5py

            def h5_array_metadata(array: Any) -> dict[str, Any]:
                return {
                    "shape": list(array.shape) if array.shape is not None else None,
                    "dtype": str(array.dtype),
                    "chunk_geometry": list(array.chunks) if array.chunks is not None else None,
                    "compression": array.compression,
                    "compression_options": array.compression_opts,
                    "shuffle": array.shuffle, "fletcher32": array.fletcher32,
                }

            with h5py.File(path, "r") as store:
                axis_storage = {}
                for axis in ("obs", "var"):
                    if axis in store and "_index" in store[axis]:
                        axis_storage[axis] = h5_array_metadata(store[axis]["_index"])
                x = store["X"]
                code = x.attrs.get("encoding-type", "array")
                if isinstance(code, bytes):
                    code = code.decode("utf-8", "replace")
                if isinstance(x, h5py.Dataset):
                    physical = h5_array_metadata(x)
                    result.update({"layout": code, "shape": physical["shape"],
                                   "dtype": physical["dtype"],
                                   "chunk_geometry": physical["chunk_geometry"],
                                   "shard_geometry": None,
                                   "compression": physical["compression"],
                                   "array_storage": physical,
                                   "index_storage": axis_storage})
                else:
                    shape = x.attrs.get("shape", [])
                    components = {name: h5_array_metadata(x[name]) for name in x
                                  if isinstance(x[name], h5py.Dataset)}
                    values = components.get("data", {})
                    result.update({"layout": code, "shape": [int(value) for value in shape],
                                   "dtype": values.get("dtype"), "chunk_geometry": None,
                                   "shard_geometry": None, "compression": None,
                                   "component_arrays": components,
                                   "index_storage": axis_storage})
        except Exception as error:
            result["metadata_error"] = f"{type(error).__name__}: {error}"
    else:
        try:
            root = path
            root_v2 = root / ".zgroup"
            root_v3 = root / "zarr.json"
            if root_v2.is_file():
                group_meta = json.loads(root_v2.read_text())
                root_attrs_path = root / ".zattrs"
                root_attrs = json.loads(root_attrs_path.read_text()) if root_attrs_path.exists() else {}
                axis_storage = {}
                for axis in ("obs", "var"):
                    axis_meta_path = root / axis / "_index" / ".zarray"
                    if axis_meta_path.is_file():
                        axis_storage[axis] = json.loads(axis_meta_path.read_text())
                xattrs_path = root / "X" / ".zattrs"
                xattrs = json.loads(xattrs_path.read_text()) if xattrs_path.exists() else {}
                xmeta_path = root / "X" / ".zarray"
                xmeta = json.loads(xmeta_path.read_text()) if xmeta_path.exists() else {}
                layout = xattrs.get("encoding-type", "array")
                if xmeta:
                    result.update({"shape": xmeta.get("shape"), "dtype": xmeta.get("dtype"),
                                   "chunk_geometry": xmeta.get("chunks"),
                                   "shard_geometry": None,
                                   "compression": xmeta.get("compressor"),
                                   "filters": xmeta.get("filters"),
                                   "array_storage": xmeta,
                                   "index_storage": axis_storage})
                elif (root / "X" / "data" / ".zarray").exists():
                    sparse_meta = json.loads((root / "X" / "data" / ".zarray").read_text())
                    components = {}
                    for component in ("data", "indices", "indptr"):
                        metadata_path = root / "X" / component / ".zarray"
                        if metadata_path.is_file():
                            components[component] = json.loads(metadata_path.read_text())
                    result.update({"shape": xattrs.get("shape"), "dtype": sparse_meta.get("dtype"),
                                   "chunk_geometry": None,
                                   "shard_geometry": None,
                                   "compression": sparse_meta.get("compressor"),
                                   "filters": sparse_meta.get("filters"),
                                   "component_arrays": components,
                                   "index_storage": axis_storage})
                result.update({"zarr_format": group_meta.get("zarr_format"), "layout": layout,
                               "root_attributes": root_attrs})
            elif root_v3.is_file():
                root_meta = json.loads(root_v3.read_text())
                x_meta = json.loads((root / "X" / "zarr.json").read_text())
                axis_storage = {}
                for axis in ("obs", "var"):
                    axis_meta_path = root / axis / "_index" / "zarr.json"
                    if axis_meta_path.is_file():
                        axis_storage[axis] = json.loads(axis_meta_path.read_text())
                attrs = x_meta.get("attributes", {})
                array_meta = x_meta
                shape = x_meta.get("shape") or attrs.get("shape")
                components = {}
                if x_meta.get("node_type") == "group":
                    for component in ("data", "indices", "indptr"):
                        data_path = root / "X" / component / "zarr.json"
                        if data_path.is_file():
                            components[component] = json.loads(data_path.read_text())
                    if "data" in components:
                        array_meta = components["data"]
                grid = array_meta.get("chunk_grid", {}).get("configuration", {})
                codecs = array_meta.get("codecs")
                shard_codecs = [codec for codec in (codecs or [])
                                if codec.get("name") in {"sharding_indexed", "sharding"}]
                shard_config = shard_codecs[0].get("configuration", {}) if shard_codecs else {}
                chunk_geometry = shard_config.get("chunk_shape") if shard_codecs else grid.get("chunk_shape")
                result.update({"zarr_format": root_meta.get("zarr_format"),
                               "layout": attrs.get("encoding-type", "array"),
                               "shape": shape, "dtype": array_meta.get("data_type"),
                               "chunk_geometry": chunk_geometry,
                               "shard_geometry": grid.get("chunk_shape") if shard_codecs else None,
                               "chunk_grid_metadata": array_meta.get("chunk_grid"),
                               "compression": codecs,
                               "component_arrays": components or None,
                               "index_storage": axis_storage})
            else:
                result["metadata_error"] = "no Zarr v2 or v3 root metadata found"
        except Exception as error:
            result["metadata_error"] = f"{type(error).__name__}: {error}"
    result["index_geometry"] = {
        axis: (value if "chunk_geometry" in value else zarr_array_geometry(value))
        for axis, value in result.get("index_storage", {}).items()
    }
    result["physical"] = fixture_evidence(path)
    return result


def copy_fixture(record: dict[str, Any], destination: Path) -> Path:
    source = Path(record["path"])
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination


def executable_evidence(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()), "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path),
        "dynamic_dependencies": tool_output(["ldd", str(path.resolve())]),
        "dynamic_loader_version": tool_output(["ldd", "--version"]),
    }


def storage_trial(
    executable: Path, fixture: dict[str, Any], action: str, trial: int,
    work_root: Path, *, limit_mib: int, timeout_seconds: int,
) -> dict[str, Any]:
    original = Path(fixture["path"])
    identity = hashlib.sha256(str(original.resolve()).encode()).hexdigest()[:12]
    run_root = work_root / (
        f"storage-{fixture['format']}-{fixture.get('runner_id', identity)}-"
        f"{fixture.get('kind', 'custom')}-{action}-{trial}"
    )
    run_root.mkdir(parents=True)
    measured_path = original
    before_copy = None
    if action in MUTATING_ACTIONS:
        measured_path = run_root / original.name
        copy_fixture(fixture, measured_path)
        before_copy = fixture_evidence(measured_path)
    command = [str(executable), action, fixture["format"], str(measured_path)]
    destination = None
    if action == "export":
        suffix = ".h5ad" if fixture["format"] == "h5ad" else ".zarr"
        destination = run_root / f"export{suffix}"
        command.append(str(destination))
    process = run_process(command, cwd=run_root, timeout_seconds=timeout_seconds,
                          limit_mib=limit_mib, capture_json=True)
    row: dict[str, Any] = {
        "format": fixture["format"], "kind": fixture.get("kind", "custom"),
        "layout": fixture.get("layout"), "action": action, "trial": trial,
        "fixture_path": str(original), "working_fixture_path": str(measured_path),
        "fixture_sha256": fixture.get("actual_metadata", {}).get("physical", {}).get("sha256"),
        "fixture_physical_bytes": fixture.get("actual_metadata", {}).get("physical", {}).get("physical_bytes"),
        "destination": str(destination) if destination else None,
        "fixture_before_mutation": before_copy,
        "command_result": process,
    }
    if before_copy is not None:
        row["fixture_after_mutation"] = fixture_evidence(measured_path)
    elif action == "export" and destination and destination.exists():
        row["exported_fixture"] = fixture_evidence(destination)
    return row


def measure_storage(
    executable: Path, fixtures: list[dict[str, Any]], actions: list[str] | None,
    work_root: Path, *, runs: int, limit_mib: int, timeout_seconds: int,
    repeat_after_timeout: bool = False, stop_file: str | None = None,
    parent_report: dict[str, Any] | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    fixture_rows = []
    scheduled_cases: list[tuple[dict[str, Any], str, int]] = []
    for fixture_number, record in enumerate(fixtures, start=1):
        path_id = hashlib.sha256(str(Path(record["path"]).resolve()).encode()).hexdigest()[:10]
        record["runner_id"] = f"{fixture_number:02d}-{path_id}"
        fixture_meta = record.get("actual_metadata") or inspect_fixture_metadata(record)
        record["actual_metadata"] = fixture_meta
        fixture_rows.append({**record, "metadata": fixture_meta})
        selected_actions = list(actions if actions else record.get("default_actions", ["open"]))
        for action in selected_actions:
            for trial in range(1, runs + 1):
                scheduled_cases.append((record, action, trial))

    result: dict[str, Any] = {
        "executable": str(executable.resolve()), "fixtures": fixture_rows,
        "executable_evidence": executable_evidence(executable),
        "rlimit_as_mib": limit_mib, "timeout_seconds": timeout_seconds,
        "repeat_after_timeout": repeat_after_timeout,
        "cache_policy": (
            "no OS cache clearing; exact fixture metadata/hash collection occurs before timed calls; "
            "first read may be warm, and mutation trials copy fixtures before timing"
        ),
        "observations": observations, "summary": [], "unrun_cases": [],
        "complete": not bool(scheduled_cases),
    }
    if parent_report is not None:
        parent_report["storage"] = result

    def refresh_summary() -> None:
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for row in observations:
            grouped.setdefault((row["format"], row.get("kind", "custom"),
                                row["fixture_path"], row["action"]), []).append(row)
        summary = []
        for (format_name, kind, fixture_path, action), group in sorted(grouped.items()):
            completed = [row for row in group if row.get("status", "completed") == "completed"]
            successful = [row for row in completed if row.get("success")]
            successful_commands = [row["command_result"] for row in successful
                                   if "command_result" in row]
            rss_values = [row["command_result"].get("max_rss_kib") for row in successful
                          if row.get("command_result", {}).get("max_rss_kib") is not None]
            operation_stages = {}
            for stage in ("open_seconds", "operation_seconds", "prepare_seconds",
                          "materialized_read_seconds"):
                stage_records = [{"success": True, "elapsed_seconds": row.get("json", {}).get(stage)}
                                 for row in successful_commands]
                operation_stages[stage] = summarize_seconds(stage_records)
            summary.append({
                "format": format_name, "kind": kind, "fixture_path": fixture_path,
                "action": action, "trials": len(completed),
                "successes": len(successful),
                "failures": sum(not row.get("success", False) for row in completed),
                "cancelled": sum(row.get("status") == "cancelled" for row in group),
                "skipped": sum(row.get("status") == "skipped" for row in group),
                "elapsed": summarize_seconds(successful_commands),
                "operation_stages": operation_stages,
                "max_rss_kib": ({
                    "samples": rss_values, "median": statistics.median(rss_values),
                    "min": min(rss_values), "max": max(rss_values),
                } if rss_values else None),
            })
        result["summary"] = summary

    timeout_blocked: set[tuple[str, str, str]] = set()
    for case_index, (record, action, trial) in enumerate(scheduled_cases):
        reason = stop_reason(stop_file)
        if reason:
            result["unrun_cases"] = [
                {"format": pending["format"], "kind": pending.get("kind", "custom"),
                 "fixture_path": pending["path"], "action": pending_action,
                 "trial": pending_trial, "reason": reason}
                for pending, pending_action, pending_trial in scheduled_cases[case_index:]
            ]
            result["complete"] = False
            result["stop_reason"] = reason
            refresh_summary()
            if checkpoint:
                checkpoint()
            break

        fixture_path = record["path"]
        key = (record["format"], fixture_path, action)
        print(f"storage: {record['format']} {record.get('kind', 'custom')} "
              f"{action} trial {trial}/{runs}", flush=True)
        if key in timeout_blocked and not repeat_after_timeout:
            row = {
                "format": record["format"], "kind": record.get("kind", "custom"),
                "action": action, "trial": trial, "fixture_path": fixture_path,
                "status": "skipped", "skipped": True,
                "skip_reason": "previous identical fixture/action trial timed out",
            }
        else:
            try:
                row = storage_trial(executable, record, action, trial, work_root,
                                    limit_mib=limit_mib,
                                    timeout_seconds=timeout_seconds)
            except Exception as error:
                row = {
                    "format": record["format"], "kind": record.get("kind", "custom"),
                    "action": action, "trial": trial,
                    "fixture_path": record["path"],
                    "runner_error": f"{type(error).__name__}: {error}",
                }
            command_result = row.get("command_result", {})
            row["success"] = row.get("success", command_result.get("success", False))
            row["status"] = "cancelled" if command_result.get("cancelled") else "completed"
            if command_result.get("timed_out"):
                row["timed_out"] = True
                if not repeat_after_timeout:
                    timeout_blocked.add(key)
        observations.append(row)
        print(f"storage complete: status={row['status']} "
              f"success={row.get('success')}", flush=True)

        if row["status"] == "cancelled":
            reason = stop_reason(stop_file) or "signal cancellation"
            result["unrun_cases"] = [
                {"format": pending["format"], "kind": pending.get("kind", "custom"),
                 "fixture_path": pending["path"], "action": pending_action,
                 "trial": pending_trial, "reason": reason}
                for pending, pending_action, pending_trial in scheduled_cases[case_index + 1:]
            ]
            result["complete"] = False
            result["stop_reason"] = reason
        refresh_summary()
        if checkpoint:
            checkpoint()
        if row["status"] == "cancelled":
            break
    result["complete"] = (
        len(observations) == len(scheduled_cases)
        and not any(row.get("status") == "cancelled" for row in observations)
    )
    if stop_reason(stop_file):
        result.setdefault("stop_reason", stop_reason(stop_file))
    if parent_report is not None:
        parent_report["storage"] = result
    return result


def benchmark(
    args: argparse.Namespace, *, checkpoint_path: Path | None = None,
    report_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(args.source).resolve()
    dependency_prefix = (Path(args.dependency_prefix).expanduser().resolve()
                         if args.dependency_prefix else source / "build" / "dependencies" / "usr")
    temp_root = Path(tempfile.mkdtemp(prefix="anndata-storage-benchmark-"))
    args._work_root = str(temp_root)
    frozen = temp_root / "frozen-source"
    source_hashes = snapshot_source(source, frozen)
    probe_hashes = add_import_probes(frozen)
    frozen_hashes = tree_manifest(frozen)
    print(f"source snapshot complete: {manifest_digest(source_hashes)}", flush=True)
    compiler = shutil.which(args.compiler) or args.compiler
    compiler_match = re.search(r"clang\+\+-(\d+)", compiler)
    scanner = f"clang-scan-deps-{compiler_match.group(1)}" if compiler_match else "clang-scan-deps"
    environment = {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_evidence(), "relevant_environment": relevant_environment(),
        "tools": {
            "compiler_command": compiler,
            "compiler": tool_output([compiler, "--version"]),
            "module_dependency_scanner": tool_output([scanner, "--version"])
            or tool_output(["clang-scan-deps", "--version"]),
            "cmake": tool_output(["cmake", "--version"]),
            "ninja": tool_output(["ninja", "--version"]),
            "gnu_time": tool_output(["/usr/bin/time", "--version"]),
            "pkg_config": tool_output(["pkg-config", "--version"]),
            "hdf5_pkg_config": tool_output(["pkg-config", "--modversion", "hdf5"]),
        },
        "python_package_versions": python_package_versions(),
        "configuration": args.configuration, "generator": "Ninja",
        "parallel_jobs": args.jobs,
        "cache_policy": "fresh CMake/Ninja build trees; operating-system page cache not cleared",
        "dependency_prefix": str(dependency_prefix),
        "dependency_prefix_exists": dependency_prefix.exists(),
        "repetitions": args.runs,
    }
    report: dict[str, Any] = {
        "schema": "anndata.storage-benchmark.v1",
        "source_root": str(source),
        "source_snapshot": {
            "git_head": tool_output(["git", "-C", str(source), "rev-parse", "HEAD"]),
            "git_status_porcelain": tool_output(["git", "-C", str(source), "status", "--short"]),
            "excluded_directory_names": sorted(EXCLUDED_DIRS),
            "source_file_hashes": source_hashes,
            "source_manifest_sha256": manifest_digest(source_hashes),
            "instrumentation_probe_hashes": probe_hashes,
            "frozen_tree_hashes": frozen_hashes,
            "frozen_tree_sha256": manifest_digest(frozen_hashes),
            "working_copy": "temporary snapshot; private and interface edits happen only in per-trial copies",
        },
        "environment": environment,
        "storage_probe_policy": {
            "timeout_seconds": args.timeout_seconds,
            "limit_mib": args.limit_mib,
            "repeat_after_timeout": args.repeat_after_timeout,
            "stop_file": (str(Path(args.stop_file).expanduser().resolve())
                          if args.stop_file else None),
        },
        "module_builds": {"trials": [], "summary_seconds": {}},
        "storage": None,
        "temporary_work_root": str(temp_root),
        "temporary_work_removed": not args.keep_work,
    }
    if args.storage_executable:
        initial_executable = Path(args.storage_executable).expanduser().resolve()
        report["storage_executable"] = (
            executable_evidence(initial_executable) if initial_executable.is_file()
            else {"path": str(initial_executable), "error": "file does not exist"}
        )
    elif args.build_storage_executable:
        report["storage_executable"] = {
            "planned": True, "target": "storage_benchmark",
            "build_tree": str(temp_root / "storage-driver-build"),
        }
    if report_state is not None:
        report_state["report"] = report

    def save_checkpoint() -> None:
        if checkpoint_path is not None:
            write_checkpoint(checkpoint_path, report)

    save_checkpoint()
    if args.generate_fixtures:
        fixtures = []
        try:
            print("generating storage fixtures", flush=True)
            fixture_root = Path(args.generate_fixtures).expanduser().resolve()
            fixtures.extend(generate_fixtures(
                fixture_root, size=args.fixture_size,
                include_huge=args.include_huge_fixtures, huge_size=args.huge_size,
                chunk=args.chunk_size,
            ))
            for size in args.comparison_fixture_size:
                fixtures.extend(generate_fixtures(
                    fixture_root / f"comparison-{size}", size=size,
                    include_huge=False, huge_size=args.huge_size, chunk=args.chunk_size,
                ))
            print(f"fixture generation complete: {len(fixtures)} stores", flush=True)
        except Exception as error:
            report["fixture_generation_error"] = f"{type(error).__name__}: {error}"
    else:
        fixtures = []
    fixtures.extend(args.fixture or [])
    fixtures.extend(args.readonly_sparse_fixture or [])
    report["fixture_generation"] = {
        "generated": [record for record in fixtures if record.get("generated")],
        "generation_requested": bool(args.generate_fixtures),
        "huge_generation_requested": args.include_huge_fixtures,
        "requested_small_size": args.fixture_size,
        "requested_comparison_sizes": args.comparison_fixture_size,
        "requested_huge_size": args.huge_size,
        "requested_chunk_size": args.chunk_size,
    }
    report["storage_fixture_inputs"] = [record for record in fixtures if not record.get("generated")]
    save_checkpoint()

    if not args.skip_module_build:
        module = report["module_builds"]
        scheduled_builds = [
            {"configuration": name, "hdf5_enabled": hdf5, "zarr_enabled": zarr,
             "trial": trial}
            for name, (hdf5, zarr) in CONFIGURATIONS.items()
            for trial in range(1, args.runs + 1)
        ]
        module["planned_cases"] = scheduled_builds

        def refresh_module_progress() -> None:
            rows = module["trials"]
            completed = {(row["configuration"], row["trial"]) for row in rows}
            module["unrun_cases"] = [
                case for case in scheduled_builds
                if (case["configuration"], case["trial"]) not in completed
            ]
            for name in CONFIGURATIONS:
                group = [row for row in rows if row["configuration"] == name]
                module["summary_seconds"][name] = {
                    stage: summarize_seconds([row[stage] for row in group if stage in row])
                    for stage in ("cold_library_build", "private_edit_build", "interface_edit_build")
                }
                module["summary_seconds"][name]["first_import_builds"] = {
                    import_name: summarize_seconds([
                        row["first_import_builds"][import_name]
                        for row in group if import_name in row.get("first_import_builds", {})
                    ]) for import_name in IMPORTS
                    if any(import_name in row.get("first_import_builds", {}) for row in group)
                }
            module["complete"] = not module["unrun_cases"] and not any(
                row.get("cancelled") for row in rows)

        stopped = False
        for name, (hdf5, zarr) in CONFIGURATIONS.items():
            for trial in range(1, args.runs + 1):
                reason = stop_reason(args.stop_file)
                if reason:
                    module["stop_reason"] = reason
                    stopped = True
                    break
                print(f"module build: {name} trial {trial}/{args.runs}", flush=True)
                try:
                    row = run_build_trial(
                        frozen, temp_root, name=name, trial=trial, hdf5=hdf5, zarr=zarr,
                        compiler=compiler, configuration=args.configuration,
                        dependency_prefix=dependency_prefix, jobs=args.jobs,
                    )
                except Exception as error:
                    row = {
                        "configuration": name, "hdf5_enabled": hdf5, "zarr_enabled": zarr,
                        "trial": trial, "errors": [f"trial orchestration failed: {type(error).__name__}: {error}"],
                        "traceback": __import__("traceback").format_exc(),
                    }
                module["trials"].append(row)
                refresh_module_progress()
                print(f"module build complete: {name} trial {trial}, errors={len(row.get('errors', []))}", flush=True)
                save_checkpoint()
                if stop_reason(args.stop_file):
                    module["stop_reason"] = stop_reason(args.stop_file)
                    stopped = True
                    break
            if stopped:
                break
        refresh_module_progress()
        if module.get("stop_reason"):
            module["complete"] = False
            save_checkpoint()
    else:
        report["module_builds"]["skipped"] = True
        report["module_builds"]["complete"] = True
        save_checkpoint()

    executable = Path(args.storage_executable).expanduser().resolve() if args.storage_executable else None
    if args.build_storage_executable:
        reason = stop_reason(args.stop_file)
        if reason:
            report["storage_driver_build"] = {
                "skipped": True, "reason": reason,
                "executable": str(temp_root / "storage-driver-build" / "storage_benchmark"),
            }
        else:
            print("building storage workload executable", flush=True)
            source = temp_root / "frozen-source"
            driver_source = temp_root / "storage-driver-source"
            shutil.copytree(source, driver_source, symlinks=True)
            driver_build = temp_root / "storage-driver-build"
            configure = configure_command(
                driver_source, driver_build, compiler=compiler,
                configuration=args.configuration, hdf5=True, zarr=True,
                dependency_prefix=dependency_prefix,
            )
            conf = run_process(configure, cwd=driver_source, timeout_seconds=900)
            conf["cmake_cache_values"] = cache_values(driver_build / "CMakeCache.txt")
            driver_report = {"configure": conf,
                             "executable": str(driver_build / "storage_benchmark")}
            if conf["success"] and not stop_reason(args.stop_file):
                build = run_process(build_command(driver_build, ["storage_benchmark"], args.jobs),
                                    cwd=driver_source, timeout_seconds=1800)
                driver_report["build"] = build
                driver_report["artifacts"] = build_artifacts(driver_build)
                if build["success"]:
                    executable = driver_build / "storage_benchmark"
                elif not (build.get("cancelled") or build.get("stopped_before_launch")):
                    driver_report["error"] = "storage_benchmark target failed to build"
            elif conf["success"]:
                driver_report["build_skipped"] = stop_reason(args.stop_file)
            elif not (conf.get("cancelled") or conf.get("stopped_before_launch")):
                driver_report["error"] = "storage driver configuration failed"
            report["storage_driver_build"] = driver_report
        save_checkpoint()
    if args.storage_executable and args.build_storage_executable:
        raise RuntimeError("--storage-executable and --build-storage-executable are mutually exclusive")
    if executable:
        if executable.is_file():
            report["storage_executable"] = executable_evidence(executable)
        else:
            report["storage_executable"] = {"path": str(executable), "error": "file does not exist"}
    if fixtures and executable:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            report["storage"] = {"executable": str(executable), "error": "not a runnable file"}
        else:
            print("starting isolated storage workload trials", flush=True)
            report["storage"] = measure_storage(
                executable, fixtures, args.action, temp_root / "storage-work",
                runs=args.runs, limit_mib=args.limit_mib,
                timeout_seconds=args.timeout_seconds,
                repeat_after_timeout=args.repeat_after_timeout,
                stop_file=args.stop_file, parent_report=report,
                checkpoint=save_checkpoint,
            )
    elif fixtures:
        storage_skipped: dict[str, Any] = {
            "skipped": True,
            "reason": "fixtures were supplied/generated without a runnable storage executable",
            "fixtures": [inspect_fixture_metadata(record) for record in fixtures],
        }
        if args.build_storage_executable and stop_reason(args.stop_file):
            pending_cases = []
            for record in fixtures:
                selected = list(args.action if args.action else
                                record.get("default_actions", ["open"]))
                for action in selected:
                    for trial in range(1, args.runs + 1):
                        pending_cases.append({
                            "format": record["format"], "kind": record.get("kind", "custom"),
                            "fixture_path": record["path"], "action": action,
                            "trial": trial, "reason": stop_reason(args.stop_file),
                        })
            storage_skipped["unrun_cases"] = pending_cases
            storage_skipped["stop_reason"] = stop_reason(args.stop_file)
        report["storage"] = storage_skipped
    elif executable:
        report["storage"] = {"skipped": True, "reason": "no fixtures supplied or generated",
                              "executable_evidence": report.get("storage_executable")}
    else:
        report["storage"] = {"skipped": True, "reason": "no executable or fixtures supplied"}
    if stop_reason(args.stop_file):
        report["stop_requested"] = True
        report["stop_reason"] = stop_reason(args.stop_file)
    save_checkpoint()
    return report


def main() -> int:
    global _STOP_REQUESTED, _STOP_REASON, _STOP_SIGNAL, _ACTIVE_PROCESS
    _STOP_REQUESTED = False
    _STOP_REASON = None
    _STOP_SIGNAL = None
    _ACTIVE_PROCESS = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[1]),
                        help="repository source root")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--runs", type=int, default=3, help="trials per build configuration and storage action")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--compiler", default="clang++-18")
    parser.add_argument("--configuration", default="Release",
                        choices=("Release", "Debug", "RelWithDebInfo"))
    parser.add_argument("--dependency-prefix", help="private storage dependency prefix passed to CMake")
    parser.add_argument("--skip-module-build", action="store_true",
                        help="skip CMake module configuration/build experiments")
    parser.add_argument("--storage-executable", help="existing storage_benchmark executable")
    parser.add_argument("--build-storage-executable", action="store_true",
                        help="build target storage_benchmark in a disposable both-format build")
    parser.add_argument("--fixture", action="append", type=parse_fixture, default=[],
                        metavar="FORMAT:PATH", help="fixture path; repeat for multiple formats/files")
    parser.add_argument("--readonly-sparse-fixture", action="append",
                        type=parse_readonly_sparse_fixture, default=[], metavar="FORMAT:PATH",
                        help="existing sparse fixture; measure only open and selected row/column reads")
    parser.add_argument("--generate-fixtures", metavar="DIR",
                        help="generate fixtures (run with .venv/bin/python for repository packages)")
    parser.add_argument("--include-huge-fixtures", action="store_true",
                        help="also generate zero-fill dense 524288 x 524288 fixtures")
    parser.add_argument("--fixture-size", type=int, default=256,
                        help="side length of generated small square matrices")
    parser.add_argument("--comparison-fixture-size", action="append", type=int, default=[],
                        help="also generate a separate matrix size for memory scaling; repeatable")
    parser.add_argument("--huge-size", type=int, default=524288,
                        help="side length of each opt-in huge logical dense matrix")
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="zero-fill large-X chunk edge and axis-index chunk length")
    parser.add_argument("--action", action="append", choices=ACTIONS,
                        help="storage action override; repeat to select multiple actions")
    parser.add_argument("--limit-mib", type=int, default=512,
                        help="per storage subprocess RLIMIT_AS in MiB; 0 disables")
    parser.add_argument("--timeout-seconds", type=int, default=30,
                        help="per storage action timeout in seconds (default: 30; pass 300 for the legacy limit)")
    parser.add_argument("--stop-file",
                        help="stop before the next build trial or storage observation when this path exists")
    parser.add_argument("--repeat-after-timeout", action="store_true",
                        help="continue identical fixture/action trials after a timeout")
    parser.add_argument("--keep-work", action="store_true",
                        help="retain frozen source copies, build trees, and trial work directories")
    args = parser.parse_args()
    if args.runs < 1 or args.jobs < 1:
        parser.error("--runs and --jobs must be positive")
    if args.limit_mib < 0:
        parser.error("--limit-mib cannot be negative")
    if args.timeout_seconds < 1 or args.fixture_size < 1 or args.huge_size < 1 or args.chunk_size < 1:
        parser.error("timeouts, fixture dimensions, and chunk size must be positive")
    if any(size < 32 for size in args.comparison_fixture_size):
        parser.error("comparison fixture dimensions must be at least 32")
    if args.comparison_fixture_size and not args.generate_fixtures:
        parser.error("--comparison-fixture-size requires --generate-fixtures")
    if len(set(args.comparison_fixture_size)) != len(args.comparison_fixture_size):
        parser.error("comparison fixture dimensions must be distinct")
    if args.include_huge_fixtures and not args.generate_fixtures:
        parser.error("--include-huge-fixtures requires --generate-fixtures")
    if args.storage_executable and args.build_storage_executable:
        parser.error("--storage-executable and --build-storage-executable are mutually exclusive")
    if args.readonly_sparse_fixture and args.action and any(
        action not in {"open", "sparse-row", "sparse-column"} for action in args.action
    ):
        parser.error("--readonly-sparse-fixture permits only open, sparse-row, and sparse-column actions")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(str(output) + ".checkpoint.json")
    temp_root: str | None = None
    state: dict[str, Any] = {}
    old_handlers: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        report = benchmark(args, checkpoint_path=checkpoint_path, report_state=state)
        temp_root = report.get("temporary_work_root")
    except Exception as error:
        report = state.get("report", {
            "schema": "anndata.storage-benchmark.v1",
            "temporary_work_root": getattr(args, "_work_root", None),
        })
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = __import__("traceback").format_exc()
        print(report["fatal_error"], file=sys.stderr)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if temp_root is None:
            temp_root = report.get("temporary_work_root") if "report" in locals() \
                else getattr(args, "_work_root", None)
        if temp_root and not args.keep_work:
            shutil.rmtree(temp_root, ignore_errors=True)
    report["temporary_work_removed"] = bool(temp_root and not args.keep_work)
    final_stop_reason = stop_reason(args.stop_file)
    if final_stop_reason:
        report["stop_requested"] = True
        report["stop_reason"] = final_stop_reason
    report["checkpoint"] = {
        "path": str(checkpoint_path),
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "state": "finalized",
    }
    atomic_write_json(checkpoint_path, report)
    atomic_write_json(output, report)
    print(json.dumps({"output": str(output), "fatal_error": report.get("fatal_error"),
                      "checkpoint": str(checkpoint_path),
                      "stop_reason": report.get("stop_reason"),
                      "source_manifest_sha256": report.get("source_snapshot", {}).get("source_manifest_sha256"),
                      "module_trial_count": len(report.get("module_builds", {}).get("trials", [])),
                      "storage_observation_count": len(report.get("storage", {}).get("observations", []))
                      if isinstance(report.get("storage"), dict) else 0,
                      "fixture_generation_error": report.get("fixture_generation_error")}, indent=2))
    module_failures = sum(len(trial.get("errors", [])) for trial in
                          report.get("module_builds", {}).get("trials", []))
    storage_failures = sum(item.get("failures", 0) for item in
                           report.get("storage", {}).get("summary", [])
                           if isinstance(item, dict)) if isinstance(report.get("storage"), dict) else 0
    storage_report = report.get("storage")
    if isinstance(storage_report, dict) and storage_report.get("error"):
        storage_failures += 1
    if report.get("storage_driver_build", {}).get("error"):
        storage_failures += 1
    if report.get("storage_executable", {}).get("error"):
        storage_failures += 1
    if report.get("fatal_error") or report.get("fixture_generation_error") \
            or module_failures or storage_failures:
        return 1
    if _STOP_SIGNAL is not None:
        return 128 + _STOP_SIGNAL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
