#!/usr/bin/env python3
"""Short paired gather experiment; snapshots and build products stay local."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("build/gather-buffer/report.json"))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture, probe = root / "benchmarks/gather-fixture", root / "benchmarks/gather.cpp"
    out = a.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    variants = {"baseline": a.baseline.resolve(), "candidate": a.candidate.resolve()}
    report = {"cap_seconds": 30, "conditions": "Release, Clang 18, Ninja, two build jobs; OS caches not cleared; no profiling", "commands": [], "runtime": [], "builds": {}}

    def save():
        temp = out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(out)

    def run(label, command):
        log = out.parent / (label + ".log")
        metrics = log.with_suffix(".time")
        wrapped = ["timeout", "30s", *map(str, command)]
        if Path("/usr/bin/time").exists():
            wrapped = ["/usr/bin/time", "-f", "%e %M", "-o", str(metrics), *wrapped]
        start = time.monotonic()
        with log.open("w") as f:
            result = subprocess.run(wrapped, stdout=f, stderr=subprocess.STDOUT)
        record = {"label": label, "command": list(map(str, command)), "exit_code": result.returncode, "wall_seconds": time.monotonic() - start, "log": str(log), "log_sha256": sha(log)}
        if metrics.exists():
            fields = metrics.read_text().splitlines()[-1].split()
            record["max_rss_kib"] = int(fields[1])
        report["commands"].append(record)
        save()
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} failed or timed out; see {log}")
        return log

    def artifacts(build):
        paths = sorted(build.rglob("*.pcm"))
        paths += list(build.glob("CMakeFiles/gather_consumer.dir/*.o"))
        return {str(x.relative_to(build)): {"sha256": sha(x), "mtime_ns": x.stat().st_mtime_ns, "bytes": x.stat().st_size} for x in paths}

    for name, source in variants.items():
        build = out.parent / (name + "-build")
        if build.exists():
            raise RuntimeError(f"fresh build required; preserve existing evidence in {build}")
        files = sorted(x for x in (source / "src").iterdir() if x.is_file())
        report["builds"][name] = {"source": str(source), "inputs": {str(x.relative_to(source)): sha(x) for x in files}}
        run(name + "-configure", ["cmake", "-S", fixture, "-B", build, "-G", "Ninja", "-DCMAKE_CXX_COMPILER=clang++-18", "-DCMAKE_BUILD_TYPE=Release", f"-DANNDATAPP_SOURCE={source}", f"-DANNDATAPP_PROBE={probe}"])
        run(name + "-cold", ["cmake", "--build", build, "--target", "gather_consumer", "--parallel", "2"])
        run(name + "-consumer-run", [build / "gather_consumer"])
        for i in range(2):
            (fixture / "consumer.cpp").touch()
            run(f"{name}-consumer-edit-{i}", ["cmake", "--build", build, "--target", "gather_consumer", "--parallel", "2"])
        before = artifacts(build)
        (source / "src/arrays.cpp").touch()
        run(name + "-private-edit", ["cmake", "--build", build, "--target", "gather_consumer", "--parallel", "2"])
        after = artifacts(build)
        report["builds"][name]["artifacts"] = after
        report["builds"][name]["private_edit_preserved_bmis_and_consumer"] = before == after
        if not before or before != after:
            raise RuntimeError(f"{name}: private edit rebuilt public artifacts")
        run(name + "-probe-build", ["cmake", "--build", build, "--target", "gather_probe", "--parallel", "2"])
        report["builds"][name]["probe_sha256"] = sha(build / "gather_probe")
        save()

    affinity = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    report["runtime_cpu"] = affinity if shutil.which("taskset") else None
    # Three short pairs, alternating order; timing is separate from allocation counters.
    for pair in range(3):
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        for name in order:
            command = [out.parent / (name + "-build/gather_probe"), "1"]
            if report["runtime_cpu"] is not None:
                command = ["taskset", "-c", str(affinity), *command]
            log = run(f"{name}-runtime-{pair}", command)
            for line in log.read_text().splitlines():
                row = json.loads(line)
                report["runtime"].append({"variant": name, "pair": pair, **row})
            save()
    report["probe_inputs"] = {str(x.relative_to(root)): sha(x) for x in [probe, fixture / "CMakeLists.txt", fixture / "consumer.cpp"]}
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
