#!/usr/bin/env python3
"""Build, validate and run the fixed AnnData++ 1.0 standard performance suite."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("direct", "acquisition", "dense_copy", "sparse", "strings", "metadata",
             "h5ad", "zarr2", "zarr3", "contexts")
SAMPLES = 5
SCENARIO_SECONDS = 30
REVISION_SECONDS = 300


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root, driver=False):
    root = Path(root).resolve()
    paths = sorted((root / "benchmarks/v1").glob("*")) if driver else [
        p for folder in (root / "src", root / "cmake") for p in folder.rglob("*") if p.is_file()]
    if not driver:
        paths += [root / "CMakeLists.txt", root / "CMakePresets.json"]
    else:
        paths += [root / "tools/v1_benchmark.py"]
    files = {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"root": str(root), "sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def command(cmd, **kwargs):
    return subprocess.run([str(s) for s in cmd], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def build(args):
    source, output = args.source.resolve(), args.build.resolve()
    before = manifest(source)
    driver = manifest(ROOT, driver=True)
    output.mkdir(parents=True, exist_ok=True)
    configure = ["cmake", "-S", ROOT / "benchmarks/v1", "-B", output, "-G", "Ninja",
                 "-DCMAKE_CXX_COMPILER=clang++-18", "-DCMAKE_C_COMPILER=clang-18",
                 "-DCMAKE_BUILD_TYPE=Release", f"-DANNDATAPP_SOURCE={source}",
                 f"-DV1_NEW_CORE={'ON' if args.new_core else 'OFF'}"]
    if args.prefix:
        configure += [f"-DCMAKE_PREFIX_PATH={args.prefix.resolve()}"]
    commands = [configure, ["cmake", "--build", output, "--target", "v1_benchmark", "v1_io_counters", "v1_alloc_counters", "-j2"]]
    records = []
    for i, cmd in enumerate(commands):
        first = time.monotonic()
        result = command(cmd)
        (output / f"build-{i}.log").write_text(result.stdout + result.stderr)
        records.append({"command": list(map(str, cmd)), "wall_seconds": time.monotonic()-first})
    if before != manifest(source) or driver != manifest(ROOT, driver=True):
        raise RuntimeError("source or driver changed during build; repeat only after freezing it")
    versions = {name: {"version": command(cmd).stdout.strip(), "executable": str(Path(shutil.which(cmd[0])).resolve()),
                       "sha256": sha(shutil.which(cmd[0]))} for name, cmd in {
        "compiler": ["clang++-18", "--version"], "scanner": ["clang-scan-deps-18", "--version"],
        "cmake": ["cmake", "--version"], "ninja": ["ninja", "--version"]}.items()}
    receipt = {"schema": "anndatapp-v1-build/1", "source": before, "driver": driver,
               "new_core": args.new_core, "commands": records, "toolchain": versions,
               "executable": str(output / "v1_benchmark"), "executable_sha256": sha(output / "v1_benchmark"),
               "interposer": str(output / "libv1_io_counters.so"),
               "interposer_sha256": sha(output / "libv1_io_counters.so"),
               "alloc_interposer": str(output / "libv1_alloc_counters.so"),
               "alloc_interposer_sha256": sha(output / "libv1_alloc_counters.so"),
               "consumer_linkage": (output / "consumer-linkage.txt").read_text()}
    write(output / "receipt.json", receipt)
    print(output / "receipt.json")


def check_receipt(path):
    receipt = json.loads(Path(path).read_text())
    for name in ("executable", "interposer", "alloc_interposer"):
        if sha(receipt[name]) != receipt[name + "_sha256"]:
            raise RuntimeError(f"{name} changed since build")
    source = receipt["source"]
    if source != manifest(source["root"]):
        raise RuntimeError("library source changed since build")
    if receipt["driver"] != manifest(ROOT, driver=True):
        raise RuntimeError("driver source changed since build")
    return receipt


def fixture_identity(path):
    path = path.resolve()
    files = {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}
    return {"path": str(path), "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "manifest": json.loads((path / "manifest.json").read_text())}


def invoke(receipt, scenario, fixture, work, mode, timeout):
    # Every process receives a disposable byte-identical store. Copying and
    # hashing are outside both the executable's timer and its process interval.
    scratch = work / "active-fixtures"
    if scratch.exists():
        shutil.rmtree(scratch)
    if scenario in ("h5ad", "zarr2", "zarr3", "contexts"):
        shutil.copytree(fixture, scratch)
    else:
        scratch.mkdir(parents=True)
    cmd = [receipt["executable"], scenario, str(scratch)]
    if mode == "validate":
        cmd.append("--validate-only")
    if mode == "diagnostic":
        cmd.append("--diagnostic")
    env = os.environ.copy(); env.pop("LD_PRELOAD",None)
    if mode == "diagnostic":
        env["LD_PRELOAD"] = receipt["alloc_interposer"] + ":" + receipt["interposer"]
    first = time.monotonic()
    result = command(cmd, env=env, timeout=timeout)
    elapsed = time.monotonic()-first
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    process = rows.pop()
    if not process.get("process_summary"):
        raise RuntimeError("missing final process RSS receipt")
    for row in rows:
        row["process_peak_rss_kib"] = process["process_peak_rss_kib"]
    if not rows or not all(row["validated"] for row in rows):
        raise RuntimeError("missing validated benchmark result")
    return {"process_wall_seconds": elapsed, "operations": rows}


def environment():
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    return {"hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "cache_policy": "normal OS caches; no cache dropping", "maximum_workers": 4,
            "timing_overlap": "suite serial; external activity uncontrolled"}


def distribution(values):
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"samples": values, "median": statistics.median(values), "min": min(values),
            "max": max(values), "p25": q[0], "p75": q[2]}


def consumer(args):
    if not args.timing_granted:
        raise RuntimeError("consumer build observations require a coordinated --timing-granted window")
    output = args.work.resolve(); output.mkdir(parents=True,exist_ok=True)
    results = {}
    for revision, path in (("baseline",args.baseline),("candidate",args.candidate)):
        receipt = check_receipt(path)
        build_dir = Path(receipt["executable"]).parent
        obj = build_dir / "CMakeFiles/v1_core_consumer.dir/core_consumer.cpp.o"
        if obj.exists():
            raise RuntimeError("fresh consumer object already exists; preserve it and use a new benchmark build")
        rows, charged = [], 0.0
        for sample in range(4):
            if sample:
                os.utime(build_dir / "core_consumer.cpp",None)
            log = output / f"{revision}-{sample}.log"
            rss = output / f"{revision}-{sample}.rss"
            cmd = ["/usr/bin/time","-f","%M","-o",str(rss),"cmake","--build",str(build_dir),
                   "--target","v1_core_consumer","-j2","--verbose"]
            first = time.monotonic()
            result = command(cmd,timeout=60-charged)
            elapsed = time.monotonic()-first; charged += elapsed
            log.write_text(result.stdout+result.stderr)
            if not obj.exists() or "core_consumer.cpp.o" not in result.stdout:
                raise RuntimeError("consumer compilation was not observed")
            rows.append({"operation": "fresh_consumer" if not sample else "source_recompile", "wall_seconds": elapsed,
                         "command_peak_rss_kib": int(rss.read_text().strip()), "log_sha256": sha(log), "object_sha256": sha(obj)})
        command([build_dir / "v1_core_consumer"])
        bmis = {str(p.relative_to(build_dir)): {"bytes":p.stat().st_size,"sha256":sha(p)}
                for p in sorted(build_dir.rglob("*.pcm"))}
        modmap = (build_dir / "CMakeFiles/v1_core_consumer.dir/core_consumer.cpp.o.modmap").read_text()
        imported = {}
        for line in modmap.splitlines():
            if not line.startswith("-fmodule-file="):
                continue
            name, relative = line.removeprefix("-fmodule-file=").split("=",1)
            p = build_dir / relative
            imported[name] = {"bytes":p.stat().st_size,"sha256":sha(p)}
        results[revision] = {"observations":rows,"charged_build_seconds":charged,"all_built_bmi_bytes":sum(p["bytes"] for p in bmis.values()),
                             "bmis":bmis,"consumer_import_bmis":imported,"consumer_import_bmi_bytes":sum(p["bytes"] for p in imported.values()),
                             "linkage": receipt["consumer_linkage"],
                             "source_sha256":receipt["source"]["sha256"]}
    report = {"scope":"one fresh core-import consumer and three source-only recompiles per revision; library BMIs prebuilt by benchmark build",
              "rss_scope":"GNU time command high-water RSS; not summed concurrent compiler RSS",
              "bmi_scope":"consumer compiler module-map import closure, plus all already-built BMI artifacts including private partitions; not parsing latency",
              "build_budget_seconds_per_revision":60,"environment":environment(),"results":results}
    path = output / "consumer-build.json"; write(path,report)
    if args.summary:
        summary = json.loads(args.summary.read_text()); summary["consumer_build"] = report
        summary["consumer_build"]["raw_sha256"] = sha(path); write(args.summary,summary)
    print(path)


def summarize(raw, diagnostics, receipts, fixture, raw_path):
    ops = {}
    for scenario in SCENARIOS:
        samples = {revision: [r for r in raw if r["scenario"] == scenario and r["revision"] == revision]
                   for revision in ("baseline", "candidate")}
        names = [p["operation"] for p in samples["baseline"][0]["operations"]]
        for name in names:
            pair = {rev: [next(p for p in r["operations"] if p["operation"] == name) for r in rows]
                    for rev, rows in samples.items()}
            b = [r["seconds"] for r in pair["baseline"]]; c = [r["seconds"] for r in pair["candidate"]]
            ratios = [y/x for x, y in zip(b, c)]
            separated = min(c) > max(b)
            record = {"scenario": scenario, "paired_median_ratio": statistics.median(ratios),
                      "regression_requires_investigation": separated and statistics.median(ratios) > 1.05,
                      "candidate_slower_ranges_separated": separated}
            for rev, rows in pair.items():
                diagnostic = next(p for r in diagnostics if r["scenario"] == scenario and r["revision"] == rev
                                  for p in r["operations"] if p["operation"] == name)
                record[rev] = {"seconds": distribution([r["seconds"] for r in rows]),
                               "iterations": rows[0]["iterations"], "payload_bytes": rows[0]["payload_bytes"],
                               "acquisition": rows[0]["acquisition"]}
                for metric in ("new_calls", "new_requested_bytes", "process_peak_rss_kib", "backend_calls", "backend_bytes",
                               "libc_read_calls", "libc_read_bytes", "libc_write_calls", "libc_write_bytes",
                               "diagnostic_operation_cpu_ns", "diagnostic_zstd_context_cpu_ns", "diagnostic_zstd_work_cpu_ns"):
                    record[rev][metric] = diagnostic[metric]
                record[rev]["codec_calls"] = diagnostic["codec_calls"]
                if rows[0]["stream_trials"] >= 0:
                    record[rev]["timed_stream_metrics"] = {key:[r[key] for r in rows] for key in
                        ("stream_trials","stream_adaptations","stream_rollbacks","stream_final_limit")}
            ops[name] = record
    return {"schema": "anndatapp-v1-standard/1", "status": "measured",
            "samples_per_revision": SAMPLES, "order": "baseline,candidate repeated five times per scenario",
            "diagnostic_samples_per_scenario_revision": 1,
            "timing_observers": "no LD_PRELOAD; ordinary global C++ allocation functions",
            "environment": environment(), "fixtures": fixture,
            "revisions": {rev: {k: receipt[k] for k in ("source", "driver", "toolchain", "executable_sha256", "interposer_sha256", "alloc_interposer_sha256", "new_core")}
                          for rev, receipt in receipts.items()},
            "budgets": {"scenario_seconds_per_revision": SCENARIO_SECONDS,
                        "benchmark_seconds_per_machine_revision": REVISION_SECONDS,
                        "charged_scope": "subprocess wall time, conservative including setup and validation; builds and fixture copies excluded"},
            "counter_scopes": {"new": "C++ global throwing new/new[], including aligned; requested bytes, not live bytes; C malloc/system allocations excluded",
                               "sampling": "counts, RSS, backend/libc/codec metrics from one separate instrumented diagnostic execution per scenario/revision; excluded from five timing samples",
                               "rss": "whole process high-water RSS, includes initialization and correctness validation",
                               "backend": "TransferContext logical transfer stats where available; null means unavailable",
                               "libc": "interposed read/pread/readv/write/pwrite/writev; not direct syscalls; includes codec file traffic",
                               "codec_cpu": "diagnostic only; sum intercepted ZSTD create/free call thread CPU and compress2/decompressStream thread CPU; operation uses process CPU; clock/interposition overhead and unobserved internal calls limit attribution",
                               "codec_calls_order": ["ZSTD_createCCtx", "ZSTD_createDCtx", "ZSTD_freeCCtx", "ZSTD_freeDCtx", "ZSTD_compress2", "ZSTD_decompressStream",
                                                     "deflateInit", "inflateInit", "deflateEnd", "inflateEnd", "H5Dread", "H5Dwrite"]},
            "raw": {"path": str(raw_path), "sha256": sha(raw_path)}, "operations": ops}


def execute(args):
    receipts = {"baseline": check_receipt(args.baseline), "candidate": check_receipt(args.candidate)}
    if receipts["baseline"]["driver"] != receipts["candidate"]["driver"]:
        raise RuntimeError("baseline and candidate must use identical driver source")
    if receipts["baseline"]["toolchain"] != receipts["candidate"]["toolchain"]:
        raise RuntimeError("baseline and candidate toolchains differ")
    fixture = fixture_identity(args.fixtures)
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    if args.action == "validate":
        rows = []
        for revision, receipt in receipts.items():
            for scenario in SCENARIOS:
                rows.append({"revision": revision, "scenario": scenario,
                             **invoke(receipt, scenario,args.fixtures,work,"validate",30)})
        write(work / "validation.json", {"fixtures": fixture, "results": rows})
        print(work / "validation.json"); return
    if not args.timing_granted:
        raise RuntimeError("run requires an explicit coordinated --timing-granted execution window")
    raw_path = work / "samples.json"
    if raw_path.exists():
        raise RuntimeError("sample receipt already exists; preserve it and use a new work directory")
    raw, diagnostics, revision_elapsed = [], [], {"baseline": 0.0, "candidate": 0.0}
    checkpoint = {"schema":"anndatapp-v1-raw/1", "environment":environment(),
                  "revisions":{name:{"source_sha256":r["source"]["sha256"],"driver_sha256":r["driver"]["sha256"],
                                      "executable_sha256":r["executable_sha256"],"toolchain":r["toolchain"]}
                               for name,r in receipts.items()},
                  "fixture_sha256":fixture["sha256"],"timings":raw,"diagnostics":diagnostics}
    write(raw_path,checkpoint)
    for scenario in SCENARIOS:
        scenario_elapsed = {"baseline": 0.0, "candidate": 0.0}
        expected_signature = None
        for sample in range(SAMPLES):
            for revision, receipt in receipts.items():
                remaining = min(SCENARIO_SECONDS-scenario_elapsed[revision], REVISION_SECONDS-revision_elapsed[revision])
                if remaining <= 0:
                    raise RuntimeError("standard suite benchmark budget exhausted")
                result = invoke(receipt,scenario,args.fixtures,work,"timing",remaining)
                signature = [(p["operation"],p["iterations"],p["payload_bytes"]) for p in result["operations"]]
                if expected_signature is not None and signature != expected_signature:
                    raise RuntimeError("scenario workload differs between samples or revisions")
                expected_signature = signature
                elapsed = result["process_wall_seconds"]
                scenario_elapsed[revision] += elapsed; revision_elapsed[revision] += elapsed
                raw.append({"scenario": scenario, "sample": sample, "revision": revision, **result})
                write(raw_path,checkpoint)
                print(f"{scenario} {sample+1}/5 {revision}: {elapsed:.3f}s", flush=True)
        for revision, receipt in receipts.items():
            remaining = min(SCENARIO_SECONDS-scenario_elapsed[revision],REVISION_SECONDS-revision_elapsed[revision])
            if remaining <= 0:
                raise RuntimeError("standard suite diagnostic budget exhausted")
            result = invoke(receipt,scenario,args.fixtures,work,"diagnostic",remaining)
            elapsed = result["process_wall_seconds"]
            scenario_elapsed[revision] += elapsed; revision_elapsed[revision] += elapsed
            diagnostics.append({"scenario": scenario, "revision": revision, **result})
            write(raw_path,checkpoint)
    if fixture != fixture_identity(args.fixtures):
        raise RuntimeError("immutable fixtures changed")
    for path in (args.baseline,args.candidate):
        check_receipt(path)
    summary = summarize(raw,diagnostics,receipts,fixture,raw_path)
    summary["charged_revision_wall_seconds"] = revision_elapsed
    write(args.output,summary)
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", type=Path, required=True); b.add_argument("--build", type=Path, required=True)
    b.add_argument("--new-core", action="store_true"); b.add_argument("--prefix", type=Path)
    c = sub.add_parser("consumer")
    for key in ("baseline","candidate","work"):
        c.add_argument("--"+key,type=Path,required=True)
    c.add_argument("--timing-granted",action="store_true"); c.add_argument("--summary",type=Path)
    for action in ("validate", "run"):
        p = sub.add_parser(action)
        for key in ("baseline", "candidate", "fixtures", "work"):
            p.add_argument("--"+key,type=Path,required=True)
        if action == "run":
            p.add_argument("--timing-granted",action="store_true")
            p.add_argument("--output",type=Path,default=ROOT / "benchmarks/results/v1-summary.json")
    args = parser.parse_args()
    try:
        build(args) if args.action == "build" else consumer(args) if args.action == "consumer" else execute(args)
    except (subprocess.SubprocessError, RuntimeError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stdout); print(exc.stderr)
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
