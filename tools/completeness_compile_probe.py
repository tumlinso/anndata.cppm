#!/usr/bin/env python3
"""Measure warm consumer imports against already built module artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def source_hashes(root: Path) -> dict:
    files=[p for p in (root/"src").glob("*") if p.suffix in (".cpp",".cppm",".hpp")]
    files.append(root/"CMakeLists.txt")
    return {str(p.relative_to(root)):digest(p) for p in sorted(files)}


def artifacts(path: Path) -> dict:
    files=sorted(p for p in path.glob("*.pcm") if p.name!="anndata.backed-storage.pcm")
    if not files: raise ValueError("no public BMI files in "+str(path))
    return {p.name:{"bytes":p.stat().st_size,"sha256":digest(p)} for p in files}


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-build",type=Path,required=True);p.add_argument("--candidate-build",type=Path,required=True)
    p.add_argument("--baseline-source",type=Path,required=True);p.add_argument("--candidate-source",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--compiler",default="clang++-18")
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    modules={"core":"anndata.core","streaming":"anndata.streaming","facade":"anndata"}
    for k,m in modules.items(): (out/(k+".cpp")).write_text(f"import {m};\nint main() {{ return 0; }}\n")
    roots={"baseline":args.baseline_source.resolve(),"candidate":args.candidate_source.resolve()}
    builds={"baseline":args.baseline_build.resolve(),"candidate":args.candidate_build.resolve()}
    record={"schema":"anndata.completeness.compile.v1","method":"one warm import-only compilation per module; OS cache not cleared",
            "compiler":args.compiler,"tool_sha256":digest(Path(__file__).resolve()),"source_hashes":{},"builds":{}}
    path=out/"result.json";tmp=out/"result.json.tmp"
    def checkpoint():
        with tmp.open("x") as f: json.dump(record,f,indent=2);f.write("\n")
        tmp.replace(path)
    for label in ("baseline","candidate"):
        record["source_hashes"][label]=source_hashes(roots[label])
        bmi=builds[label]/"CMakeFiles/anndata.dir"
        data={"bmi_directory":str(bmi),"public_bmis":artifacts(bmi),"imports":{}}
        data["public_bmi_bytes"]=sum(x["bytes"] for x in data["public_bmis"].values());record["builds"][label]=data
    checkpoint()
    for label in ("baseline","candidate"):
        bmi=Path(record["builds"][label]["bmi_directory"])
        for name in modules:
            source=out/(name+".cpp");obj=out/(label+"-"+name+".o");measure=out/(label+"-"+name+".time")
            command=["/usr/bin/time","-f","%e %M","-o",str(measure),args.compiler,"-std=c++20","-O3","-DNDEBUG",
                     "-fprebuilt-module-path="+str(bmi),"-c",str(source),"-o",str(obj)]
            entry={"command":command,"status":"running"};record["builds"][label]["imports"][name]=entry;checkpoint()
            start=time.monotonic()
            process=subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            try:
                stdout,stderr=process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: stdout,stderr=process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    stdout,stderr=process.communicate()
                entry.update(status="timeout",wall_seconds=time.monotonic()-start,stdout=stdout,stderr=stderr);checkpoint();return 1
            entry.update(returncode=process.returncode,wall_seconds=time.monotonic()-start,stdout=stdout,stderr=stderr,
                         time_text=measure.read_text() if measure.exists() else None,status="ok" if process.returncode==0 else "failed")
            if obj.exists():entry["object_sha256"]=digest(obj)
            checkpoint()
            if process.returncode:return 1
    for label in roots:
        if source_hashes(roots[label])!=record["source_hashes"][label]:
            record["status"]="source_changed";checkpoint();return 1
    record["status"]="ok";checkpoint()
    print(json.dumps({"status":"ok","sha256":digest(path)}));return 0


if __name__=="__main__":raise SystemExit(main())
