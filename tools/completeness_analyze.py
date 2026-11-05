#!/usr/bin/env python3
"""Compare checkpointed observations with independent source-shard indexes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

ROWS = (270,64,270,9,256,511,257,3,95,10,448,1,384,65,258,0)
COLS = (520,4,520,129,513,255,127,6)
CASES = {"full_4":("full","zarr3_4"),"full_16":("full","zarr3_16"),"full_64":("full","zarr3_64"),
         "partial_16":("partial","zarr3_16"),"gather_stride":("gather_stride","zarr3_16"),
         "gather_view":("gather_view","zarr3_16"),"huge_gap":("huge_gap","huge_gap")}
MISSING = (1<<64)-1


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def parts(positions, chunk: int) -> dict[int,set[int]]:
    result: dict[int,set[int]]={}
    for value in positions: result.setdefault(value//chunk,set()).add(value%chunk)
    return result


def expected(root: Path, name: str, spec: dict) -> dict:
    action,fixture=CASES[name]
    shape=(1<<30,1<<20) if fixture=="huge_gap" else (512,1024)
    if action=="full": rows,cols=range(256),range(512)
    elif action=="partial": rows,cols=range(150,350),range(300,700)
    elif action=="huge_gap": rows,cols=(0,shape[0]-1),(0,shape[1]-1)
    else: rows,cols=ROWS,COLS
    inner=tuple(spec["inner"]); shard=tuple(spec["shard"]); index_bytes=spec["index_bytes"]
    metadata=json.loads((root/fixture/"X"/"zarr.json").read_text())
    conf=metadata["codecs"][0]["configuration"]
    if conf["index_location"]!="end" or [x["name"] for x in conf["index_codecs"]]!=["bytes","crc32c"]:
        raise ValueError("index prediction requires little-endian end index with CRC32C")
    byteorder=conf["index_codecs"][0].get("configuration",{}).get("endian","little")
    if byteorder!="little" or [x["name"] for x in conf["codecs"]]!=["bytes","zstd"]:
        raise ValueError("fixture codec differs from the measured assumptions")
    rowparts,colparts=parts(rows,inner[0]),parts(cols,inner[1])
    factor=(shard[0]//inner[0],shard[1]//inner[1])
    selected: dict[tuple[int,int],dict[tuple[int,int],bool]]={}
    for r,rset in rowparts.items():
        for c,cset in colparts.items():
            sr,sc=r//factor[0],c//factor[1]
            nr=min(inner[0],shape[0]-r*inner[0]);nc=min(inner[1],shape[1]-c*inner[1])
            selected.setdefault((sr,sc),{})[(r%factor[0],c%factor[1])]=len(rset)==nr and len(cset)==nc
    result={"changed_chunks":0,"partial_present_chunks":0,"full_present_chunks":0,"shards":{}}
    for (sr,sc),updates in sorted(selected.items()):
        file=root/fixture/"X"/"c"/str(sr)/str(sc)
        entries=[]
        if file.exists():
            size=file.stat().st_size
            with file.open("rb") as f: f.seek(-index_bytes,2);raw=f.read(index_bytes)
            if len(raw)!=index_bytes or len(raw)-4!=16*factor[0]*factor[1]: raise ValueError("invalid source index size")
            for i in range(factor[0]*factor[1]):
                offset,length=struct.unpack_from("<QQ",raw,16*i)
                if offset==MISSING and length==MISSING: entries.append(None);continue
                if offset==MISSING or length==MISSING or offset>size-index_bytes or length>size-index_bytes-offset:
                    raise ValueError("invalid source index entry")
                entries.append(length)
        else: entries=[None]*(factor[0]*factor[1])
        old_read=0;untouched=0;partial=0;full=0
        for i,length in enumerate(entries):
            if length is None: continue
            coord=(i//factor[1],i%factor[1])
            if coord not in updates: old_read+=length;untouched+=1
            elif updates[coord]: full+=1
            else: old_read+=length;partial+=1
        result["changed_chunks"]+=len(updates);result["partial_present_chunks"]+=partial;result["full_present_chunks"]+=full
        result["shards"][f"c/{sr}/{sc}"]={"source_exists":file.exists(),"index_bytes":index_bytes,
            "changed_chunks":len(updates),"partial_present_chunks":partial,"fully_overwritten_present_chunks":full,
            "untouched_present_chunks":untouched,"old_payload_bytes_needed_at_most_once":old_read}
    return result


def summarize(entry: dict, chunk_bytes: int) -> dict:
    c=entry["counters"]; z=c["codec_api_stream_calls"]["zstd"]
    if z["decode_bytes_out"]%chunk_bytes or z["encode_bytes_in"]%chunk_bytes:
        raise ValueError("Zstd byte totals do not match complete fixture chunks")
    return {"seconds":entry["result"]["seconds"],"useful_bytes":entry["result"]["payload_bytes"],
            "intercepted_read_bytes":c["intercepted_read_bytes"],"intercepted_write_bytes":c["intercepted_write_bytes"],
            "malloc_calls":c["alloc"]["malloc"]["calls"],"malloc_requested_bytes":c["alloc"]["malloc"]["requested_bytes"],
            "zstd_decode_api_calls":z["decode_calls"],"zstd_encode_api_calls":z["encode_calls"],
            "decoded_complete_chunks":z["decode_bytes_out"]//chunk_bytes,
            "encoded_complete_chunks":z["encode_bytes_in"]//chunk_bytes,
            "successful_shard_renames":sum(s["rename_successes"] for s in c["shards"]),
            "index_position_reads":sum(s["index_read_calls"] for s in c["shards"]),
            "index_position_bytes":sum(s["index_read_bytes"] for s in c["shards"]),
            "index_end_write_candidates":sum(s["index_end_write_candidates"] for s in c["shards"]),
            "retained_scratch_bytes":entry["result"]["scratch_bytes"],
            "process_max_rss_kib":entry["result"]["process_max_rss_kib"]}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True);p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--fixtures",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise SystemExit("refusing to overwrite analysis")
    old,new=json.loads(args.baseline.read_text()),json.loads(args.candidate.read_text())
    for k in ("fixtures_sha256","seed_sha256","interposer_sha256","runner_sha256","budget"):
        if old[k]!=new[k]: raise ValueError("paired condition differs: "+k)
    manifest=json.loads((args.fixtures/"manifest.json").read_text())
    if digest(args.fixtures/"manifest.json")!=old["fixtures_sha256"]: raise ValueError("fixture manifest differs")
    cases={}
    for name,(action,fixture) in CASES.items():
        a,b=old["cases"][name],new["cases"][name]
        if a["status"]!="ok" or b["status"]!="ok" or a["validation"]!=b["validation"]:
            raise ValueError("paired validation missing or differs: "+name)
        for observation in (a,b):
            counters=observation["counters"]
            if counters["read_rejections"] or counters["write_rejections"] or counters["shard_path_overflow"]:
                raise ValueError("I/O cap or shard-path limit reached: "+name)
        spec=manifest["huge_gap"] if fixture=="huge_gap" else manifest["layouts"][fixture]
        prediction=expected(args.fixtures,name,spec);chunk_bytes=spec["inner"][0]*spec["inner"][1]*4
        before,after=summarize(a,chunk_bytes),summarize(b,chunk_bytes)
        if after["successful_shard_renames"]!=len(prediction["shards"]): raise ValueError("not one replacement per shard: "+name)
        if after["encoded_complete_chunks"]!=prediction["changed_chunks"]: raise ValueError("changed chunks encoded more than once: "+name)
        if after["decoded_complete_chunks"]!=prediction["partial_present_chunks"]: raise ValueError("changed chunks decoded incorrectly: "+name)
        observed={s["path"]:s for s in b["counters"]["shards"]}
        if set(observed)!=set(prediction["shards"]): raise ValueError("touched shard paths differ: "+name)
        for path,specification in prediction["shards"].items():
            measured=observed[path]
            if measured["rename_successes"]!=1: raise ValueError("multiple shard replacements: "+name+":"+path)
            if measured["rename_failures"]: raise ValueError("shard replacement failed: "+name+":"+path)
            if measured["index_read_bytes"]>specification["index_bytes"]: raise ValueError("repeated index read: "+name+":"+path)
            allowance=specification["old_payload_bytes_needed_at_most_once"]+specification["index_bytes"]
            if measured["read_bytes"]>allowance: raise ValueError("repeated old payload reads: "+name+":"+path)
        cases[name]={"source_index_prediction":prediction,"baseline":before,"candidate":after,"independent_validation":b["validation"]}
    adaptive={}
    for name in ("fixed_1","fixed_2","fixed_4","fixed_8","adaptive"):
        a,b=old["cases"][name],new["cases"][name]
        if a["status"]!="ok" or b["status"]!="ok" or a["validation"]!=b["validation"]:
            raise ValueError("adaptive paired validation missing or differs: "+name)
        for observation in (a,b):
            counters=observation["counters"]
            if counters["read_rejections"] or counters["write_rejections"] or counters["shard_path_overflow"]:
                raise ValueError("I/O cap or shard-path limit reached: "+name)
        adaptive[name]={version:{"seconds":v["result"]["seconds"],"max_outstanding":v["result"]["max_outstanding"],
                                  "max_concurrent_x_pread":v["counters"]["max_concurrent_x_pread"],
                                  "trials":v["result"]["trials"],"adaptations":v["result"]["adaptations"],
                                  "trace":v["result"]["trace"]} for version,v in (("baseline",a),("candidate",b))}
    result={"schema":"anndatapp.completeness.analysis.v1","baseline_sha256":digest(args.baseline),
            "candidate_sha256":digest(args.candidate),"fixture_manifest_sha256":digest(args.fixtures/"manifest.json"),
            "method":"Independent end-index length parsing; total Zstd output/input bytes divided by full inner chunk bytes. Interposed calls exclude direct syscalls and mmap. One observation per case; source fixture, codec and settings match.",
            "cases":cases,"adaptive":adaptive}
    with args.output.open("x") as f: json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({"write_cases":len(cases),"adaptive_cases":len(adaptive),"sha256":digest(args.output)}))


if __name__=="__main__": main()
