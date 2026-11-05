#!/usr/bin/env python3
"""Compare one grouped-read candidate with the preserved head-to-head baseline."""
import argparse
import json
import os
from pathlib import Path
import time

from run import atomic_json, digest, expected_sha, one


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--python',required=True)
    p.add_argument('--driver',required=True,type=Path)
    p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--library',required=True,type=Path)
    p.add_argument('--counter-small',required=True,type=Path)
    p.add_argument('--counter-large',required=True,type=Path)
    args=p.parse_args()
    base=args.root.resolve(); baseline=args.baseline.resolve(); old=json.loads(baseline.read_text()); out=base/'gather-candidate.json'
    if out.exists():raise SystemExit(f'refusing to overwrite {out}')
    cpu=old['conditions']['logical_cpu']
    if cpu not in os.sched_getaffinity(0):raise SystemExit('baseline CPU is outside current affinity')
    os.sched_setaffinity(0,{cpu})
    binary=args.binary.resolve();driver=args.driver.resolve();library=args.library.resolve()
    sources={'candidate_binary_sha256':digest(binary),'candidate_library_sha256':digest(library),
             'candidate_backed_array_sha256':digest(Path(__file__).resolve().parents[2]/'src/backed_array.cpp'),
             'fixture_manifest_sha256':digest(base/'fixtures'/'fixture-manifest.json'),
             'baseline_campaign_sha256':digest(baseline),'driver_sha256':digest(driver),
             'original_cpp_driver_sha256':digest(Path(__file__).with_name('cpp_driver.cpp')),
             'counter_small_sha256':digest(args.counter_small),'counter_large_sha256':digest(args.counter_large),
             'comparison_source_sha256':digest(Path(__file__))}
    if sources['fixture_manifest_sha256']!=old['sources']['fixture_manifest_sha256'] or sources['driver_sha256']!=old['sources']['python_driver_sha256'] or sources['original_cpp_driver_sha256']!=old['sources']['cpp_driver_sha256']:
        raise SystemExit('baseline fixture or driver identity mismatch')
    cases=[(1024,'zarr3_gather_read'),(4096,'zarr3_gather_read'),(4096,'zarr3_sharded_read')]
    def pressure():
        return {'captured_unix':time.time(),'load_average':os.getloadavg(),
                'cpu_pressure':Path('/proc/pressure/cpu').read_text() if Path('/proc/pressure/cpu').exists() else None,
                'selected_cpu_ticks':next((x for x in Path('/proc/stat').read_text().splitlines() if x.startswith(f'cpu{cpu} ')),None)}
    report={'schema':'anndatapp.gather-candidate.v1','sources':sources,'conditions':old['conditions'],
            'shared_host_before':pressure(),'cases':{}}
    for n,case in cases:
        key=f'{n}/{case}';probe_old=old['cases'][f'{key}/cpp/probe'];timed_old=old['cases'][f'{key}/cpp/0'];python_old=old['cases'][f'{key}/python/0']
        counter=args.counter_small.resolve() if n==1024 else args.counter_large.resolve()
        result={'baseline_probe':probe_old,'baseline_cpp':timed_old,'baseline_python':python_old,'status':'running'}
        report['cases'][key]=result;atomic_json(out,report)
        probe=one(base,args.python,binary,driver,counter,n,case,'cpp','probe',True,30)
        probe['independent_output_match']=probe.get('verification',{}).get('sha256')==expected_sha(base,n,case)
        result['candidate_probe']=probe;atomic_json(out,report)
        if probe['status']!='complete' or not probe['independent_output_match']:
            result['status']='failed_preflight';atomic_json(out,report);print(key,result['status'],flush=True);break
        timed=one(base,args.python,binary,driver,counter,n,case,'cpp','0',False,30)
        timed['independent_output_match']=timed.get('verification',{}).get('sha256')==expected_sha(base,n,case)
        result['candidate_cpp']=timed
        result['status']='complete' if timed['status']=='complete' and timed['independent_output_match'] else 'failed_timed'
        atomic_json(out,report)
        print(key,result['status'],flush=True)
        if result['status']!='complete':break
    report['shared_host_after']=pressure();atomic_json(out,report)
    print('campaign',out,flush=True)


if __name__=='__main__':main()
