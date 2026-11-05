#!/usr/bin/env python3
"""Checkpoint each matched trial before validation; keep raw evidence local."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import time

CASES = (
    'h5ad_eager_read_dense', 'h5ad_eager_write_dense', 'h5py_dense_read_direct',
    'h5ad_sparse_rows_csr', 'h5ad_sparse_cols_csc', 'zarr3_sharded_read',
    'zarr3_gather_read', 'zarr3_full_write', 'zarr3_partial_write', 'numpy_fused_loop',
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def line_with_deadline(p: subprocess.Popen, deadline: float, collected: bytearray) -> bytes:
    s = selectors.DefaultSelector(); s.register(p.stdout, selectors.EVENT_READ)
    try:
        while b'\n' not in collected:
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError('30-second process phase limit')
            if not s.select(min(remaining, 0.1)):
                if p.poll() is not None: raise RuntimeError(f'process exited {p.returncode} before checkpoint')
                status = Path(f'/proc/{p.pid}/status')
                if status.exists():
                    for line in status.read_text().splitlines():
                        if line.startswith('VmRSS:') and int(line.split()[1]) > 1024*1024:
                            raise MemoryError('1 GiB process RSS limit')
                continue
            data = os.read(p.stdout.fileno(), 65536)
            if not data: raise RuntimeError(f'process closed stdout before checkpoint, status={p.poll()}')
            collected.extend(data)
        n=collected.index(b'\n')+1; result=bytes(collected[:n]); del collected[:n]; return result
    finally: s.close()


def terminate(p: subprocess.Popen) -> None:
    if p.poll() is None:
        try: os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()


def fixture(base: Path, n: int, case: str) -> Path:
    name = 'csr.h5ad' if case=='h5ad_sparse_rows_csr' else 'csc.h5ad' if case=='h5ad_sparse_cols_csc' else 'sharded.zarr' if case.startswith('zarr3_') else 'dense.h5ad'
    return base / 'fixtures' / str(n) / name


def warm(path: Path) -> int:
    """Warm every source file immediately before each trial, outside the timer."""
    files=[path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    total=0
    for item in files:
        with item.open('rb') as f:
            for block in iter(lambda:f.read(1 << 20),b''):total+=len(block)
    return total


def expected_sha(base: Path, n: int, case: str) -> str:
    x=json.loads((base/'fixtures'/str(n)/'expected.json').read_text())
    if case=='zarr3_gather_read':return x['gather']['sha256']
    if case in ('numpy_fused_loop','numba_fused_loop'):return x['fused']['sha256']
    if case=='h5ad_sparse_rows_csr':return x['sparse_csr_rows']['sha256']
    if case=='h5ad_sparse_cols_csc':return x['sparse_csc_columns']['sha256']
    if case=='zarr3_full_write':return hashlib.sha256(struct.pack('<f',7.5)*(256*512)).hexdigest()
    if case=='zarr3_partial_write':return hashlib.sha256(struct.pack('<f',7.5)*(96*128)).hexdigest()
    return x['dense_x_sha256']


def one(base: Path, python: str, binary: Path, driver: Path, counters: Path,
        n: int, case: str, language: str, label: str, instrumented: bool, deadline: int) -> dict:
    root=base/'runs'/str(n)/case/f'{language}-{label}'
    root.mkdir(parents=True, exist_ok=False)
    inp=fixture(base,n,case); output=root/'output.h5ad'
    if case in ('zarr3_full_write','zarr3_partial_write'):
        output=root/'output.zarr'; shutil.copytree(inp,output)
    warm_bytes=warm(output if case in ('zarr3_full_write','zarr3_partial_write') else inp)
    cmd=([str(binary),case,str(inp),str(output)] if language=='cpp' else [python,str(driver),case,str(inp),str(output)])
    env=os.environ.copy()
    env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'NUMEXPR_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1','BLOSC_NTHREADS':'1','PYTHONHASHSEED':'0'})
    env.pop('ANNDATAPP_INTEROP_ZARR_WAKEUP',None)
    if instrumented: env.update({'LD_PRELOAD':str(counters),'HEAD_TO_HEAD_PROBE':'1'})
    else: env.pop('LD_PRELOAD',None);env.pop('HEAD_TO_HEAD_PROBE',None)
    result={'case':case,'size':n,'language':language,'label':label,'instrumented':instrumented,
            'command':cmd,'fixture':str(inp),'output':str(output),'warmup_read_bytes':warm_bytes,'status':'starting'}
    atomic_json(root/'command.json',result)
    begin=time.monotonic();buf=bytearray()
    with (root/'stderr.log').open('wb',buffering=0) as err:
        p=subprocess.Popen(cmd, stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=env,start_new_session=True)
        try:
            line=line_with_deadline(p,begin+deadline,buf)
            measurement=json.loads(line)
            if measurement.get('phase')!='measurement': raise ValueError(f'unexpected output {measurement}')
            result.update({'measurement':measurement,'acquisition_wall_seconds':time.monotonic()-begin,'status':'measured'})
            # This durable record precedes sending the validation acknowledgement.
            atomic_json(root/'measurement.json',result)
            p.stdin.write(b'continue\n');p.stdin.flush()
            verify=json.loads(line_with_deadline(p,time.monotonic()+deadline,buf))
            if verify.get('phase')!='verification':raise ValueError(f'unexpected verification {verify}')
            result['verification']=verify
            p.stdin.close();p.wait(timeout=deadline)
            if p.returncode:raise RuntimeError(f'process returned {p.returncode}')
            result['status']='complete'
        except BaseException as e:
            result.update(status='failed',error=f'{type(e).__name__}: {e}')
            terminate(p)
        finally:
            terminate(p)
    result['process_returncode']=p.returncode
    result['stderr_sha256']=digest(root/'stderr.log')
    if instrumented:
        for line in (root/'stderr.log').read_text(errors='replace').splitlines():
            if line.startswith('ANNDATAPP_COMPLETENESS_PROBE '):
                result['counters']=json.loads(line.split(' ',1)[1])
        if 'counters' not in result: result.update(status='failed',error='missing bounded counter receipt')
        elif result['counters']['read_rejections'] or result['counters']['write_rejections']:
            result.update(status='failed',error='intercepted I/O cap reached')
    atomic_json(root/'result.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--python',required=True)
    parser.add_argument('--driver',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--counters',type=Path,required=True)
    parser.add_argument('--sizes',nargs='+',type=int,default=[1024,4096])
    parser.add_argument('--cases',nargs='+',default=list(CASES))
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--numba-control',action='store_true')
    parser.add_argument('--reuse-pilots',type=Path,help='reuse successful non-write pilot records with matching driver and fixture identities')
    parser.add_argument('--timeout-seconds',type=int,default=30)
    parser.add_argument('--cpu',type=int,help='one allowed logical CPU; defaults to first in current affinity')
    args=parser.parse_args()
    if args.repetitions<0 or args.repetitions>3 or not 0<args.timeout_seconds<=30:parser.error('maximum three repetitions and 30 seconds per phase')
    base=args.root.resolve();binary=args.binary.resolve();driver=args.driver.resolve();counters=args.counters.resolve()
    assert all(x in CASES for x in args.cases)
    allowed=sorted(os.sched_getaffinity(0));cpu=allowed[0] if args.cpu is None else args.cpu
    if cpu not in allowed:parser.error('selected CPU is outside current affinity')
    os.sched_setaffinity(0,{cpu})
    summary=base/'campaign.json';data=json.loads(summary.read_text()) if summary.exists() else {'schema':'anndatapp.head-to-head.raw.v1','cases':{},'sources':{}}
    source_dir=Path(__file__).resolve().parent
    identities={'binary_sha256':digest(binary),'python_driver_sha256':digest(driver),'counters_sha256':digest(counters),
                'runner_sha256':digest(Path(__file__)),'fixture_manifest_sha256':digest(base/'fixtures'/'fixture-manifest.json'),
                'cpp_driver_sha256':digest(source_dir/'cpp_driver.cpp'),'fixture_generator_sha256':digest(source_dir/'make_fixture.py'),
                'write_validator_sha256':digest(source_dir/'validate_writes.py'),
                'io_counters_source_sha256':digest(source_dir/'io_counters.c'),
                'base_counters_source_sha256':digest(source_dir.parent/'completeness_probe'/'counters.c')}
    for n in (1024,4096):
        p=base/'fixtures'/str(n)/'expected.json'
        if p.exists():identities[f'expected_{n}_sha256']=digest(p)
    if data['sources'] and data['sources']!=identities:raise RuntimeError('source, binary, or fixture identity changed: preserve this campaign and use a new root')
    data['sources']=identities
    conditions={'logical_cpu':cpu,'cpu_policy':'one CPU chosen from caller affinity', 'omp_threads':1,'blas_threads':1,
                'blosc_threads':1,'zarr_async_concurrency':1,'zarr_threading_max_workers':1,
                'cache':'whole input files read immediately before each trial; OS cache not dropped',
                'probe_timeout_seconds':args.timeout_seconds,'intercepted_read_cap_bytes':512<<20,'intercepted_write_cap_bytes':512<<20,
                'process_rss_limit_kib':1024*1024,'probe_scope':'instrumented pilots only; uninstrumented timing repetitions require successful pilots'}
    if data.get('conditions') and data['conditions']!=conditions:raise RuntimeError('campaign conditions changed: use a new root')
    data['conditions']=conditions
    prior=json.loads(args.reuse_pilots.read_text()) if args.reuse_pilots else None
    if prior:
        for key in ('binary_sha256','python_driver_sha256','cpp_driver_sha256','fixture_manifest_sha256','expected_1024_sha256','expected_4096_sha256'):
            if prior['sources'].get(key)!=identities.get(key):raise RuntimeError(f'prior pilot {key} does not match')
    def acquire(n,case,lang,label,probe):
        root=base/'runs'/str(n)/case/f'{lang}-{label}'
        saved=root/'result.json'
        if saved.exists():
            r=json.loads(saved.read_text())
            needs_write=case in ('h5ad_eager_write_dense','zarr3_full_write','zarr3_partial_write')
            if r['status']!='complete' or (r.get('independent_output_match') and (not needs_write or 'independent_write_validation' in r)):
                return r
        else:r=one(base,args.python,binary,driver,counters,n,case,lang,label,probe,args.timeout_seconds)
        if r['status']=='complete':
            expected=expected_sha(base,n,case); actual=r['verification'].get('sha256')
            r['independent_expected_sha256']=expected
            r['independent_output_match']=actual==expected
            if actual!=expected:r.update(status='failed',error='output hash does not match independently generated expected values')
        if r['status']=='complete' and case in ('h5ad_eager_write_dense','zarr3_full_write','zarr3_partial_write'):
            verifier=Path(__file__).resolve().parent/'validate_writes.py'
            cmd=[args.python,str(verifier),case,str(fixture(base,n,case)),r['output']]
            env=os.environ.copy()
            env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','BLOSC_NTHREADS':'1'})
            env.pop('ANNDATAPP_INTEROP_ZARR_WAKEUP',None);env.pop('LD_PRELOAD',None)
            try:
                verify=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=args.timeout_seconds,text=True)
                (root/'independent-validation.stdout').write_text(verify.stdout)
                (root/'independent-validation.stderr').write_text(verify.stderr)
                r['independent_write_validation']={'command':cmd,'returncode':verify.returncode,
                    'stdout_sha256':digest(root/'independent-validation.stdout'),'stderr_sha256':digest(root/'independent-validation.stderr')}
                if verify.returncode:r.update(status='failed',error='independent write validation failed')
            except subprocess.TimeoutExpired as e:r.update(status='failed',error=f'independent write validation timeout: {e}')
        atomic_json(saved,r)
        data['cases'][f'{n}/{case}/{lang}/{label}']=r
        atomic_json(summary,data)
        print(f'{n} {case} {lang} {label} {r["status"]}',flush=True)
        return r
    small_failed=set()
    for n in args.sizes:
        for case in args.cases:
            if n>min(args.sizes) and case in small_failed:
                print(f'{n} {case}: skipped after smaller-case failure',flush=True);continue
            keys=[f'{n}/{case}/{lang}/probe' for lang in ('cpp','python')]
            reuse=prior and 'write' not in case and all(prior['cases'].get(k,{}).get('status')=='complete' for k in keys)
            if reuse:
                probes=[]
                for key in keys:
                    value=dict(prior['cases'][key]);value['reused_from']={'campaign':str(args.reuse_pilots),
                        'campaign_sha256':digest(args.reuse_pilots),'sources':prior['sources']}
                    data['cases'][key]=value;probes.append(value)
                atomic_json(summary,data)
                print(f'{n} {case}: reused matching, successful bounded pilots',flush=True)
            else:probes=[acquire(n,case,lang,'probe',True) for lang in ('cpp','python')]
            if any(r['status']!='complete' for r in probes):
                if n==min(args.sizes):small_failed.add(case)
                continue
            if probes[0]['verification'].get('sha256')!=probes[1]['verification'].get('sha256'):
                print(f'{n} {case}: PILOT HASH MISMATCH, no timing repetitions',flush=True)
                if n==min(args.sizes):small_failed.add(case)
                continue
            if args.preflight_only:continue
            for rep in range(args.repetitions):
                order=('cpp','python') if (rep+(n==4096))%2==0 else ('python','cpp')
                trials=[acquire(n,case,lang,str(rep),False) for lang in order]
                if any(r['status']!='complete' for r in trials) or trials[0]['verification'].get('sha256')!=trials[1]['verification'].get('sha256'):
                    print(f'{n} {case}: trial failed or hashes differ; stopping case',flush=True)
                    if n==min(args.sizes):small_failed.add(case)
                    break
    if args.numba_control and 4096 in args.sizes:
        r=acquire(4096,'numba_fused_loop','python','probe',True)
        if r['status']=='complete' and not args.preflight_only:
            for rep in range(args.repetitions):
                if acquire(4096,'numba_fused_loop','python',str(rep),False)['status']!='complete':break
    print('campaign records',len(data['cases']),flush=True)


if __name__=='__main__':main()
