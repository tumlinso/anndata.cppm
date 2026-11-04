#!/usr/bin/env python3
"""Summarize successful paired observations; preserve failed cases explicitly."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def spread(values):
    return {'median':statistics.median(values),'min':min(values),'max':max(values),'samples':values}


def payload(records):
    m=[x['measurement'] for x in records]
    def f(key):return spread([x[key] for x in m])
    return {'operation_seconds':f('seconds'),'operation_cpu_seconds':f('cpu_seconds'),
            'preparation_seconds':f('prepare_seconds'),
            'acquisition_wall_seconds':spread([x['acquisition_wall_seconds'] for x in records]),
            'rss_before_kib':f('rss_before_kib'),'rss_after_kib':f('rss_after_kib'),
            'process_hwm_kib':f('lifetime_hwm_kib'),
            'threads_before':sorted(set(x['threads_before'] for x in m)),
            'threads_after':sorted(set(x['threads_after'] for x in m))}


def counters(r):
    c=r.get('counters',{});io=c.get('io',{});alloc=c.get('alloc',{})
    return {'intercepted_read_bytes':c.get('intercepted_read_bytes'),'intercepted_write_bytes':c.get('intercepted_write_bytes'),
            'intercepted_read_calls':sum(io.get(k,{}).get('calls',0) for k in ('read','pread','readv','preadv')),
            'intercepted_write_calls':sum(io.get(k,{}).get('calls',0) for k in ('write','pwrite','writev','pwritev')),
            'malloc_calls':alloc.get('malloc',{}).get('calls'),'malloc_requested_bytes':alloc.get('malloc',{}).get('requested_bytes'),
            'codec_api_stream_calls':None,'coverage':'libc symbols only; codec hooks disabled to avoid bundled-wheel ABI conflicts; direct syscalls and hidden APIs may bypass interception'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--repetitions',type=int,default=1,choices=(1,2,3));args=p.parse_args()
    if args.output.exists():raise SystemExit(f'refusing to overwrite {args.output}')
    root=args.root.resolve();raw=root/'campaign.json';data=json.loads(raw.read_text());records=data['cases'];cases={}
    for size in (1024,4096):
        names=sorted({key.split('/')[1] for key in records if key.startswith(f'{size}/') and '/numba_fused_loop/' not in key})
        for case in names:
            key=f'{size}/{case}';out={'size':size,'case':case}
            pilots={lang:records.get(f'{key}/{lang}/probe') for lang in ('cpp','python')}
            out['pilots']={lang:({'status':v['status'],'error':v.get('error'),'counters':counters(v) if v['status']=='complete' else None} if v else None) for lang,v in pilots.items()}
            trials={lang:[records.get(f'{key}/{lang}/{i}') for i in range(args.repetitions)] for lang in ('cpp','python')}
            ok=all(x and x['status']=='complete' for v in trials.values() for x in v)
            out['complete_paired_trials']=sum(all(trials[lang][i] and trials[lang][i]['status']=='complete' for lang in ('cpp','python')) for i in range(args.repetitions))
            if ok:
                out['cpp']=payload(trials['cpp']);out['python']=payload(trials['python'])
                cpp=out['cpp']['operation_seconds'];py=out['python']['operation_seconds']
                out['python_over_cpp_median_time']=py['median']/cpp['median']
                out['classification']='cpp_faster_observed' if cpp['max']<py['min'] else 'python_faster_observed' if py['max']<cpp['min'] else 'overlapping_ranges_inconclusive'
                out['verification']={'all_independent_expected_hashes_match':all(x['independent_output_match'] for v in trials.values() for x in v),
                    'all_writes_independently_validated':all(x.get('independent_write_validation',{}).get('returncode')==0 for v in trials.values() for x in v) if 'write' in case else None}
            else:
                out['status']='incomplete_or_failed'
                out['trial_status']={lang:[({'status':x['status'],'error':x.get('error')} if x else None) for x in v] for lang,v in trials.items()}
            cases[key]=out
    numba=[records.get(f'4096/numba_fused_loop/python/{i}') for i in range(args.repetitions)]
    result={'schema':'anndata.cppm.head-to-head.summary.v1','raw_campaign_sha256':sha(raw),
            'baseline_revision':'dc946845e1457299351b0ffc4ba36ef91d49db7b','source_identities':data['sources'],
            'conditions':data['conditions'],'cases':cases,
            'numba_control':payload(numba) if all(x and x['status']=='complete' for x in numba) else {'status':'incomplete_or_unavailable'},
            'limitations':[f'{args.repetitions} paired observation(s) per case; no statistical confidence interval.',
                'Process VmHWM covers imports and setup through the operation, not operation-only allocations. VmRSS snapshots reflect resident pages and allocator retention.',
                'C++ package HDF5 1.10.10 differs from Python h5py HDF5 2.0.0; numcodecs Zstd 1.5.6 differs from native 1.5.5.',
                'I/O and allocation counters are separately instrumented pilots, not timed repetitions; interposed symbols do not cover hidden APIs or direct syscalls.',
                'OS file cache warm; fresh processes; ordinary Python AnnData and efficient low-level HDF5/Zarr paths are labeled separately.',
                'Small corpus on one system; no general package-wide performance or biology claim.',
                'Numba may specialize to the host ISA; the portable C++ Release build does not use -march=native.']}
    if all(x and x['status']=='complete' for x in numba):
        result['numba_control']['jit_compile_seconds']=spread([x['measurement']['buffer_contract']['jit_compile_seconds'] for x in numba])
        result['numba_control']['numba_import_seconds']=spread([x['measurement']['buffer_contract']['numba_import_seconds'] for x in numba])
        result['numba_control']['target_cpu']=numba[0]['measurement']['buffer_contract']['numba_target_cpu']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'summary':str(args.output),'sha256':sha(args.output),'cases':len(cases),'complete':sum('cpp' in x for x in cases.values())},indent=2))


if __name__=='__main__':main()
