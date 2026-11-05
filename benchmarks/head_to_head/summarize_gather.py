#!/usr/bin/env python3
"""Extract compact paired gather results from preserved raw checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def operation(record):
    m=record['measurement']
    return {k:m.get(k) for k in ('seconds','cpu_seconds','prepare_seconds','rss_before_kib','rss_after_kib',
                                 'lifetime_hwm_kib','logical_read_calls','logical_read_bytes','scratch_bytes')}


def counter(record):
    c=record['counters'];io=c['io']
    return {'read_calls':sum(io[k]['calls'] for k in ('read','pread','readv','preadv')),
            'read_bytes':c['intercepted_read_bytes'],'write_bytes':c['intercepted_write_bytes'],
            'malloc_calls':c['alloc']['malloc']['calls'],'malloc_requested_bytes':c['alloc']['malloc']['requested_bytes'],
            'read_rejections':c['read_rejections'],'write_rejections':c['write_rejections']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise SystemExit(f'refusing to overwrite {a.output}')
    raw=json.loads(a.raw.read_text()); cases={}
    for key,r in raw['cases'].items():
        if r['status']!='complete':
            cases[key]={'status':r['status']};continue
        v={'status':r['status'],'baseline_cpp':operation(r['baseline_cpp']),
           'baseline_python':operation(r['baseline_python']),'candidate_cpp':operation(r['candidate_cpp']),
           'baseline_cpp_probe':counter(r['baseline_probe']),'candidate_cpp_probe':counter(r['candidate_probe']),
           'candidate_probe_output_matches':r['candidate_probe']['independent_output_match'],
           'candidate_output_matches':r['candidate_cpp']['independent_output_match'],
           'sha256':r['candidate_cpp']['verification']['sha256']}
        v['candidate_speed_ratio']=v['baseline_cpp']['seconds']/v['candidate_cpp']['seconds']
        if key.startswith('1024/'):
            v['baseline_cpp_probe']['zstd_stream_decode_calls']=r['baseline_probe']['counters']['codec_api_stream_calls']['zstd']['decode_calls']
            v['candidate_cpp_probe']['zstd_stream_decode_calls']=r['candidate_probe']['counters']['codec_api_stream_calls']['zstd']['decode_calls']
        cases[key]=v
    out={'schema':'anndatapp.gather-read.summary.v1','raw_sha256':sha(a.raw),
         'baseline_revision':'dc946845e1457299351b0ffc4ba36ef91d49db7b','candidate_base_revision':'c159d88',
         'sources':raw['sources'],'conditions':raw['conditions'],'shared_host_before':raw['shared_host_before'],
         'shared_host_after':raw['shared_host_after'],'cases':cases,
         'limits':['One candidate and one earlier baseline observation per case; no confidence interval.',
                   'Python times are preserved from the earlier baseline screen, not rerun with the candidate.',
                   'Scratch bytes count retained buffers after the call, not temporary per-axis grouping allocations or process peak.',
                   'Native logical transfer bytes now count full codec-tile reads; they are not the user-visible selected output bytes.',
                   'I/O and codec counts are from instrumented pilots and exclude direct syscalls, mmap and hidden symbols.',
                   'HDF5 and codec versions differ between C++ and Python; OS page cache was warm.',
                   'One CPU was selected from the allowed affinity set on a shared host; other work may contend.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'summary':str(a.output),'sha256':sha(a.output),'cases':len(cases)}))


if __name__=='__main__':main()
