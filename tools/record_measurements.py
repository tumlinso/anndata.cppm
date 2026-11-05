#!/usr/bin/env python3
"""Historical record campaign recovery utility, tied to this checkout and its preserved artifacts.

This is not a general benchmark: it assumes the original final2 trees and dependency
prefix, takes one sample, and mutates disposable campaign copies. See docs/v1-build.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/record-measurements'
PREFIX = '/home/tumlinson/anndatapp/build/dependencies/usr'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts(build):
    return {str(p.relative_to(build)): {'bytes': p.stat().st_size, 'sha256': sha(p),
                                       'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(build.rglob('*')) if p.is_file() and p.suffix in ('.pcm', '.o')}


def run(cmd, log):
    start = time.monotonic()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - start
    log.write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f'{cmd}: exit {p.returncode}; see {log}')
    return {'command': cmd, 'wall_seconds': elapsed, 'log': str(log), 'log_sha256': sha(log)}


def resume_final(src):
    """Finish the interrupted final2 run incrementally, keeping its original logs."""
    target = OUT / 'final2-work'
    work, build = target/'source', target/'build'
    output = OUT/'final-incremental-raw.json'
    if output.exists(): raise RuntimeError('incremental report already exists')
    baseline = OUT/'final2-review'
    changes = []
    for p in sorted((src/'src').glob('*')):
        old = baseline/'src'/p.name
        if p.is_file() and (not old.is_file() or sha(p) != sha(old)):
            changes.append(p.name)
            shutil.copy2(p, work/'src'/p.name)
    for name in ('core', 'facade'):
        with (work/'CMakeLists.txt').open('a') as f:
            f.write(f'\nadd_executable(record_current_{name} bench-probes/{name}.cpp)\n'
                    f'target_link_libraries(record_current_{name} PRIVATE anndatapp::anndatapp)\n')
    review = OUT/'final-current-review'
    if review.exists(): raise RuntimeError('current review snapshot already exists')
    shutil.copytree(work, review)
    report = {'schema': 'anndatapp.record.incremental.v1', 'changes_since_final2': changes,
              'source_sha256': {p.name: sha(p) for p in sorted((work/'src').glob('*')) if p.is_file()},
              'snapshot': str(review), 'cache': 'existing final2 build tree; OS page cache not cleared',
              'jobs': 2, 'config': 'Release (-O3 -DNDEBUG)', 'load_at_start': os.getloadavg()}
    def save(): output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    def build_target(name, log):
        return run(['cmake', '--build', str(build), '--target', name, '--parallel', '2', '--verbose'], target/log)
    save()
    report['incremental_library'] = build_target('anndatapp', 'current-library.log'); save()
    for name in ('core', 'facade'):
        report[name] = build_target('record_current_'+name, 'current-'+name+'.log'); save()
        run([str(build/('record_current_'+name))], target/('current-'+name+'-run.log'))
    report['benchmark_build'] = build_target('record_native_bench', 'current-benchmark-build.log'); save()
    report['benchmark_binary_sha256'] = sha(build/'record_native_bench')
    report['before'] = artifacts(build); save()
    impl = work/'src/records.cpp'
    with impl.open('a') as f: f.write('\nnamespace { [[maybe_unused]] constexpr int record_measurement_private_edit = 1; }\n')
    report['private_edit'] = build_target('record_current_facade', 'current-private-edit.log')
    report['after_private'] = artifacts(build); save()
    iface = work/'src/anndatapp.records.cppm'
    content = iface.read_text(); position = content.rfind('\n}')
    if position < 0: raise RuntimeError('record interface namespace marker not found')
    iface.write_text(content[:position] + '\ninline constexpr int record_measurement_public_edit = 1;\n' + content[position:])
    report['public_edit'] = build_target('record_current_facade', 'current-public-edit.log')
    report['after_public'] = artifacts(build); save()
    print(f'{output}: {sha(output)}')


def prepare_runtime(src):
    """Rebuild the runtime executable after the disposable public edit."""
    target = OUT/'final2-work'
    work, build = target/'source', target/'build'
    output = OUT/'runtime-build-raw.json'
    if output.exists(): raise RuntimeError('runtime build report already exists')
    for p in sorted((src/'src').glob('*')):
        if p.is_file():
            dst = work/'src'/p.name
            if not dst.exists() or sha(p) != sha(dst):
                shutil.copy2(p, dst)
                os.utime(dst, None)
    review = OUT/'final-runtime-review'
    if review.exists(): raise RuntimeError('runtime review snapshot already exists')
    shutil.copytree(work, review)
    report = {'schema': 'anndatapp.record.runtime-build.v1', 'snapshot': str(review),
              'source_sha256': {p.name: sha(p) for p in sorted((work/'src').glob('*')) if p.is_file()},
              'benchmark_source_sha256': sha(work/'benchmarks/records.cpp'), 'jobs': 2}
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    report['build'] = run(['cmake', '--build', str(build), '--target', 'record_native_bench',
                           '--parallel', '2', '--verbose'], target/'runtime-build.log')
    report['binary_sha256'] = sha(build/'record_native_bench')
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(f'{output}: {sha(output)}')


def prepare_candidate(src):
    """Change only record sources, preserving the previous executable and workload."""
    target = OUT/'final2-work'
    work, build = target/'source', target/'build'
    output = OUT/'candidate-build-raw.json'
    if output.exists(): raise RuntimeError('candidate build report already exists')
    before = json.loads((OUT/'runtime-build-raw.json').read_text())
    prior = OUT/'runtime-before-bench'
    if sha(prior) != before['binary_sha256']: raise RuntimeError('baseline binary identity mismatch')
    for name in ('anndatapp.records.cppm', 'records.cpp'):
        dst = work/'src'/name
        shutil.copy2(src/'src'/name, dst); os.utime(dst, None)
    changed = [p.name for p in sorted((work/'src').glob('*')) if p.is_file()
               and before['source_sha256'].get(p.name) != sha(p)]
    if set(changed) != {'anndatapp.records.cppm', 'records.cpp'}:
        raise RuntimeError(f'unexpected source delta: {changed}')
    review = OUT/'candidate-review'
    if review.exists(): raise RuntimeError('candidate snapshot already exists')
    shutil.copytree(work, review)
    report = {'schema': 'anndatapp.record.candidate-build.v1', 'snapshot': str(review),
              'changes_from_runtime': changed, 'jobs': 2,
              'source_sha256': {p.name: sha(p) for p in sorted((work/'src').glob('*')) if p.is_file()},
              'benchmark_source_sha256': sha(work/'benchmarks/records.cpp'),
              'prior_binary_sha256': sha(prior)}
    if report['benchmark_source_sha256'] != before['benchmark_source_sha256']:
        raise RuntimeError('benchmark source changed')
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    report['build'] = run(['cmake', '--build', str(build), '--target', 'record_native_bench',
                           '--parallel', '2', '--verbose'], target/'candidate-build.log')
    report['binary_sha256'] = sha(build/'record_native_bench')
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(f'{output}: {sha(output)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acknowledge-campaign-recovery', action='store_true',
                        help='acknowledge checkout-specific historical recovery and one-sample scope')
    parser.add_argument('label', choices=['baseline', 'final'])
    parser.add_argument('--source', required=True)
    parser.add_argument('--tag', help='unique artifact prefix for a fresh retry')
    parser.add_argument('--resume-final', action='store_true')
    parser.add_argument('--prepare-runtime', action='store_true')
    parser.add_argument('--prepare-candidate', action='store_true')
    args = parser.parse_args()
    if not args.acknowledge_campaign_recovery:
        parser.error('historical checkout-specific recovery only; pass --acknowledge-campaign-recovery')
    src = Path(args.source).resolve()
    if args.prepare_candidate:
        if args.label != 'final': raise ValueError('--prepare-candidate requires final')
        prepare_candidate(src)
        return
    if args.prepare_runtime:
        if args.label != 'final': raise ValueError('--prepare-runtime requires final')
        prepare_runtime(src)
        return
    if args.resume_final:
        if args.label != 'final': raise ValueError('--resume-final requires final')
        resume_final(src)
        return
    tag = args.tag or args.label
    if not tag.replace('-', '').replace('_', '').isalnum(): raise ValueError('invalid tag')
    target = OUT / (tag + '-work')
    if target.exists():
        raise RuntimeError(f'{target} already exists; preserve prior evidence')
    target.mkdir(parents=True)
    work = target / 'source'
    shutil.copytree(src, work, ignore=shutil.ignore_patterns('.git', '.todo-orchestrator', '.agents',
                    '.codex', '.aws', '.venv', 'build', 'out', '__pycache__'))
    inputs = {str(p.relative_to(work)): sha(p) for p in sorted(work.rglob('*'))
              if p.is_file() and (p.suffix in ('.cpp', '.cppm', '.cmake', '.h', '.hpp')
                                  or p.name in ('CMakeLists.txt', 'CMakePresets.json'))}
    probes = work / 'bench-probes'
    probes.mkdir()
    for name, mod in [('core', 'anndatapp.core'), ('facade', 'anndatapp')]:
        (probes / f'{name}.cpp').write_text(f'import {mod};\nint main() {{ auto a=anndatapp::AnnData::empty(2,2); a.set_x(anndatapp::Array::zeros<float>({{2,2}})); return a.x().extent(0)==2 ? 0 : 1; }}\n')
    with (work / 'CMakeLists.txt').open('a') as f:
        for name in ('core', 'facade'):
            f.write(f'\nadd_executable(record_probe_{name} bench-probes/{name}.cpp)\ntarget_link_libraries(record_probe_{name} PRIVATE anndatapp::anndatapp)\n')
        if args.label == 'final':
            f.write('\nadd_executable(record_native_bench benchmarks/records.cpp)\ntarget_link_libraries(record_native_bench PRIVATE anndatapp::anndatapp)\n')
    build = target / 'build'
    conf = ['cmake', '-S', str(work), '-B', str(build), '-G', 'Ninja',
            '-DCMAKE_CXX_COMPILER=clang++-18', '-DCMAKE_BUILD_TYPE=Release',
            '-DANNDATAPP_BUILD_TESTS=OFF', '-DANNDATAPP_BUILD_EXAMPLES=OFF',
            '-DANNDATAPP_BUILD_BENCHMARKS=OFF', '-DANNDATAPP_ENABLE_INTEROP_TESTS=OFF',
            f'-DANNDATAPP_DEPENDENCY_PREFIX={PREFIX}', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    report = {'label': args.label, 'source': str(src), 'os': platform.platform(),
              'cpu': next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), None),
              'compiler': subprocess.check_output(['clang++-18', '--version'], text=True).splitlines()[0],
              'load_at_start': os.getloadavg(), 'jobs': 2, 'config': 'Release (-O3 -DNDEBUG)',
              'cache': 'fresh build tree, OS page cache not cleared', 'samples': 1}
    report['build_input_sha256'] = inputs
    report['configure'] = run(conf, target/'configure.log')
    def build_target(name, log):
        return run(['cmake', '--build', str(build), '--target', name, '--parallel', '2', '--verbose'], target/log)
    report['library'] = build_target('anndatapp', 'library.log')
    for name in ('core', 'facade'):
        report[name] = build_target('record_probe_'+name, name+'.log')
        run([str(build/('record_probe_'+name))], target/(name+'-run.log'))
    if args.label == 'final':
        try:
            report['benchmark_build'] = build_target('record_native_bench', 'benchmark-build.log')
            report['benchmark_binary_sha256'] = sha(build/'record_native_bench')
        except RuntimeError as error:
            report['benchmark_build_error'] = str(error)
    report['before'] = artifacts(build)
    if args.label == 'final':
        impl = work/'src/records.cpp'
        iface = work/'src/anndatapp.records.cppm'
        if not impl.exists() or not iface.exists():
            raise RuntimeError('record source paths not found')
        with impl.open('a') as f: f.write('\nnamespace { [[maybe_unused]] constexpr int record_measurement_private_edit = 1; }\n')
        report['private_edit'] = build_target('record_probe_facade', 'private-edit.log')
        report['after_private'] = artifacts(build)
        content = iface.read_text()
        position = content.rfind('\n}')
        if position < 0: raise RuntimeError('record interface namespace marker not found')
        iface.write_text(content[:position] + '\ninline constexpr int record_measurement_public_edit = 1;\n' + content[position:])
        report['public_edit'] = build_target('record_probe_facade', 'public-edit.log')
        report['after_public'] = artifacts(build)
    output = OUT / (tag+'-raw.json')
    output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(f'{output}: {sha(output)}')


if __name__ == '__main__': main()
