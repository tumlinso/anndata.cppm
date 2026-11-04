#!/usr/bin/env python3
"""Capture host and backend identities for the bounded comparison."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import h5py
import numcodecs.blosc
import numcodecs.zstd
import numpy
import scipy
import zarr


def run(args):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
    return {'command':args,'exit_code':result.returncode,'output':result.stdout.strip()}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv)!=3:raise SystemExit('usage: environment.py OUTPUT CPP_BINARY')
    output=Path(sys.argv[1]);binary=Path(sys.argv[2]).resolve()
    if output.exists():raise SystemExit(f'refusing to overwrite {output}')
    root=Path(__file__).resolve().parents[2]
    pkg=['anndata','numpy','pandas','scipy','h5py','zarr','numcodecs','numba','llvmlite']
    versions={key:importlib.metadata.version(key) for key in pkg}
    versions.update(python=sys.version,h5py_hdf5=h5py.version.hdf5_version,
                    numcodecs_zstd=numcodecs.zstd.__version__,numcodecs_blosc=numcodecs.blosc.VERSION_STRING)
    inputs={str(p.relative_to(root)):sha(p) for p in sorted((root/'src').glob('*')) if p.is_file()}
    report={'schema':'anndata.cppm.head-to-head.environment.v1','platform':platform.platform(),
            'allowed_cpus':sorted(os.sched_getaffinity(0)), 'cpu':run(['lscpu']),
            'filesystem':run(['stat','-f','-c','%T',str(output.parent)]),
            'load_average':os.getloadavg(),'packages':versions,'python_executable':sys.executable,
            'numpy_configuration':run([sys.executable,'-c','import numpy; numpy.show_config()']),
            'cpp_toolchain':run(['clang++-18','--version']),'cmake':run(['cmake','--version']),
            'ninja':run(['ninja','--version']),'cpp_hdf5':run(['pkg-config','--modversion','hdf5']),
            'cpp_zstd':run(['pkg-config','--modversion','libzstd']),'cpp_binary':str(binary),
            'cpp_binary_sha256':sha(binary),'cpp_shared_libraries':run(['ldd',str(binary)]),
            'python_h5py_shared_libraries':run(['ldd',h5py.h5.__file__]),
            'python_zstd_shared_libraries':run(['ldd',numcodecs.zstd.__file__]),
            'source_revision':run(['git','-C',str(root),'rev-parse','HEAD']),
            'source_changes':run(['git','-C',str(root),'diff','--','src']),
            'source_sha256':inputs,'flags':'Clang Release -O3 -DNDEBUG; driver adds -ffp-contract=off; no -march=native',
            'timing_environment':'normal unsandboxed asyncio; one CPU selected dynamically from allowed affinity; thread env=1; warm input cache'}
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'path':str(output),'sha256':sha(output),'versions':versions},indent=2))


if __name__=='__main__':main()
