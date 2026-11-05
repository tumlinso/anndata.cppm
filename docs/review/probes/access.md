# Backed access correctness probe

`access.cpp` tests the frozen Debug library with (1) four independent contexts prepared serially from the same read/write session, concurrently reading X and writing four separate layer arrays for ten iterations, checking every value; and (2) a deliberately malformed Zarr v2 hierarchy whose required `/obs` and `/var` nodes are encoded as `dict`. The first is a narrow correctness test, not a race-detector or performance test. The second demonstrates a process crash in the reviewed snapshot; use a disposable process and disable core dumps.

The following reproduces the review setup using its frozen source tree after its Debug build has completed. It makes new disposable output in a fresh temporary directory, uses the CMake-generated module mapping, and never writes to the source fixture:

```sh
review_root=/home/tumlinson/anndatapp
frozen_build="$review_root/build/thin-review-20261105/source/build/debug"
probe_source="$review_root/docs/review/probes/access.cpp"
probe_run=$(mktemp -d /tmp/anndatapp-access-review.XXXXXXXX)
cd "$frozen_build"
clang++-18 -g -std=c++20 \
  @CMakeFiles/test_transfer_context.dir/tests/test_transfer_context.cpp.o.modmap \
  "$probe_source" -o "$probe_run/probe" libanndatapp.a \
  -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial:/home/tumlinson/anndatapp/build/dependencies/usr/lib/x86_64-linux-gnu \
  /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
  /usr/lib/x86_64-linux-gnu/libz.so /usr/lib/x86_64-linux-gnu/libzstd.so \
  /home/tumlinson/anndatapp/build/dependencies/usr/lib/x86_64-linux-gnu/libblosc.so -lpthread
"$probe_run/probe" v2 "$probe_run/v2.zarr"
"$probe_run/probe" v3 "$probe_run/v3.zarr"
python3 - "$probe_run/malformed.zarr" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for node in (root, root / 'obs', root / 'var'):
    node.mkdir(parents=True, exist_ok=True)
    (node / '.zgroup').write_text(json.dumps({'zarr_format': 2}))
(root / '.zattrs').write_text(json.dumps({'encoding-type': 'anndata', 'encoding-version': '0.1.0'}))
for node in (root / 'obs', root / 'var'):
    (node / '.zattrs').write_text(json.dumps({'encoding-type': 'dict', 'encoding-version': '0.1.0'}))
PY
ulimit -c 0
"$probe_run/probe" malformed "$probe_run/malformed.zarr"
```

The link paths are specific to this host and captured environment; use the same CMake-generated module map and dependency paths for another build. The frozen Debug run printed `same-session independent contexts passed v2` and `same-session independent contexts passed v3`; the malformed case exited 139. Keep probe output and stores separate from tracked repository files.
