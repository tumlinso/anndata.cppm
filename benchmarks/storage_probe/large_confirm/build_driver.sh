#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/anndata-storage-b-confirm-20261103-mnXjYU
CAND=/tmp/anndata-dense-transfer-candidate
DEPENDENCY_PREFIX=/home/tumlinson/anndata.cppm/build/dependencies/usr
BUILD=$ROOT/build-v2

if [[ -e "$BUILD" ]]; then
  echo "refusing to reuse build directory: $BUILD" >&2
  exit 73
fi
mkdir -p "$BUILD/obj"

python3 - "$CAND/candidate-b-identity.json" "$CAND/libanndata.a" "$CAND/candidate-b-source.sha256" <<'PY'
import hashlib, json, pathlib, sys
identity_path, archive_path, manifest_path = map(pathlib.Path, sys.argv[1:])
identity = json.loads(identity_path.read_text())
expected = {
    "label": "B_destination_codec_units",
    "source_manifest_sha256": "39d5e66667103eff4de17f8e4c70b2166108a942e71c6304f8b2ad3f2377b89e",
    "archive_sha256": "213270a9e5adfa2565a97b2a761f5573fb7afcd758fc8c8f3b7d47a9f70c8299",
}
for key, value in expected.items():
    if identity.get(key) != value:
        raise SystemExit(f"Candidate B identity mismatch for {key}: {identity.get(key)!r}")
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if digest != expected["archive_sha256"]:
    raise SystemExit(f"Candidate B archive hash mismatch: {digest}")
manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if manifest_hash != expected["source_manifest_sha256"]:
    raise SystemExit(f"Candidate B source manifest hash mismatch: {manifest_hash}")
PY

cp "$CAND/candidate-b-identity.json" "$BUILD/candidate-b-identity.json"
cp "$CAND/candidate-b-source.sha256" "$BUILD/candidate-b-source.sha256"
cp "$CAND/libanndata.a" "$BUILD/input-libanndata.a"
cp "$ROOT/confirm_driver.cpp" "$BUILD/confirm_driver.cpp"
cp "$ROOT/codec_counts_limited_256.c" "$BUILD/codec_counts_limited_256.c"

MODULE_DIR=$CAND/CMakeFiles/anndata.dir
MODULE_MAP=$BUILD/obj/consumer.modmap
printf '%s\n' \
  "-fmodule-file=anndata=$MODULE_DIR/anndata.pcm" \
  "-fmodule-file=anndata.core=$MODULE_DIR/anndata.core.pcm" \
  "-fmodule-file=anndata.backed=$MODULE_DIR/anndata.backed.pcm" \
  "-fmodule-file=anndata.h5ad=$MODULE_DIR/anndata.h5ad.pcm" \
  "-fmodule-file=anndata.zarr=$MODULE_DIR/anndata.zarr.pcm" \
  "-fmodule-file=anndata.core:arrays=$MODULE_DIR/anndata.core-arrays.pcm" \
  "-fmodule-file=anndata.core:data=$MODULE_DIR/anndata.core-data.pcm" \
  "-fmodule-file=anndata.core:strings=$MODULE_DIR/anndata.core-strings.pcm" \
  "-fmodule-file=anndata.backed:storage=$MODULE_DIR/anndata.backed-storage.pcm" \
  > "$MODULE_MAP"

(
  cd "$CAND"
  /usr/bin/clang++-18 \
    -DANNDATA_HAS_HDF5=1 -DANNDATA_HAS_ZARR=1 \
    -isystem /usr/include/hdf5/serial \
    -isystem "$DEPENDENCY_PREFIX/include" \
    -O3 -DNDEBUG -std=c++20 \
    -MD -MT "$BUILD/obj/confirm_driver.cpp.o" \
    -MF "$BUILD/obj/confirm_driver.cpp.o.d" \
    @"$MODULE_MAP" \
    -o "$BUILD/obj/confirm_driver.cpp.o" \
    -c "$BUILD/confirm_driver.cpp"

  /usr/bin/clang++-18 -O3 -DNDEBUG \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial:"$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu" \
    "$BUILD/obj/confirm_driver.cpp.o" \
    "$CAND/libanndata.a" \
    /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
    /usr/lib/x86_64-linux-gnu/libz.so \
    /usr/lib/x86_64-linux-gnu/libzstd.so \
    "$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu/libblosc.so" \
    -ldl -o "$BUILD/confirm_driver"
)

/usr/bin/cc -std=gnu11 -O2 -fPIC -shared \
  -I"$DEPENDENCY_PREFIX/include" \
  "$BUILD/codec_counts_limited_256.c" -ldl \
  "$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu/libblosc.so" \
  /usr/lib/x86_64-linux-gnu/libzstd.so \
  -Wl,-rpath,"$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu" \
  -o "$BUILD/libcodec_counts_limited_256.so"

sha256sum \
  "$BUILD/candidate-b-identity.json" \
  "$BUILD/candidate-b-source.sha256" \
  "$BUILD/input-libanndata.a" \
  "$BUILD/confirm_driver.cpp" \
  "$BUILD/obj/consumer.modmap" \
  "$BUILD/obj/confirm_driver.cpp.o" \
  "$BUILD/confirm_driver" \
  "$BUILD/codec_counts_limited_256.c" \
  "$BUILD/libcodec_counts_limited_256.so" \
  > "$BUILD/build-hashes.sha256"
{
  printf 'candidate_label=B_destination_codec_units\n'
  printf 'candidate_source_manifest_sha256=39d5e66667103eff4de17f8e4c70b2166108a942e71c6304f8b2ad3f2377b89e\n'
  printf 'candidate_archive_sha256=213270a9e5adfa2565a97b2a761f5573fb7afcd758fc8c8f3b7d47a9f70c8299\n'
  printf 'scratch_bytes=67108864\nintercepted_read_cap_bytes=268435456\nintercepted_write_cap_bytes=268435456\n'
  /usr/bin/clang++-18 --version
  sha256sum "$MODULE_MAP" "$BUILD/confirm_driver.cpp" "$BUILD/confirm_driver" \
    "$BUILD/codec_counts_limited_256.c" "$BUILD/libcodec_counts_limited_256.so"
} > "$BUILD/build-context.txt"
printf 'prepared B confirmation build at %s\n' "$BUILD"
