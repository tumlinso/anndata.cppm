#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: build_candidate_driver.sh OUTPUT_DIR" >&2
  exit 64
fi

TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired384-z2
PAIRED96=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96
CAND=/tmp/anndata-dense-transfer-candidate
DEPENDENCY_PREFIX=/home/tumlinson/anndata.cppm/build/dependencies/usr
SOURCE=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/bulk_export.cpp
MODULE_MAP=$PAIRED96/candidate/build-run/obj/consumer.modmap
OUTPUT_DIR=$1

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to reuse output directory: $OUTPUT_DIR" >&2
  exit 73
fi
mkdir -p "$OUTPUT_DIR/obj" "$OUTPUT_DIR/source"
cp "$SOURCE" "$OUTPUT_DIR/source/bulk_export.cpp"
cp "$CAND/candidate-identity.json" "$OUTPUT_DIR/candidate-identity.json"
cp "$CAND/candidate-source.sha256" "$OUTPUT_DIR/candidate-source.sha256"

(
  cd "$CAND"
  /usr/bin/clang++-18 -O3 -DNDEBUG -std=c++20 \
    -MD -MT "$OUTPUT_DIR/obj/bulk_export.cpp.o" \
    -MF "$OUTPUT_DIR/obj/bulk_export.cpp.o.d" \
    @"$MODULE_MAP" \
    -o "$OUTPUT_DIR/obj/bulk_export.cpp.o" \
    -c "$SOURCE"

  /usr/bin/clang++-18 -O3 -DNDEBUG \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial:"$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu" \
    "$OUTPUT_DIR/obj/bulk_export.cpp.o" \
    "$CAND/libanndata.a" \
    /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
    /usr/lib/x86_64-linux-gnu/libz.so \
    /usr/lib/x86_64-linux-gnu/libzstd.so \
    "$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu/libblosc.so" \
    -ldl -o "$OUTPUT_DIR/bulk_export"
)

{
  printf 'lane=A2_candidate_zarr2\nsource_root=%s\nbuild_root=%s\ninput_archive=%s\n' \
    "$CAND" "$CAND" "$CAND/libanndata.a"
  /usr/bin/clang++-18 --version
  sha256sum "$CAND/libanndata.a" "$MODULE_MAP" "$SOURCE" \
    "$OUTPUT_DIR/obj/bulk_export.cpp.o" "$OUTPUT_DIR/bulk_export"
} > "$OUTPUT_DIR/build-context.txt"
sha256sum "$CAND/libanndata.a" "$MODULE_MAP" "$SOURCE" \
  "$OUTPUT_DIR/obj/bulk_export.cpp.o" "$OUTPUT_DIR/bulk_export" \
  > "$OUTPUT_DIR/build-hashes.sha256"
printf 'built candidate driver at %s\n' "$OUTPUT_DIR"
