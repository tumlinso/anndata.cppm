#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: build_profile.sh LANE SOURCE_ROOT BUILD_ROOT INPUT_LIBANNDATA_A OUTPUT_DIR" >&2
  exit 64
fi

TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96
DEPENDENCY_PREFIX=/home/tumlinson/anndata.cppm/build/dependencies/usr
LANE=$1
SOURCE_ROOT=$(realpath "$2")
BUILD_ROOT=$(realpath "$3")
INPUT_ARCHIVE=$(realpath "$4")
OUTPUT_DIR=$5
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to reuse output directory: $OUTPUT_DIR" >&2
  exit 73
fi
mkdir -p "$OUTPUT_DIR/source/src" "$OUTPUT_DIR/obj"

cp "$SOURCE_ROOT/src/backed_zarr.cpp" "$OUTPUT_DIR/source/src/backed_zarr.cpp"
cp "$SOURCE_ROOT/src/zarr_engine.hpp" "$OUTPUT_DIR/source/src/zarr_engine.hpp"
(cd "$OUTPUT_DIR/source" && patch --batch --fuzz=0 -p1 < "$TASK/logical-profile.patch")

MODULE_DIR=$BUILD_ROOT/CMakeFiles/anndata.dir
CONSUMER_MAP=$OUTPUT_DIR/obj/consumer.modmap
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
  > "$CONSUMER_MAP"

(
  cd "$BUILD_ROOT"
  /usr/bin/clang++-18 \
    -DANNDATA_HAS_HDF5=1 -DANNDATA_HAS_ZARR=1 \
    -isystem /usr/include/hdf5/serial \
    -isystem "$DEPENDENCY_PREFIX/include" \
    -I"$SOURCE_ROOT/src" \
    -O3 -DNDEBUG -std=c++20 -fPIC -Wall -Wextra -Wpedantic \
    -MD -MT "$OUTPUT_DIR/obj/backed_zarr.cpp.o" \
    -MF "$OUTPUT_DIR/obj/backed_zarr.cpp.o.d" \
    @CMakeFiles/anndata.dir/src/backed_zarr.cpp.o.modmap \
    -o "$OUTPUT_DIR/obj/backed_zarr.cpp.o" \
    -c "$OUTPUT_DIR/source/src/backed_zarr.cpp"

  cp "$INPUT_ARCHIVE" "$OUTPUT_DIR/libanndata-profiled.a"
  llvm-ar d "$OUTPUT_DIR/libanndata-profiled.a" backed_zarr.cpp.o
  llvm-ar r "$OUTPUT_DIR/libanndata-profiled.a" "$OUTPUT_DIR/obj/backed_zarr.cpp.o"
  llvm-ranlib "$OUTPUT_DIR/libanndata-profiled.a"

  /usr/bin/clang++-18 \
    -DANNDATA_HAS_HDF5=1 -DANNDATA_HAS_ZARR=1 \
    -O3 -DNDEBUG -std=c++20 \
    -MD -MT "$OUTPUT_DIR/obj/bulk_export_paired96.cpp.o" \
    -MF "$OUTPUT_DIR/obj/bulk_export_paired96.cpp.o.d" \
    @"$CONSUMER_MAP" \
    -o "$OUTPUT_DIR/obj/bulk_export_paired96.cpp.o" \
    -c "$TASK/bulk_export_paired96.cpp"

  /usr/bin/clang++-18 -O3 -DNDEBUG \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial:"$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu" \
    "$OUTPUT_DIR/obj/bulk_export_paired96.cpp.o" \
    "$OUTPUT_DIR/libanndata-profiled.a" \
    /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
    /usr/lib/x86_64-linux-gnu/libz.so \
    /usr/lib/x86_64-linux-gnu/libzstd.so \
    "$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu/libblosc.so" \
    -ldl -o "$OUTPUT_DIR/bulk_export_paired96"
)

/usr/bin/cc -std=gnu11 -O2 -fPIC -shared \
  "$TASK/alloc_counts.c" -o "$OUTPUT_DIR/liballoc_counts.so"

sha256sum "$SOURCE_ROOT/src/backed_zarr.cpp" "$INPUT_ARCHIVE" \
  "$OUTPUT_DIR/source/src/backed_zarr.cpp" \
  "$OUTPUT_DIR/obj/backed_zarr.cpp.o" "$OUTPUT_DIR/libanndata-profiled.a" \
  "$TASK/bulk_export_paired96.cpp" "$OUTPUT_DIR/bulk_export_paired96" \
  "$TASK/alloc_counts.c" "$OUTPUT_DIR/liballoc_counts.so" \
  > "$OUTPUT_DIR/build-hashes.sha256"
{
  printf 'lane=%s\nsource_root=%s\nbuild_root=%s\ninput_archive=%s\n' \
    "$LANE" "$SOURCE_ROOT" "$BUILD_ROOT" "$INPUT_ARCHIVE"
  /usr/bin/clang++-18 --version
  sha256sum "$BUILD_ROOT/CMakeFiles/anndata.dir/src/backed_zarr.cpp.o.modmap" \
    "$CONSUMER_MAP"
} > "$OUTPUT_DIR/build-context.txt"
if [[ -f "$BUILD_ROOT/candidate-identity.json" ]]; then
  cp "$BUILD_ROOT/candidate-identity.json" "$OUTPUT_DIR/candidate-identity.json"
fi
if [[ -f "$BUILD_ROOT/candidate-source.sha256" ]]; then
  cp "$BUILD_ROOT/candidate-source.sha256" "$OUTPUT_DIR/candidate-source.sha256"
fi
printf 'built lane=%s output=%s\n' "$LANE" "$OUTPUT_DIR"
