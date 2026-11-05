#!/usr/bin/env bash
set -euo pipefail
TASK=/tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/geometry64k
P96=/tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/paired96/candidate/build-run
BUILD=/tmp/anndatapp-dense-transfer-candidate
DEPENDENCY_PREFIX=/home/tumlinson/anndatapp/build/dependencies/usr
OUT="$TASK/build-run"
if [[ -e "$OUT" ]]; then
  echo "refusing to reuse output directory: $OUT" >&2
  exit 73
fi
mkdir -p "$OUT/source" "$OUT/obj"
cp "$TASK/bulk_export_geometry64k.cpp" "$OUT/source/"
cp "$P96/profile-sources/alloc_counts.c" "$OUT/source/"
cp "$P96/profile-sources/codec_counts_limited.c" "$OUT/source/"
cp "$P96/liballoc_counts.so" "$OUT/"
/usr/bin/cc -std=gnu11 -O2 -fPIC -shared "$OUT/source/codec_counts_limited.c" \
  -I"$DEPENDENCY_PREFIX/include" -L"$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu" \
  -lblosc -lzstd -ldl -pthread -o "$OUT/libcodec_counts_limited.so"
(
  cd "$BUILD"
  /usr/bin/clang++-18 \
    -DANNDATAPP_HAS_HDF5=1 -DANNDATAPP_HAS_ZARR=1 \
    -isystem /usr/include/hdf5/serial \
    -isystem "$DEPENDENCY_PREFIX/include" \
    -O3 -DNDEBUG -std=c++20 \
    -MD -MT "$OUT/obj/bulk_export_geometry64k.cpp.o" \
    -MF "$OUT/obj/bulk_export_geometry64k.cpp.o.d" \
    @"$P96/obj/consumer.modmap" \
    -o "$OUT/obj/bulk_export_geometry64k.cpp.o" \
    -c "$OUT/source/bulk_export_geometry64k.cpp"
  /usr/bin/clang++-18 -O3 -DNDEBUG \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial:"$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu" \
    "$OUT/obj/bulk_export_geometry64k.cpp.o" \
    "$P96/libanndatapp-profiled.a" \
    /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
    /usr/lib/x86_64-linux-gnu/libz.so \
    /usr/lib/x86_64-linux-gnu/libzstd.so \
    "$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu/libblosc.so" \
    -ldl -o "$OUT/bulk_export_geometry64k"
)
cp "$P96/candidate-identity.json" "$OUT/"
cp "$P96/candidate-source.sha256" "$OUT/"
sha256sum \
  "$TASK/bulk_export_geometry64k.cpp" \
  "$P96/libanndatapp-profiled.a" \
  "$OUT/obj/bulk_export_geometry64k.cpp.o" \
  "$OUT/bulk_export_geometry64k" \
  "$OUT/source/alloc_counts.c" "$OUT/liballoc_counts.so" \
  "$OUT/source/codec_counts_limited.c" "$OUT/libcodec_counts_limited.so" \
  "$P96/obj/consumer.modmap" "$BUILD/CMakeFiles/anndatapp.dir/src/backed_zarr.cpp.o.modmap" \
  > "$OUT/build-hashes.sha256"
{
  printf 'lane=A2_budget_rectangles_profiled_driver_geometry64k\n'
  printf 'source_root=/home/tumlinson/anndatapp\nbuild_root=%s\n' "$BUILD"
  printf 'input_candidate_archive=/tmp/anndatapp-dense-transfer-candidate/libanndatapp.a\n'
  printf 'profiled_archive=%s\n' "$P96/libanndatapp-profiled.a"
  /usr/bin/clang++-18 --version
  printf 'candidate_archive_sha256='
  sha256sum "$BUILD/libanndatapp.a"
  printf 'profiled_archive_sha256='
  sha256sum "$P96/libanndatapp-profiled.a"
  printf 'consumer_modmap_sha256='
  sha256sum "$P96/obj/consumer.modmap"
  printf 'backend_modmap_sha256='
  sha256sum "$BUILD/CMakeFiles/anndatapp.dir/src/backed_zarr.cpp.o.modmap"
} > "$OUT/build-context.txt"
printf 'built output=%s\n' "$OUT"
