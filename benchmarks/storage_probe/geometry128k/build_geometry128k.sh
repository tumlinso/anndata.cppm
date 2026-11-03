#!/usr/bin/env bash
set -euo pipefail
TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k
P96=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96/candidate/build-run
BUILD=/tmp/anndata-dense-transfer-candidate
DEPENDENCY_PREFIX=/home/tumlinson/anndata.cppm/build/dependencies/usr
OUT="$TASK/build-run-v2"
CODEC_IO_PRELOAD=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/libcodec_counts_limited.so
EXPECTED_CANDIDATE_SHA=b910afcf8e0f9fc1eca3f18015943117706e639512db5cabc3ff1fd01c8dc9a6
EXPECTED_PROFILED_SHA=ccb2343c7226e9ebcfd78b3c3a04b9af530ab40ee3f717e00c233deb1e1f03f8
EXPECTED_ALLOC_SHA=740aaf7c85d6089b93210e487c268dbf1cf36930f0f352ff2398bf18834132cb
EXPECTED_CODEC_IO_SHA=45ff53af1e7d5014278965215832113f965348711a94c03a16ce657c40c8305d
[[ "$(sha256sum "$BUILD/libanndata.a" | cut -d' ' -f1)" == "$EXPECTED_CANDIDATE_SHA" ]]
[[ "$(sha256sum "$P96/libanndata-profiled.a" | cut -d' ' -f1)" == "$EXPECTED_PROFILED_SHA" ]]
[[ "$(sha256sum "$P96/liballoc_counts.so" | cut -d' ' -f1)" == "$EXPECTED_ALLOC_SHA" ]]
[[ "$(sha256sum "$CODEC_IO_PRELOAD" | cut -d' ' -f1)" == "$EXPECTED_CODEC_IO_SHA" ]]
if [[ -e "$OUT" ]]; then
  echo "refusing to reuse output directory: $OUT" >&2
  exit 73
fi
mkdir -p "$OUT/source" "$OUT/obj"
cp "$TASK/bulk_export_geometry128k.cpp" "$OUT/source/"
cp "$P96/profile-sources/alloc_counts.c" "$OUT/source/"
cp "$P96/profile-sources/codec_counts_limited.c" "$OUT/source/"
cp "$P96/liballoc_counts.so" "$OUT/"
cp "$CODEC_IO_PRELOAD" "$OUT/"
(
  cd "$BUILD"
  /usr/bin/clang++-18 \
    -DANNDATA_HAS_HDF5=1 -DANNDATA_HAS_ZARR=1 \
    -isystem /usr/include/hdf5/serial \
    -isystem "$DEPENDENCY_PREFIX/include" \
    -O3 -DNDEBUG -std=c++20 \
    -MD -MT "$OUT/obj/bulk_export_geometry128k.cpp.o" \
    -MF "$OUT/obj/bulk_export_geometry128k.cpp.o.d" \
    @"$P96/obj/consumer.modmap" \
    -o "$OUT/obj/bulk_export_geometry128k.cpp.o" \
    -c "$OUT/source/bulk_export_geometry128k.cpp"
  /usr/bin/clang++-18 -O3 -DNDEBUG \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial:"$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu" \
    "$OUT/obj/bulk_export_geometry128k.cpp.o" \
    "$P96/libanndata-profiled.a" \
    /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so \
    /usr/lib/x86_64-linux-gnu/libz.so \
    /usr/lib/x86_64-linux-gnu/libzstd.so \
    "$DEPENDENCY_PREFIX/lib/x86_64-linux-gnu/libblosc.so" \
    -ldl -o "$OUT/bulk_export_geometry128k"
)
cp "$P96/candidate-identity.json" "$OUT/"
cp "$P96/candidate-source.sha256" "$OUT/"
sha256sum \
  "$TASK/bulk_export_geometry128k.cpp" \
  "$P96/libanndata-profiled.a" \
  "$OUT/obj/bulk_export_geometry128k.cpp.o" \
  "$OUT/bulk_export_geometry128k" \
  "$OUT/source/alloc_counts.c" "$OUT/liballoc_counts.so" \
  "$OUT/source/codec_counts_limited.c" "$OUT/libcodec_counts_limited.so" \
  "$P96/obj/consumer.modmap" "$BUILD/CMakeFiles/anndata.dir/src/backed_zarr.cpp.o.modmap" \
  > "$OUT/build-hashes.sha256"
{
  printf 'lane=A2_budget_rectangles_profiled_driver_geometry128k\n'
  printf 'source_root=/home/tumlinson/anndata.cppm\nbuild_root=%s\n' "$BUILD"
  printf 'input_candidate_archive=/tmp/anndata-dense-transfer-candidate/libanndata.a\n'
  printf 'input_candidate_archive_sha256=%s\n' "$EXPECTED_CANDIDATE_SHA"
  printf 'profiled_archive=%s\n' "$P96/libanndata-profiled.a"
  printf 'codec_io_preload_source=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96/candidate/build-run/profile-sources/codec_counts_limited.c\n'
  printf 'codec_io_preload_binary=%s\n' "$CODEC_IO_PRELOAD"
  printf 'allocator_preload_sha256=%s\n' "$EXPECTED_ALLOC_SHA"
  printf 'codec_io_preload_sha256=%s\n' "$EXPECTED_CODEC_IO_SHA"
  /usr/bin/clang++-18 --version
  printf 'candidate_archive_sha256='
  sha256sum "$BUILD/libanndata.a"
  printf 'profiled_archive_sha256='
  sha256sum "$P96/libanndata-profiled.a"
  printf 'consumer_modmap_sha256='
  sha256sum "$P96/obj/consumer.modmap"
  printf 'backend_modmap_sha256='
  sha256sum "$BUILD/CMakeFiles/anndata.dir/src/backed_zarr.cpp.o.modmap"
} > "$OUT/build-context.txt"
printf 'built output=%s\n' "$OUT"
