#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: run_geometry128k_once.sh BUILD_DIR SOURCE DESTINATION RECORD_DIR" >&2
  exit 64
fi
TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/geometry128k
BUILD=$(realpath "$1")
SOURCE=$(realpath "$2")
DESTINATION=$3
RECORD=$4
[[ -x "$BUILD/bulk_export_geometry128k" ]]
[[ -f "$BUILD/liballoc_counts.so" ]]
[[ -f "$BUILD/libcodec_counts_limited.so" ]]
[[ -d "$SOURCE" ]]
[[ ! -e "$DESTINATION" ]]
[[ ! -e "$RECORD" ]]
mkdir -p "$RECORD"
printf 'source=%s\ndestination=%s\n' "$SOURCE" "$DESTINATION" > "$RECORD/run-context.txt"
printf 'source_x_shape=512x512 float32\nsource_x_inner_chunks=32x64\nsource_x_shards=256x512\nselected_rows=[0,512)\nselected_columns=all 512\nuseful_payload_bytes=1048576\nobs_var_index_chunk_labels=64\nsource_scratch_bytes=131072\nwriter_scratch_bytes=131072\nrun_timeout_seconds=30\ninterposed_libc_read_cap_bytes=536870912\ninterposed_libc_write_cap_bytes=536870912\n' >> "$RECORD/run-context.txt"
sha256sum "$SOURCE/X/zarr.json" "$SOURCE/obs/_index/zarr.json" "$SOURCE/var/_index/zarr.json" > "$RECORD/source-metadata-hashes.sha256"
find "$SOURCE" -type f -printf '%s %p\n' | sort -n > "$RECORD/source-files.txt"
find "$SOURCE" -type f -print0 | sort -z | xargs -0 sha256sum > "$RECORD/source-file-hashes.sha256"
cp "$TASK/fixture-evidence.json" "$RECORD/fixture-evidence.json"
cp "$BUILD/build-context.txt" "$RECORD/build-context.txt"
cp "$BUILD/build-hashes.sha256" "$RECORD/build-hashes.sha256"
cp "$BUILD/candidate-identity.json" "$RECORD/candidate-identity.json"
cp "$BUILD/candidate-source.sha256" "$RECORD/candidate-source.sha256"
sha256sum "$TASK/bulk_export_geometry128k.cpp" "$TASK/compare_geometry128k.py" "$TASK/run_geometry128k_once.sh" "$BUILD/bulk_export_geometry128k" "$BUILD/liballoc_counts.so" "$BUILD/libcodec_counts_limited.so" > "$RECORD/run-source-binary-hashes.sha256"
set +e
/usr/bin/timeout --signal=TERM --kill-after=1s 30s \
  /usr/bin/env \
  LD_PRELOAD="$BUILD/liballoc_counts.so:$BUILD/libcodec_counts_limited.so" \
  "$BUILD/bulk_export_geometry128k" "$SOURCE" "$DESTINATION" \
  > "$RECORD/run.stdout" 2> "$RECORD/run.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$RECORD/exit-code.txt"
if [[ $status -eq 0 ]]; then
  /home/tumlinson/anndata.cppm/.venv/bin/python \
    "$TASK/compare_geometry128k.py" "$SOURCE" "$DESTINATION" "$RECORD/validation.json" \
    > "$RECORD/validation.stdout" 2> "$RECORD/validation.stderr"
else
  printf 'not validated: run exit status %s; this was the sole authorized attempt\n' "$status" > "$RECORD/validation-skipped.txt"
fi
if [[ -e "$DESTINATION" ]]; then
  /usr/bin/du -sb "$DESTINATION" > "$RECORD/destination-du.txt"
  find "$DESTINATION" -type f -printf '%s %p\n' | sort -n > "$RECORD/destination-files.txt"
  for relative in zarr.json X/zarr.json obs/_index/zarr.json var/_index/zarr.json; do
    if [[ -f "$DESTINATION/$relative" ]]; then
      name=${relative//\//_}
      cp "$DESTINATION/$relative" "$RECORD/destination-$name"
    fi
  done
fi
sha256sum "$BUILD/bulk_export_geometry128k" "$BUILD/liballoc_counts.so" "$BUILD/libcodec_counts_limited.so" > "$RECORD/run-binary-hashes.sha256"
printf 'exit_status=%s record_dir=%s\n' "$status" "$RECORD"
exit "$status"
