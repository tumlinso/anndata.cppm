#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: run_once.sh PROFILE_BUILD_DIR SOURCE_ZARR3 DESTINATION_ZARR3 RECORD_DIR" >&2
  exit 64
fi

TASK=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/paired96
CODEC_IO_PRELOAD=/tmp/anndata-storage-bulk-probe-20261103-kwCan4/libcodec_counts_limited.so
PROFILE_BUILD=$(realpath "$1")
SOURCE=$(realpath "$2")
DESTINATION=$3
RECORD_DIR=$4

[[ -x "$PROFILE_BUILD/bulk_export_paired96" ]]
[[ -f "$PROFILE_BUILD/liballoc_counts.so" ]]
[[ -f "$CODEC_IO_PRELOAD" ]]
[[ -d "$SOURCE" ]]
[[ ! -e "$DESTINATION" ]]
[[ ! -e "$RECORD_DIR" ]]
mkdir -p "$RECORD_DIR"

printf 'source=%s\ndestination=%s\n' "$SOURCE" "$DESTINATION" > "$RECORD_DIR/run-context.txt"
printf 'source_X_shape=4096x4096 float32\nsource_X_shards=2048x4096\nsource_X_inner_chunks=256x512\nselected_rows=[2000,2096)\nselected_columns=all 4096\nuseful_payload_bytes=1572864\nrun_timeout_seconds=30\ninterposed_libc_read_cap_bytes=536870912\ninterposed_libc_write_cap_bytes=536870912\n' \
  >> "$RECORD_DIR/run-context.txt"
sha256sum "$SOURCE/X/zarr.json" > "$RECORD_DIR/source-x-metadata.sha256"
find "$SOURCE/X" -type f -printf '%s %p\n' | sort -n \
  > "$RECORD_DIR/source-x-files-by-size.txt"
sha256sum "$TASK/bulk_export_paired96.cpp" "$TASK/alloc_counts.c" \
  "$TASK/logical-profile.patch" "$TASK/compare_zarr3.py" \
  "$TASK/build_profile.sh" "$TASK/run_once.sh" "$TASK/recreate_fixture.py" \
  > "$RECORD_DIR/profile-source-hashes.sha256"
cp "$PROFILE_BUILD/build-context.txt" "$RECORD_DIR/build-context.txt"
cp "$PROFILE_BUILD/build-hashes.sha256" "$RECORD_DIR/build-hashes.sha256"
if [[ -f "$PROFILE_BUILD/candidate-identity.json" ]]; then
  cp "$PROFILE_BUILD/candidate-identity.json" "$RECORD_DIR/candidate-identity.json"
fi
if [[ -f "$PROFILE_BUILD/candidate-source.sha256" ]]; then
  cp "$PROFILE_BUILD/candidate-source.sha256" "$RECORD_DIR/candidate-source.sha256"
fi
sha256sum "$PROFILE_BUILD/bulk_export_paired96" \
  "$PROFILE_BUILD/liballoc_counts.so" "$CODEC_IO_PRELOAD" \
  > "$RECORD_DIR/binary-hashes.sha256"

set +e
/usr/bin/timeout --signal=TERM --kill-after=1s 30s \
  /usr/bin/env \
  LD_PRELOAD="$PROFILE_BUILD/liballoc_counts.so:$CODEC_IO_PRELOAD" \
  "$PROFILE_BUILD/bulk_export_paired96" "$SOURCE" "$DESTINATION" \
  > "$RECORD_DIR/run.stdout" 2> "$RECORD_DIR/run.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$RECORD_DIR/exit-code.txt"

if [[ $status -eq 0 ]]; then
  /home/tumlinson/anndata.cppm/.venv/bin/python \
    "$TASK/compare_zarr3.py" "$SOURCE" "$DESTINATION" \
    "$RECORD_DIR/validation.json" \
    > "$RECORD_DIR/validation.stdout" 2> "$RECORD_DIR/validation.stderr"
else
  printf 'not validated: run exit status %s (timeout=124, allocation-cap failure=2 expected)\n' \
    "$status" > "$RECORD_DIR/validation-skipped.txt"
fi

if [[ -e "$DESTINATION" ]]; then
  /usr/bin/du -sb "$DESTINATION" > "$RECORD_DIR/destination-du.txt"
  find "$DESTINATION" -type f -printf '%s %p\n' | sort -n \
    > "$RECORD_DIR/destination-files.txt"
fi
sha256sum "$PROFILE_BUILD/bulk_export_paired96" \
  "$PROFILE_BUILD/liballoc_counts.so" "$CODEC_IO_PRELOAD" \
  > "$RECORD_DIR/binary-hashes.sha256"
printf 'exit_status=%s record_dir=%s\n' "$status" "$RECORD_DIR"
exit "$status"
