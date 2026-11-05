#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: run_candidate_once.sh DRIVER_BUILD SOURCE_ZARR2 DESTINATION_ZARR2 RECORD_DIR" >&2
  exit 64
fi

TASK=/tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/paired384-z2
CODEC_IO_PRELOAD=/tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/libcodec_counts_limited.so
PROFILE_BUILD=$(realpath "$1")
SOURCE=$(realpath "$2")
DESTINATION=$3
RECORD_DIR=$4

[[ -x "$PROFILE_BUILD/bulk_export" ]]
[[ -f "$CODEC_IO_PRELOAD" ]]
[[ -d "$SOURCE" ]]
[[ ! -e "$DESTINATION" ]]
[[ ! -e "$RECORD_DIR" ]]
mkdir -p "$RECORD_DIR/profile-sources"

cp "$TASK/compare_zarr2.py" "$RECORD_DIR/profile-sources/"
cp "$TASK/run_candidate_once.sh" "$RECORD_DIR/profile-sources/"
cp /tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/bulk_export.cpp \
  "$RECORD_DIR/profile-sources/"
cp /tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/codec_counts_limited.c \
  "$RECORD_DIR/profile-sources/"
cp "$CODEC_IO_PRELOAD" "$RECORD_DIR/profile-sources/libcodec_counts_limited.so"
cp "$PROFILE_BUILD/build-context.txt" "$RECORD_DIR/build-context.txt"
cp "$PROFILE_BUILD/build-hashes.sha256" "$RECORD_DIR/build-hashes.sha256"
cp "$PROFILE_BUILD/candidate-identity.json" "$RECORD_DIR/candidate-identity.json"
cp "$PROFILE_BUILD/candidate-source.sha256" "$RECORD_DIR/candidate-source.sha256"

printf 'source=%s\ndestination=%s\n' "$SOURCE" "$DESTINATION" \
  > "$RECORD_DIR/run-context.txt"
printf 'source_X_shape=4096x4096 float32\nsource_X_chunks=512x512\nsource_X_codec=Blosc/LZ4\nselected_rows=[1792,2176)\nselected_columns=all 4096\nuseful_payload_bytes=6291456\nsource_scratch_bytes=67108864\nwriter_scratch_bytes=67108864\nrun_timeout_seconds=30\ninterposed_libc_read_cap_bytes=536870912\ninterposed_libc_write_cap_bytes=536870912\n' \
  >> "$RECORD_DIR/run-context.txt"
sha256sum "$SOURCE/X/.zarray" > "$RECORD_DIR/source-x-metadata.sha256"
find "$SOURCE/X" -type f -printf '%s %p\n' | sort -n \
  > "$RECORD_DIR/source-x-files-by-size.txt"
sha256sum "$PROFILE_BUILD/bulk_export" "$CODEC_IO_PRELOAD" \
  "$TASK/compare_zarr2.py" "$TASK/run_candidate_once.sh" \
  /tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/bulk_export.cpp \
  /tmp/anndatapp-storage-bulk-probe-20261103-kwCan4/codec_counts_limited.c \
  > "$RECORD_DIR/run-hashes.sha256"

set +e
/usr/bin/timeout --signal=TERM --kill-after=1s 30s \
  /usr/bin/env LD_PRELOAD="$CODEC_IO_PRELOAD" \
  "$PROFILE_BUILD/bulk_export" zarr2 "$SOURCE" "$DESTINATION" \
  > "$RECORD_DIR/run.stdout" 2> "$RECORD_DIR/run.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$RECORD_DIR/exit-code.txt"

if [[ $status -eq 0 ]]; then
  /home/tumlinson/anndatapp/.venv/bin/python \
    "$TASK/compare_zarr2.py" "$SOURCE" "$DESTINATION" \
    "$RECORD_DIR/validation.json" \
    > "$RECORD_DIR/validation.stdout" 2> "$RECORD_DIR/validation.stderr"
else
  printf 'not validated: run exit status %s (timeout=124, intercepted I/O cap failure=2 expected)\n' \
    "$status" > "$RECORD_DIR/validation-skipped.txt"
fi

if [[ -e "$DESTINATION" ]]; then
  /usr/bin/du -sb "$DESTINATION" > "$RECORD_DIR/destination-du.txt"
  find "$DESTINATION" -type f -printf '%s %p\n' | sort -n \
    > "$RECORD_DIR/destination-files.txt"
  /home/tumlinson/anndatapp/.venv/bin/python - "$DESTINATION" "$RECORD_DIR/destination.sha256" <<'PY'
import hashlib
from pathlib import Path
import sys
root=Path(sys.argv[1])
lines=[]
for path in sorted(p for p in root.rglob('*') if p.is_file()):
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
Path(sys.argv[2]).write_text("\n".join(lines)+"\n")
PY
fi
printf 'exit_status=%s record_dir=%s\n' "$status" "$RECORD_DIR"
exit "$status"
