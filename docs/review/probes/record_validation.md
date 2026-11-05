# Backed rec-array validation probe

This probe exercises a valid HDF5 `rec-array/0.2.0` with one numeric and one
UTF-8 text field. It then writes an invalid UTF-8 byte into the text field and
checks that full `validate_all()` and full backed export report the error even
when the source session was opened with `Validation::none`. Same-mode exports
provide controls that distinguish HDF5 rec-array support from the
operation-local validation adapter.

Run from the repository root after building and installing the frozen Debug
package:

```sh
export DEPS=/home/tumlinson/anndatapp/build/dependencies/usr
export PREFIX=/home/tumlinson/anndatapp/build/thin-review-20261105/source/build/debug/installed-test/prefix
export SCRATCH=/home/tumlinson/anndatapp/build/thin-review-20261105/record-verification
export PKG_CONFIG_PATH="$DEPS/lib/x86_64-linux-gnu/pkgconfig:$DEPS/lib/pkgconfig"
cmake -S docs/review/probes -B "$SCRATCH/consumer" -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-18 \
  -DCMAKE_PREFIX_PATH="$PREFIX;$DEPS"
cmake --build "$SCRATCH/consumer" --target record_validation --parallel 2
"$SCRATCH/consumer/record_validation" valid \
  "$SCRATCH/valid.h5ad" "$SCRATCH/full-mismatch.h5ad" \
  "$SCRATCH/none-same.h5ad" "$SCRATCH/full-same.h5ad"
python3 docs/review/probes/record_validation_corrupt.py "$SCRATCH/valid.h5ad"
"$SCRATCH/consumer/record_validation" invalid \
  "$SCRATCH/valid.h5ad" "$SCRATCH/full-mismatch-invalid.h5ad" \
  "$SCRATCH/full-same-invalid.h5ad" "$SCRATCH/none-same-invalid.h5ad"
```

Each mode returns nonzero when any expectation fails, but prints every case
before returning. Valid same-mode full and none exports should preserve the
numeric and text values. Full validation of malformed text should report an
`invalid_argument` diagnostic naming `invalid UTF-8 string`; same-mode none
export should still copy the bytes. The mismatched operations are the assigned
claim under test. Build products, H5AD files, and raw logs belong in the scratch
directory; the driver and mutation helper are source-backed probe inputs.

## Frozen Debug result

The build cache names
`build/thin-review-20261105/source` as its source tree. The installed public
`anndatapp.backed.cppm` and `anndatapp.h5ad.cppm` hashes match that tree's
interfaces, and `build/debug/libanndatapp.a` and the installed static library
both hash to
`dcb67df70229b3c2df1eaeb7e7ea5a0de998b365a3de794456c74e884ba46c9f`.
The downstream driver was built with Clang 18.1.3, CMake 3.28.3, Ninja, and
HDF5 1.10.10.

Observed output:

| Case | Result |
| --- | --- |
| valid rec-array, open none, `validate_all()` | Fails with `unsupported_encoding` at `/uns/record_validation`: `record arrays are unsupported by this backend` |
| valid rec-array, open none, full export | Fails with the same unsupported-encoding diagnostic |
| valid rec-array, open none, none export | Passes; numeric and UTF-8 fields round-trip |
| valid rec-array, open full, full export | Passes; numeric and UTF-8 fields round-trip |
| malformed UTF-8, open none, `validate_all()` | Fails with the unsupported-encoding diagnostic before reporting malformed text |
| malformed UTF-8, open none, full export | Fails with the unsupported-encoding diagnostic before reporting malformed text |
| malformed UTF-8, open full, full export | Correctly reports `invalid_argument` at `/uns/record_validation[label]`: `invalid UTF-8 string` |
| malformed UTF-8, open none, none export | Passes, as expected for explicit opt-out |

The source pinpoints two dispatch gaps. `validate_all()` calls `export_store`
with `Sink`; `Sink` does not override `Backend::create_record`, whose default
throws the observed error (`src/backed_copy.cpp:191-211, 380-382`;
`src/anndatapp.backed.storage.cppm:128-146`). When an operation's validation
differs from the open option, `validation_view()` installs a `BorrowedBackend`
that also omits the record methods. `copy_record()` then calls those inherited
throwing methods for numeric/text record reads (`src/backed_copy.cpp:181-189`;
`src/backed_records.cpp:221-255`). The HDF5 backend itself implements those
methods, as shown by the successful same-mode controls (`src/backed_hdf5.cpp:
905-939, 980-1010, 1118-1159`).

The captured reproduction logs are `build/thin-review-20261105/record-verification/
{valid,invalid}.log`; the malformed source H5AD SHA-256 is
`a965fbb37b12071d5f00cbec999e71454e0e61b0709c90c2fdbc3d951bfed633`. The
frozen implementation source hashes are:

| File | SHA-256 |
| --- | --- |
| `src/backed_copy.cpp` | `99415247cbe7150f416b84ee6a76fb10218f7208ff2a33323796a7c3d3500383` |
| `src/backed_records.cpp` | `7b77a6f1abbf0d1eeb49047bbcef60c412a2522a63b0e90d5070609b96d6e728` |
| `src/anndatapp.backed.storage.cppm` | `4f9d715c704d63e4dfc0313ef6942d19c8c239b6bd15163651961b12b6ac5777` |
