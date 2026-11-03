# Compact representation cleanups

This implements the straightforward findings from the
[repository audit](compliance-review.md). The baseline is the preserved working
tree at the start of this step, including earlier uncommitted selection work,
not the older Git HEAD. Raw snapshots, binaries and logs remain under ignored
`build/compliance-fixes/`; historical selection/gather evidence is preserved.

## Changes

- **R1/R4:** remove unused duplicated Zarr attributes and per-chunk full metadata
  copies. Payload codecs borrow the array description, selected codec list and
  explicit scratch budget. Shard specifications stay alive through codec calls;
  original JSON and node attributes remain available for faithful rewriting.
- **R2:** packed strings retain one owner, containing the two moved vectors or
  the external lifetime token. Numerical buffers keep their addresses; shared,
  borrowed and adopted storage retain their existing contracts.
- **R3:** backed index comparisons retain one tagged table handle instead of a
  callback plus duplicate optional owners. Native names compare as views;
  backed comparisons retain their bounded staging and diagnostics.
- **R5:** remove unused category-plan budget state and keep refill counts local.
- **S1–S4/S6:** share aligned-element wrapping, group scalar-copy dispatch,
  remove unreachable empty-axis arithmetic, reuse writer coordinates and expose
  resize/nullable-copy sequencing. Scalar dtype and byte order are preserved.
- **S7:** scope the H5AD benchmark owner directly in its write branch, removing
  its unnecessary heap wrapper. Compact the example and strengthen STYLE.md's
  representation guidance without changing the protected Cellerator guide.

A short paired check exposed slower full string validation after the ownership
change. A bounded ASCII-prefix scan now reads eight bytes at a time using
`memcpy` and a byte-high-bit mask; it never reads past a string's extent. The
existing UTF-8 decoder handles the first non-ASCII word and the remainder.
No alignment directives or new dependencies were introduced. The original
slow candidate and diagnostic measurements remain preserved. The precise cause
of that slowdown was not established.

Public module sources are unchanged relative to this step's baseline. Acquired
numerical loops gain no dispatch, checks, copying or synchronization.

## Scope kept separate

R6's active metadata payload, R7's pruning-map representation, category-state
tagging and S5's shared string-reader traversal remain deferred. They require
separate ownership/representation decisions. The rejected gather-buffer trial
remains rejected. This patch does not claim complete repository compliance.

## Validation and measurements

The [validation record](../tests/results/compliance-fixes.json) contains source
hashes, commands and artifact identities. All 36 CTest groups passed across
Debug, core-only, shared, sanitizers and the selected Python semantic/H5AD
checks. Seven targeted Python storage cases passed: rich v2/v3 mutation,
independent C++ output reads/cross-format exports, all three boundary-source
formats, and malformed shard/index/codec diagnostics. Dense/CSR H5AD benchmark
write/read smoke checks also passed with equal checksums.

ASan/UBSan used the documented runner settings (`detect_leaks=0`,
`halt_on_error=1`); Zarr used the test-only async wakeup. Combined rich and
boundary checks each reached their 30-second limit once. Splitting independent
format/source cases kept each targeted Python storage case below 30 seconds without
changing fixture sizes or geometry. The full Python storage campaign was not
restarted.

Tests cover retained/adopted/shared/borrowed string
lifetimes, pointer identity, validation failure cleanup, scalar dtypes and byte
order, aligned indexes, category paging and selected numerical writes.
The writer regression includes a 3×2×1200 annotation, a 4096-byte budget and
reverse/gather observations: each row crosses the 1024-float transfer boundary.
A 1024-byte trial was rejected by the existing codec budget; the test uses a
budget that accommodates decoded chunks while still requiring multiple tiles.

Five alternating runtime pairs used Clang 18.1.3, Release `-O3`, CPU 0 and the
same eight-byte ASCII fixtures. The committed probe matches the measured
source byte-for-byte. It includes equal input copies, construction/destruction
and retained-view checks; setup, output and allocation counters are outside
timing. Each sample copies up to 64 MiB, capped at 4096 iterations.

| Strings | Validation | Baseline median | Final median | Allocations |
|---:|---|---:|---:|---:|
| 0 | full | 329 ns | 282 ns | 5 → 4 |
| 16 | full | 627 ns | 520 ns | 6 → 5 |
| 4096 | full | 39.93 µs | 17.59 µs | 6 → 5 |
| 65536 | full | 1.324 ms | 0.878 ms | 6 → 5 |
| 4096 | none | 4.41 µs | 4.21 µs | 6 → 5 |

Requested allocation bytes decreased by 48 per construction at every size.
No tested final runtime case regressed; this is a small ASCII workload, not a
whole-dataset speed claim. Unicode-heavy runtime remains unmeasured. Counts
cover ordinary throwing `new`/`new[]`, excluding aligned, C and system
allocations. Host speed varied between experiments; compare paired samples.

The isolated core/import build took 14.61 s before and 13.65 s after; peak
compiler memory was 245760 KiB in both. Two consumer recompiles took
0.77/0.79 s before and 0.70/0.71 s after. These few observations show no build
regression, not a compilation speedup. The four core BMI files totalled
19862080/19862164 bytes in the separate source-copy paths. All nine public
module sources match the baseline exactly. Touching the final private string
implementation rebuilt its object and relinked; public BMI and consumer object
hashes **and modification times stayed unchanged**. No public-interface edit
or unrelated compiler/optional-format benchmark campaign was repeated.

No dependencies were added or installed; no installation commands were needed.
