# Retained gather buffer: short experiment

**Decision: retain the current production buffer.** A flat shared-array
candidate reduced allocation calls, but the short runtime comparison was mixed
and showed no import/build advantage. Its additional ownership paths do not yet
justify replacing the working representation. The tested candidate is retained
as a [patch](../benchmarks/gather-fixture/candidate.patch); this result does not
establish a general advantage for either standard containers or custom storage.

The baseline is the exact working tree after the compact-selection cleanup,
including its uncommitted changes, rather than the older Git HEAD. Baseline and
candidate sources, binaries, command logs and measurements remain under ignored
`build/gather-buffer/`. The [compact record](../benchmarks/results/gather-buffer.json)
identifies them by hashes. Earlier selection measurements remain unchanged.

## Candidate and ownership

The candidate stores `shared_ptr<const size_t[]>` instead of a shared vector,
with an explicit empty-gather tag. Raw and composed gathers allocate their
final index storage once. Normalized vectors retain their moved vector owner
through an aliasing shared pointer, preserving adoption without an extra payload
copy. Public selection calls, retained lifetimes, mutability, AnnData behavior,
and caller numerical loops are unchanged.

The C++20 array and aliasing constructors were qualified on Clang 18.1.3 with
libstdc++ 13. No dependencies were added. The canonical written Cellerator guide
was followed; its current implementation was not used as a style example.
`AGENTS.md` now references that guide and explicitly preserves fast module
imports and consumer recompilation as very high priorities for `anndata.cppm`.

## Allocation and runtime checks

Three short pairs alternated baseline/candidate order on the same CPU. Each
probe process took less than 0.3 seconds, with allocation counters disabled
during timing. Fixtures and output were excluded. Full order, pointer identity,
ownership, composed numeric values, AnnData labels and checksums were verified.
Raw/normalized construction timing samples three returned indexes; retained
reuse traverses all indexes. AnnData timing also traverses observation labels.

| Operation | Baseline allocations | Candidate allocations |
|---|---:|---:|
| Raw gather | 2 | 1 |
| Normalized vector handoff, including input-vector copy | 2 | 2 |
| Retained copy and traversal | 0 | 0 |
| Array-view gather composition | 4 | 3 |
| AnnData slice and label traversal | 14 | 14 |

Counts were unchanged across 64, 4,096 and 65,536 indexes. Requested bytes were
identical between versions for these nonempty cases; fewer allocation calls do
not establish a reduction in total requested bytes or peak memory. Counters
cover ordinary throwing C++ `new`/`delete`, not every allocator API.

Selected median observations, in microseconds per operation:

| Operation | Indexes | Baseline | Candidate |
|---|---:|---:|---:|
| Raw gather | 64 | 0.048 | 0.036 |
| Raw gather | 65,536 | 34.384 | 34.353 |
| Composition | 4,096 | 22.695 | 24.859 |
| Composition | 65,536 | 504.567 | 475.630 |
| Retained copy and traversal | 65,536 | 127.960 | 128.738 |
| AnnData slice and label traversal | 65,536 | 1,205.056 | 1,239.648 |

The 4,096-index composition was about 9.5% slower in this probe; candidate and
baseline sample ranges did not overlap there. Larger composition improved,
while other cases were close or variable. This is a bounded screening result,
not a general runtime ranking. No further profiling or timing campaign followed.

## Module boundary and compilation

The fixture compiles the actual arrays, strings, data and core module files,
then links a small consumer using `import anndata.core;`. Both versions use
Clang 18, CMake module scanning, Ninja, Release optimization and two build jobs.
Backend modules and whole-package installation are outside this comparison.

| Check | Baseline | Candidate |
|---|---:|---:|
| Fresh core and consumer build, one each | 13.242 s | 13.198 s |
| Consumer source recompilation, two-sample range | 0.655–0.672 s | 0.682–0.698 s |
| Private implementation timestamp edit | 2.342 s | 2.512 s |
| Total four public BMIs | 19,862,044 bytes | 19,862,724 bytes |

Both private edits preserved public BMI and consumer-object hashes and
timestamps; consumers only relinked. OS caches were not cleared, and source
paths differ between snapshots. These short observations establish no compile
speed improvement. Every command stayed within its 30-second cap.

## Correctness and reproduction

The candidate passed all seven core-only CTest groups, including an independent
installed consumer. Its isolated array suite also passed ASan/UBSan with leak
detection disabled for the runner's ptrace restriction. The restored baseline
passed the enhanced array suite in a fresh scanned-module build. Added empty,
composed and normalized-gather ownership regressions remain in the project.
No full storage interoperability or deep profiling campaign was rerun.

To repeat, copy the current `src/` tree into separate baseline/candidate source
directories, apply `candidate.patch` to the candidate using `patch -p1`, then run:

```sh
python3 tools/gather_benchmark.py \
  --baseline /absolute/path/to/baseline \
  --candidate /absolute/path/to/candidate \
  --out build/gather-repeat/report.json
```

The runner requires a fresh output build directory, checkpoints each command,
caps commands at 30 seconds, and stops on failure or timeout. It performs no
profiling. Compare input hashes with the compact record before interpreting
results from a changed tree. Baseline restoration refreshed source timestamps
so cached compiler modules cannot conceal the restored representation.

No installation commands were needed for this experiment.
