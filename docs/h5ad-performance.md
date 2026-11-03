# H5AD performance evidence

The H5AD measurements use [h5ad_benchmark.py](../tools/h5ad_benchmark.py)
with [h5ad.cpp](../benchmarks/h5ad.cpp). The script copies the current source
tree to temporary directories, leaving the checkout untouched apart from the
explicit JSON report destination.

The report records source hashes, compiler and storage-library versions,
machine details, per-build module artifacts, consumer object files, elapsed
time, maximum resident-set size, file size, payload size, and checksums. It
compares fresh core-only and HDF5-enabled builds, first compilation of core,
facade, and H5AD importers, and the invalidation behavior after a backend or
public H5AD interface edit. The recorded run is in
[h5ad.json](../benchmarks/results/h5ad.json).

The run used Clang 18.1.3, clang-scan-deps 18.1.3, CMake 3.28.3, Ninja 1.11.1,
HDF5 1.10.10, and two Ninja jobs on an Intel Xeon E5-2698 v4 at 2.20 GHz
(Linux 6.8, x86_64). Each of the six build trials used a fresh source/build
snapshot. The host page cache was not cleared, so these are fresh-build-tree
measurements, not cold-cache measurements. Revision d4c9ecb is the Git
foundation commit; the H5AD implementation and benchmark inputs were
uncommitted, and their exact hashes are recorded in the JSON manifest.
The exact measurement-script revision is also identified separately in the
report; narrative documentation, report files, and the harness itself do not
enter the compiler-input manifest.

Median fresh library build time was 13.40 s with HDF5 disabled and 13.54 s with
it enabled. The first import builds took 0.162 s for core and 0.178 s for the
facade in the core-only build; with HDF5 enabled they took 0.211 s, 0.217 s,
and 0.223 s for core, facade, and H5AD respectively. These include consumer
compilation and linking. The HDF5-enabled importer builds did not require
consumers to parse HDF5 headers.

The final build produced 5 core-only BMIs totalling 20,150,040 bytes (19.2
MiB), and 6 HDF5-enabled BMIs totalling 22,311,556 bytes (21.3 MiB). Building
the H5AD import-and-I/O executable took a median 2.25 s. These figures include
the full interface and implementation source from the measured tree.

An implementation edit rebuilt the private H5AD implementation object in
6.52 s while preserving every public BMI and consumer object. A public H5AD
interface edit rebuilt the H5AD and facade BMIs plus the corresponding
consumer objects in 7.05 s. These builds also scan dependencies and relink
consumers. The reports distinguish AnnData library objects from actual
importing consumer objects.

I/O trials used a 4,096 × 2,048 float32 dense matrix (33,554,432 payload
bytes) and a CSR matrix with 524,288 entries (4,227,080 value/index/offset
payload bytes). Data generation and checksum traversal were outside the timed
serialization calls. Each combination has three read and three write trials;
the 48 raw observations cover no compression or gzip level 4 and native or
big-endian files. Both reads and writes used the API default
Validation::full. Maximum RSS covers the entire benchmark process, including
dataset setup and checksum work; the operation timer covers only the
read_h5ad or write_h5ad call.

| Matrix | Compression | File order | Write median (range) | Read median (range) | File bytes | Max RSS, write/read |
|---|---|---|---:|---:|---:|---:|
| Dense | none | native | 0.0344 s (0.0335–0.0433) | 0.0246 s (0.0242–0.0249) | 33,832,354 | 49,304 / 48,000 KiB |
| Dense | none | big | 0.0385 s (0.0377–0.0388) | 0.0331 s (0.0322–0.0353) | 33,832,354 | 49,304 / 48,640 KiB |
| Dense | gzip | native | 1.2928 s (1.2791–1.2951) | 0.1861 s (0.1780–0.1900) | 30,221,013 | 51,188 / 52,544 KiB |
| Dense | gzip | big | 1.3911 s (1.3795–1.4294) | 0.1854 s (0.1831–0.1868) | 30,221,071 | 51,564 / 52,524 KiB |
| CSR | none | native | 0.0105 s (0.0104–0.0106) | 0.0083 s (0.0081–0.0085) | 4,506,698 | 20,664 / 19,840 KiB |
| CSR | none | big | 0.0114 s (0.0112–0.0115) | 0.0093 s (0.0085–0.0099) | 4,506,698 | 20,664 / 19,840 KiB |
| CSR | gzip | native | 0.0982 s (0.0981–0.1027) | 0.0224 s (0.0219–0.0266) | 2,107,722 | 21,916 / 22,472 KiB |
| CSR | gzip | big | 0.1066 s (0.1052–0.1067) | 0.0237 s (0.0223–0.0240) | 2,119,443 | 22,556 / 22,472 KiB |

All eight write/read checksum groups matched across trials. The C++ benchmark
prints checksums at default stream precision, so these are sanity checks; the
independent Python interoperability suite supplies exact value verification.
GNU time's %M reports maximum resident set for each waited process. The
synthetic value distributions and one cached filesystem limit how these
numbers generalize. No speedup or broad performance claim follows from this
single-host run.

Run the complete measurement after the H5AD implementation and CMake benchmark
target are integrated:

    python3 tools/h5ad_benchmark.py --output benchmarks/results/h5ad.json --keep-builds

The defaults use three isolated build trials per configuration and three
paired I/O trials per workload. The options --repetitions, --rows, --columns,
and --per-row change those workloads. The option --keep-builds preserves
temporary source snapshots and build trees for inspection. The optional
--io-executable mode skips compilation measurements and measures I/O with a
previously built h5ad_benchmark. The --reclassify-saved-report option can
correct object classification and repeat only dependency invalidation checks
after source preservation, leaving measured I/O samples unchanged.

The operating-system page cache is not cleared. GNU time's %M reports maximum
resident set for each waited process; it is not a sum of concurrently running
processes. Builds use two Ninja jobs. No performance improvement is claimed
without paired measurements on the same host and build conditions. Reports
should state whether the checkout had implementation changes during collection
and identify each measurement source revision.
