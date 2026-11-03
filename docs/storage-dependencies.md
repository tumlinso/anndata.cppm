# Storage dependencies

The C++ core requires no storage library or Python runtime. Default builds enable HDF5 and Zarr. Storage headers, JSON and codecs remain private to implementation units.

| Dependency | Purpose | Version available for this milestone |
|---|---|---|
| HDF5 C | H5AD storage | 1.10.10 |
| zlib | Gzip/zlib chunk codecs | 1.3 |
| Zstandard | Zarr chunk codec | 1.5.5 |
| C-Blosc | Blosc/LZ4 and shuffle codecs | 1.21.5 |
| nlohmann JSON | Private Zarr metadata parsing | 3.11.3 |

On Ubuntu, system development packages can be installed with:

```sh
sudo apt-get install pkg-config libhdf5-dev zlib1g-dev libzstd-dev libblosc-dev nlohmann-json3-dev
```

HDF5, zlib and Zstandard development packages were already installed on this host. Sudo required an interactive password, so the missing packages were downloaded and extracted into an ignored private prefix instead. The effective installation commands were:

```sh
mkdir -p build/dependencies/debs
cd build/dependencies/debs
apt-get download libblosc-dev libblosc1 nlohmann-json3-dev
for package in ./*.deb; do
    dpkg-deb -x "$package" ..
done
```

The extracted prefix is `build/dependencies/usr`. CMake discovers it through `ANNDATA_DEPENDENCY_PREFIX`, which can be changed explicitly. No packages were installed into the system environment by this private-prefix procedure. The downloaded Ubuntu package versions were `libblosc-dev`/`libblosc1` 1.21.5+ds-1build1 and `nlohmann-json3-dev` 3.11.3-1.

HDF5, zlib and Zstandard discovery uses pkg-config consistently to avoid mixing a system HDF5 library with unrelated Conda runtime libraries. Installed consumers must be able to discover the corresponding binary dependencies; nlohmann JSON is required to build the implementation, not to compile importing consumers.

Python is development-only. `tests/requirements-interop.txt` pins the reference AnnData, Zarr, NumPy, pandas, SciPy, h5py and numcodecs versions. This milestone uses the existing workspace reference environment; it does not require installing Python to consume the C++ library.

For native-record interoperability, `tests/requirements-records.txt` adds
Scanpy 1.11.5 to those pins. The record milestone reused the native dependencies
above and installed Python packages only in `/tmp/anndata-record-reference`.
The effective bootstrap commands on this host were:

```sh
uv venv --allow-existing --no-project \
  --python /home/tumlinson/anndata.cppm/.venv/bin/python3 \
  --cache-dir /tmp/anndata-record-reference-uv-cache /tmp/anndata-record-reference
uv pip install --python /tmp/anndata-record-reference/bin/python \
  --cache-dir /tmp/anndata-record-reference-uv-cache pip
/tmp/anndata-record-reference/bin/python -m pip install --disable-pip-version-check \
  -r tests/requirements-records.txt
```

The actual install used `-r tests/requirements-interop.txt scanpy==1.11.5`,
which is the same pinned package request. The original workspace environment
and the system installation were unchanged. Systems with a working `venv`
module may use the shorter commands in [development.md](development.md).
