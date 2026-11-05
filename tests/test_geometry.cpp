#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

import anndatapp;

using namespace anndatapp;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void geometry(BackedAnnData& store) {
    auto x = store.x().storage_geometry();
    check(x.shape == std::vector<std::size_t>({5, 3}) && x.source_shape == x.shape, "logical and source shapes");
    check(x.data.shape == x.shape && x.axes.size() == 2 && x.axes[0].affine, "physical and affine geometry");
    auto selected = store.x().select({indices{4, 0, 4}, range{1, 3}}).storage_geometry();
    check(selected.shape == std::vector<std::size_t>({3, 2}), "selected logical shape");
    check(selected.data.shape == x.data.shape && selected.data.chunks == x.data.chunks, "selection retains physical geometry");
    check(!selected.axes[0].affine && selected.axes[1].affine && selected.axes[1].start == 1, "gather and range mapping");
    auto sparse = store.layer("counts").storage_geometry();
    check(sparse.layout == Layout::csr && sparse.source_shape == x.shape, "sparse logical geometry");
    check(sparse.data.shape == std::vector<std::size_t>{4} && sparse.indices.shape == sparse.data.shape &&
          sparse.offsets.shape == std::vector<std::size_t>{6}, "sparse physical buffers");
    check(!sparse.parallel_writes, "sparse structural writes are serial");
}
int main(int argc, char** argv) {
    try {
        const auto root = std::filesystem::path(argc > 1 ? argv[1] : "build/geometry-tests");
        std::filesystem::create_directories(root);
        auto data = AnnData::empty(5, 3);
        data.set_x(Array::zeros<float>({5, 3}));
        auto a = Array::csr<float, std::int32_t, std::int64_t>(5, 3, 4);
        auto b = a.edit().csr<float, std::int32_t, std::int64_t>();
        const std::int64_t p[]{0, 2, 2, 3, 4, 4};
        for (int i = 0; i < 6; ++i) b.unsafe_offsets()[i] = p[i];
        for (int i = 0; i < 4; ++i) { b.unsafe_indices()[i] = i % 3; b.values()[i] = float(i); }
        data.set_layer("counts", std::move(a));
#if ANNDATAPP_HAS_HDF5
        for (bool compressed : {false, true}) {
            auto path = (root / (compressed ? "chunked.h5ad" : "contiguous.h5ad")).string();
            WriteOptions options; options.compression = compressed ? Compression::gzip : Compression::none;
            write_h5ad(path, data, options); auto store = open_h5ad(path); geometry(store);
            auto g = store.x().storage_geometry();
            check(g.data.organization == (compressed ? StorageOrganization::chunked : StorageOrganization::contiguous), "HDF5 layout");
            check(!g.parallel_writes && !g.writable && g.parallel_reads == hdf5_library_threadsafe(), "HDF5 capabilities");
            if (compressed) check(g.data.chunks.size() == 2, "HDF5 chunk dimensions");
        }
#endif
#if ANNDATAPP_HAS_ZARR
        for (auto format : {ZarrFormat::v2, ZarrFormat::v3}) {
            auto path = (root / (format == ZarrFormat::v2 ? "v2.zarr" : "v3.zarr")).string();
            ZarrWriteOptions options; options.format = format;
            write_zarr(path, data, options); BackedOptions access; access.mode = AccessMode::read_write;
            auto store = open_zarr(path, access); geometry(store);
            auto g = store.x().storage_geometry();
            check(g.data.organization == StorageOrganization::chunked && g.parallel_reads && g.parallel_writes && g.writable,
                  "Zarr chunk and thread capabilities");
            check(g.data.chunks == g.data.inner_chunks && g.data.chunks == g.data.write_unit, "unsharded conflict units");
        }
#endif
        std::cout << "storage geometry passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
