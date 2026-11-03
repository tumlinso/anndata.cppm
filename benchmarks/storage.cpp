// Opt-in, process-isolated backed-storage workload. Fixture construction and
// process RSS/limits are owned by tools/storage_benchmark.py.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

import anndata;

namespace {
using namespace anndata;
using Clock = std::chrono::steady_clock;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::uint64_t hash_bytes(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ULL) {
    const auto* data = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ data[i]) * 1099511628211ULL;
    return h;
}
BackedAnnData open(std::string_view format, const std::string& path, bool writable = false) {
    BackedOptions options;
    options.mode = writable ? AccessMode::read_write : AccessMode::read_only;
    options.scratch_bytes = 64u * 1024u * 1024u;
    options.decoded_cache_bytes = 0;
    if (format == "h5ad") return open_h5ad(path, options);
    if (format == "zarr2" || format == "zarr3") return open_zarr(path, options);
    throw std::runtime_error("format must be h5ad, zarr2 or zarr3");
}
// Aligned for any supported scalar, including on implementations where byte
// vector storage does not satisfy a numerical type's alignment.
struct Buffer {
    explicit Buffer(std::size_t bytes) : words(std::max(std::size_t{1}, (bytes + 7) / 8)), size(bytes) {}
    void* data() { return words.data(); }
    std::vector<std::uint64_t> words;
    std::size_t size;
};
void require_equal(ArrayView a, ArrayView b) {
    require(a.layout() == b.layout() && a.dtype() == b.dtype() && a.rank() == b.rank(),
            "array metadata differs");
    for (std::size_t i = 0; i < a.rank(); ++i) require(a.extent(i) == b.extent(i), "array shape differs");
    if (a.layout() == Layout::dense) {
        auto x = a.bytes(), y = b.bytes();
        require(x.contiguous() && y.contiguous(), "comparison needs contiguous dense arrays");
        std::size_t count = x.element_size();
        for (std::size_t i = 0; i < x.rank(); ++i) count *= x.extent(i);
        require(!count || std::memcmp(x.data(), y.data(), count) == 0, "dense values differ");
    } else {
        require(a.index_dtype() == b.index_dtype() && a.offset_dtype() == b.offset_dtype(),
                "sparse types differ");
        auto x = a.sparse_bytes(), y = b.sparse_bytes();
        auto n = x.nnz();
        require(n == y.nnz(), "sparse nnz differs");
        const auto major = a.layout() == Layout::csr ? x.rows() : x.columns();
        require((!n || std::memcmp(x.values(), y.values(), n * x.value_size()) == 0) &&
                (!n || std::memcmp(x.indices(), y.indices(), n * x.index_size()) == 0) &&
                std::memcmp(x.offsets(), y.offsets(), (major + 1) * x.offset_size()) == 0,
                "sparse values or topology differ");
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5) throw std::runtime_error(
            "usage: storage_benchmark ACTION FORMAT PATH [DEST]");
        const std::string action(argv[1]), format(argv[2]), path(argv[3]);
        const auto open_start = Clock::now();
        auto backed = open(format, path, action == "write" || action == "reindex" || action == "structure");
        const double open_seconds = seconds(open_start);
        const auto rows = backed.n_obs(), columns = backed.n_vars();
        require(backed.has_x() && rows >= 32 && columns >= 32, "fixture needs X and at least 32 rows and columns");
        double operation = 0, prepare = 0, read_time = 0;
        std::uint64_t hash = 0;
        std::size_t payload = 0, nnz = 0;
        bool verified = true;
        unsigned threads = 1;
        if (action == "dense" || action == "huge") {
            auto selected = backed.slice(range{0, 32, 1}, range{0, 32, 1}).x();
            require(selected.layout() == Layout::dense, "dense action requires dense X");
            auto t = Clock::now();
            auto expected = selected.read();
            read_time = seconds(t);
            auto dense = expected.view().bytes();
            require(dense.contiguous(), "materialized selection must be contiguous");
            payload = 32 * 32 * dense.element_size();
            Buffer buffer(payload);
            void* const original = buffer.data();
            t = Clock::now();
            auto prepared = selected.prepare_read();
            prepare = seconds(t);
            DenseBuffer dest{buffer.data(), selected.dtype(), {32, 32}, {}};
            t = Clock::now();
            prepared.read_into(dest);
            operation = seconds(t);
            require(dest.data == original && buffer.data() == original, "destination pointer changed");
            require(std::memcmp(buffer.data(), dense.data(), payload) == 0, "dense read_into differs from read");
            hash = hash_bytes(buffer.data(), payload);
            std::memset(buffer.data(), 0xa5, payload);
            prepared.read_into(dest);
            require(dest.data == original && std::memcmp(buffer.data(), dense.data(), payload) == 0,
                    "reused dense output differs from read");
            if (action == "huge") {
                const auto* bytes = static_cast<const unsigned char*>(buffer.data());
                require(std::all_of(bytes, bytes + payload, [](unsigned char b) { return b == 0; }),
                        "huge fill fixture did not return zeros");
            }
        } else if (action == "sparse-row" || action == "sparse-column") {
            auto selected = action == "sparse-row"
                ? backed.slice(range{0, 32, 1}, all).x()
                : backed.slice(all, range{0, 32, 1}).x();
            require(selected.layout() != Layout::dense, "sparse action requires sparse X");
            auto t = Clock::now();
            auto prepared = selected.prepare_read();
            auto req = prepared.requirements();
            prepare = seconds(t);
            const auto major = req.layout == Layout::csr ? req.shape[0] : req.shape[1];
            nnz = req.nnz;
            Buffer values(nnz * dtype_size(req.dtype)), indices(nnz * dtype_size(req.index_dtype));
            Buffer offsets((major + 1) * dtype_size(req.offset_dtype));
            payload = values.size + indices.size + offsets.size;
            SparseBuffer dest{values.data(), indices.data(), offsets.data(), req.dtype, req.index_dtype,
                req.offset_dtype, nnz, nnz, major + 1};
            const auto *vptr = dest.values, *iptr = dest.indices, *optr = dest.offsets;
            t = Clock::now();
            prepared.read_into(dest);
            operation = seconds(t);
            t = Clock::now();
            auto expected = prepared.read();
            read_time = seconds(t);
            auto raw = expected.view().sparse_bytes();
            require(raw.nnz() == nnz && (!values.size || std::memcmp(values.data(), raw.values(), values.size) == 0) &&
                    (!indices.size || std::memcmp(indices.data(), raw.indices(), indices.size) == 0) &&
                    std::memcmp(offsets.data(), raw.offsets(), offsets.size) == 0,
                    "prepared sparse read_into differs from read");
            hash = hash_bytes(values.data(), values.size);
            hash = hash_bytes(indices.data(), indices.size, hash);
            hash = hash_bytes(offsets.data(), offsets.size, hash);
            if (values.size) std::memset(values.data(), 0xa5, values.size);
            if (indices.size) std::memset(indices.data(), 0xa5, indices.size);
            std::memset(offsets.data(), 0xa5, offsets.size);
            prepared.read_into(dest);
            require(dest.values == vptr && dest.indices == iptr && dest.offsets == optr &&
                    (!values.size || std::memcmp(values.data(), raw.values(), values.size) == 0) &&
                    (!indices.size || std::memcmp(indices.data(), raw.indices(), indices.size) == 0) &&
                    std::memcmp(offsets.data(), raw.offsets(), offsets.size) == 0,
                    "reused sparse output differs from read");
        } else if (action == "map") {
            auto t = Clock::now();
            auto mapped = backed.x().map_readonly();
            operation = seconds(t);
            payload = mapped.byte_size();
            require(payload != 0 && mapped.data() != nullptr, "empty mapping");
            hash = hash_bytes(mapped.data(), std::min(payload, std::size_t{4096}));
            auto tile = backed.slice(range{0, 1, 1}, range{0, 32, 1}).x().read();
            auto tile_bytes = tile.view().bytes();
            require(tile_bytes.contiguous() &&
                    std::memcmp(mapped.data(), tile_bytes.data(), 32 * tile_bytes.element_size()) == 0,
                    "mapped values differ from selected read");
        } else if (action == "parallel") {
            require(format != "h5ad" || hdf5_library_threadsafe(),
                    "parallel HDF5 workload requires thread-safe linked HDF5");
            require(backed.x().layout() == Layout::dense, "parallel action requires dense X");
            threads = 4;
            std::uint64_t hashes[4]{};
            std::exception_ptr errors[4]{};
            std::jthread workers[4];
            auto t = Clock::now();
            for (unsigned i = 0; i < threads; ++i) workers[i] = std::jthread([&, i] {
                try {
                    // Independent sessions make ownership and backend synchronization explicit.
                    auto local = open(format, path);
                    auto selected = local.slice(range{0, 32, 1}, range{0, 32, 1}).x();
                    const auto width = dtype_size(selected.dtype());
                    Buffer output(1024 * width);
                    selected.read_into(DenseBuffer{output.data(), selected.dtype(), {32, 32}, {}});
                    hashes[i] = hash_bytes(output.data(), output.size);
                } catch (...) { errors[i] = std::current_exception(); }
            });
            for (auto& worker : workers) worker.join();
            operation = seconds(t);
            for (unsigned i = 0; i < threads; ++i) {
                if (errors[i]) std::rethrow_exception(errors[i]);
                require(hashes[i] == hashes[0], "parallel reads disagree");
            }
            hash = hashes[0];
            payload = 1024 * dtype_size(backed.x().dtype());
        } else if (action == "write") {
            auto selected = backed.x().select({range{0, 32, 1}, range{0, 32, 1}});
            require(selected.layout() == Layout::dense && selected.dtype() == DType::float32,
                    "write fixture needs dense float32 X");
            auto data = Array::empty<float>({32, 32});
            auto view = data.edit().dense<float>();
            std::fill_n(view.data(), 32 * 32, 17.0f);
            auto t = Clock::now();
            selected.write(data.view());
            backed.flush();
            operation = seconds(t);
            backed.close();
            auto reopened = open(format, path);
            auto result = reopened.slice(range{0, 32, 1}, range{0, 32, 1}).x().read();
            auto result_view = result.view().dense<float>();
            require(std::all_of(result_view.data(), result_view.data() + 1024,
                                [](float x) { return x == 17.0f; }), "write verification failed");
            payload = 1024 * sizeof(float);
            hash = hash_bytes(result_view.data(), payload);
        } else if (action == "structure") {
            auto data = Array::empty<float>({rows});
            auto view = data.edit().dense<float>();
            std::fill_n(view.data(), rows, 23.0f);
            auto t = Clock::now();
            backed.set("/obs/storage_benchmark", Element(Column::numeric(std::move(data))));
            backed.flush();
            operation = seconds(t);
            backed.close();
            auto result = open(format, path);
            auto observed = result.obs().column("storage_benchmark").values().read();
            auto values = observed.view().dense<float>();
            require(values.extent(0) == rows && values.data()[0] == 23.0f && values.data()[rows - 1] == 23.0f,
                    "observation column update differs after reopening");
            payload = rows * sizeof(float);
        } else if (action == "export") {
            require(argc == 5, "export requires destination path");
            auto reference = backed.slice(range{0, 32, 1}, range{0, 32, 1}).x().read();
            auto t = Clock::now();
            if (format == "h5ad") write_h5ad(argv[4], backed.view());
            else {
                ZarrWriteOptions options;
                options.format = format == "zarr2" ? ZarrFormat::v2 : ZarrFormat::v3;
                write_zarr(argv[4], backed.view(), options);
            }
            operation = seconds(t);
            auto result = open(format, argv[4]);
            require(result.n_obs() == rows && result.n_vars() == columns,
                    "export dimensions changed");
            auto observed = result.slice(range{0, 32, 1}, range{0, 32, 1}).x().read();
            require_equal(reference.view(), observed.view());
        } else if (action == "reindex") {
            auto reference = backed.slice(range{0, 31, 1}, range{0, 32, 1}).x().read();
            auto t = Clock::now();
            auto edit = backed.edit_structure();
            edit.reindex(range{0, static_cast<std::ptrdiff_t>(rows - 1), 1}, all);
            edit.apply();
            backed.flush();
            operation = seconds(t);
            backed.close();
            auto result = open(format, path);
            require(result.n_obs() == rows - 1 && result.n_vars() == columns,
                    "reindexed dimensions differ");
            auto observed = result.slice(range{0, 31, 1}, range{0, 32, 1}).x().read();
            require_equal(reference.view(), observed.view());
        } else if (action != "open") throw std::runtime_error("unknown action");
        std::cout << "{\"status\":\"ok\",\"action\":\"" << action
                  << "\",\"format\":\"" << format << "\",\"rows\":" << rows
                  << ",\"columns\":" << columns << ",\"open_seconds\":" << open_seconds
                  << ",\"operation_seconds\":" << operation << ",\"prepare_seconds\":" << prepare
                  << ",\"materialized_read_seconds\":" << read_time << ",\"payload_bytes\":" << payload
                  << ",\"nnz\":" << nnz << ",\"hash\":" << hash
                  << ",\"threads\":" << threads
                  << ",\"verified\":" << (verified ? "true" : "false") << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
