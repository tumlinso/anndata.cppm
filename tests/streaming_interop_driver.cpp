#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import anndatapp;

namespace {
using namespace anndatapp;
constexpr std::size_t n = 35, m = 67;

void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}

BackedAnnData open(std::string_view kind, const std::string& path, AccessMode mode) {
    BackedOptions options;
    options.mode = mode;
    options.scratch_bytes = 8 * 1024 * 1024;
    return kind.starts_with("hdf5") ? open_h5ad(path, options) : open_zarr(path, options);
}

void geometry(std::string_view kind, const BackedArray& x) {
    auto g = x.storage_geometry();
    require(g.layout == Layout::dense && g.dtype == DType::float32, "dense schema");
    require(g.shape == std::vector<std::size_t>{n, m} && g.source_shape == g.shape, "dense shape");
    require(g.axes.size() == 2 && g.axes[0].affine && g.axes[0].start == 0 && g.axes[0].step == 1,
            "full array source mapping");
    if (kind == "hdf5-contiguous") {
        require(g.data.organization == StorageOrganization::contiguous && g.data.chunks.empty(),
                "HDF5 contiguous storage geometry");
        require(!g.parallel_writes, "HDF5 must not promise parallel writes");
    } else if (kind == "hdf5-chunked" || kind == "zarr2") {
        require(g.data.organization == StorageOrganization::chunked, "chunked storage geometry");
        require(g.data.chunks == std::vector<std::size_t>{8, 16}, "chunk dimensions");
        if (kind == "zarr2") require(g.parallel_writes, "Zarr chunk write capability");
        else require(!g.parallel_writes, "HDF5 must not promise parallel writes");
    } else {
        require(g.data.organization == StorageOrganization::sharded, "sharded storage geometry");
        require(g.data.chunks == std::vector<std::size_t>{16, 32}, "shard dimensions");
        require(g.data.inner_chunks == std::vector<std::size_t>{8, 16}, "inner chunk dimensions");
        require(g.data.write_unit == std::vector<std::size_t>{16, 32}, "shard write unit");
        require(g.parallel_writes, "Zarr shard write capability");
    }
    auto selection = x.select({range{2, 30, 1}, range{3, 65, 2}}).storage_geometry();
    require(selection.source_shape == g.source_shape && selection.data.chunks == g.data.chunks,
            "selection must retain source geometry");
    require(selection.axes[0].affine && selection.axes[0].start == 2 && selection.axes[0].step == 1 &&
            selection.axes[1].affine && selection.axes[1].start == 3 && selection.axes[1].step == 2,
            "strided source mapping");
    auto gathered = x.select({indices{3, 1, 3}, all}).storage_geometry();
    require(!gathered.axes[0].affine && gathered.source_shape == g.source_shape, "gather source mapping");
}

void dense_read(const BackedArray& x) {
    auto ctx = x.prepare_transfer();
    std::vector<float> output(32, -1.0f);
    DenseBuffer target = DenseBuffer::from(output.data() + 4, {4, 5},
                                           {std::ptrdiff_t(8*sizeof(float)), -std::ptrdiff_t(sizeof(float))});
    ctx.read_into({7, 15}, {4, 5}, target);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            require(output[i*8 + 4-j] == float((7+i)*m + 15+j), "negative-stride boundary read");
    ctx.read_into({7, 15}, {4, 5}, target);
    auto selected = x.select({indices{9, 2, 9}, indices{17, 3}});
    auto gather = selected.prepare_transfer();
    float values[6]{};
    gather.read_into({0, 0}, {3, 2}, DenseBuffer::from(values, {3, 2}));
    require(values[0] == 9*m+17 && values[1] == 9*m+3 &&
            values[2] == 2*m+17 && values[3] == 2*m+3 &&
            values[4] == 9*m+17 && values[5] == 9*m+3, "gather order and repetition");
}

void sparse_read(const BackedArray& array, bool csr) {
    auto g = array.storage_geometry();
    require(g.layout == (csr ? Layout::csr : Layout::csc), "sparse orientation");
    auto ctx = array.prepare_transfer();
    const std::vector<std::size_t> start = csr ? std::vector<std::size_t>{4, 0} : std::vector<std::size_t>{0, 4};
    const std::vector<std::size_t> count = csr ? std::vector<std::size_t>{5, m} : std::vector<std::size_t>{n, 5};
    auto r = ctx.requirements(start, count);
    require(r.layout == g.layout && r.nnz == 15 && r.dtype == DType::float32 &&
            r.index_dtype == DType::int32 && r.offset_dtype == DType::int64, "sparse requirements and widths");
    std::vector<float> values(15);
    std::vector<std::int32_t> positions(15);
    std::vector<std::int64_t> offsets(6);
    SparseBuffer output{values.data(), positions.data(), offsets.data(), DType::float32,
                        DType::int32, DType::int64, values.size(), positions.size(), offsets.size()};
    ctx.read_into(start, count, output);
    for (std::size_t i = 0; i < 5; ++i) {
        auto row = i + 4, minor = csr ? m : n;
        require(offsets[i] == std::int64_t(3*i), "sparse offsets");
        require(values[3*i] == float(row+1) && values[3*i+1] == 0 && values[3*i+2] == float(row+2),
                "sparse values and explicit zero");
        require(positions[3*i] == std::int32_t(row%minor) && positions[3*i+1] == positions[3*i] &&
                positions[3*i+2] == std::int32_t((3*row+1)%minor), "sparse indices and duplicates");
    }
    require(offsets[5] == 15, "sparse final offset");
    output.values_capacity = 14;
    bool failed = false;
    try { ctx.read_into(start, count, output); } catch (const Error&) { failed = true; }
    require(failed, "undersized sparse buffer must fail");
}

void stream_read(std::string_view kind, const std::string& path, const BackedArray& x, StreamMode mode) {
    auto g = x.storage_geometry();
    if (kind.starts_with("zarr")) require(g.parallel_reads, "Zarr must expose parallel read capability");
    StreamOptions options;
    options.mode = mode;
    options.operation = StreamOperation::read;
    options.min_block_bytes = 128;
    options.target_block_bytes = 512;
    options.max_block_bytes = 1024;
    options.max_in_flight = 4;
    options.scratch_bytes_per_worker = 8 * 1024 * 1024;
    BlockStream stream(g, options);
    auto workers = g.parallel_reads ? 4u : 1u;
    if (mode == StreamMode::manual)
        require(stream.settings().in_flight_limit == workers, "backend read concurrency limit");
    std::vector<BackedAnnData> stores;
    std::vector<TransferContext> contexts;
    stores.reserve(workers); contexts.reserve(workers);
    for (std::size_t i = 0; i < workers; ++i) {
        stores.push_back(open(kind, path, AccessMode::read_write));
        contexts.push_back(stores.back().x().prepare_transfer());
    }
    std::vector<float> output(n*m, -1);
    bool saw_parallel = false;
    while (!stream.finished()) {
        std::vector<BlockRequest> requests;
        for (std::size_t i = 0; i < workers; ++i) {
            auto r = stream.next();
            if (!r) break;
            requests.push_back(std::move(*r));
        }
        require(!requests.empty(), "read stream cannot make progress");
        saw_parallel |= requests.size() > 1;
        std::barrier gate{std::ptrdiff_t(requests.size())};
        std::vector<std::future<BlockCompletion>> jobs;
        for (std::size_t i = 0; i < requests.size(); ++i) {
            jobs.push_back(std::async(std::launch::async, [&, i] {
                const auto& r = requests[i];
                gate.arrive_and_wait();
                auto begin = std::chrono::steady_clock::now();
                auto *p = output.data() + r.start[0]*m + r.start[1];
                contexts[i].read_into(r.start, r.count,
                    DenseBuffer::from(p, r.count, {std::ptrdiff_t(m*sizeof(float)), std::ptrdiff_t(sizeof(float))}));
                auto end = std::chrono::steady_clock::now();
                const auto secs = [](auto t) { return std::chrono::duration<double>(t.time_since_epoch()).count(); };
                return BlockCompletion{r.count[0]*r.count[1]*sizeof(float),
                                       std::chrono::duration<double>(end-begin).count(), secs(begin), secs(end)};
            }));
        }
        for (std::size_t i = 0; i < requests.size(); ++i) {
            auto result = jobs[i].get();
            stream.complete(requests[i], result);
        }
    }
    if (g.parallel_reads && mode == StreamMode::manual)
        require(saw_parallel, "backend produced no parallel read requests");
    for (std::size_t i = 0; i < n*m; ++i) require(output[i] == float(i), "parallel read values");
}

void stream_write(std::string_view kind, const std::string& path, const BackedArray& x) {
    auto g = x.storage_geometry();
    StreamOptions options;
    options.mode = StreamMode::manual;
    options.operation = StreamOperation::write;
    options.min_block_bytes = 128;
    options.target_block_bytes = 512;
    options.max_block_bytes = 1024;
    options.max_in_flight = 4;
    options.scratch_bytes_per_worker = 8 * 1024 * 1024;
    BlockStream stream(g, options);
    auto workers = stream.settings().in_flight_limit;
    require(workers == (kind.starts_with("hdf5") ? 1 : 4), "backend concurrency limit");
    std::vector<BackedAnnData> stores;
    std::vector<TransferContext> contexts;
    stores.reserve(workers); contexts.reserve(workers);
    for (std::size_t i = 0; i < workers; ++i) {
        stores.push_back(open(kind, path, AccessMode::read_write));
        contexts.push_back(stores.back().x().prepare_transfer());
    }
    bool saw_parallel = false;
    while (!stream.finished()) {
        std::vector<BlockRequest> requests;
        for (std::size_t i = 0; i < workers; ++i) {
            auto r = stream.next();
            if (!r) break;
            requests.push_back(std::move(*r));
        }
        require(!requests.empty(), "stream cannot make progress");
        for (const auto& r : requests)
            require(r.start.size() == 2 && r.count.size() == 2, "dense block rank");
        saw_parallel |= requests.size() > 1;
        std::barrier gate{std::ptrdiff_t(requests.size())};
        std::vector<std::future<void>> jobs;
        for (std::size_t i = 0; i < requests.size(); ++i) {
            jobs.push_back(std::async(std::launch::async, [&, i] {
                const auto& r = requests[i];
                std::vector<float> values(r.count[0]*r.count[1]);
                for (std::size_t row = 0; row < r.count[0]; ++row)
                    for (std::size_t col = 0; col < r.count[1]; ++col)
                        values[row*r.count[1] + col] = float((r.start[0]+row)*m + r.start[1]+col) + 1000;
                gate.arrive_and_wait();
                contexts[i].write_from(r.start, r.count, DenseBuffer::from(values.data(), r.count));
            }));
        }
        for (std::size_t i = 0; i < requests.size(); ++i) {
            jobs[i].get();
            stream.complete(requests[i], requests[i].count[0]*requests[i].count[1]*sizeof(float));
        }
    }
    if (!kind.starts_with("hdf5")) require(saw_parallel, "Zarr produced no parallel requests");
    if (kind.starts_with("hdf5")) for (auto& store : stores) store.flush();
}

void codec_roundtrip(const BackedArray& x) {
    auto g = x.storage_geometry();
    require(g.data.organization == StorageOrganization::chunked &&
            g.data.chunks == std::vector<std::size_t>{8, 16}, "codec chunk geometry");
    auto ctx = x.prepare_transfer();
    for (std::size_t row = 0; row < n; row += 8) for (std::size_t col = 0; col < m; col += 16) {
        const std::size_t rows = std::min<std::size_t>(8, n-row), cols = std::min<std::size_t>(16, m-col);
        std::vector<float> block(rows*cols);
        auto buffer = DenseBuffer::from(block.data(), {rows, cols});
        ctx.read_into({row, col}, {rows, cols}, buffer);
        for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < cols; ++j) {
            require(block[i*cols+j] == float((row+i)*m+col+j), "codec native read values");
            block[i*cols+j] += 1000;
        }
        ctx.write_from({row, col}, {rows, cols}, buffer);
        ctx.read_into({row, col}, {rows, cols}, buffer);
        for (std::size_t i = 0; i < rows; ++i) for (std::size_t j = 0; j < cols; ++j)
            require(block[i*cols+j] == float((row+i)*m+col+j)+1000, "codec native re-read values");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: streaming_interop_driver KIND PATH\n"; return 2; }
    try {
        const std::string kind = argv[1], path = argv[2];
        auto store = open(kind, path, AccessMode::read_write);
        if (kind.starts_with("zarr") && kind.find('-') != std::string::npos) {
            codec_roundtrip(store.x());
            return 0;
        }
        geometry(kind, store.x());
        dense_read(store.x());
        sparse_read(store.layer("csr"), true);
        sparse_read(store.layer("csc"), false);
        stream_read(kind, path, store.x(), StreamMode::manual);
        stream_read(kind, path, store.x(), StreamMode::adaptive);
        stream_write(kind, path, store.x());
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
