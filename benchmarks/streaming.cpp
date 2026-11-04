// Compare the same native caller buffers and logical blocks through the old
// select/prepare path and the reusable context. Counters are externally armed.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

import anndata;

extern "C" int anndata_probe_begin(std::uint64_t, std::uint64_t) __attribute__((weak));
extern "C" void anndata_probe_end() __attribute__((weak));
using Clock = std::chrono::steady_clock;
using namespace anndata;

static std::uint64_t hash_bytes(std::uint64_t h, const void* p, std::size_t n) {
    auto x = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ x[i])*1099511628211ULL;
    return h;
}
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
static BackedAnnData open(const std::string& format, const std::string& path) {
    BackedOptions options;
    if (format == "h5ad") return open_h5ad(path, options);
    if (format == "zarr2" || format == "zarr3") return open_zarr(path, options);
    throw std::runtime_error("format must be h5ad, zarr2 or zarr3");
}
static std::vector<Selection> selection(const std::vector<std::size_t>& start,
                                        const std::vector<std::size_t>& count) {
    std::vector<Selection> result;
    for (std::size_t i = 0; i < start.size(); ++i)
        result.emplace_back(range{std::ptrdiff_t(start[i]), std::ptrdiff_t(start[i] + count[i]), 1});
    return result;
}
int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: streaming_probe dense|adaptive|sparse FORMAT PATH ROW_BLOCK COLUMN_BLOCK");
        const std::string action = argv[1], format = argv[2], path = argv[3];
        const auto br = std::stoul(argv[4]), bc = std::stoul(argv[5]);
        if (!br || !bc) throw std::runtime_error("block dimensions must be positive");
        auto store = open(format, path);
        auto x = store.x();
        const auto shape = x.shape();
        if (shape.size() != 2) throw std::runtime_error("fixture must be rank two");
        const auto layout = x.layout();
        if ((action != "sparse") != (layout == Layout::dense)) throw std::runtime_error("fixture layout differs from action");
        if (action != "dense" && action != "adaptive" && action != "sparse") throw std::runtime_error("unknown action");
#if !ANNDATA_STREAM_CANDIDATE
        if (action == "adaptive") throw std::runtime_error("baseline has no adaptive stream");
#endif

        // Dense fixture values are float32 and sparse values are int32.
        // Index buffers hold raw bytes and are never read as uint64_t values.
        const auto capacity = layout == Layout::dense ? br*bc :
            (layout == Layout::csr ? br*shape[1] : bc*shape[0]);
        std::vector<float> dense_values(layout == Layout::dense ? capacity + 1 : 0);
        std::vector<std::int32_t> sparse_values(layout == Layout::dense ? 0 : capacity + 1);
        std::vector<std::uint64_t> indices((capacity*8 + 7)/8 + 1);
        std::vector<std::uint64_t> offsets((std::max(br, bc) + 2)*2);
        double prepare = 0, io = 0;
        std::size_t calls = 0, payload = 0;
        std::uint64_t hash = 1469598103934665603ULL;
        if (!anndata_probe_begin || !anndata_probe_end) throw std::runtime_error("bounded probe interposer is required");
        if (anndata_probe_begin(512ULL*1024*1024, 512ULL*1024*1024))
            throw std::runtime_error("could not arm bounded probe");
        auto total_start = Clock::now();
#if ANNDATA_STREAM_CANDIDATE
        auto t = Clock::now();
        auto transfer = x.prepare_transfer();
        StreamStats stream_stats{};
        prepare += elapsed(t);
#endif
        auto block = [&](std::vector<std::size_t> start, std::vector<std::size_t> count) {
            auto t = Clock::now();
#if ANNDATA_STREAM_CANDIDATE
            auto req = transfer.requirements(start, count);
#else
            auto chosen = x.select(selection(start, count));
            auto prepared = chosen.prepare_read();
            auto req = prepared.requirements();
#endif
            prepare += elapsed(t);
            t = Clock::now();
            if (layout == Layout::dense) {
                if (req.dtype != DType::float32) throw std::runtime_error("dense fixture must use float32");
                DenseBuffer b{dense_values.data(), req.dtype, count, {}};
#if ANNDATA_STREAM_CANDIDATE
                transfer.read_into(start, count, b);
#else
                prepared.read_into(b);
#endif
                io += elapsed(t);
                const auto bytes = count[0]*count[1]*dtype_size(req.dtype);
                const auto* p = dense_values.data();
                for (std::size_t i = 0; i < count[0]; ++i)
                    for (std::size_t j = 0; j < count[1]; ++j) {
                        const auto r = start[0] + i, c = start[1] + j;
                        const auto expected = float((17*r + 13*c + (r*c)%97)%1009)/1009.0f;
                        if (p[i*count[1] + j] != expected) throw std::runtime_error("dense fixture values differ");
                    }
                hash = hash_bytes(hash, dense_values.data(), bytes); payload += bytes;
            } else {
                if (req.dtype != DType::int32) throw std::runtime_error("sparse fixture must use int32");
                const auto major = layout == Layout::csr ? count[0] : count[1];
                if (req.nnz > capacity) throw std::runtime_error("sparse fixture exceeds buffer capacity");
                SparseBuffer b{sparse_values.data(), indices.data(), offsets.data(), req.dtype, req.index_dtype,
                    req.offset_dtype, capacity, capacity, major + 1};
#if ANNDATA_STREAM_CANDIDATE
                transfer.read_into(start, count, b);
#else
                prepared.read_into(b);
#endif
                io += elapsed(t);
                const auto v = req.nnz*dtype_size(req.dtype), i = req.nnz*dtype_size(req.index_dtype);
                const auto o = (major + 1)*dtype_size(req.offset_dtype);
                hash = hash_bytes(hash, sparse_values.data(), v); hash = hash_bytes(hash, indices.data(), i);
                hash = hash_bytes(hash, offsets.data(), o); payload += v + i + o;
            }
            ++calls;
        };
        if (action == "adaptive") {
#if ANNDATA_STREAM_CANDIDATE
            StreamOptions options;
            options.min_block_bytes = br*bc;
            options.target_block_bytes = br*bc*2;
            options.max_block_bytes = br*bc*4;
            options.statistics = true;
            BlockStream stream(transfer.geometry(), options);
            while (auto request = stream.next()) {
                const auto before = payload;
                const auto io_before = io;
                block(request->start, request->count);
                stream.complete(*request, payload - before, io - io_before);
            }
            if (!stream.finished()) throw std::runtime_error("adaptive stream did not finish");
            stream_stats = stream.stats();
#endif
        } else if (layout == Layout::dense) {
            for (std::size_t i = 0; i < shape[0]; i += br)
                for (std::size_t j = 0; j < shape[1]; j += bc)
                    block({i, j}, {std::min(br, shape[0] - i), std::min(bc, shape[1] - j)});
        } else if (layout == Layout::csr) {
            for (std::size_t i = 0; i < shape[0]; i += br)
                block({i, 0}, {std::min(br, shape[0] - i), shape[1]});
        } else {
            for (std::size_t j = 0; j < shape[1]; j += bc)
                block({0, j}, {shape[0], std::min(bc, shape[1] - j)});
        }
        const auto total = elapsed(total_start);
#if ANNDATA_STREAM_CANDIDATE
        const auto stats = transfer.stats();
#endif
        anndata_probe_end();
        std::cout << "{\"mode\":\"" << (action == "adaptive" ? "adaptive" : ANNDATA_STREAM_CANDIDATE ? "context" : "select")
                  << "\",\"calls\":" << calls << ",\"payload_bytes\":" << payload
                  << ",\"prepare_seconds\":" << prepare << ",\"transfer_seconds\":" << io
                  << ",\"total_seconds\":" << total << ",\"hash\":" << hash
#if ANNDATA_STREAM_CANDIDATE
                  << ",\"backend_reads\":" << stats.read_calls << ",\"backend_read_bytes\":" << stats.read_bytes
                  << ",\"scratch_allocations\":" << stats.scratch_allocations << ",\"scratch_bytes\":" << stats.scratch_bytes
                  << ",\"adaptation_trials\":" << stream_stats.trials << ",\"adaptations\":" << stream_stats.adaptations
#endif
                  << "}\n";
        return 0;
    } catch (const std::exception& e) {
        if (anndata_probe_end) anndata_probe_end();
        std::cerr << e.what() << '\n';
        return 2;
    }
}
