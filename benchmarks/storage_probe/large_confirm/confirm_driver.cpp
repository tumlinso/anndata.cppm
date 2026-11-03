#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

import anndata;

extern "C" void bulk_probe_begin() __attribute__((weak));
extern "C" void bulk_probe_end() __attribute__((weak));

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kScratchBytes = 64u * 1024u * 1024u;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::uint64_t hash_bytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
}

struct ProbeScope {
    ProbeScope() {
        require(bulk_probe_begin != nullptr && bulk_probe_end != nullptr,
                "capped I/O interposer gate was not resolved");
        bulk_probe_begin();
        active = true;
    }
    ~ProbeScope() {
        if (active) bulk_probe_end();
    }
    void stop() {
        if (active) {
            bulk_probe_end();
            active = false;
        }
    }
    bool active = false;
};

anndata::BackedAnnData open_store(std::string_view format, const std::string& path,
                                  bool writable) {
    anndata::BackedOptions options;
    options.mode = writable ? anndata::AccessMode::read_write : anndata::AccessMode::read_only;
    options.scratch_bytes = kScratchBytes;
    options.decoded_cache_bytes = 0;
    if (format == "zarr2" || format == "zarr3") return anndata::open_zarr(path, options);
    throw std::runtime_error("format must be zarr2 or zarr3");
}

std::uint64_t read_baseline_reference_tile(anndata::BackedAnnData& store,
                                           std::size_t rows) {
    auto expected = store.slice(anndata::range{0, static_cast<std::ptrdiff_t>(rows), 1},
                                anndata::range{0, 32, 1}).x().read();
    auto bytes = expected.view().bytes();
    require(bytes.contiguous(), "baseline reference tile is not contiguous");
    const auto byte_count = rows * 32 * sizeof(float);
    return hash_bytes(bytes.data(), byte_count);
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 4 || argc == 5,
                "usage: confirm_driver export|reindex zarr2|zarr3 SOURCE [DESTINATION]");
        const std::string_view action(argv[1]);
        const std::string_view format(argv[2]);
        const std::string source(argv[3]);
        require((action == "export" && argc == 5) || (action == "reindex" && argc == 4),
                "argument count differs from action");
        require(std::filesystem::exists(source), "source fixture does not exist");
        const std::string destination = argc == 5 ? argv[4] : std::string{};
        if (action == "export") require(!std::filesystem::exists(destination),
                                        "export destination already exists");

        const auto open_start = Clock::now();
        auto store = open_store(format, source, action == "reindex");
        const double open_seconds = seconds(open_start);
        require(store.n_obs() == 4096 && store.n_vars() == 4096,
                "source dimensions differ from the frozen 4096 fixture");
        require(store.has_x() && store.x().layout() == anndata::Layout::dense &&
                store.x().dtype() == anndata::DType::float32,
                "source X differs from dense float32 fixture");

        // Match benchmarks/storage.cpp: materialize and retain a small reference
        // tile before timing. This warms some source storage and is outside the
        // I/O-cap ROI; full independent validation follows the target process.
        const std::size_t reference_rows = action == "export" ? 32 : 31;
        const auto reference_hash = read_baseline_reference_tile(store, reference_rows);

        double operation_seconds = 0.0;
        if (action == "export") {
            anndata::ZarrWriteOptions options;
            options.format = format == "zarr2" ? anndata::ZarrFormat::v2 : anndata::ZarrFormat::v3;
            options.scratch_bytes = kScratchBytes;
            ProbeScope probe;
            const auto operation_start = Clock::now();
            anndata::write_zarr(destination, store.view(), options);
            operation_seconds = seconds(operation_start);
            probe.stop();
            std::printf(
                "{\"status\":\"ok\",\"action\":\"export\",\"format\":\"%.*s\","
                "\"rows\":4096,\"columns\":4096,\"payload_bytes\":67108864,"
                "\"scratch_bytes\":67108864,\"decoded_cache_bytes\":0,"
                "\"reference_rows\":32,\"reference_columns\":32,"
                "\"reference_hash\":%llu,\"open_seconds\":%.9f,"
                "\"operation_seconds\":%.9f}\n",
                static_cast<int>(format.size()), format.data(),
                static_cast<unsigned long long>(reference_hash), open_seconds, operation_seconds);
        } else {
            ProbeScope probe;
            const auto operation_start = Clock::now();
            auto edit = store.edit_structure();
            edit.reindex(anndata::range{0, 4095, 1}, anndata::all);
            edit.apply();
            store.flush();
            operation_seconds = seconds(operation_start);
            probe.stop();
            const auto close_start = Clock::now();
            store.close();
            const double close_seconds = seconds(close_start);
            std::printf(
                "{\"status\":\"ok\",\"action\":\"reindex\",\"format\":\"%.*s\","
                "\"input_rows\":4096,\"output_rows\":4095,\"columns\":4096,"
                "\"payload_bytes\":67092480,\"scratch_bytes\":67108864,"
                "\"decoded_cache_bytes\":0,\"reference_rows\":31,"
                "\"reference_columns\":32,\"reference_hash\":%llu,"
                "\"open_seconds\":%.9f,\"operation_seconds\":%.9f,"
                "\"close_seconds_outside_roi\":%.9f}\n",
                static_cast<int>(format.size()), format.data(),
                static_cast<unsigned long long>(reference_hash), open_seconds,
                operation_seconds, close_seconds);
        }
        return 0;
    } catch (const std::exception& error) {
        if (bulk_probe_end) bulk_probe_end();
        std::fprintf(stderr, "confirm_driver: %s\n", error.what());
        return 1;
    }
}
