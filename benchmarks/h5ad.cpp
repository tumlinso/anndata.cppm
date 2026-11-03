// Small end-to-end H5AD workload for the opt-in benchmark target.
import anndata;

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {
using Clock = std::chrono::steady_clock;

std::size_t parse_size(std::string_view value, std::size_t fallback) {
    if (value.empty()) return fallback;
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size() || result == 0)
        throw std::runtime_error("dimension must be a positive integer");
    return result;
}

std::uint32_t next_value(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

anndata::AnnData make_dense(std::size_t rows, std::size_t columns) {
    auto result = anndata::AnnData::empty(rows, columns);
    auto values = anndata::Array::empty<float>({rows, columns});
    auto view = values.edit().dense<float>();
    auto* data = view.data();
    std::uint32_t state = 0x41f2ac53u;
    for (std::size_t i = 0, count = rows * columns; i < count; ++i) {
        const auto bits = next_value(state) >> 8;
        data[i] = static_cast<float>(bits) * (1.0f / 16777216.0f);
    }
    result.set_x(std::move(values));
    return result;
}

anndata::AnnData make_csr(std::size_t rows, std::size_t columns, std::size_t per_row) {
    const auto nnz = rows * per_row;
    auto result = anndata::AnnData::empty(rows, columns);
    auto values = anndata::Array::csr<float, std::uint32_t, std::uint64_t>(rows, columns, nnz);
    auto view = values.edit().csr<float, std::uint32_t, std::uint64_t>();
    std::uint32_t state = 0x86b4d20fu;
    auto* data = view.values();
    auto* indices = view.unsafe_indices();
    auto* offsets = view.unsafe_offsets();
    for (std::size_t row = 0; row <= rows; ++row)
        offsets[row] = static_cast<std::uint64_t>(row * per_row);
    for (std::size_t i = 0; i < nnz; ++i) {
        data[i] = static_cast<float>(next_value(state) >> 8) * (1.0f / 16777216.0f);
        indices[i] = static_cast<std::uint32_t>((i * 37 + 11) % columns);
    }
    result.set_x(std::move(values));
    return result;
}

double checksum(anndata::ArrayView values) {
    double sum = 0.0;
    if (values.layout() == anndata::Layout::dense) {
        auto typed = values.dense<float>();
        for (std::size_t row = 0; row < typed.extent(0); ++row)
            for (std::size_t column = 0; column < typed.extent(1); ++column)
                sum += typed.data()[static_cast<std::ptrdiff_t>(row) * typed.stride(0)
                                    + static_cast<std::ptrdiff_t>(column) * typed.stride(1)];
    } else {
        auto typed = values.csr<float, std::uint32_t, std::uint64_t>();
        for (std::size_t i = 0; i < typed.nnz(); ++i) sum += typed.values()[i];
    }
    return sum;
}

std::size_t payload_bytes(anndata::ArrayView values) {
    if (values.layout() == anndata::Layout::dense)
        return values.extent(0) * values.extent(1) * sizeof(float);
    auto typed = values.csr<float, std::uint32_t, std::uint64_t>();
    return typed.nnz() * (sizeof(float) + sizeof(std::uint32_t))
        + (typed.rows() + 1) * sizeof(std::uint64_t);
}

anndata::WriteOptions write_options(std::string_view compression,
                                    std::string_view byte_order) {
    anndata::WriteOptions options;
    if (compression == "gzip") {
        options.compression = anndata::Compression::gzip;
        options.gzip_level = 4;
    } else if (compression != "none") {
        throw std::runtime_error("compression must be none or gzip");
    }
    if (byte_order == "native") options.byte_order = anndata::FileByteOrder::native;
    else if (byte_order == "big") options.byte_order = anndata::FileByteOrder::big;
    else if (byte_order == "little") options.byte_order = anndata::FileByteOrder::little;
    else throw std::runtime_error("byte order must be native, little, or big");
    return options;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 9) {
        std::cerr << "usage: h5ad_benchmark <seed|write|read> <dense|csr> <none|gzip> <file> [rows] [columns] [per-row] [native|little|big]\n";
        return 2;
    }
    const std::string_view action(argv[1]);
    const std::string_view kind(argv[2]);
    const std::string_view compression(argv[3]);
    const std::string filename(argv[4]);
    const auto rows = argc > 5 ? parse_size(argv[5], 4096) : 4096;
    const auto columns = argc > 6 ? parse_size(argv[6], 2048) : 2048;
    const auto per_row = argc > 7 ? parse_size(argv[7], 128) : 128;
    const std::string_view byte_order = argc > 8 ? argv[8] : "native";
    if (kind != "dense" && kind != "csr") throw std::runtime_error("kind must be dense or csr");
    if (rows > std::numeric_limits<std::size_t>::max() / columns)
        throw std::runtime_error("matrix dimensions overflow");

    auto make_data = [&] {
        if (kind == "dense") return make_dense(rows, columns);
        if (columns > std::numeric_limits<std::uint32_t>::max()
            || rows > std::numeric_limits<std::uint64_t>::max() / per_row)
            throw std::runtime_error("CSR dimensions exceed benchmark index widths");
        return make_csr(rows, columns, per_row);
    };

    auto options = write_options(compression, byte_order);
    std::unique_ptr<anndata::AnnData> output;
    if (action == "seed" || action == "write") {
        output = std::make_unique<anndata::AnnData>(make_data());
    }
    double sum = 0.0;
    std::size_t payload = 0;
    if (action == "seed" || action == "write") {
        sum = checksum(output->x());
        payload = payload_bytes(output->x());
        if (action == "seed") {
            anndata::write_h5ad(filename, *output, options);
            return 0;
        }
        const auto start = Clock::now();
        anndata::write_h5ad(filename, *output, options);
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::ifstream input(filename, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("could not inspect benchmark output file");
        const auto file_bytes = static_cast<std::uint64_t>(input.tellg());
        std::cout << "{\"action\":\"" << action << "\",\"kind\":\"" << kind
                  << "\",\"byte_order\":\"" << byte_order
                  << "\",\"compression\":\"" << compression << "\",\"rows\":" << rows
                  << ",\"columns\":" << columns << ",\"payload_bytes\":" << payload
                  << ",\"file_bytes\":" << file_bytes << ",\"seconds\":" << seconds
                  << ",\"checksum\":" << sum << "}\n";
        return 0;
    } else if (action == "read") {
        const auto start = Clock::now();
        auto data = anndata::read_h5ad(filename);
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (data.n_obs() != rows || data.n_vars() != columns)
            throw std::runtime_error("read dimensions differ from expected dimensions");
        sum = checksum(data.x());
        payload = payload_bytes(data.x());
        std::ifstream input(filename, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("could not inspect benchmark input file");
        const auto file_bytes = static_cast<std::uint64_t>(input.tellg());
        std::cout << "{\"action\":\"" << action << "\",\"kind\":\"" << kind
                  << "\",\"byte_order\":\"" << byte_order
                  << "\",\"compression\":\"" << compression << "\",\"rows\":" << rows
                  << ",\"columns\":" << columns << ",\"payload_bytes\":" << payload
                  << ",\"file_bytes\":" << file_bytes << ",\"seconds\":" << seconds
                  << ",\"checksum\":" << sum << "}\n";
        return 0;
    } else {
        throw std::runtime_error("action must be seed, write, or read");
    }
}
