#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

import anndatapp;

using namespace anndatapp;

namespace {
std::string_view code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::invalid_argument: return "invalid_argument";
        case ErrorCode::out_of_bounds: return "out_of_bounds";
        case ErrorCode::dtype_mismatch: return "dtype_mismatch";
        case ErrorCode::unsupported_layout: return "unsupported_layout";
        case ErrorCode::shape_mismatch: return "shape_mismatch";
        case ErrorCode::invalid_sparse_structure: return "invalid_sparse_structure";
        case ErrorCode::non_contiguous: return "non_contiguous";
        case ErrorCode::io_error: return "io_error";
        case ErrorCode::unsupported_encoding: return "unsupported_encoding";
        case ErrorCode::unsupported_version: return "unsupported_version";
        case ErrorCode::unsupported_dtype: return "unsupported_dtype";
        case ErrorCode::unavailable_filter: return "unavailable_filter";
    }
    return "unknown";
}

template<class F>
void expect_unsupported_dtype(std::string_view stage, F&& operation) {
    try {
        std::forward<F>(operation)();
    } catch (const Error& error) {
        if (error.code() != ErrorCode::unsupported_dtype)
            throw std::runtime_error(std::string(stage) + " returned " +
                                     std::string(code_name(error.code())) +
                                     ", expected unsupported_dtype");
        std::cout << stage << ": unsupported_dtype at " << error.path() << '\n';
        return;
    }
    throw std::runtime_error(std::string(stage) + " accepted float16 X");
}

#if ANNDATAPP_HAS_HDF5
void check_h5ad(std::string_view path) {
    expect_unsupported_dtype("eager H5AD read", [&] { (void)read_h5ad(path); });

    auto metadata = open_h5ad(path);
    expect_unsupported_dtype("backed H5AD X metadata", [&] { (void)metadata.x().dtype(); });

    auto backed = open_h5ad(path);
    expect_unsupported_dtype("backed H5AD X read", [&] { (void)backed.x().read(); });
}
#endif

#if ANNDATAPP_HAS_ZARR
void check_zarr(std::string_view path) {
    expect_unsupported_dtype("eager Zarr read", [&] { (void)read_zarr(path); });

    auto metadata = open_zarr(path);
    expect_unsupported_dtype("backed Zarr X metadata", [&] { (void)metadata.x().dtype(); });

    auto backed = open_zarr(path);
    expect_unsupported_dtype("backed Zarr X read", [&] { (void)backed.x().read(); });
}
#endif
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: dtype_rejection_driver h5ad|zarr2|zarr3 /path/to/store\n";
        return 2;
    }
    try {
        const std::string_view backend(argv[1]);
        if (backend == "h5ad") {
#if ANNDATAPP_HAS_HDF5
            check_h5ad(argv[2]);
#else
            throw std::runtime_error("HDF5 backend is disabled");
#endif
        } else if (backend == "zarr2" || backend == "zarr3") {
#if ANNDATAPP_HAS_ZARR
            check_zarr(argv[2]);
#else
            throw std::runtime_error("Zarr backend is disabled");
#endif
        } else {
            throw std::runtime_error("backend must be h5ad, zarr2, or zarr3");
        }
    } catch (const Error& error) {
        std::cerr << "unexpected anndatapp error[" << code_name(error.code()) << "] at "
                  << error.path() << ": " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
