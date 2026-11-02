// Compare a checked typed view's pointer loop to the same kernel using its raw pointer.
// Access acquisition and initialization are outside the timed single-thread kernels.
import anndata.core;

#include <algorithm>
#include <barrier>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Result {
    const char* access;
    const char* mode;
    std::size_t trial;
    std::size_t threads;
    double seconds;
    double checksum;
};

struct DenseAcquisitionResult {
    const char* access;
    std::size_t trial;
    std::size_t iterations;
    double seconds;
    std::uintptr_t pointer_token_checksum;
    bool pointer_identity;
    bool checksum_matches;
    double storage_checksum;
};

struct SparseWrapResult {
    const char* validation;
    std::size_t trial;
    std::size_t iterations;
    double seconds;
    bool pointer_identity;
    bool checksum_matches;
    double buffer_checksum;
};

std::size_t parse_size(const char* text, std::size_t fallback) {
    if (!text) return fallback;
    std::size_t result = 0;
    const std::string_view input{text};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), result);
    if (error != std::errc{} || end != input.data() + input.size() || !result) {
        std::cerr << "expected a positive integer, received: " << input << '\n';
        std::terminate();
    }
    return result;
}

void kernel(const float* input, float* output, std::size_t begin, std::size_t end) {
    constexpr float scale = 1.0001f;
    constexpr float bias = 0.125f;
    for (std::size_t i = begin; i < end; ++i)
        output[i] = input[i] * scale + bias;
}

std::pair<std::size_t, std::size_t> block_range(std::size_t count,
                                               std::size_t blocks,
                                               std::size_t block) {
    const auto width = count / blocks;
    const auto remainder = count % blocks;
    const auto begin = block * width + std::min(block, remainder);
    return {begin, begin + width + (block < remainder ? 1 : 0)};
}

double checksum(const float* values, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i)
        sum += static_cast<double>(values[i]);
    return sum;
}

double sparse_checksum(const float* values, std::size_t nnz,
                       const std::uint32_t* indices,
                       const std::uint64_t* offsets,
                       std::size_t offset_count) {
    double sum = checksum(values, nnz);
    for (std::size_t i = 0; i < nnz; ++i) sum += static_cast<double>(indices[i]);
    for (std::size_t i = 0; i < offset_count; ++i) sum += static_cast<double>(offsets[i]);
    return sum;
}

template<class Acquire>
DenseAcquisitionResult timed_dense_acquisition(
        const char* access, std::size_t trial, std::size_t iterations,
        const float* expected_pointer, bool pointer_identity,
        double storage_checksum, Acquire&& acquire) {
    std::uintptr_t pointer_token_checksum = 0;
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        auto descriptor = acquire();
        pointer_token_checksum += reinterpret_cast<std::uintptr_t>(descriptor.data());
    }
    const auto stop = Clock::now();
    const auto expected = reinterpret_cast<std::uintptr_t>(expected_pointer) * iterations;
    return {access, trial, iterations,
            std::chrono::duration<double>(stop - start).count(),
            pointer_token_checksum, pointer_identity,
            pointer_token_checksum == expected, storage_checksum};
}

template<class Wrap>
SparseWrapResult timed_sparse_wrap(const char* validation, std::size_t trial,
                                   std::size_t iterations, bool pointer_identity,
                                   bool checksum_matches, double buffer_checksum,
                                   Wrap&& wrap) {
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        [[maybe_unused]] auto array = wrap();
    }
    const auto stop = Clock::now();
    return {validation, trial, iterations,
            std::chrono::duration<double>(stop - start).count(),
            pointer_identity, checksum_matches, buffer_checksum};
}

template<class Function>
Result timed_single(const char* access, std::size_t trial, std::size_t count,
                    Function&& function, const float* output) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    return {access, "serial", trial, 1,
            std::chrono::duration<double>(stop - start).count(), checksum(output, count)};
}

template<class Function>
Result timed_parallel(const char* access, std::size_t trial, std::size_t count,
                      std::size_t threads, Function&& function, const float* output) {
    // Workers and synchronization objects are ready before the timer starts.
    std::barrier start_line(static_cast<std::ptrdiff_t>(threads + 1));
    std::barrier finish_line(static_cast<std::ptrdiff_t>(threads + 1));
    std::vector<std::jthread> workers;
    workers.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            start_line.arrive_and_wait();
            function(t);
            finish_line.arrive_and_wait();
        });
    }
    const auto begin = Clock::now();
    start_line.arrive_and_wait();
    finish_line.arrive_and_wait();
    const auto end = Clock::now();
    workers.clear();
    return {access, "caller_threads", trial, threads,
            std::chrono::duration<double>(end - begin).count(), checksum(output, count)};
}

void report(const Result& result, std::size_t bytes) {
    const double gib_per_second = static_cast<double>(bytes) / result.seconds / (1024.0 * 1024.0 * 1024.0);
    std::cout << "{\"access\":\"" << result.access
              << "\",\"mode\":\"" << result.mode
              << "\",\"trial\":" << result.trial
              << ",\"threads\":" << result.threads
              << ",\"seconds\":" << result.seconds
              << ",\"effective_gib_per_second\":" << gib_per_second
              << ",\"checksum\":" << result.checksum << "}\n";
}

void report(const DenseAcquisitionResult& result) {
    std::cout << "{\"kind\":\"dense_acquisition_trial\",\"access\":\"" << result.access
              << "\",\"trial\":" << result.trial
              << ",\"iterations\":" << result.iterations
              << ",\"seconds\":" << result.seconds
              << ",\"nanoseconds_per_acquisition\":"
              << result.seconds * 1.0e9 / static_cast<double>(result.iterations)
              << ",\"pointer_identity\":" << (result.pointer_identity ? "true" : "false")
              << ",\"pointer_token_checksum\":" << result.pointer_token_checksum
              << ",\"pointer_token_checksum_matches\":" << (result.checksum_matches ? "true" : "false")
              << ",\"storage_checksum\":" << result.storage_checksum << "}\n";
}

void report(const SparseWrapResult& result) {
    std::cout << "{\"kind\":\"sparse_wrap_trial\",\"validation\":\"" << result.validation
              << "\",\"trial\":" << result.trial
              << ",\"iterations\":" << result.iterations
              << ",\"seconds\":" << result.seconds
              << ",\"nanoseconds_per_wrap\":"
              << result.seconds * 1.0e9 / static_cast<double>(result.iterations)
              << ",\"pointer_identity\":" << (result.pointer_identity ? "true" : "false")
              << ",\"buffer_checksum_matches\":" << (result.checksum_matches ? "true" : "false")
              << ",\"buffer_checksum\":" << result.buffer_checksum << "}\n";
}

double quantile(const std::vector<double>& sorted, double fraction) {
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(sorted.size() - 1));
    return sorted[index];
}

void report_distributions(const std::vector<Result>& results) {
    for (const char* access : {"typed", "raw_pointer"}) {
        for (const char* mode : {"serial", "caller_threads"}) {
            std::vector<double> times;
            for (const auto& result : results)
                if (std::string_view(result.access) == access && std::string_view(result.mode) == mode)
                    times.push_back(result.seconds);
            if (times.empty()) continue;
            std::sort(times.begin(), times.end());
            const double median = times.size() % 2
                ? times[times.size() / 2]
                : (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0;
            std::cout << "{\"kind\":\"timing_distribution\",\"access\":\"" << access
                      << "\",\"mode\":\"" << mode << "\",\"trials\":" << times.size()
                      << ",\"min_seconds\":" << times.front()
                      << ",\"p25_seconds\":" << quantile(times, 0.25)
                      << ",\"median_seconds\":" << median
                      << ",\"p75_seconds\":" << quantile(times, 0.75)
                      << ",\"max_seconds\":" << times.back() << "}\n";
        }
    }
}

void report_acquisition_distributions(const std::vector<DenseAcquisitionResult>& results) {
    for (const char* access : {"checked_dense", "unsafe_dense"}) {
        std::vector<double> times;
        for (const auto& result : results)
            if (std::string_view(result.access) == access)
                times.push_back(result.seconds * 1.0e9 / static_cast<double>(result.iterations));
        if (times.empty()) continue;
        std::sort(times.begin(), times.end());
        const double median = times.size() % 2
            ? times[times.size() / 2]
            : (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0;
        std::cout << "{\"kind\":\"dense_acquisition_distribution\",\"access\":\"" << access
                  << "\",\"trials\":" << times.size()
                  << ",\"min_nanoseconds_per_acquisition\":" << times.front()
                  << ",\"p25_nanoseconds_per_acquisition\":" << quantile(times, 0.25)
                  << ",\"median_nanoseconds_per_acquisition\":" << median
                  << ",\"p75_nanoseconds_per_acquisition\":" << quantile(times, 0.75)
                  << ",\"max_nanoseconds_per_acquisition\":" << times.back() << "}\n";
    }
}

void report_sparse_wrap_distributions(const std::vector<SparseWrapResult>& results) {
    for (const char* validation : {"full", "none"}) {
        std::vector<double> times;
        for (const auto& result : results)
            if (std::string_view(result.validation) == validation)
                times.push_back(result.seconds * 1.0e9 / static_cast<double>(result.iterations));
        if (times.empty()) continue;
        std::sort(times.begin(), times.end());
        const double median = times.size() % 2
            ? times[times.size() / 2]
            : (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0;
        std::cout << "{\"kind\":\"sparse_wrap_distribution\",\"validation\":\"" << validation
                  << "\",\"trials\":" << times.size()
                  << ",\"min_nanoseconds_per_wrap\":" << times.front()
                  << ",\"p25_nanoseconds_per_wrap\":" << quantile(times, 0.25)
                  << ",\"median_nanoseconds_per_wrap\":" << median
                  << ",\"p75_nanoseconds_per_wrap\":" << quantile(times, 0.75)
                  << ",\"max_nanoseconds_per_wrap\":" << times.back() << "}\n";
    }
}
}

int main(int argc, char** argv) {
    const auto count = parse_size(argc > 1 ? argv[1] : nullptr, std::size_t{1} << 22);
    const auto hardware_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const auto threads = parse_size(argc > 2 ? argv[2] : nullptr,
                                    std::min({std::size_t{8}, hardware_threads, count}));
    const auto trials = parse_size(argc > 3 ? argv[3] : nullptr, 7);
    const auto acquisition_iterations = parse_size(argc > 4 ? argv[4] : nullptr, 100000);
    const auto sparse_wrap_iterations = parse_size(argc > 5 ? argv[5] : nullptr, 64);
    if (count > static_cast<std::size_t>(-1) / (2 * sizeof(float))) {
        std::cerr << "requested array size overflows byte count\n";
        return 2;
    }
    if (threads >= static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        trials > static_cast<std::size_t>(-1) / 4) {
        std::cerr << "thread or trial count is too large\n";
        return 2;
    }

    auto input = anndata::Array::empty<float>({count});
    auto output = anndata::Array::empty<float>({count});
    auto input_access = input.edit().dense<float>();
    auto output_access = output.edit().dense<float>();
    float* input_raw = input_access.data();
    float* output_raw = output_access.data();
    for (std::size_t i = 0; i < count; ++i)
        input_raw[i] = static_cast<float>((i % 1021) * 0.001);
    auto input_view = input.view();
    const float* input_checked = input_view.dense<float>().data();
    float* output_checked = output.edit().dense<float>().data();
    const float* input_direct = input_raw;
    float* output_direct = output_raw;
    const std::size_t traffic = count * 2 * sizeof(float);
    const double input_storage_checksum = checksum(input_raw, count);
    const bool dense_pointer_identity = [&] {
        auto checked = input_view.dense<float>();
        auto unchecked = input_view.unsafe_dense<float>();
        return checked.data() == unchecked.data() && checked.data() == input_raw;
    }();
    if (!dense_pointer_identity) {
        std::cerr << "checked and unchecked dense views did not preserve pointer identity\n";
        return 2;
    }

    constexpr std::size_t sparse_rows = 1U << 14;
    constexpr std::size_t sparse_columns = 1U << 14;
    constexpr std::size_t sparse_nnz = sparse_rows * 2;
    std::vector<float> sparse_values(sparse_nnz);
    std::vector<std::uint32_t> sparse_indices(sparse_nnz);
    std::vector<std::uint64_t> sparse_offsets(sparse_rows + 1);
    for (std::size_t row = 0; row < sparse_rows; ++row) {
        sparse_offsets[row] = static_cast<std::uint64_t>(row * 2);
        sparse_indices[row * 2] = 0;
        sparse_indices[row * 2 + 1] = 1;
        sparse_values[row * 2] = static_cast<float>((row % 997) * 0.001);
        sparse_values[row * 2 + 1] = static_cast<float>(((row + 1) % 997) * 0.001);
    }
    sparse_offsets[sparse_rows] = sparse_nnz;
    const double expected_sparse_checksum = sparse_checksum(
        sparse_values.data(), sparse_nnz, sparse_indices.data(), sparse_offsets.data(), sparse_rows + 1);
    auto inspect_sparse = [&](anndata::Validation validation) {
        auto wrapped = anndata::Array::borrow_sparse<float, std::uint32_t, std::uint64_t>(
            anndata::Layout::csr, sparse_rows, sparse_columns, sparse_nnz,
            sparse_values.data(), sparse_indices.data(), sparse_offsets.data(), validation);
        const auto view = wrapped.view().csr<float, std::uint32_t, std::uint64_t>();
        const bool same_buffers = view.values() == sparse_values.data() &&
                                  view.indices() == sparse_indices.data() &&
                                  view.offsets() == sparse_offsets.data() &&
                                  view.rows() == sparse_rows && view.columns() == sparse_columns &&
                                  view.nnz() == sparse_nnz;
        const double observed = sparse_checksum(view.values(), view.nnz(), view.indices(), view.offsets(),
                                                view.rows() + 1);
        return std::pair{same_buffers, observed};
    };
    const auto [full_sparse_pointer_identity, full_sparse_checksum] = inspect_sparse(anndata::Validation::full);
    const auto [unchecked_sparse_pointer_identity, unchecked_sparse_checksum] = inspect_sparse(anndata::Validation::none);
    const bool full_sparse_checksum_matches = full_sparse_checksum == expected_sparse_checksum;
    const bool unchecked_sparse_checksum_matches = unchecked_sparse_checksum == expected_sparse_checksum;
    if (!full_sparse_pointer_identity || !unchecked_sparse_pointer_identity ||
        !full_sparse_checksum_matches || !unchecked_sparse_checksum_matches) {
        std::cerr << "sparse external wrapping changed buffer identity or checksum\n";
        return 2;
    }

    std::cout << "{\"kind\":\"configuration\",\"compiler\":\"" << __VERSION__
              << "\",\"cplusplus\":" << __cplusplus
              << ",\"hardware_concurrency\":" << std::thread::hardware_concurrency()
              << ",\"elements\":" << count
              << ",\"bytes_read_written\":" << traffic << ",\"threads\":" << threads
              << ",\"trials\":" << trials
              << ",\"dense_acquisition_iterations\":" << acquisition_iterations
              << ",\"dense_storage_checksum\":" << input_storage_checksum
              << ",\"dense_pointer_identity\":true"
              << ",\"sparse_rows\":" << sparse_rows
              << ",\"sparse_columns\":" << sparse_columns
              << ",\"sparse_nnz\":" << sparse_nnz
              << ",\"sparse_wrap_iterations\":" << sparse_wrap_iterations
              << ",\"sparse_pointer_identity_full\":true"
              << ",\"sparse_pointer_identity_none\":true"
              << ",\"sparse_buffer_checksum\":" << expected_sparse_checksum
              << ",\"kernel\":\"output[i]=input[i]*1.0001f+0.125f\"}\n";

    auto acquire_checked = [&] { return input_view.dense<float>(); };
    auto acquire_unsafe = [&] { return input_view.unsafe_dense<float>(); };
    std::vector<DenseAcquisitionResult> acquisition_results;
    acquisition_results.reserve(trials * 2);
    for (std::size_t trial = 0; trial < trials; ++trial) {
        const auto run_checked = [&] {
            return timed_dense_acquisition("checked_dense", trial, acquisition_iterations,
                                           input_raw, dense_pointer_identity,
                                           input_storage_checksum, acquire_checked);
        };
        const auto run_unsafe = [&] {
            return timed_dense_acquisition("unsafe_dense", trial, acquisition_iterations,
                                           input_raw, dense_pointer_identity,
                                           input_storage_checksum, acquire_unsafe);
        };
        DenseAcquisitionResult checked_result;
        DenseAcquisitionResult unsafe_result;
        if (trial % 2 == 0) {
            unsafe_result = run_unsafe();
            checked_result = run_checked();
        } else {
            checked_result = run_checked();
            unsafe_result = run_unsafe();
        }
        if (!checked_result.checksum_matches || !unsafe_result.checksum_matches) {
            std::cerr << "dense acquisition returned an unexpected pointer checksum\n";
            return 2;
        }
        acquisition_results.push_back(checked_result);
        acquisition_results.push_back(unsafe_result);
        report(checked_result);
        report(unsafe_result);
    }

    auto wrap_full = [&] {
        return anndata::Array::borrow_sparse<float, std::uint32_t, std::uint64_t>(
            anndata::Layout::csr, sparse_rows, sparse_columns, sparse_nnz,
            sparse_values.data(), sparse_indices.data(), sparse_offsets.data(), anndata::Validation::full);
    };
    auto wrap_none = [&] {
        return anndata::Array::borrow_sparse<float, std::uint32_t, std::uint64_t>(
            anndata::Layout::csr, sparse_rows, sparse_columns, sparse_nnz,
            sparse_values.data(), sparse_indices.data(), sparse_offsets.data(), anndata::Validation::none);
    };
    std::vector<SparseWrapResult> sparse_wrap_results;
    sparse_wrap_results.reserve(trials * 2);
    for (std::size_t trial = 0; trial < trials; ++trial) {
        const auto run_full = [&] {
            return timed_sparse_wrap("full", trial, sparse_wrap_iterations,
                                     full_sparse_pointer_identity, full_sparse_checksum_matches,
                                     full_sparse_checksum, wrap_full);
        };
        const auto run_none = [&] {
            return timed_sparse_wrap("none", trial, sparse_wrap_iterations,
                                     unchecked_sparse_pointer_identity, unchecked_sparse_checksum_matches,
                                     unchecked_sparse_checksum, wrap_none);
        };
        SparseWrapResult full_result;
        SparseWrapResult none_result;
        if (trial % 2 == 0) {
            none_result = run_none();
            full_result = run_full();
        } else {
            full_result = run_full();
            none_result = run_none();
        }
        sparse_wrap_results.push_back(full_result);
        sparse_wrap_results.push_back(none_result);
        report(full_result);
        report(none_result);
    }

    std::vector<Result> results;
    results.reserve(trials * (threads > 1 ? 4 : 2));
    for (std::size_t trial = 0; trial < trials; ++trial) {
        const bool raw_first = (trial % 2) == 0;
        auto checked_serial = [&] { kernel(input_checked, output_checked, 0, count); };
        auto raw_serial = [&] { kernel(input_direct, output_direct, 0, count); };
        Result serial_checked;
        Result serial_raw;
        if (raw_first) {
            serial_raw = timed_single("raw_pointer", trial, count, raw_serial, output_direct);
            serial_checked = timed_single("typed", trial, count, checked_serial, output_checked);
            results.push_back(serial_raw);
            results.push_back(serial_checked);
            report(serial_raw, traffic);
            report(serial_checked, traffic);
        } else {
            serial_checked = timed_single("typed", trial, count, checked_serial, output_checked);
            serial_raw = timed_single("raw_pointer", trial, count, raw_serial, output_direct);
            results.push_back(serial_checked);
            results.push_back(serial_raw);
            report(serial_checked, traffic);
            report(serial_raw, traffic);
        }

        if (threads > 1) {
            auto checked_partition = [&](std::size_t t) {
                const auto [begin, end] = block_range(count, threads, t);
                kernel(input_checked, output_checked, begin, end);
            };
            auto raw_partition = [&](std::size_t t) {
                const auto [begin, end] = block_range(count, threads, t);
                kernel(input_direct, output_direct, begin, end);
            };
            if (raw_first) {
                auto raw_result = timed_parallel("raw_pointer", trial, count, threads,
                                                 raw_partition, output_direct);
                auto typed_result = timed_parallel("typed", trial, count, threads,
                                                   checked_partition, output_checked);
                results.push_back(raw_result);
                results.push_back(typed_result);
                report(raw_result, traffic);
                report(typed_result, traffic);
            } else {
                auto typed_result = timed_parallel("typed", trial, count, threads,
                                                   checked_partition, output_checked);
                auto raw_result = timed_parallel("raw_pointer", trial, count, threads,
                                                 raw_partition, output_direct);
                results.push_back(typed_result);
                results.push_back(raw_result);
                report(typed_result, traffic);
                report(raw_result, traffic);
            }
        }
    }
    report_acquisition_distributions(acquisition_results);
    report_sparse_wrap_distributions(sparse_wrap_results);
    report_distributions(results);
}
