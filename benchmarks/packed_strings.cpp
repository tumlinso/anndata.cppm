import anndata.core;

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
bool count_allocations{};
std::size_t new_calls{}, delete_calls{}, new_bytes{};
volatile std::uint64_t sink{};

void* allocate(std::size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc{};
    if (count_allocations) { ++new_calls; new_bytes += n; }
    return p;
}

void release(void* p) noexcept {
    if (p) {
        if (count_allocations) ++delete_calls;
        std::free(p);
    }
}

struct Fixture {
    std::vector<char> bytes;
    std::vector<std::uint64_t> offsets;
};

Fixture make_fixture(std::size_t n) {
    Fixture f;
    f.bytes.reserve(8 * n);
    f.offsets.reserve(n + 1);
    f.offsets.push_back(0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < 8; ++j)
            f.bytes.push_back(char('a' + (i + j) % 26));
        f.offsets.push_back(static_cast<std::uint64_t>(f.bytes.size()));
    }
    return f;
}

bool matches(std::string_view s, std::size_t i) {
    if (s.size() != 8) return false;
    for (std::size_t j = 0; j < 8; ++j)
        if (s[j] != char('a' + (i + j) % 26)) return false;
    return true;
}

std::uint64_t run_once(const Fixture& seed, std::size_t n,
                       anndata::Validation validation) {
    auto bytes = seed.bytes;
    auto offsets = seed.offsets;
    std::vector<std::size_t> shape{n};
    const char* expected_bytes = bytes.empty() ? nullptr : bytes.data();
    const auto* expected_offsets = offsets.data();
    anndata::StringArrayView view;
    {
        auto owner = anndata::StringArray::from_packed(
            std::move(shape), std::move(bytes), std::move(offsets), validation);
        view = owner.view();
    }

    if (view.rank() != 1 || view.extent(0) != n || view.size() != n)
        throw std::runtime_error("packed string shape mismatch");
    if (view.storage_byte_size() != seed.bytes.size() ||
        view.storage_bytes() != expected_bytes ||
        view.storage_offsets() != expected_offsets ||
        view.storage_offsets()[0] != 0 ||
        view.storage_offsets()[n] != seed.bytes.size())
        throw std::runtime_error("packed buffer ownership or offset mismatch");

    std::uint64_t h = static_cast<std::uint64_t>(view.size());
    h = h * 131 + static_cast<std::uint64_t>(view.storage_byte_size());
    h = h * 131 + view.storage_offsets()[n];
    if (n) {
        const auto first = view.string(0), last = view.string(n - 1);
        if (!matches(first, 0) || !matches(last, n - 1))
            throw std::runtime_error("packed string content mismatch");
        for (const auto c : first) h = h * 131 + static_cast<unsigned char>(c);
        for (const auto c : last) h = h * 131 + static_cast<unsigned char>(c);
    } else if (view.storage_bytes() != nullptr) {
        throw std::runtime_error("empty packed bytes must be null");
    }
    return h;
}

const char* validation_name(anndata::Validation v) {
    return v == anndata::Validation::full ? "full" : "none";
}

struct Trial { double ns_per_op{}; std::uint64_t checksum{}; };
} // namespace

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t) noexcept { release(p); }

int main(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t byte_budget = 1u << 26, max_iterations = 4096;
    constexpr std::array<std::size_t, 4> sizes{0, 16, 4096, 65536};
    constexpr std::array<anndata::Validation, 2> validations{
        anndata::Validation::full, anndata::Validation::none};
    constexpr const char* counter_scope =
        "ordinary throwing new/new[] and delete/delete[] only; aligned, malloc, and system allocations excluded";
    std::size_t trial_count = 3;
    if (argc > 2) { std::cerr << "usage: packed_strings [trials:1..3]\n"; return 2; }
    if (argc == 2) {
        char* end{};
        const auto value = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end || value < 1 || value > 3) {
            std::cerr << "trials must be an integer from 1 to 3\n";
            return 2;
        }
        trial_count = static_cast<std::size_t>(value);
    }
    (void)Clock::now();

    for (const auto n : sizes) {
        const auto seed = make_fixture(n);
        const auto source_bytes = seed.bytes.size() + seed.offsets.size() * sizeof(std::uint64_t);
        const auto iterations = std::min(max_iterations,
            std::max(std::size_t{1}, byte_budget / std::max(source_bytes, std::size_t{1})));
        for (const auto validation : validations) {
            new_calls = delete_calls = new_bytes = 0;
            count_allocations = true;
            const auto checksum = run_once(seed, n, validation);
            count_allocations = false;
            const auto allocs = new_calls, frees = delete_calls, bytes = new_bytes;

            std::array<Trial, 3> samples{};
            for (std::size_t t = 0; t < trial_count; ++t) {
                const auto start = Clock::now();
                std::uint64_t sum = 0;
                for (std::size_t i = 0; i < iterations; ++i)
                    sum += run_once(seed, n, validation);
                const auto stop = Clock::now();
                if (sum != checksum * iterations)
                    throw std::runtime_error("timed checksum differs from allocation sample");
                sink = sink ^ sum;
                samples[t] = {
                    std::chrono::duration<double, std::nano>(stop - start).count() /
                        static_cast<double>(iterations), sum};
            }
            for (std::size_t t = 0; t < trial_count; ++t) {
                const auto& sample = samples[t];
                std::cout << "{\"sample\":\"time\",\"operation\":\"from_packed\",\"n\":"
                          << n << ",\"strings\":" << n << ",\"validation\":\""
                          << validation_name(validation) << "\",\"iterations\":" << iterations
                          << ",\"trial\":" << t << ",\"ns_per_op\":" << std::fixed
                          << std::setprecision(2) << sample.ns_per_op
                          << ",\"allocation_calls\":" << allocs
                          << ",\"deallocation_calls\":" << frees
                          << ",\"allocation_bytes\":" << bytes
                          << ",\"checksum\":" << sample.checksum
                          << ",\"counter_scope\":\"" << counter_scope << "\"}\n";
            }
        }
    }
    return 0;
}
