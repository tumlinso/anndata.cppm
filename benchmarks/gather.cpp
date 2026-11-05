// Paired gather probe: allocation counts cover regular throwing scalar/array
// new and delete only; aligned/nothrow forms and direct C allocation are outside
// this counter. The measured cases use ordinary size_t-aligned allocations.
import anndatapp.core;

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace allocation_probe {
struct Counts {
    std::size_t calls = 0, bytes = 0, deletes = 0;
};

thread_local bool active = false;
thread_local Counts counts{};

void* allocate(std::size_t bytes) {
    const auto actual = bytes ? bytes : 1;
    auto* p = std::malloc(actual);
    if (!p) throw std::bad_alloc{};
    if (active) { ++counts.calls; counts.bytes += bytes; }
    return p;
}

void release(void* p) noexcept {
    if (active && p) ++counts.deletes;
    std::free(p);
}
}

void* operator new(std::size_t bytes) { return allocation_probe::allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocation_probe::allocate(bytes); }
void operator delete(void* p) noexcept { allocation_probe::release(p); }
void operator delete[](void* p) noexcept { allocation_probe::release(p); }
void operator delete(void* p, std::size_t) noexcept { allocation_probe::release(p); }
void operator delete[](void* p, std::size_t) noexcept { allocation_probe::release(p); }

namespace {
using namespace anndatapp;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t hash_seed = 1469598103934665603ULL;
constexpr std::uint64_t hash_prime = 1099511628211ULL;

void require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}

std::uint64_t fold(std::uint64_t hash, std::size_t value) {
    return (hash ^ static_cast<std::uint64_t>(value)) * hash_prime;
}

std::uint64_t axis_hash(const Axis& axis, std::size_t n) {
    auto hash = hash_seed;
    for (std::size_t i = 0; i < n; ++i) hash = fold(hash, axis.index(i));
    return hash;
}

std::uint64_t axis_probe_hash(const Axis& axis, std::size_t n) {
    auto hash = fold(hash_seed, n);
    const std::size_t sample[] = {0, n / 2, n - 1};
    for (const auto i : sample) hash = fold(hash, axis.index(i));
    return hash;
}

std::uint64_t check_axis(const Axis& axis, const std::vector<std::size_t>& expected) {
    require(axis.is_gather() && axis.extent(expected.size()) == expected.size(),
            "gather axis has an incorrect kind or extent");
    auto hash = hash_seed;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto value = axis.index(i);
        require(value == expected[i], "gather axis changed order");
        hash = fold(hash, value);
    }
    return hash;
}

std::vector<std::size_t> positions(std::size_t n, std::size_t step, std::size_t start = 0) {
    std::vector<std::size_t> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = (i * step + start) % n;
    return out;
}

std::size_t iterations(std::size_t n) {
    constexpr std::size_t position_budget = 4 * 1024 * 1024;
    return std::min<std::size_t>(8192, std::max<std::size_t>(1, position_budget / n));
}

template<class F>
allocation_probe::Counts count_once(F&& operation) {
    allocation_probe::counts = {};
    allocation_probe::active = true;
    try { (void)operation(); }
    catch (...) { allocation_probe::active = false; throw; }
    allocation_probe::active = false;
    return allocation_probe::counts;
}

struct Trial { double ns_per_op = 0; std::uint64_t checksum = 0; };

template<class F>
std::array<Trial, 3> time_case(std::size_t count, std::size_t repeat,
        std::uint64_t checksum_per_op, F&& operation) {
    std::array<Trial, 3> out{};
    const auto expected = checksum_per_op * repeat;
    allocation_probe::active = false;
    for (std::size_t t = 0; t < count; ++t) {
        std::uint64_t checksum = 0;
        const auto begin = Clock::now();
        for (std::size_t i = 0; i < repeat; ++i) checksum += operation();
        const auto end = Clock::now();
        require(checksum == expected, "timed checksum differs from allocation sample");
        out[t] = {std::chrono::duration<double, std::nano>(end - begin).count() / repeat,
                  checksum};
    }
    return out;
}

void report(const char* operation, std::size_t n, std::size_t repeat,
        std::size_t trial_count, allocation_probe::Counts counts,
        const std::array<Trial, 3>& trials) {
    allocation_probe::active = false;
    std::cout << "{\"sample\":\"allocation\",\"operation\":\"" << operation
              << "\",\"n\":" << n << ",\"iterations\":1,\"allocation_calls\":"
              << counts.calls << ",\"allocation_bytes\":" << counts.bytes
              << ",\"deallocation_calls\":" << counts.deletes << "}\n";
    for (std::size_t t = 0; t < trial_count; ++t)
        std::cout << "{\"sample\":\"time\",\"operation\":\"" << operation
                  << "\",\"n\":" << n << ",\"iterations\":" << repeat
                  << ",\"trial\":" << t << ",\"ns_per_op\":" << std::fixed
                  << std::setprecision(2) << trials[t].ns_per_op << ",\"checksum\":"
                  << trials[t].checksum << "}\n";
}

std::uint64_t raw_gather(const std::vector<std::size_t>& source) {
    auto axis = Axis::gather(source.data(), source.size());
    return axis_probe_hash(axis, source.size());
}

std::uint64_t normalized_gather(const std::vector<std::size_t>& source) {
    auto copy = source; // Same input-vector copy is included for both revisions.
    auto axis = detail::gather_axis(std::move(copy));
    return axis_probe_hash(axis, source.size());
}

std::uint64_t compose_view(const ArrayView& parent_view, const Axis& child,
        std::size_t n, const float* pointer, std::uint64_t map_hash) {
    auto selected = parent_view.select({child});
    auto dense = selected.dense<float>();
    require(dense.data() == pointer && dense.rank() == 1 && dense.extent(0) == n
            && dense.stride(0) == 0, "composed view changed pointer or shape");
    return fold(map_hash, dense.extent(0));
}

std::uint64_t label_hash(std::string_view label, std::uint64_t hash) {
    hash = fold(hash, label.size());
    for (unsigned char c : label) hash = fold(hash, c);
    return hash;
}

template<bool check_order>
std::uint64_t ann_data_slice(const AnnData& data, const Selection& selection,
        const std::vector<std::size_t>& expected,
        const std::vector<std::string>& labels, std::size_t n,
        const float* pointer) {
    auto selected = data.slice(selection, all);
    if constexpr (check_order)
        require(selected.n_obs() == n && selected.n_vars() == 1 && selected.has_x(),
                "AnnData slice changed dimensions or lost X");
    auto dense = selected.x().dense<float>();
    if constexpr (check_order)
        require(dense.data() == pointer && dense.rank() == 2 && dense.extent(0) == n
                && dense.extent(1) == 1 && dense.stride(0) == 0 && dense.stride(1) == 0,
                "AnnData slice changed X pointer or shape");
    const auto obs = selected.obs();
    auto hash = fold(hash_seed, dense.rank());
    hash = fold(fold(hash, dense.extent(0)), dense.extent(1));
    for (std::size_t i = 0; i < n; ++i) {
        const auto name = obs.name(i);
        if constexpr (check_order)
            require(name == labels[expected[i]], "AnnData slice changed observation order");
        hash = label_hash(name, hash);
    }
    return hash;
}

void run_size(std::size_t n, std::size_t trial_count) {
    const auto repeat = iterations(n);
    const auto indices = positions(n, 17);
    const auto child_indices = positions(n, 29, 3);
    std::vector<std::size_t> composed_indices(n);
    for (std::size_t i = 0; i < n; ++i) composed_indices[i] = indices[child_indices[i]];

    Axis owned;
    {
        auto temporary = indices;
        owned = Axis::gather(temporary.data(), temporary.size());
        temporary[0] = n;
    }
    const auto expected_axis_hash = check_axis(owned, indices);
    auto retained = Axis::gather(indices.data(), n);
    require(check_axis(retained, indices) == expected_axis_hash,
            "retained gather differs from source-owned check");
    const auto sparse_axis_hash = axis_probe_hash(retained, n);
    auto raw_before = Axis::gather(indices.data(), n);
    require(check_axis(raw_before, indices) == expected_axis_hash,
            "raw gather order changed before timing");
    auto normalized_before_input = indices;
    auto normalized_before = detail::gather_axis(std::move(normalized_before_input));
    require(check_axis(normalized_before, indices) == expected_axis_hash,
            "normalized gather order changed before timing");

    const auto raw_count = count_once([&] { return raw_gather(indices); });
    const auto raw_trials = time_case(trial_count, repeat, sparse_axis_hash,
        [&] {
            auto axis = Axis::gather(indices.data(), n);
            return axis_probe_hash(axis, n);
        });
    report("raw_axis_gather", n, repeat, trial_count, raw_count, raw_trials);

    const auto normalized_count = count_once([&] { return normalized_gather(indices); });
    const auto normalized_trials = time_case(trial_count, repeat, sparse_axis_hash,
        [&] {
            auto copy = indices;
            auto axis = detail::gather_axis(std::move(copy));
            return axis_probe_hash(axis, n);
        });
    report("normalized_gather_axis", n, repeat, trial_count, normalized_count,
           normalized_trials);
    auto raw_after = Axis::gather(indices.data(), n);
    require(check_axis(raw_after, indices) == expected_axis_hash,
            "raw gather order changed after timing");
    auto normalized_after_input = indices;
    auto normalized_after = detail::gather_axis(std::move(normalized_after_input));
    require(check_axis(normalized_after, indices) == expected_axis_hash,
            "normalized gather order changed after timing");

    const auto reuse_count = count_once([&] {
        Axis copy = retained;
        return check_axis(copy, indices);
    });
    const auto reuse_trials = time_case(trial_count, repeat, expected_axis_hash,
        [&] { Axis copy = retained; return axis_hash(copy, n); });
    report("retained_axis_copy_index", n, repeat, trial_count, reuse_count, reuse_trials);

    float scalar = 7.25f;
    const auto parent = Axis::gather(indices.data(), n);
    const auto child = Axis::gather(child_indices.data(), n);
    auto source = Array::borrow_dense(&scalar, {n}, {0}, Validation::none);
    const auto parent_view = source.view().select({parent});
    auto expected_map_hash = hash_seed;
    // A broadcast source exposes pointer and shape, so validate its composed
    // order through the retained parent and child axes.
    for (std::size_t i = 0; i < n; ++i) {
        require(parent.index(child.index(i)) == composed_indices[i],
                "parent-child gather composition changed order");
        expected_map_hash = fold(expected_map_hash, composed_indices[i]);
    }
    std::vector<float> ordered_values(n);
    for (std::size_t i = 0; i < n; ++i) ordered_values[i] = static_cast<float>(i);
    auto ordered_source = Array::borrow_dense(ordered_values.data(), {n}, {1}, Validation::none);
    const auto ordered_parent = ordered_source.view().select({parent});
    auto materialized = ordered_parent.select({child}).copy();
    const auto ordered = materialized.view().dense<float>();
    require(ordered.data() != ordered_values.data() && ordered.rank() == 1
            && ordered.extent(0) == n, "composed gather materialization changed ownership or shape");
    for (std::size_t i = 0; i < n; ++i)
        require(ordered.data()[i] == static_cast<float>(composed_indices[i]),
                "composed ArrayView changed selected order");
    const auto compose_checksum = compose_view(parent_view, child, n, &scalar, expected_map_hash);
    const auto compose_count = count_once([&] {
        return compose_view(parent_view, child, n, &scalar, expected_map_hash);
    });
    const auto compose_trials = time_case(trial_count, repeat, compose_checksum,
        [&] { return compose_view(parent_view, child, n, &scalar, expected_map_hash); });
    report("array_view_composition", n, repeat, trial_count, compose_count, compose_trials);

    std::vector<std::string> labels;
    labels.reserve(n);
    std::vector<std::ptrdiff_t> selected_indices;
    selected_indices.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        labels.push_back(std::to_string(i));
        selected_indices.push_back(static_cast<std::ptrdiff_t>(indices[i]));
    }
    auto data = AnnData::empty(n, 1);
    data.set_obs(Table::with_index(labels));
    data.set_x(Array::borrow_dense(&scalar, {n, 1}, {0, 0}, Validation::none));
    const Selection selection(anndatapp::indices{std::move(selected_indices)});
    const auto ann_checksum = ann_data_slice<true>(data, selection, indices, labels, n, &scalar);
    const auto ann_count = count_once([&] {
        return ann_data_slice<true>(data, selection, indices, labels, n, &scalar);
    });
    const auto ann_trials = time_case(trial_count, repeat, ann_checksum,
        [&] { return ann_data_slice<false>(data, selection, indices, labels, n, &scalar); });
    report("anndatapp_slice", n, repeat, trial_count, ann_count, ann_trials);

    require(check_axis(retained, indices) == expected_axis_hash,
            "retained gather changed after probe cases");
    require(check_axis(owned, indices) == expected_axis_hash,
            "gather lost ownership after its source buffer expired");
    require(compose_view(parent_view, child, n, &scalar, expected_map_hash)
            == compose_checksum, "composed view changed after probe cases");
    for (std::size_t i = 0; i < n; ++i)
        require(parent.index(child.index(i)) == composed_indices[i],
                "parent-child gather order changed after probe cases");
    require(ann_data_slice<true>(data, selection, indices, labels, n, &scalar) == ann_checksum,
            "AnnData slice changed after probe cases");
}

std::size_t trial_count(int argc, char** argv) {
    if (argc == 1) return 3;
    require(argc == 2, "usage: gather_probe [trial-count: 1..3]");
    char* end = nullptr;
    const auto value = std::strtoul(argv[1], &end, 10);
    require(argv[1][0] && end && *end == '\0' && value >= 1 && value <= 3,
            "trial count must be 1, 2, or 3");
    return static_cast<std::size_t>(value);
}
}

int main(int argc, char** argv) {
    try {
        const auto trials = trial_count(argc, argv);
        std::cout << "{\"sample\":\"metadata\",\"scope\":\"local paired gather probe\","
                     "\"allocation_counter\":\"regular throwing scalar/array new and delete; excludes aligned/nothrow and direct C allocation\"}\n";
        for (const std::size_t n : {64, 4096, 65536}) run_size(n, trials);
        return 0;
    } catch (const std::exception& error) {
        allocation_probe::active = false;
        std::cerr << "gather_probe: " << error.what() << '\n';
        return 1;
    }
}
