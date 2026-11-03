// Count C++ allocations for descriptor acquisition and release only.
// This probe makes no timing or zero-allocation claim.
import anndata.core;

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace allocation_probe {
struct Counts {
    std::size_t calls = 0;
    std::size_t bytes = 0;
    std::size_t deletes = 0;
};

thread_local bool active = false;
thread_local Counts counts{};

void record_allocation(std::size_t bytes) noexcept {
    if (active) {
        ++counts.calls;
        counts.bytes += bytes;
    }
}

void record_delete(void* p) noexcept {
    if (active && p) ++counts.deletes;
}

void* allocate(std::size_t bytes, std::size_t alignment) {
    const std::size_t actual = bytes ? bytes : 1;
    if (alignment > alignof(std::max_align_t)
        && actual > std::numeric_limits<std::size_t>::max() - (alignment - 1))
        throw std::bad_alloc{};
    for (;;) {
        void* p = nullptr;
        if (alignment <= alignof(std::max_align_t)) {
            p = std::malloc(actual);
        } else {
            const std::size_t rounded = ((actual + alignment - 1) / alignment) * alignment;
            p = std::aligned_alloc(alignment, rounded);
        }
        if (p) {
            record_allocation(bytes);
            return p;
        }
        const auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc{};
        handler();
    }
}

struct Scope {
    Scope() noexcept { counts = {}; active = true; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    ~Scope() { active = false; }
    Counts close() noexcept {
        active = false;
        return counts;
    }
};
}

void* operator new(std::size_t n) { return allocation_probe::allocate(n, alignof(std::max_align_t)); }
void* operator new[](std::size_t n) { return allocation_probe::allocate(n, alignof(std::max_align_t)); }
void* operator new(std::size_t n, std::align_val_t a) {
    return allocation_probe::allocate(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
    return allocation_probe::allocate(n, static_cast<std::size_t>(a));
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n, alignof(std::max_align_t)); }
    catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n, alignof(std::max_align_t)); }
    catch (...) { return nullptr; }
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n, static_cast<std::size_t>(a)); }
    catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    try { return allocation_probe::allocate(n, static_cast<std::size_t>(a)); }
    catch (...) { return nullptr; }
}

void operator delete(void* p) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete[](void* p) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete(void* p, std::size_t) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { allocation_probe::record_delete(p); std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
    allocation_probe::record_delete(p); std::free(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
    allocation_probe::record_delete(p); std::free(p);
}

namespace {
using anndata::Array;
using anndata::Axis;
using anndata::AnnData;
using anndata::Column;
using anndata::DenseView;
using anndata::Error;

void require(bool condition, const char* message) {
    if (!condition) throw Error(anndata::ErrorCode::invalid_argument, "selection_probe", message);
}

template<class F>
allocation_probe::Counts measure(F&& f) {
    allocation_probe::Scope scope;
    f();
    return scope.close();
}

void report_counts(const char* kind, const char* operation, const char* size_key,
                   std::size_t size, const allocation_probe::Counts& c) {
    std::cout << "{\"kind\":\"" << kind << "\",\"operation\":\"" << operation
              << "\",\"" << size_key << "\":" << size
              << ",\"allocation_calls\":" << c.calls
              << ",\"requested_bytes\":" << c.bytes << ",\"deallocation_calls\":" << c.deletes << "}\n";
}

template<class F>
allocation_probe::Counts selection_sample(F&& select, const float* expected,
        std::size_t columns, std::ptrdiff_t column_stride) {
    return measure([&] {
        auto selected = select();
        auto dense = selected.template dense<float>();
        require(dense.data() == expected, "selection changed the data pointer");
        require(dense.rank() == 2, "selection changed rank");
        require(dense.extent(0) == 1 && dense.extent(1) == columns,
                "selection produced an incorrect shape");
        require(dense.stride(0) == 0 && dense.stride(1) == column_stride,
                "selection produced an incorrect stride");
        require(dense.data()[0] == *expected, "selection changed the value");
    });
}

void check_same_counts(const allocation_probe::Counts& a,
                       const allocation_probe::Counts& b, const char* message) {
    require(a.calls == b.calls && a.bytes == b.bytes,
            message);
}

void selection_probe() {
    constexpr std::size_t lengths[] = {
        16, std::size_t{1} << 20, std::size_t{1} << 32
    };
    allocation_probe::Counts reference[4]{};
    bool have_reference[4]{};

    for (std::size_t n : lengths) {
        float scalar = 7.25f;
        auto source = Array::borrow_dense(&scalar, {1, n}, {0, 0});
        auto view = source.view();

        const auto all = selection_sample(
            [&] { return view.select({Axis::all(), Axis::all()}); }, &scalar, n, 0);
        const auto forward = selection_sample(
            [&] { return view.select({Axis::all(), Axis::stride(0, n / 2, 2)}); },
            &scalar, n / 2, 0);
        const auto reverse = selection_sample(
            [&] { return view.select({Axis::all(), Axis::stride(n - 1, n, 1, true)}); },
            &scalar, n, 0);
        const auto nested = selection_sample([&] {
            auto parent = view.select({Axis::all(), Axis::stride(0, n / 2, 2)});
            return parent.select({Axis::all(), Axis::stride(1, n / 4, 2)});
        }, &scalar, n / 4, 0);
        const allocation_probe::Counts result[] = {all, forward, reverse, nested};
        const char* names[] = {"all", "forward", "reverse", "nested_affine"};
        for (std::size_t i = 0; i < 4; ++i) {
            if (have_reference[i])
                check_same_counts(reference[i], result[i],
                                  "selection allocation totals depend on logical length");
            else {
                reference[i] = result[i];
                have_reference[i] = true;
            }
            report_counts("selection", names[i], "logical_length", n, result[i]);
        }
    }
}

void add_columns(AnnData& data, std::size_t count, bool observations) {
    for (std::size_t i = 0; i < count; ++i) {
        const bool target = i + 1 == count;
        std::string name = "c" + std::to_string(i);
        if (target) name = "target";
        const float first = target ? 13.25f : static_cast<float>(i);
        const float second = target ? 29.5f : first + 0.5f;
        auto column = Column::from_values<float>({first, second});
        if (observations) data.set_obs_column(std::move(name), std::move(column));
        else data.set_var_column(std::move(name), std::move(column));
    }
}

void check_target(const DenseView<const float>& dense) {
    require(dense.data() != nullptr && dense.rank() == 1 && dense.extent(0) == 2,
            "metadata column has an incorrect shape");
    require(dense.stride(0) == 1, "metadata column has an incorrect stride");
    require(dense.data()[0] == 13.25f && dense.data()[1] == 29.5f,
            "metadata column has incorrect values");
}

void metadata_probe() {
    constexpr std::size_t column_counts[] = {8, 128, 1024};
    allocation_probe::Counts direct_obs_reference{}, direct_var_reference{};
    bool have_reference = false;

    for (std::size_t n : column_counts) {
        auto data = AnnData::empty(2, 2);
        add_columns(data, n, true);
        add_columns(data, n, false);
        require(data.n_obs() == 2 && data.n_vars() == 2,
                "metadata fixture dimensions are incorrect");

        const float* obs_pointer = nullptr;
        const float* var_pointer = nullptr;
        const auto direct_obs = measure([&] {
            auto dense = data.obs_column("target").values().dense<float>();
            check_target(dense);
            obs_pointer = dense.data();
        });
        const auto table_obs = measure([&] {
            auto dense = data.obs().column("target").values().dense<float>();
            check_target(dense);
            require(dense.data() == obs_pointer, "obs routes return different storage");
        });
        const auto direct_var = measure([&] {
            auto dense = data.var_column("target").values().dense<float>();
            check_target(dense);
            var_pointer = dense.data();
        });
        const auto table_var = measure([&] {
            auto dense = data.var().column("target").values().dense<float>();
            check_target(dense);
            require(dense.data() == var_pointer, "var routes return different storage");
        });

        if (have_reference) {
            check_same_counts(direct_obs_reference, direct_obs,
                              "direct obs acquisition totals depend on unrelated columns");
            check_same_counts(direct_var_reference, direct_var,
                              "direct var acquisition totals depend on unrelated columns");
        } else {
            direct_obs_reference = direct_obs;
            direct_var_reference = direct_var;
            have_reference = true;
        }
        report_counts("metadata", "obs_direct", "columns_per_axis", n, direct_obs);
        report_counts("metadata", "obs_table_route", "columns_per_axis", n, table_obs);
        report_counts("metadata", "var_direct", "columns_per_axis", n, direct_var);
        report_counts("metadata", "var_table_route", "columns_per_axis", n, table_var);
        std::cout << "{\"kind\":\"metadata_semantics\",\"columns_per_axis\":" << n
                  << ",\"obs_pointer_matches\":true,\"var_pointer_matches\":true"
                  << ",\"contents_match\":true}\n";
    }
}
}

int main() {
    static_assert(sizeof(std::size_t) == 8, "selection probe requires 64-bit size_t");
    std::cout << "{\"kind\":\"probe\",\"name\":\"anndata_selection_probe\","
                 "\"counter\":\"global C++ new/delete; calls and requested bytes\","
                 "\"measurement\":\"acquisition plus returned descriptor release; setup and JSON output excluded\","
                 "\"retained_descriptor_zero_allocation_claim\":false,"
                 "\"lookup_time_may_remain_linear_in_column_count\":true}\n";
    try {
        selection_probe();
        metadata_probe();
    } catch (const std::exception& error) {
        std::cerr << "selection probe failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
