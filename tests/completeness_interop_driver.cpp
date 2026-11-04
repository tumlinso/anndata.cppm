#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

import anndata;

namespace { std::atomic<bool> tracking{false}; std::atomic<std::size_t> largest{0}; }
// Allocation measurement runs in native/ASan builds; TSan owns these symbols.
#if !__has_feature(thread_sanitizer)
void* operator new(std::size_t n) {
    if (tracking.load(std::memory_order_relaxed)) {
        auto old = largest.load(std::memory_order_relaxed);
        while (old < n && !largest.compare_exchange_weak(old, n, std::memory_order_relaxed)) {}
    }
    if (auto* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#endif

namespace {
using namespace anndata;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
BackedAnnData open(std::string_view kind, const std::string& path, std::size_t scratch = 64u*1024u*1024u) {
    BackedOptions options; options.mode = AccessMode::read_write; options.scratch_bytes = scratch;
    return kind == "hdf5" ? open_h5ad(path, options) : open_zarr(path, options);
}
void gather_negative(BackedAnnData& store) {
    auto x = store.x().select({indices{34, 1, 17, 1, 0, 16, 34, 8},
                               indices{66, 2, 33, 2, 0, 31, 66, 16}});
    auto ctx = x.prepare_transfer();
    std::array<float, 64> source{};
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(10000+i);
    ctx.write_from({0, 0}, {8, 8}, DenseBuffer::from(source.data()+63, {8, 8}, {-32, -4}));
}
void broadcast(BackedAnnData& store) {
    auto x = store.x().select({indices{2, 19, 2, 32}, indices{1, 17, 33, 49, 65}});
    auto ctx = x.prepare_transfer();
    float source[]{20200, 20201, 20202, 20203, 20204};
    ctx.write_from({0, 0}, {4, 5}, DenseBuffer::from(source, {4, 5}, {0, 4}));
}
void gathered_source(BackedAnnData& store) {
    auto source = Array::empty<float>({9, 10});
    auto p = source.edit().dense<float>().data();
    for (std::size_t i = 0; i < 90; ++i) p[i] = float(20000+i);
    auto selection = source.view().select({Axis::gather({8, 0, 8, 2, 5, 7}),
                                           Axis::gather({9, 1, 9, 0, 2, 8, 3})});
    auto x = store.x().select({indices{1, 18, 1, 34, 16, 0}, indices{2, 34, 2, 66, 32, 1, 65}});
    x.write(selection);
}
void reverse(BackedAnnData& store) {
    auto x = store.x().select({range{34, 3, -3}, range{66, 0, -5}});
    auto ctx = x.prepare_transfer();
    std::array<float, 154> source{};
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(11000+i);
    ctx.write_from({0, 0}, {11, 14}, DenseBuffer::from(source.data(), {11, 14}));
}
void rectangle(BackedAnnData& store) {
    auto ctx = store.x().prepare_transfer();
    std::vector<float> source(27*51);
    for (std::size_t i = 0; i < 27; ++i)
        for (std::size_t j = 0; j < 51; ++j) source[i*51+j] = float(13000+i*100+j);
    ctx.write_from({4, 9}, {27, 51}, DenseBuffer::from(source.data(), {27, 51}));
}
void empty(BackedAnnData& store) {
    auto ctx = store.x().prepare_transfer();
    ctx.write_from({0, 0}, {0, 5}, DenseBuffer::from(static_cast<float*>(nullptr), {0, 5}));
}
void special(BackedAnnData& store) {
    auto target = store.x().select({indices{34, 0}, indices{66, 0}}).prepare_transfer();
    float source[]{-0.0f, std::numeric_limits<float>::infinity(),
                   -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    target.write_from({0, 0}, {2, 2}, DenseBuffer::from(source, {2, 2}));
}
void budget(BackedAnnData& store) {
    float source = 5;
    bool rejected = false;
    try {
        auto ctx = store.x().prepare_transfer(1024);
        ctx.write_from({0, 0}, {1, 1}, DenseBuffer::from(&source, {1, 1}));
    }
    catch (const Error& e) { rejected = e.code() == ErrorCode::invalid_argument; }
    require(rejected, "indivisible shard index must fail the scratch budget");
}
void huge(BackedAnnData& store) {
    constexpr std::ptrdiff_t n = std::ptrdiff_t{1} << 30, m = std::ptrdiff_t{1} << 20;
    auto x = store.uns().at("huge").array();
    require(x.shape() == std::vector<std::size_t>{std::size_t(n), std::size_t(m)}, "huge fixture shape");
    auto target = x.select({indices{n-1, 0, n-1, n/2}, indices{m-1, 0, m-1, m/2}}).prepare_transfer();
    std::array<float, 16> source{};
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(43000+i);
    target.write_from({0, 0}, {4, 4}, DenseBuffer::from(source.data(), {4, 4}));
}
void bounded_source(BackedAnnData& store) {
    constexpr std::size_t n = 2048, m = 1024;
    auto data = Array::empty<float>({2, 2});
    auto p = data.edit().dense<float>().data();
    for (std::size_t i = 0; i < 4; ++i) p[i] = float(51000+i);
    std::vector<std::size_t> rows(n), cols(m);
    for (std::size_t i = 0; i < n; ++i) rows[i] = i%2;
    for (std::size_t i = 0; i < m; ++i) cols[i] = i%2;
    auto source = data.view().select({Axis::gather(rows.data(), rows.size()), Axis::gather(cols.data(), cols.size())});
    auto target = store.uns().at("large").array();
    require(target.shape() == std::vector<std::size_t>{n, m}, "bounded source fixture shape");
    largest.store(0, std::memory_order_relaxed);
    tracking.store(true, std::memory_order_relaxed);
    try { target.write(source); }
    catch (...) { tracking.store(false, std::memory_order_relaxed); throw; }
    tracking.store(false, std::memory_order_relaxed);
#if !__has_feature(thread_sanitizer)
    require(largest.load(std::memory_order_relaxed) < 4u*1024u*1024u,
            "gathered source created a full-size numerical allocation");
#else
    std::cout << "allocation observation omitted under ThreadSanitizer\n";
#endif
}
template<class T> void typed(BackedAnnData& store, const char* key) {
    auto x = store.uns().at(key).array().select({indices{8, 1, 8, 0}, indices{12, 2, 12, 0, 5}});
    auto ctx = x.prepare_transfer();
    std::array<T, 20> source{};
    for (std::size_t i = 0; i < source.size(); ++i) {
        if constexpr (std::is_same_v<T, bool>) source[i] = bool(i%2);
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) source[i] = T(std::numeric_limits<T>::lowest()+T(i));
        else if constexpr (std::is_integral_v<T>) source[i] = T(std::numeric_limits<T>::max()-T(i));
        else source[i] = T(0.125*double(i)+0.25);
    }
    ctx.write_from({0, 0}, {4, 5}, DenseBuffer::from(source.data(), {4, 5}));
}
void all_types(BackedAnnData& store) {
    typed<bool>(store, "typed_b"); typed<std::int8_t>(store, "typed_i1"); typed<std::uint8_t>(store, "typed_u1");
    typed<std::int16_t>(store, "typed_i2"); typed<std::uint16_t>(store, "typed_u2");
    typed<std::int32_t>(store, "typed_i4"); typed<std::uint32_t>(store, "typed_u4");
    typed<std::int64_t>(store, "typed_i8"); typed<std::uint64_t>(store, "typed_u8");
    typed<float>(store, "typed_f4"); typed<double>(store, "typed_f8");
}
struct Worker {
    BackedAnnData store;
    TransferContext target;
    std::array<float, 9> source{};
    Worker(const std::string& path, int i)
        : store(open("zarr", path)),
          target(store.x().select({indices{(i/2)*16+1, (i/2)*16, (i/2)*16+1},
                                   indices{(i%2)*32+2, (i%2)*32+1, (i%2)*32+2}}).prepare_transfer()) {
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) source[r*3+c] = float(32000+100*i+10*r+c);
    }
};
void parallel(const std::string& path) {
    std::vector<Worker> workers; workers.reserve(4);
    for (int i = 0; i < 4; ++i) workers.emplace_back(path, i);
    for (const auto& w : workers) require(w.target.geometry().parallel_writes, "parallel Zarr capability absent");
    std::barrier gate{std::ptrdiff_t{4}};
    std::vector<std::future<void>> jobs;
    for (int i = 0; i < 4; ++i) jobs.push_back(std::async(std::launch::async, [&, i] {
        gate.arrive_and_wait();
        auto& w = workers[i];
        w.target.write_from({0, 0}, {3, 3}, DenseBuffer::from(w.source.data(), {3, 3}));
    }));
    for (auto& job : jobs) job.get();
}
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: completeness_interop_driver MODE BACKEND PATH\n"; return 2; }
    try {
        const std::string mode = argv[1], kind = argv[2], path = argv[3];
        if (mode == "parallel") { require(kind == "zarr", "parallel fixture requires Zarr"); parallel(path); return 0; }
        auto store = open(kind, path, mode == "budget" ? 1024 : mode == "bounded-source" ? 1024*1024 : 64u*1024u*1024u);
        if (mode == "gather-negative") gather_negative(store);
        else if (mode == "broadcast") broadcast(store);
        else if (mode == "gathered-source") gathered_source(store);
        else if (mode == "reverse") reverse(store);
        else if (mode == "rectangle") rectangle(store);
        else if (mode == "empty") empty(store);
        else if (mode == "special") special(store);
        else if (mode == "budget") budget(store);
        else if (mode == "huge") huge(store);
        else if (mode == "bounded-source") bounded_source(store);
        else if (mode == "all-types") all_types(store);
        else throw std::runtime_error("unknown fixture mode");
        store.flush();
    } catch (const std::exception& e) { tracking.store(false, std::memory_order_relaxed); std::cerr << e.what() << '\n'; return 1; }
}
