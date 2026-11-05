// Bounded single-thread probe: the same AoS kernel through a raw row, a retained
// member, and an acquired pointer/byte stride. Allocation counts use global new.
import anndatapp.core;

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace {
bool counting = false;
std::size_t allocations = 0, allocated_bytes = 0;
}
void* operator new(std::size_t n) {
    if (void* p = std::malloc(n ? n : 1)) {
        if (counting) { ++allocations; allocated_bytes += n; }
        return p;
    }
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using Clock = std::chrono::steady_clock;
struct Row { double x, y; anndatapp::Utf8Ref label; };
struct Measurement { double seconds; std::size_t allocations, bytes; };

template<class F> Measurement measure(F f) {
    allocations = allocated_bytes = 0; counting = true;
    const auto begin = Clock::now(); f(); const auto end = Clock::now();
    counting = false;
    return {std::chrono::duration<double>(end - begin).count(), allocations, allocated_bytes};
}

[[gnu::noinline]] void raw(Row* rows, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) rows[i].x += 0.125*rows[i].y;
}
[[gnu::noinline]] void member(const anndatapp::RecordMemberView<double>& x,
                              const anndatapp::RecordMemberView<double>& y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) x[i] += 0.125*y[i];
}
[[gnu::noinline]] void stride(double* x, double* y, std::ptrdiff_t step, std::size_t n) {
    auto* px = reinterpret_cast<std::byte*>(x);
    auto* py = reinterpret_cast<std::byte*>(y);
    for (std::size_t i = 0; i < n; ++i, px += step, py += step)
        *reinterpret_cast<double*>(px) += 0.125*(*reinterpret_cast<double*>(py));
}
void print(const char* name, Measurement m) {
    std::cout << '"' << name << "\":{\"seconds\":" << m.seconds
              << ",\"new_calls\":" << m.allocations << ",\"new_bytes\":" << m.bytes << '}';
}
}

int main() {
    constexpr std::size_t n = 32768, repeats = 100, acquisitions = 10000;
    std::vector<Row> rows(n);
    std::vector<std::string> labels(n);
    for (std::size_t i = 0; i < n; ++i) {
        labels[i] = std::string(48, char('a' + i % 26));
        rows[i] = {0.0, double(i % 31), {labels[i].data(), labels[i].size()}};
    }
    anndatapp::RecordLayout layout{sizeof(Row), alignof(Row), {
        {"x", anndatapp::RecordFieldKind::numeric, anndatapp::DType::float64, offsetof(Row, x), sizeof(double)},
        {"y", anndatapp::RecordFieldKind::numeric, anndatapp::DType::float64, offsetof(Row, y), sizeof(double)},
        {"label", anndatapp::RecordFieldKind::utf8_ref, anndatapp::DType::uint8, offsetof(Row, label), sizeof(anndatapp::Utf8Ref)}}};
    auto records = anndatapp::RecordArray::borrow_interleaved(rows.data(), {n}, layout);
    auto view = records.view();
    auto x = records.edit_member<double>("x"), y = records.edit_member<double>("y");
    bool pointers_ok = x.data() == &rows[0].x && y.data() == &rows[0].y &&
                       x.stride_bytes(0) == sizeof(Row) && y.stride_bytes(0) == sizeof(Row);
    auto acquisition = measure([&] {
        for (std::size_t i = 0; i < acquisitions; ++i) {
            auto p = view.member<double>("x");
            pointers_ok &= p.data() == &rows[0].x;
        }
    });
    auto reset = [&] { for (auto& row : rows) row.x = 0.0; };
    auto sum = [&] { double s = 0; for (auto& row : rows) s += row.x; return s; };
    raw(rows.data(), n); member(x, y, n); stride(x.data(), y.data(), x.stride_bytes(0), n); reset();
    auto a = measure([&] { for (std::size_t i = 0; i < repeats; ++i) raw(rows.data(), n); });
    const auto raw_sum = sum();
    bool values_ok = true;
    for (auto& row : rows) values_ok &= row.x == 12.5*row.y;
    reset();
    auto b = measure([&] { for (std::size_t i = 0; i < repeats; ++i) member(x, y, n); });
    const auto member_sum = sum();
    for (auto& row : rows) values_ok &= row.x == 12.5*row.y;
    reset();
    auto c = measure([&] { for (std::size_t i = 0; i < repeats; ++i) stride(x.data(), y.data(), x.stride_bytes(0), n); });
    const auto stride_sum = sum();
    for (auto& row : rows) values_ok &= row.x == 12.5*row.y;
    anndatapp::RecordArray copy;
    auto d = measure([&] { copy = records.copy(); });
    auto text = copy.view().text("label");
    bool text_ok = text.size() == n;
    for (std::size_t i = 0; i < n; ++i)
        text_ok &= text.string(i) == labels[i] && text.string(i).data() != labels[i].data();
    labels[0][0] = 'z';
    text_ok &= text.string(0)[0] == 'a';
    const bool ok = pointers_ok && raw_sum == member_sum && raw_sum == stride_sum && text_ok && values_ok;
    std::cout << "{\"rows\":" << n << ",\"row_bytes\":" << sizeof(Row)
              << ",\"repeats\":" << repeats << ",\"text_bytes_per_row\":48,\"acquisitions\":" << acquisitions
              << ",\"ok\":" << (ok ? "true" : "false") << ",\"checksum\":" << raw_sum << ',';
    print("acquire", acquisition); std::cout << ','; print("raw", a); std::cout << ',';
    print("member", b); std::cout << ','; print("stride", c); std::cout << ',';
    print("deep_copy", d); std::cout << "}\n";
    return ok ? 0 : 1;
}
