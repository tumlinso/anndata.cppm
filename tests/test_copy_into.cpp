#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

import anndatapp.core;

namespace { bool measure = false; std::size_t largest = 0; }
void* operator new(std::size_t n) {
    if (measure && n > largest) largest = n;
    if (auto* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace anndatapp;
void check(bool ok, const char* s) { if (!ok) throw std::runtime_error(s); }
template<class F> void bad(ErrorCode code, F f) {
    try { f(); } catch (const Error& e) { check(e.code() == code, "wrong error code"); return; }
    throw std::runtime_error("missing copy error");
}
template<class T> void numeric() {
    auto a = Array::empty<T>({2, 3}), b = Array::empty<T>({2, 3});
    auto x = a.edit().template dense<T>(), y = b.edit().template dense<T>();
    const T values[]{T(0), T(1), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max(), T(0), T(1)};
    for (int i = 0; i < 6; ++i) x.data()[i] = values[i];
    auto* p = y.data();
    a.view().copy_into(b.edit());
    check(y.data() == p && std::memcmp(x.data(), p, 6*sizeof(T)) == 0, "native dtype or pointer changed");
}
void gathered_and_strided() {
    auto a = Array::empty<float>({4, 5});
    auto x = a.edit().dense<float>();
    for (int i = 0; i < 20; ++i) x.data()[i] = float(i);
    auto view = a.view().select({Axis::gather({3, 0, 3}), Axis::gather({4, 1, 4})});
    std::array<float, 11> out; out.fill(-99);
    auto b = Array::borrow_dense(out.data()+9, {3, 3}, {-3, -1});
    view.copy_into(b.edit());
    const float expected[]{19, 16, 19, 4, 1, 4, 19, 16, 19};
    for (int i = 0; i < 9; ++i) check(out[9-i] == expected[i], "gather or destination stride changed values");
    check(out.front() == -99 && out.back() == -99, "copy escaped destination span");
    auto reverse = Array::borrow_dense(x.data()+19, {4, 5}, {-5, -1});
    auto selected = reverse.view().select({Axis::gather({0, 3, 1}), Axis::gather({4, 0})});
    auto c = Array::empty<float>({3, 2}); selected.copy_into(c.edit());
    const float reversed[]{15, 19, 0, 4, 10, 14};
    check(std::memcmp(c.view().dense<float>().data(), reversed, sizeof(reversed)) == 0, "signed source strides changed");
    auto mismatch = Array::zeros<double>({3, 3});
    bad(ErrorCode::dtype_mismatch, [&]{ view.copy_into(mismatch.edit()); });
    auto shape = Array::zeros<float>({9});
    bad(ErrorCode::shape_mismatch, [&]{ view.copy_into(shape.edit()); });
    auto sparse = Array::csr<float, std::int32_t, std::int32_t>(3, 3, 0);
    bad(ErrorCode::unsupported_layout, [&]{ view.copy_into(sparse.edit()); });
    bad(ErrorCode::unsupported_layout, [&]{ sparse.view().copy_into(b.edit()); });
}
void lifetime_and_allocation() {
    ArrayView selected;
    {
        auto a = Array::empty<std::int32_t>({4096});
        auto x = a.edit().dense<std::int32_t>();
        for (int i = 0; i < 4096; ++i) x.data()[i] = i;
        selected = a.view().select({Axis::gather({4095, 0, 1024, 7, 4095})});
    }
    std::array<std::int32_t, 5> out{};
    auto b = Array::borrow_dense(out.data(), {5}, {1});
    selected.copy_into(b.edit(), Validation::none);
    check(out == std::array<std::int32_t, 5>{4095, 0, 1024, 7, 4095}, "retained source lifetime");
    auto large = Array::empty<float>({65536}), target = Array::empty<float>({65536});
    auto p = large.edit().dense<float>().data();
    for (int i = 0; i < 65536; ++i) p[i] = float(i);
    largest = 0; measure = true;
    for (int i = 0; i < 3; ++i) large.view().copy_into(target.edit());
    measure = false;
    check(largest < 1024, "copy allocated numerical-sized temporary storage");
    check(target.view().dense<float>().data()[65535] == 65535, "large copy changed values");
}
void edges() {
    const auto max_step = std::numeric_limits<std::size_t>::max();
    const float small[]{2, 9}; std::array<float, 4> spaced{};
    auto singleton = Array::borrow_dense(small, {2, 1}, {1, 1});
    auto singleton_out = Array::borrow_dense(spaced.data(), {2, 1}, {2, 1});
    singleton.view().select({Axis::all(), Axis::stride(0, 1, max_step)}).copy_into(singleton_out.edit());
    check(spaced[0] == 2 && spaced[2] == 9, "irrelevant singleton selection step");
    auto wide_stride = Array::borrow_dense(small, {1, 2}, {std::numeric_limits<std::ptrdiff_t>::max(), 1});
    auto wide_out = Array::borrow_dense(spaced.data(), {1, 2}, {std::numeric_limits<std::ptrdiff_t>::max(), 2});
    wide_stride.view().copy_into(wide_out.edit());
    check(spaced[0] == 2 && spaced[2] == 9, "irrelevant singleton memory stride");
    const float value = 17;
    auto broadcast = Array::borrow_dense(&value, {max_step}, {0}, Validation::none);
    auto broadcast_out = Array::borrow_dense(spaced.data(), {2}, {2});
    broadcast.view().select({Axis::stride(0, 2, max_step-1)}).copy_into(broadcast_out.edit(), Validation::none);
    check(spaced[0] == 17 && spaced[2] == 17, "large broadcast selection step");
    auto native = Array::empty<long long>({2}), fixed = Array::empty<std::int64_t>({2});
    native.edit().dense<long long>().data()[0] = std::numeric_limits<long long>::lowest();
    native.edit().dense<long long>().data()[1] = -7;
    native.view().copy_into(fixed.edit());
    check(fixed.view().dense<std::int64_t>().data()[0] == std::numeric_limits<std::int64_t>::lowest() &&
          fixed.view().dense<std::int64_t>().data()[1] == -7, "same-width native numerical types");
    auto a = Array::empty<float>({}), b = Array::empty<float>({});
    a.edit().dense<float>().data()[0] = std::bit_cast<float>(std::uint32_t{0x7fc01234});
    a.view().copy_into(b.edit());
    check(std::memcmp(a.view().dense<float>().data(), b.view().dense<float>().data(), sizeof(float)) == 0, "scalar NaN bits");
    const auto max = std::numeric_limits<std::size_t>::max();
    auto empty = Array::empty<double>({max, 0}), other = Array::empty<double>({max, 0});
    empty.view().copy_into(other.edit());
    auto nd = Array::empty<std::uint16_t>({2, 2, 2, 2, 2}), output = Array::empty<std::uint16_t>({2, 2, 2, 2, 2});
    auto p = nd.edit().dense<std::uint16_t>().data();
    for (int i = 0; i < 32; ++i) p[i] = std::uint16_t(i);
    nd.view().select({Axis::stride(1, 2, 1, true), Axis::all(), Axis::all(), Axis::all(), Axis::all()}).copy_into(output.edit());
    auto q = output.view().dense<std::uint16_t>().data();
    for (int i = 0; i < 32; ++i) check(q[i] == ((i+16) % 32), "N-D reversed axis");
}
}
int main() {
    try {
        numeric<bool>(); numeric<std::int8_t>(); numeric<std::uint8_t>(); numeric<std::int16_t>(); numeric<std::uint16_t>();
        numeric<std::int32_t>(); numeric<std::uint32_t>(); numeric<std::int64_t>(); numeric<std::uint64_t>(); numeric<float>(); numeric<double>();
        gathered_and_strided(); lifetime_and_allocation(); edges();
        std::cout << "copy_into passed\n";
    } catch (const std::exception& e) { measure = false; std::cerr << e.what() << '\n'; return 1; }
}
