module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow) && __has_builtin(__builtin_add_overflow)
#define ANNDATAPP_CHECKED_INTRINSICS 1
#endif
#endif
#ifndef ANNDATAPP_CHECKED_INTRINSICS
#define ANNDATAPP_CHECKED_INTRINSICS 0
#endif

module anndatapp.core;
import :arrays;

namespace anndatapp {
static_assert(std::numeric_limits<unsigned char>::digits == 8 && sizeof(bool) == 1 &&
    sizeof(float) == 4 && sizeof(double) == 8 && std::numeric_limits<float>::is_iec559 &&
    std::numeric_limits<double>::is_iec559, "anndatapp requires 8-bit bytes and IEEE binary32/binary64 scalars");
namespace detail {
struct State {
    DType type = DType::float32;
    Layout format = Layout::dense;
    DType index_type = DType::int32;
    DType offset_type = DType::int32;
    ByteOrder value_byte_order = ByteOrder::native;
    ByteOrder indices_byte_order = ByteOrder::native;
    ByteOrder offsets_byte_order = ByteOrder::native;
    NativeType native = NativeType::single;
    NativeType native_index = NativeType::integer;
    NativeType native_offset = NativeType::integer;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    std::size_t nnz = 0;
    void* data = nullptr;
    void* indices = nullptr;
    void* offsets = nullptr;
    bool writable = true;
    bool indices_writable = true;
    bool offsets_writable = true;
    std::shared_ptr<void> data_owner;
    std::shared_ptr<void> indices_owner;
    std::shared_ptr<void> offsets_owner;
};
struct Selection {
    // Each axis maps view coordinates directly to coordinates in State.
    std::vector<Axis> axes;
};
}

Error::Error(ErrorCode code, std::string path, std::string message)
    : std::runtime_error(path.empty() ? std::move(message) : path + ": " + message), code_(code), path_(std::move(path)) {}
ErrorCode Error::code() const noexcept { return code_; }
const std::string& Error::path() const noexcept { return path_; }

Axis Axis::all() noexcept { return {}; }
Axis Axis::stride(std::size_t first, std::size_t count, std::size_t step, bool reverse) {
    Axis a;
    a.all_ = false; a.first_ = first; a.count_ = count; a.step_ = step; a.reverse_ = reverse;
    return a;
}
Axis Axis::gather(const std::size_t* positions, std::size_t count) {
    if (count && !positions) throw Error(ErrorCode::invalid_argument, "selection", "null gather positions");
    std::vector<std::size_t> copy;
    if (count) copy.assign(positions, positions + count);
    return detail::gather_axis(std::move(copy));
}
Axis Axis::gather(std::initializer_list<std::size_t> positions) {
    return gather(positions.begin(), positions.size());
}
Axis detail::gather_axis(std::vector<std::size_t>&& positions) {
    Axis a;
    a.all_ = false;
    a.count_ = positions.size();
    a.positions_ = std::make_shared<const std::vector<std::size_t>>(std::move(positions));
    return a;
}
Axis detail::compose_axis(const Axis& parent, const Axis& child, std::size_t source_extent) {
    const auto n = parent.extent(source_extent);
    if (child.is_all()) return parent;
    const auto count = child.extent(n);
    if (child.is_gather()) {
        for (std::size_t i = 0; i < count; ++i)
            if (child.index(i) >= n) throw Error(ErrorCode::out_of_bounds, "selection", "selection index out of bounds");
    } else if (count) {
        const auto first = child.first();
        if (first >= n) throw Error(ErrorCode::out_of_bounds, "selection", "selection index out of bounds");
        const auto available = child.reverse() ? first : n - 1 - first;
        if (count > 1 && child.step() && count - 1 > available / child.step())
            throw Error(ErrorCode::out_of_bounds, "selection", "selection exceeds axis bounds");
    }
    if (parent.is_all()) return child;
    if (!count) return Axis::stride(0, 0);
    if (!parent.is_gather() && !child.is_gather()) {
        const auto first = parent.index(child.first());
        std::size_t step = 1;
        if (count > 1) {
            if (child.step() && parent.step() > std::numeric_limits<std::size_t>::max()/child.step())
                throw Error(ErrorCode::out_of_bounds, "selection", "selection step overflows");
            step = parent.step()*child.step();
        }
        return Axis::stride(first, count, step, parent.reverse() != child.reverse());
    }
    std::vector<std::size_t> positions;
    positions.reserve(count);
    for (std::size_t i = 0; i < count; ++i) positions.push_back(parent.index(child.index(i)));
    return gather_axis(std::move(positions));
}

std::size_t dtype_size(DType type) noexcept {
    switch (type) {
    case DType::boolean: return sizeof(bool);
    case DType::int8: case DType::uint8: return 1;
    case DType::int16: case DType::uint16: return 2;
    case DType::int32: case DType::uint32: case DType::float32: return 4;
    case DType::int64: case DType::uint64: case DType::float64: return 8;
    }
    return 0;
}

namespace {
using State = detail::State;
using ArraySelection = detail::Selection;

[[noreturn]] void fail(ErrorCode code, const char* message) { throw Error(code, "array", message); }
void require_state(const std::shared_ptr<State>& s) { if (!s) fail(ErrorCode::invalid_argument, "array is empty or moved from"); }

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) fail(ErrorCode::invalid_argument, "array size overflow");
    return a * b;
}
std::size_t checked_byte_span(std::size_t count, std::size_t width) {
    const auto bytes = checked_mul(count, width);
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        fail(ErrorCode::invalid_argument, "buffer byte extent exceeds addressable range");
    return bytes;
}
std::size_t element_count(const std::vector<std::size_t>& shape) {
    // An empty array has zero elements even if the product of earlier
    // dimensions would overflow size_t.
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) return 0;
    std::size_t n = 1;
    for (auto d : shape) n = checked_mul(n, d);
    return n;
}
std::shared_ptr<void> allocate_bytes(std::size_t n, bool zero = false) {
    if (!n) return {};
    auto* p = ::operator new(n);
    std::shared_ptr<void> owner(p, [](void* q) { ::operator delete(q); });
    if (zero) std::memset(p, 0, n);
    return owner;
}
std::vector<std::ptrdiff_t> row_major_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::ptrdiff_t> result(shape.size());
    const bool empty = std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end();
    std::size_t s = 1;
    for (std::size_t i = shape.size(); i > 0; --i) {
        if (s > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
            if (!empty) fail(ErrorCode::invalid_argument, "stride overflow");
            s = 0; // Unreachable axes of an empty array need no addressable stride.
        }
        result[i - 1] = static_cast<std::ptrdiff_t>(s);
        if (shape[i - 1] && s > std::numeric_limits<std::size_t>::max() / shape[i - 1]) {
            if (!empty) fail(ErrorCode::invalid_argument, "array size overflow");
            s = 0;
        } else s *= shape[i - 1];
    }
    return result;
}
bool is_index_dtype(DType t) {
    switch (t) {
    case DType::int8: case DType::uint8: case DType::int16: case DType::uint16:
    case DType::int32: case DType::uint32: case DType::int64: case DType::uint64:
        return true;
    default: return false;
    }
}
template<class T> std::size_t read_nonnegative(const void* p, std::size_t i) {
    T value;
    std::memcpy(&value, static_cast<const std::byte*>(p) + i * sizeof(T), sizeof(T));
    if constexpr (std::numeric_limits<T>::is_signed) if (value < 0) fail(ErrorCode::invalid_sparse_structure, "negative sparse index or offset");
    const auto v = static_cast<std::uint64_t>(value);
    if (v > std::numeric_limits<std::size_t>::max()) fail(ErrorCode::invalid_sparse_structure, "sparse index overflows size_t");
    return static_cast<std::size_t>(v);
}
template<class F> decltype(auto) visit_index(DType type, F&& f) {
    switch (type) {
    case DType::int8: return f.template operator()<std::int8_t>();
    case DType::uint8: return f.template operator()<std::uint8_t>();
    case DType::int16: return f.template operator()<std::int16_t>();
    case DType::uint16: return f.template operator()<std::uint16_t>();
    case DType::int32: return f.template operator()<std::int32_t>();
    case DType::uint32: return f.template operator()<std::uint32_t>();
    case DType::int64: return f.template operator()<std::int64_t>();
    case DType::uint64: return f.template operator()<std::uint64_t>();
    default: fail(ErrorCode::invalid_argument, "sparse indices and offsets must be integers");
    }
}
template<class T> void write_checked(void* p, std::size_t i, std::size_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) fail(ErrorCode::invalid_sparse_structure, "sparse index does not fit original dtype");
    const auto converted = static_cast<T>(value);
    std::memcpy(static_cast<std::byte*>(p) + i * sizeof(T), &converted, sizeof(T));
}
void validate_dense_address_range(const State& s, std::size_t element_count);
void validate_state(const State& s) {
    if (s.format == Layout::dense) {
        if (s.strides.size() != s.shape.size()) fail(ErrorCode::shape_mismatch, "strides and shape must have equal rank");
        auto n = element_count(s.shape);
        checked_mul(n, dtype_size(s.type));
        if (n && !s.data) fail(ErrorCode::invalid_argument, "nonempty dense array has null data");
        if (s.data && reinterpret_cast<std::uintptr_t>(s.data) % dtype_size(s.type)) fail(ErrorCode::invalid_argument, "dense data alignment does not match dtype");
        validate_dense_address_range(s, n);
        return;
    }
    if (s.format != Layout::csr && s.format != Layout::csc) fail(ErrorCode::unsupported_layout, "invalid sparse layout");
    if (s.shape.size() != 2) fail(ErrorCode::shape_mismatch, "compressed sparse array must be two-dimensional");
    if (!is_index_dtype(s.index_type) || !is_index_dtype(s.offset_type)) fail(ErrorCode::invalid_argument, "sparse indices and offsets must be 8-bit to 64-bit integers");
    const auto major = s.format == Layout::csr ? s.shape[0] : s.shape[1];
    const auto minor = s.format == Layout::csr ? s.shape[1] : s.shape[0];
    if (major == std::numeric_limits<std::size_t>::max()) fail(ErrorCode::invalid_argument, "sparse major dimension overflow");
    // Reject impossible buffer extents before reading any caller-supplied data.
    // The caller still owns the contract that actual buffers cover these spans.
    checked_byte_span(s.nnz, dtype_size(s.type));
    checked_byte_span(s.nnz, dtype_size(s.index_type));
    checked_byte_span(major + 1, dtype_size(s.offset_type));
    if (!s.offsets || (s.nnz && (!s.data || !s.indices))) fail(ErrorCode::invalid_sparse_structure, "sparse array has a null buffer");
    if ((s.data && reinterpret_cast<std::uintptr_t>(s.data) % dtype_size(s.type)) ||
        (s.indices && reinterpret_cast<std::uintptr_t>(s.indices) % dtype_size(s.index_type)) ||
        reinterpret_cast<std::uintptr_t>(s.offsets) % dtype_size(s.offset_type)) fail(ErrorCode::invalid_argument, "sparse buffer alignment does not match dtype");
    visit_index(s.offset_type, [&]<class Offset> {
        if (read_nonnegative<Offset>(s.offsets, 0) != 0)
            fail(ErrorCode::invalid_sparse_structure, "sparse first offset must be zero");
        std::size_t last = 0;
        for (std::size_t i = 1; i <= major; ++i) {
            const auto next = read_nonnegative<Offset>(s.offsets, i);
            if (next < last || next > s.nnz) fail(ErrorCode::invalid_sparse_structure, "sparse offsets must be monotone and within nnz");
            last = next;
        }
        if (last != s.nnz) fail(ErrorCode::invalid_sparse_structure, "sparse final offset differs from nnz");
    });
    visit_index(s.index_type, [&]<class Index> {
        for (std::size_t i = 0; i < s.nnz; ++i)
            if (read_nonnegative<Index>(s.indices, i) >= minor)
                fail(ErrorCode::invalid_sparse_structure, "sparse index exceeds minor dimension");
    });
}
std::size_t view_extent(const State& s, const std::shared_ptr<const ArraySelection>& sel, std::size_t axis) {
    if (axis >= s.shape.size()) fail(ErrorCode::out_of_bounds, "axis out of bounds");
    return sel ? sel->axes[axis].extent(s.shape[axis]) : s.shape[axis];
}
std::size_t original_index(const std::shared_ptr<const ArraySelection>& sel, std::size_t axis, std::size_t i) {
    return sel ? sel->axes[axis].index(i) : i;
}
bool has_selection(const std::shared_ptr<const ArraySelection>& sel) {
    if (!sel) return false;
    for (const auto& a : sel->axes) if (!a.is_all()) return true;
    return false;
}
bool is_contiguous(const detail::DenseDimensions& dimensions) {
    for (std::size_t i = 0; i < dimensions.size; ++i) if (!dimensions.extent(i)) return true;
    std::size_t expected = 1;
    for (std::size_t i = dimensions.size; i > 0; --i) {
        const auto extent = dimensions.extent(i - 1);
        const auto stride = dimensions.stride(i - 1);
        if (extent > 1 && (stride < 0 || static_cast<std::size_t>(stride) != expected)) return false;
        expected = checked_mul(expected, extent);
    }
    return true;
}
std::ptrdiff_t checked_signed_mul(std::ptrdiff_t a, std::ptrdiff_t b) {
#if ANNDATAPP_CHECKED_INTRINSICS
    std::ptrdiff_t result;
    if (__builtin_mul_overflow(a, b, &result)) fail(ErrorCode::invalid_argument, "stride or offset overflow");
    return result;
#else
    const bool negative = (a < 0) != (b < 0);
    const auto x = a < 0 ? std::uintmax_t(-(a + 1)) + 1 : std::uintmax_t(a);
    const auto y = b < 0 ? std::uintmax_t(-(b + 1)) + 1 : std::uintmax_t(b);
    const auto limit = std::uintmax_t(std::numeric_limits<std::ptrdiff_t>::max()) + negative;
    if (y && x > limit/y) fail(ErrorCode::invalid_argument, "stride or offset overflow");
    const auto n = x*y;
    if (negative && n == limit) return std::numeric_limits<std::ptrdiff_t>::min();
    return negative ? -std::ptrdiff_t(n) : std::ptrdiff_t(n);
#endif
}
std::ptrdiff_t checked_signed_add(std::ptrdiff_t a, std::ptrdiff_t b) {
#if ANNDATAPP_CHECKED_INTRINSICS
    std::ptrdiff_t result;
    if (__builtin_add_overflow(a, b, &result)) fail(ErrorCode::invalid_argument, "stride or offset overflow");
    return result;
#else
    if ((b > 0 && a > std::numeric_limits<std::ptrdiff_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<std::ptrdiff_t>::min() - b))
        fail(ErrorCode::invalid_argument, "stride or offset overflow");
    return a + b;
#endif
}
std::ptrdiff_t signed_index(std::size_t i) {
    if (i > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) fail(ErrorCode::invalid_argument, "index exceeds addressable range");
    return static_cast<std::ptrdiff_t>(i);
}

std::ptrdiff_t extent_stride(std::size_t count, std::ptrdiff_t stride) {
    if (!count || !stride) return 0;
    const auto positive_limit = static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto negative_limit = positive_limit + std::uintmax_t{1};
    const auto magnitude = stride < 0
        ? static_cast<std::uintmax_t>(-(stride + 1)) + std::uintmax_t{1}
        : static_cast<std::uintmax_t>(stride);
    const auto limit = stride < 0 ? negative_limit : positive_limit;
    if (static_cast<std::uintmax_t>(count) > limit / magnitude)
        fail(ErrorCode::invalid_argument, "dense strided element offset exceeds addressable range");
    const auto product = static_cast<std::uintmax_t>(count) * magnitude;
    if (stride >= 0) return static_cast<std::ptrdiff_t>(product);
    if (product == negative_limit) return std::numeric_limits<std::ptrdiff_t>::min();
    return -static_cast<std::ptrdiff_t>(product);
}

void validate_dense_address_range(const State& s, std::size_t n) {
    if (!n) return;
    std::ptrdiff_t minimum = 0, maximum = 0;
    for (std::size_t axis = 0; axis < s.shape.size(); ++axis) {
        const auto term = extent_stride(s.shape[axis] - 1, s.strides[axis]);
        if (term < 0) minimum = checked_signed_add(minimum, term);
        else maximum = checked_signed_add(maximum, term);
    }
    const auto width = static_cast<std::ptrdiff_t>(dtype_size(s.type));
    // Include the last byte of the last element, not just element origins.
    const auto min_byte = checked_signed_mul(minimum, width);
    const auto max_byte = checked_signed_mul(maximum, width);
    const auto max_end_byte = checked_signed_add(max_byte, width - 1);
    // Both directions must fit individually, and so must their combined
    // buffer span; subtracting the signed endpoints directly could overflow.
    const auto left = min_byte < 0
        ? static_cast<std::uintmax_t>(-(min_byte + 1)) + std::uintmax_t{1}
        : std::uintmax_t{0};
    const auto right = static_cast<std::uintmax_t>(max_end_byte);
    const auto limit = static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (left >= limit || right >= limit - left)
        fail(ErrorCode::invalid_argument, "dense buffer byte extent exceeds addressable range");
}
}

std::ptrdiff_t detail::checked_byte_stride(std::ptrdiff_t stride, std::size_t element_size) {
    if (element_size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        fail(ErrorCode::invalid_argument, "element size exceeds addressable range");
    return checked_signed_mul(stride, static_cast<std::ptrdiff_t>(element_size));
}

detail::DenseDescriptor detail::borrowed_dense_descriptor(const void* data, std::size_t rank,
    const std::size_t* shape, const std::ptrdiff_t* strides, std::size_t width, Validation validation) {
    if (rank && (!shape || !strides)) fail(ErrorCode::invalid_argument, "null dimension metadata");
    DenseDescriptor d; d.data = const_cast<void*>(data); d.dimensions.size = rank;
    if (rank > DenseDimensions::inline_rank) {
        d.dimensions.large = std::make_shared<LargeDimensions>();
        d.dimensions.large->shape.assign(shape, shape + rank);
        d.dimensions.large->strides.assign(strides, strides + rank);
    } else for (std::size_t a = 0; a < rank; ++a) {
        d.dimensions.shape[a] = shape[a]; d.dimensions.strides[a] = strides[a];
    }
    bool empty = false;
    for (std::size_t a = 0; a < rank; ++a) empty = empty || shape[a] == 0;
    std::size_t n = empty ? 0 : 1;
    if (!empty) for (std::size_t a = 0; a < rank; ++a) n = checked_mul(n, shape[a]);
    if (validation == Validation::full) checked_mul(n, width);
    if (validation == Validation::full && n) {
        if (!data || reinterpret_cast<std::uintptr_t>(data) % width)
            fail(ErrorCode::invalid_argument, "dense pointer is null or misaligned");
        std::ptrdiff_t lo = 0, hi = 0;
        for (std::size_t a = 0; a < rank; ++a) {
            const auto k = extent_stride(shape[a] - 1, strides[a]);
            if (k < 0) lo = checked_signed_add(lo, k);
            else hi = checked_signed_add(hi, k);
        }
        const auto low = checked_signed_mul(lo, std::ptrdiff_t(width));
        const auto high = checked_signed_add(checked_signed_mul(hi, std::ptrdiff_t(width)), std::ptrdiff_t(width) - 1);
        const auto left = low < 0 ? std::uintmax_t(-(low + 1)) + 1 : std::uintmax_t{0};
        if (left >= std::uintmax_t(PTRDIFF_MAX) || std::uintmax_t(high) >= std::uintmax_t(PTRDIFF_MAX) - left)
            fail(ErrorCode::invalid_argument, "dense byte span exceeds addressable range");
    }
    d.contiguous = is_contiguous(d.dimensions);
    return d;
}

Array detail::empty_like(const ArrayView& source, std::size_t length) {
    auto out = Array::allocate_dense(source.dtype(), source.native_type(), {length}, false);
    out.set_serialization_byte_order(source.serialization_byte_order());
    return out;
}

Array::Array(std::shared_ptr<State> state) : state_(std::move(state)) {}
Array::~Array() = default;
Array::Array(Array&&) noexcept = default;
Array& Array::operator=(Array&&) noexcept = default;

Array Array::allocate_dense(DType type, detail::NativeType native, std::vector<std::size_t> shape, bool zero) {
    auto s = std::make_shared<State>();
    s->type = type; s->native = native;
    s->shape = std::move(shape);
    s->strides = row_major_strides(s->shape);
    s->data_owner = allocate_bytes(checked_byte_span(element_count(s->shape), dtype_size(type)), zero);
    s->data = s->data_owner.get();
    return Array(std::move(s));
}
Array Array::allocate_sparse(Layout layout, DType type, DType index, DType offset, detail::NativeType native, detail::NativeType native_index, detail::NativeType native_offset, std::size_t rows, std::size_t columns, std::size_t nnz) {
    if (layout != Layout::csr && layout != Layout::csc) fail(ErrorCode::unsupported_layout, "compressed sparse layout must be CSR or CSC");
    if (!is_index_dtype(index) || !is_index_dtype(offset)) fail(ErrorCode::invalid_argument, "sparse index and offset dtype must be 8-bit to 64-bit integer");
    auto major = layout == Layout::csr ? rows : columns;
    if (major == std::numeric_limits<std::size_t>::max()) fail(ErrorCode::invalid_argument, "sparse major dimension overflow");
    auto s = std::make_shared<State>();
    s->format = layout; s->type = type; s->index_type = index; s->offset_type = offset; s->shape = {rows, columns}; s->nnz = nnz;
    s->native = native; s->native_index = native_index; s->native_offset = native_offset;
    const auto data_bytes = checked_byte_span(nnz, dtype_size(type));
    const auto index_bytes = checked_byte_span(nnz, dtype_size(index));
    const auto offset_bytes = checked_byte_span(major + 1, dtype_size(offset));
    s->data_owner = allocate_bytes(data_bytes);
    s->indices_owner = allocate_bytes(index_bytes);
    s->offsets_owner = allocate_bytes(offset_bytes, true);
    s->data = s->data_owner.get(); s->indices = s->indices_owner.get(); s->offsets = s->offsets_owner.get();
    return Array(std::move(s));
}
Array Array::wrap_dense(DType type, detail::NativeType native, void* data, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides, std::shared_ptr<void> owner, bool writable, Validation validation) {
    auto s = std::make_shared<State>();
    s->type = type; s->native = native; s->shape = std::move(shape); s->strides = std::move(strides); s->data = data; s->data_owner = std::move(owner); s->writable = writable;
    if (validation == Validation::full) validate_state(*s);
    return Array(std::move(s));
}
Array Array::wrap_sparse(Layout layout, DType type, DType index, DType offset, detail::NativeType native, detail::NativeType native_index, detail::NativeType native_offset, std::size_t rows, std::size_t columns, std::size_t nnz, void* values, void* indices, void* offsets, std::shared_ptr<void> owner, bool writable, bool indices_writable, bool offsets_writable, Validation validation) {
    auto s = std::make_shared<State>();
    s->format = layout; s->type = type; s->index_type = index; s->offset_type = offset; s->shape = {rows, columns}; s->nnz = nnz;
    s->native = native; s->native_index = native_index; s->native_offset = native_offset;
    s->data = values; s->indices = indices; s->offsets = offsets; s->data_owner = std::move(owner); s->writable = writable;
    s->indices_writable = indices_writable; s->offsets_writable = offsets_writable;
    if (validation == Validation::full) validate_state(*s);
    return Array(std::move(s));
}
Array Array::from_sparse_buffers(Layout layout, Array values, Array indices, Array offsets,
        std::size_t rows, std::size_t columns, Validation validation) {
    if (layout != Layout::csr && layout != Layout::csc)
        fail(ErrorCode::unsupported_layout, "compressed sparse layout must be CSR or CSC");
    auto check_component = [](const Array& component) {
        require_state(component.state_);
        if (component.state_->format != Layout::dense || component.state_->shape.size() != 1 ||
            (component.state_->shape[0] > 1 && component.state_->strides[0] != 1))
            fail(ErrorCode::unsupported_layout, "sparse components must be contiguous one-dimensional arrays");
    };
    check_component(values); check_component(indices); check_component(offsets);
    if (!is_index_dtype(indices.dtype()) || !is_index_dtype(offsets.dtype()))
        fail(ErrorCode::invalid_argument, "sparse index and offset dtype must be 8-bit to 64-bit integer");
    const auto major = layout == Layout::csr ? rows : columns;
    if (major == std::numeric_limits<std::size_t>::max()) fail(ErrorCode::invalid_argument, "sparse major dimension overflow");
    if (values.extent(0) != indices.extent(0) || offsets.extent(0) != major + 1)
        fail(ErrorCode::shape_mismatch, "sparse component lengths do not match compressed shape");
    auto s = std::make_shared<State>();
    s->format = layout; s->shape = {rows, columns}; s->nnz = values.extent(0);
    s->type = values.dtype(); s->index_type = indices.dtype(); s->offset_type = offsets.dtype();
    s->native = values.native_type(); s->native_index = indices.native_type(); s->native_offset = offsets.native_type();
    s->value_byte_order = values.serialization_byte_order();
    s->indices_byte_order = indices.serialization_byte_order();
    s->offsets_byte_order = offsets.serialization_byte_order();
    s->data = values.state_->data; s->indices = indices.state_->data; s->offsets = offsets.state_->data;
    s->data_owner = std::shared_ptr<void>(values.state_, s->data);
    s->indices_owner = std::shared_ptr<void>(indices.state_, s->indices);
    s->offsets_owner = std::shared_ptr<void>(offsets.state_, s->offsets);
    s->writable = values.state_->writable;
    s->indices_writable = indices.state_->writable;
    s->offsets_writable = offsets.state_->writable;
    if (validation == Validation::full) validate_state(*s);
    return Array(std::move(s));
}
std::size_t Array::rank() const noexcept { return state_ ? state_->shape.size() : 0; }
std::size_t Array::extent(std::size_t axis) const { return view().extent(axis); }
DType Array::dtype() const { return view().dtype(); }
ByteOrder Array::serialization_byte_order() const { return view().serialization_byte_order(); }
ByteOrder Array::index_serialization_byte_order() const { return view().index_serialization_byte_order(); }
ByteOrder Array::offset_serialization_byte_order() const { return view().offset_serialization_byte_order(); }
Layout Array::layout() const { return view().layout(); }
NativeType Array::native_type() const { return view().native_type(); }
ArrayView Array::view() const { require_state(state_); return ArrayView(state_); }
ArrayEdit Array::edit() { require_state(state_); if (!state_->writable) fail(ErrorCode::invalid_argument, "array was created from read-only storage"); return ArrayEdit(state_); }
Array Array::copy(Validation validation) const { return view().copy(validation); }
void Array::set_serialization_byte_order(ByteOrder value) {
    require_state(state_); state_->value_byte_order = value;
}
void Array::set_index_serialization_byte_order(ByteOrder value) {
    require_state(state_);
    if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse indices");
    state_->indices_byte_order = value;
}
void Array::set_offset_serialization_byte_order(ByteOrder value) {
    require_state(state_);
    if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse offsets");
    state_->offsets_byte_order = value;
}
void Array::validate() const { view().validate(); }

ArrayView::ArrayView(std::shared_ptr<State> state, std::shared_ptr<const ArraySelection> selection) : state_(std::move(state)), selection_(std::move(selection)) {}
ArrayEdit::ArrayEdit(std::shared_ptr<State> state) : state_(std::move(state)) {}
ArrayView ArrayEdit::view() const { require_state(state_); return ArrayView(state_); }
DenseBytesView<std::byte> ArrayEdit::bytes() const {
    auto v = view();
    return {v.dense_descriptor(state_->type, state_->native, true), dtype_size(state_->type)};
}
SparseBytesView<std::byte> ArrayEdit::sparse_bytes() const {
    require_state(state_);
    if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse buffers");
    const auto descriptor = view().sparse_descriptor(state_->format, state_->type, state_->index_type,
        state_->offset_type, state_->native, state_->native_index, state_->native_offset);
    return {descriptor, dtype_size(state_->type), dtype_size(state_->index_type), dtype_size(state_->offset_type)};
}
std::size_t ArrayView::rank() const noexcept { return state_ ? state_->shape.size() : 0; }
std::size_t ArrayView::extent(std::size_t axis) const { require_state(state_); return view_extent(*state_, selection_, axis); }
DType ArrayView::dtype() const { require_state(state_); return state_->type; }
Layout ArrayView::layout() const { require_state(state_); return state_->format; }
NativeType ArrayView::native_type() const { require_state(state_); return state_->native; }
DType ArrayView::index_dtype() const { require_state(state_); if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse indices"); return state_->index_type; }
DType ArrayView::offset_dtype() const { require_state(state_); if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse offsets"); return state_->offset_type; }
ByteOrder ArrayView::serialization_byte_order() const { require_state(state_); return state_->value_byte_order; }
ByteOrder ArrayView::index_serialization_byte_order() const { require_state(state_); if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse indices"); return state_->indices_byte_order; }
ByteOrder ArrayView::offset_serialization_byte_order() const { require_state(state_); if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse offsets"); return state_->offsets_byte_order; }
NativeType ArrayView::index_native_type() const { require_state(state_); if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse indices"); return state_->native_index; }
NativeType ArrayView::offset_native_type() const { require_state(state_); if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse offsets"); return state_->native_offset; }
void ArrayView::validate() const { require_state(state_); validate_state(*state_); }

ArrayView ArrayView::select(const Axis* axes, std::size_t count) const {
    require_state(state_);
    if (count != rank()) fail(ErrorCode::shape_mismatch, "number of selectors differs from array rank");
    if (count && !axes) fail(ErrorCode::invalid_argument, "null axis selectors");
    bool same = true;
    for (std::size_t a = 0; a < count; ++a) same = same && axes[a].is_all();
    if (same) return *this;
    auto result = std::make_shared<ArraySelection>();
    result->axes.resize(rank());
    for (std::size_t a = 0; a < rank(); ++a) {
        const auto parent = selection_ ? selection_->axes[a] : Axis::all();
        result->axes[a] = detail::compose_axis(parent, axes[a], state_->shape[a]);
    }
    return ArrayView(state_, std::move(result));
}

detail::DenseDescriptor ArrayView::dense_descriptor(DType type, detail::NativeType native, bool checked) const {
    require_state(state_);
    if (state_->format != Layout::dense) fail(ErrorCode::unsupported_layout, "array is not dense");
    if (checked && (state_->type != type || state_->native != native)) fail(ErrorCode::dtype_mismatch, "requested C++ type does not match array storage type");
    detail::DenseDescriptor d;
    d.owner = state_; d.data = state_->data;
    d.dimensions.size = rank();
    if (rank() > detail::DenseDimensions::inline_rank) {
        d.dimensions.large = std::make_shared<detail::LargeDimensions>();
        d.dimensions.large->shape.resize(rank());
        d.dimensions.large->strides = state_->strides;
    }
    std::ptrdiff_t origin = 0;
    bool empty = false;
    for (std::size_t a = 0; a < rank(); ++a) if (extent(a) == 0) { empty = true; break; }
    for (std::size_t a = 0; a < rank(); ++a) {
        const auto e = extent(a);
        if (d.dimensions.large) d.dimensions.large->shape[a] = e;
        else { d.dimensions.shape[a] = e; d.dimensions.strides[a] = state_->strides[a]; }
        if (empty) continue; // No coordinate in this view can be dereferenced.
        if (!selection_ || selection_->axes[a].is_all()) continue;
        const auto& v = selection_->axes[a];
        // All indices along a broadcast axis address the same element, even
        // when those indices cannot themselves be represented in ptrdiff_t.
        if (state_->strides[a] == 0) continue;
        auto start = signed_index(v.index(0));
        std::ptrdiff_t step = 1;
        if (e > 1) {
            if (v.is_gather()) {
                step = signed_index(v.index(1)) - start;
                for (std::size_t i = 2; i < e; ++i) {
                    auto expected = checked_signed_add(start, checked_signed_mul(signed_index(i), step));
                    if (signed_index(v.index(i)) != expected) fail(ErrorCode::non_contiguous, "gathered selection has no direct strided view; copy it first");
                }
            } else {
                step = signed_index(v.step());
                if (v.reverse()) step = -step;
            }
        }
        origin = checked_signed_add(origin, checked_signed_mul(start, state_->strides[a]));
        auto stride = checked_signed_mul(state_->strides[a], step);
        if (d.dimensions.large) d.dimensions.large->strides[a] = stride;
        else d.dimensions.strides[a] = stride;
    }
    if (d.data && !empty) {
        auto bytes = checked_signed_mul(origin, static_cast<std::ptrdiff_t>(dtype_size(state_->type)));
        d.data = static_cast<std::byte*>(d.data) + bytes;
    }
    d.contiguous = is_contiguous(d.dimensions);
    return d;
}
detail::SparseDescriptor ArrayView::sparse_descriptor(Layout layout, DType type, DType index, DType offset, detail::NativeType native, detail::NativeType native_index, detail::NativeType native_offset) const {
    require_state(state_);
    if (state_->format != layout) fail(ErrorCode::unsupported_layout, "requested sparse layout differs from array layout");
    if (state_->type != type || state_->index_type != index || state_->offset_type != offset ||
        state_->native != native || state_->native_index != native_index || state_->native_offset != native_offset)
        fail(ErrorCode::dtype_mismatch, "requested sparse buffer C++ type differs from stored type");
    if (has_selection(selection_)) fail(ErrorCode::non_contiguous, "selected sparse data requires explicit copy before buffer access");
    return {state_, state_->data, state_->indices, state_->offsets, state_->shape[0], state_->shape[1], state_->nnz,
            state_->indices_writable, state_->offsets_writable};
}
DenseBytesView<const std::byte> ArrayView::bytes() const {
    require_state(state_);
    return {dense_descriptor(state_->type, state_->native, true), dtype_size(state_->type)};
}
SparseBytesView<const std::byte> ArrayView::sparse_bytes() const {
    require_state(state_);
    if (state_->format == Layout::dense) fail(ErrorCode::unsupported_layout, "dense arrays have no sparse buffers");
    const auto descriptor = sparse_descriptor(state_->format, state_->type, state_->index_type,
        state_->offset_type, state_->native, state_->native_index, state_->native_offset);
    return {descriptor, dtype_size(state_->type), dtype_size(state_->index_type), dtype_size(state_->offset_type)};
}

void ArrayView::copy_into(const ArrayEdit& destination, Validation validation) const {
    require_state(state_);
    const auto target = destination.view();
    if (layout() != Layout::dense || target.layout() != Layout::dense)
        fail(ErrorCode::unsupported_layout, "copy_into requires dense numerical arrays");
    if (dtype() != target.dtype()) fail(ErrorCode::dtype_mismatch, "copy_into dtypes differ");
    const auto r = rank(), width = dtype_size(state_->type);
    if (r != target.rank()) fail(ErrorCode::shape_mismatch, "copy_into ranks differ");
    std::vector<std::size_t> shape(r), coordinate(r);
    for (std::size_t a = 0; a < r; ++a) {
        shape[a] = extent(a);
        if (shape[a] != target.extent(a)) fail(ErrorCode::shape_mismatch, "copy_into shapes differ");
    }
    if (validation == Validation::full) { validate(); target.validate(); }
    const auto n = element_count(shape), byte_count = checked_mul(n, width);
    if (!n) return;
    if (state_->strides.size() != r) fail(ErrorCode::shape_mismatch, "source stride rank differs");
    auto out = destination.bytes();
    const auto* input = static_cast<const std::byte*>(state_->data);
    if (!input || !out.data()) fail(ErrorCode::invalid_argument, "copy_into has a null buffer");
    bool gathered = false;
    if (selection_) for (const auto& a : selection_->axes) gathered = gathered || a.is_gather();
    if (!gathered && out.contiguous()) {
        auto in = bytes();
        if (in.contiguous()) { std::memcpy(out.data(), in.data(), byte_count); return; }
    }
    std::vector<std::ptrdiff_t> source_stride(r), target_stride(r);
    for (std::size_t a = 0; a < r; ++a) {
        source_stride[a] = state_->shape[a] > 1 ? detail::checked_byte_stride(state_->strides[a], width) : 0;
        target_stride[a] = shape[a] > 1 ? out.stride_bytes(a) : 0;
    }
    std::size_t run = 1;
    if (r && shape.back() > 1 && source_stride.back() && target_stride.back() == static_cast<std::ptrdiff_t>(width)) {
        const auto axis = selection_ ? selection_->axes.back() : Axis::all();
        if (!axis.is_gather()) {
            auto step = axis.is_all() ? std::ptrdiff_t{1} : signed_index(axis.step());
            if (axis.reverse()) step = -step;
            if (checked_signed_mul(source_stride.back(), step) == static_cast<std::ptrdiff_t>(width)) run = shape.back();
        }
    }
    const auto run_bytes = checked_mul(run, width), outer = n / run;
    const auto dimensions = r - (run > 1 ? 1 : 0);
    // Validated strides and selectors make these offsets ordinary arithmetic.
    // Trusted callers retain the same usable-buffer preconditions as borrowing.
    auto transfer = [&]<std::size_t Width>() {
        for (std::size_t i = 0; i < outer; ++i) {
            std::ptrdiff_t from = 0, to = 0;
            for (std::size_t a = 0; a < r; ++a) {
                if (source_stride[a]) from += static_cast<std::ptrdiff_t>(original_index(selection_, a, coordinate[a])) * source_stride[a];
                to += static_cast<std::ptrdiff_t>(coordinate[a]) * target_stride[a];
            }
            if (run == 1) std::memcpy(out.data() + to, input + from, Width);
            else std::memcpy(out.data() + to, input + from, run_bytes);
            for (std::size_t a = dimensions; a > 0; --a) {
                if (++coordinate[a-1] < shape[a-1]) break;
                coordinate[a-1] = 0;
            }
        }
    };
    switch (width) {
    case 1: transfer.template operator()<1>(); break;
    case 2: transfer.template operator()<2>(); break;
    case 4: transfer.template operator()<4>(); break;
    case 8: transfer.template operator()<8>(); break;
    default: fail(ErrorCode::unsupported_dtype, "copy_into has an unsupported scalar width");
    }
}

Array ArrayView::copy(Validation validation) const {
    require_state(state_);
    if (state_->format == Layout::dense) {
        std::vector<std::size_t> shape;
        for (std::size_t a = 0; a < rank(); ++a) shape.push_back(extent(a));
        auto result = Array::allocate_dense(state_->type, state_->native, shape, false);
        result.state_->value_byte_order = state_->value_byte_order;
        // Owning dense copy historically trusts the source storage. Reuse the
        // caller-buffer walker without adding whole-source validation (notably
        // for trusted zero-stride broadcasts with enormous logical extents).
        copy_into(result.edit(), Validation::none);
        return result;
    }
    const std::size_t major_axis = state_->format == Layout::csr ? 0 : 1, minor_axis = 1 - major_axis;
    if (!has_selection(selection_)) {
        auto result = Array::allocate_sparse(state_->format, state_->type, state_->index_type,
                                              state_->offset_type, state_->native, state_->native_index,
                                              state_->native_offset, state_->shape[0], state_->shape[1], state_->nnz);
        result.state_->value_byte_order = state_->value_byte_order;
        result.state_->indices_byte_order = state_->indices_byte_order;
        result.state_->offsets_byte_order = state_->offsets_byte_order;
        if (state_->nnz) {
            std::memcpy(result.state_->data, state_->data, checked_mul(state_->nnz, dtype_size(state_->type)));
            std::memcpy(result.state_->indices, state_->indices, checked_mul(state_->nnz, dtype_size(state_->index_type)));
        }
        std::memcpy(result.state_->offsets, state_->offsets,
                    checked_mul(state_->shape[major_axis] + 1, dtype_size(state_->offset_type)));
        return result;
    }
    const auto out_major = extent(major_axis), out_minor = extent(minor_axis);
    if (out_major == std::numeric_limits<std::size_t>::max())
        fail(ErrorCode::invalid_argument, "sparse major dimension overflow");
    // Check both the stored offsets and the size_t staging vector before
    // building a minor-index map or traversing the selected major axis.
    checked_byte_span(out_major + 1, dtype_size(state_->offset_type));
    checked_byte_span(out_major + 1, sizeof(std::size_t));
    if (validation == Validation::full) validate_state(*state_);
    const auto width = dtype_size(state_->type), index_width = dtype_size(state_->index_type);
    const auto minor = selection_->axes[minor_axis];
    std::vector<std::size_t> offsets(out_major + 1, 0);
    auto allocate = [&](std::size_t nnz) {
        auto out = Array::allocate_sparse(state_->format, state_->type, state_->index_type, state_->offset_type,
            state_->native, state_->native_index, state_->native_offset, extent(0), extent(1), nnz);
        out.state_->value_byte_order = state_->value_byte_order;
        out.state_->indices_byte_order = state_->indices_byte_order;
        out.state_->offsets_byte_order = state_->offsets_byte_order;
        return out;
    };
    auto add_entries = [](std::size_t& total, std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() - total)
            fail(ErrorCode::invalid_sparse_structure, "selected sparse entry count overflows");
        total += n;
    };
    if (minor.is_all()) return visit_index(state_->offset_type, [&]<class Offset> {
        for (std::size_t m = 0; m < out_major; ++m) {
            const auto i = original_index(selection_, major_axis, m);
            auto n = offsets[m];
            add_entries(n, read_nonnegative<Offset>(state_->offsets, i + 1) - read_nonnegative<Offset>(state_->offsets, i));
            offsets[m + 1] = n;
        }
        auto out = allocate(offsets.back());
        for (std::size_t m = 0; m < out_major; ++m) {
            const auto i = original_index(selection_, major_axis, m);
            const auto from = read_nonnegative<Offset>(state_->offsets, i), to = offsets[m], n = offsets[m + 1] - to;
            if (n) {
                std::memcpy(static_cast<std::byte*>(out.state_->data) + to*width,
                    static_cast<const std::byte*>(state_->data) + from*width, n*width);
                std::memcpy(static_cast<std::byte*>(out.state_->indices) + to*index_width,
                    static_cast<const std::byte*>(state_->indices) + from*index_width, n*index_width);
            }
        }
        for (std::size_t m = 0; m <= out_major; ++m) write_checked<Offset>(out.state_->offsets, m, offsets[m]);
        return out;
    });
    auto selected = [&]<class Match>(Match match) {
        return visit_index(state_->offset_type, [&]<class Offset> {
            return visit_index(state_->index_type, [&]<class Index> {
                for (std::size_t m = 0; m < out_major; ++m) {
                    const auto i = original_index(selection_, major_axis, m);
                    const auto a = read_nonnegative<Offset>(state_->offsets, i), b = read_nonnegative<Offset>(state_->offsets, i + 1);
                    auto n = offsets[m];
                    for (auto k = a; k < b; ++k)
                        match(read_nonnegative<Index>(state_->indices, k), [&](std::size_t) { add_entries(n, 1); });
                    offsets[m + 1] = n;
                }
                auto out = allocate(offsets.back());
                for (std::size_t m = 0; m < out_major; ++m) {
                    const auto i = original_index(selection_, major_axis, m);
                    const auto a = read_nonnegative<Offset>(state_->offsets, i), b = read_nonnegative<Offset>(state_->offsets, i + 1);
                    auto n = offsets[m];
                    for (auto k = a; k < b; ++k) match(read_nonnegative<Index>(state_->indices, k), [&](std::size_t j) {
                        std::memcpy(static_cast<std::byte*>(out.state_->data) + n*width,
                            static_cast<const std::byte*>(state_->data) + k*width, width);
                        write_checked<Index>(out.state_->indices, n++, j);
                    });
                }
                for (std::size_t m = 0; m <= out_major; ++m) write_checked<Offset>(out.state_->offsets, m, offsets[m]);
                return out;
            });
        });
    };
    if (!minor.is_gather()) return selected([&](std::size_t i, auto emit) {
        const auto first = minor.first(), step = minor.step();
        if (!step) { if (i == first) for (std::size_t j = 0; j < out_minor; ++j) emit(j); return; }
        if (minor.reverse() ? i > first : i < first) return;
        const auto d = minor.reverse() ? first - i : i - first;
        if (d % step == 0 && d/step < out_minor) emit(d/step);
    });
    // A compact source axis earns a direct table; cap its size and keep the
    // scratch proportional to the requested gather, including repetitions.
    const auto source_minor = state_->shape[minor_axis];
    if (source_minor <= (std::size_t{1} << 20) &&
        out_minor >= source_minor/4 + (source_minor % 4 != 0)) {
        checked_byte_span(out_minor, sizeof(std::size_t));
        const auto none = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> head(source_minor, none), next(out_minor);
        for (std::size_t j = out_minor; j > 0; --j) {
            const auto i = minor.index(j - 1);
            next[j - 1] = head[i]; head[i] = j - 1;
        }
        return selected([&](std::size_t i, auto emit) {
            for (auto j = head[i]; j != none; j = next[j]) emit(j);
        });
    }
    std::vector<std::pair<std::size_t, std::size_t>> lookup;
    lookup.reserve(out_minor);
    for (std::size_t j = 0; j < out_minor; ++j) lookup.emplace_back(minor.index(j), j);
    std::sort(lookup.begin(), lookup.end());
    return selected([&](std::size_t i, auto emit) {
        auto p = std::lower_bound(lookup.begin(), lookup.end(), i, [](const auto& x, std::size_t j) { return x.first < j; });
        for (; p != lookup.end() && p->first == i; ++p) emit(p->second);
    });
}
} // namespace anndatapp
