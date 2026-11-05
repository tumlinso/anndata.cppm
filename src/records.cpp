module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

module anndatapp.core;
import :records;

namespace anndatapp {
namespace detail {
struct RecordImpl {
    std::vector<std::size_t> shape;
    std::vector<RecordField> fields;
    std::vector<Array> numeric;
    std::vector<StringArray> strings;
    std::size_t item_size = 0, alignment = 1;
    std::vector<std::ptrdiff_t> strides;
    void* data = nullptr;
    bool interleaved = false, writable = true;
    std::shared_ptr<const void> owner;
};
struct RecordViewImpl {
    std::vector<std::size_t> shape;
    std::vector<RecordField> fields;
    std::vector<ArrayView> numeric;
    std::vector<StringArrayView> strings;
    std::size_t item_size = 0, alignment = 1;
    std::vector<std::ptrdiff_t> strides;
    void* data = nullptr;
    bool interleaved = false;
    std::shared_ptr<const void> owner;
};
}

namespace {
using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::ptrdiff_t>;
using View = detail::RecordViewImpl;
using Descriptor = detail::RecordDescriptor;

void reserve_field(detail::RecordImpl& p) {
    const auto n = p.fields.size() + 1;
    if (p.fields.capacity() >= n && p.numeric.capacity() >= n && p.strings.capacity() >= n) return;
    const auto limit = p.fields.max_size();
    const auto capacity = p.fields.capacity() > limit/2 ? limit : std::max(n, 2*p.fields.capacity());
    p.fields.reserve(capacity); p.numeric.reserve(capacity); p.strings.reserve(capacity);
}
[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, "record-array" + (path.empty() ? std::string{} : "/" + std::string(path)), std::move(message));
}
std::size_t mul(std::size_t a, std::size_t b) {
    if (b && a > std::size_t(PTRDIFF_MAX)/b) fail(ErrorCode::invalid_argument, {}, "byte size or shape product overflows");
    return a*b;
}
std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::size_t(PTRDIFF_MAX) || b > std::size_t(PTRDIFF_MAX) - a)
        fail(ErrorCode::invalid_argument, {}, "byte size overflows");
    return a + b;
}
std::size_t count(const Shape& shape) {
    for (auto n : shape) if (!n) return 0;
    std::size_t n = 1;
    for (auto k : shape) n = mul(n, k);
    return n;
}
template<class F> decltype(auto) visit_numeric(DType t, F&& f) {
    switch (t) {
    case DType::boolean: return f.template operator()<bool>();
    case DType::int8: return f.template operator()<std::int8_t>();
    case DType::uint8: return f.template operator()<std::uint8_t>();
    case DType::int16: return f.template operator()<std::int16_t>();
    case DType::uint16: return f.template operator()<std::uint16_t>();
    case DType::int32: return f.template operator()<std::int32_t>();
    case DType::uint32: return f.template operator()<std::uint32_t>();
    case DType::int64: return f.template operator()<std::int64_t>();
    case DType::uint64: return f.template operator()<std::uint64_t>();
    case DType::float32: return f.template operator()<float>();
    case DType::float64: return f.template operator()<double>();
    }
    fail(ErrorCode::unsupported_dtype, {}, "unknown numerical field dtype");
}
bool utf8(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned c = static_cast<unsigned char>(s[i++]), n = 0;
        std::uint32_t x = 0, low = 0;
        if (c < 0x80) continue;
        if (c >= 0xc2 && c <= 0xdf) { n = 1; x = c & 0x1f; low = 0x80; }
        else if (c >= 0xe0 && c <= 0xef) { n = 2; x = c & 0xf; low = 0x800; }
        else if (c >= 0xf0 && c <= 0xf4) { n = 3; x = c & 7; low = 0x10000; }
        else return false;
        if (n > s.size() - i) return false;
        for (unsigned k = 0; k < n; ++k) {
            c = static_cast<unsigned char>(s[i++]);
            if ((c & 0xc0) != 0x80) return false;
            x = (x << 6) | (c & 0x3f);
        }
        if (x < low || x > 0x10ffff || (x >= 0xd800 && x <= 0xdfff)) return false;
    }
    return true;
}
void check_name(std::string_view name) {
    if (name.empty() || name.find('\0') != name.npos || !utf8(name))
        fail(ErrorCode::invalid_argument, name, "field names must be nonempty UTF-8 without NUL");
}
template<class Fields> std::size_t find(const Fields& fields, std::string_view name) {
    for (std::size_t i = 0; i < fields.size(); ++i) if (fields[i].name == name) return i;
    fail(ErrorCode::out_of_bounds, name, "field does not exist");
}
void require(const std::shared_ptr<View>& p) {
    if (!p) fail(ErrorCode::invalid_argument, {}, "empty record view");
}
std::size_t width(const RecordField& f) {
    return f.kind == RecordFieldKind::numeric ? dtype_size(f.dtype) :
           f.kind == RecordFieldKind::utf8_ref ? sizeof(Utf8Ref) : f.width;
}
std::size_t alignment(const RecordField& f) {
    return f.kind == RecordFieldKind::numeric ? visit_numeric(f.dtype, []<class T> { return alignof(T); }) :
           f.kind == RecordFieldKind::utf8_ref ? alignof(Utf8Ref) : 1;
}
void check_layout(RecordLayout& layout) {
    if (!layout.item_size || layout.item_size > std::size_t(PTRDIFF_MAX) ||
        !layout.alignment || (layout.alignment & (layout.alignment - 1)) ||
        layout.alignment > std::size_t(PTRDIFF_MAX) || layout.item_size % layout.alignment)
        fail(ErrorCode::invalid_argument, {}, "record size/alignment is invalid");
    for (std::size_t j = 0; j < layout.fields.size(); ++j) {
        auto& f = layout.fields[j]; check_name(f.name);
        if (f.kind != RecordFieldKind::numeric && f.kind != RecordFieldKind::fixed_utf8 && f.kind != RecordFieldKind::utf8_ref)
            fail(ErrorCode::unsupported_layout, f.name, "interleaved text requires fixed_utf8 or utf8_ref storage");
        const auto w = width(f), a = alignment(f);
        if (!w || (f.width && f.width != w) || f.offset > layout.item_size || w > layout.item_size - f.offset)
            fail(ErrorCode::invalid_argument, f.name, "member exceeds its record or has an invalid width");
        if (layout.alignment < a || f.offset % a)
            fail(ErrorCode::invalid_argument, f.name, "member offset or record alignment is unsuitable");
        f.width = w;
        for (std::size_t k = 0; k < j; ++k) {
            const auto& g = layout.fields[k];
            if (f.name == g.name) fail(ErrorCode::invalid_argument, f.name, "duplicate field name");
            if (f.offset < g.offset + g.width && g.offset < f.offset + f.width)
                fail(ErrorCode::invalid_argument, f.name, "record members overlap");
        }
    }
}
Strides contiguous_strides(const Shape& shape, std::size_t item_size) {
    Strides strides(shape.size()); std::size_t n = item_size;
    const bool empty = count(shape) == 0;
    for (std::size_t a = shape.size(); a > 0; --a) {
        strides[a - 1] = std::ptrdiff_t(n);
        n = empty && shape[a - 1] && n > std::size_t(PTRDIFF_MAX)/shape[a - 1] ? 0 : mul(n, shape[a - 1]);
    }
    return strides;
}
void check_buffer(const Shape& shape, std::size_t item_size, std::size_t alignment, const Strides& strides, const void* data) {
    const auto n = count(shape);
    if (strides.size() != shape.size()) fail(ErrorCode::shape_mismatch, {}, "record strides do not cover the outer rank");
    if (!n) return;
    if (!data || reinterpret_cast<std::uintptr_t>(data) % alignment)
        fail(ErrorCode::invalid_argument, {}, "record pointer is null or misaligned");
    std::ptrdiff_t low = 0, high = 0;
    for (std::size_t a = 0; a < shape.size(); ++a) {
        if (shape[a] <= 1) continue;
        if (strides[a] % std::ptrdiff_t(alignment))
            fail(ErrorCode::invalid_argument, {}, "record stride does not preserve alignment");
        const auto stride = strides[a];
        const auto magnitude = stride < 0 ? std::uintmax_t(-(stride + 1)) + 1 : std::uintmax_t(stride);
        const auto limit = std::uintmax_t(PTRDIFF_MAX) + (stride < 0);
        if (magnitude && shape[a] - 1 > limit/magnitude)
            fail(ErrorCode::invalid_argument, {}, "record address range overflows");
        const auto n = std::uintmax_t(shape[a] - 1)*magnitude;
        const auto k = stride < 0 ? n == limit ? PTRDIFF_MIN : -std::ptrdiff_t(n) : std::ptrdiff_t(n);
        if (k < 0) {
            if (low < PTRDIFF_MIN - k) fail(ErrorCode::invalid_argument, {}, "record address range overflows");
            low += k;
        } else {
            if (high > PTRDIFF_MAX - k) fail(ErrorCode::invalid_argument, {}, "record address range overflows");
            high += k;
        }
    }
    if (high > PTRDIFF_MAX - std::ptrdiff_t(item_size) ||
        (low < 0 && high > PTRDIFF_MAX + low - std::ptrdiff_t(item_size)))
        fail(ErrorCode::invalid_argument, {}, "record address span overflows");
}
Descriptor descriptor(std::shared_ptr<const void> owner, void* data, const Shape& shape,
                      const Strides& strides, std::size_t w) {
    Descriptor d; d.owner = std::move(owner); d.data = data; d.width = w; d.dimensions.size = shape.size();
    d.first_stride = strides.empty() ? 0 : strides[0];
    if (shape.size() > detail::DenseDimensions::inline_rank) {
        d.dimensions.large = std::make_shared<detail::LargeDimensions>();
        d.dimensions.large->shape = shape; d.dimensions.large->strides = strides;
    } else for (std::size_t a = 0; a < shape.size(); ++a) {
        d.dimensions.shape[a] = shape[a]; d.dimensions.strides[a] = strides[a];
    }
    return d;
}
Descriptor numeric_descriptor(const std::shared_ptr<View>& p, std::size_t j) {
    const auto& f = p->fields[j];
    if (f.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, f.name, "field is not numerical");
    if (p->interleaved) {
        auto* b = static_cast<std::byte*>(p->data);
        return descriptor(p, b ? b + f.offset : nullptr, p->shape, p->strides, width(f));
    }
    auto b = p->numeric[j].bytes(); Descriptor d;
    d.owner = p; d.data = const_cast<std::byte*>(b.data()); d.width = b.element_size();
    d.dimensions.size = b.rank();
    bool empty = false;
    for (std::size_t a = 0; a < b.rank(); ++a) empty = empty || b.extent(a) == 0;
    d.first_stride = b.rank() && !empty && b.extent(0) > 1 ? b.stride_bytes(0) : 0;
    if (b.rank() > detail::DenseDimensions::inline_rank) {
        d.dimensions.large = std::make_shared<detail::LargeDimensions>();
        d.dimensions.large->shape.resize(b.rank()); d.dimensions.large->strides.resize(b.rank());
        for (std::size_t a = 0; a < b.rank(); ++a) {
            d.dimensions.large->shape[a] = b.extent(a);
            d.dimensions.large->strides[a] = !empty && b.extent(a) > 1 ? b.stride_bytes(a) : 0;
        }
    } else for (std::size_t a = 0; a < b.rank(); ++a) {
        d.dimensions.shape[a] = b.extent(a);
        d.dimensions.strides[a] = !empty && b.extent(a) > 1 ? b.stride_bytes(a) : 0;
    }
    return d;
}
std::ptrdiff_t offset(const Shape& shape, const Strides& strides, std::size_t i) {
    std::ptrdiff_t k = 0;
    for (std::size_t a = shape.size(); a > 0; --a) {
        k += std::ptrdiff_t(i % shape[a - 1])*strides[a - 1]; i /= shape[a - 1];
    }
    return k;
}
std::string_view text_at(const View& p, std::size_t j, std::size_t i) {
    const auto& f = p.fields[j];
    if (!p.interleaved) return p.strings[j].string(i);
    auto* b = static_cast<const std::byte*>(p.data) + offset(p.shape, p.strides, i) + f.offset;
    if (f.kind == RecordFieldKind::utf8_ref) {
        Utf8Ref s; std::memcpy(&s, b, sizeof s);
        if ((!s.data && s.size) || s.size > std::size_t(PTRDIFF_MAX))
            fail(ErrorCode::invalid_argument, f.name, "invalid text pointer/length");
        return s.size ? std::string_view(s.data, s.size) : std::string_view{};
    }
    auto* s = reinterpret_cast<const char*>(b); auto n = f.width;
    while (n && !s[n - 1]) --n;
    return {s, n};
}
void validate_view(const std::shared_ptr<View>& p, bool content) {
    require(p); const auto n = count(p->shape);
    if (p->interleaved) {
        check_buffer(p->shape, p->item_size, p->alignment, p->strides, p->data);
    }
    for (std::size_t j = 0; j < p->fields.size(); ++j) {
        const auto& f = p->fields[j]; check_name(f.name);
        for (std::size_t k = 0; k < j; ++k) if (p->fields[k].name == f.name)
            fail(ErrorCode::invalid_argument, f.name, "duplicate field name");
        if (f.kind == RecordFieldKind::numeric) {
            if (!p->interleaved) {
                if (p->numeric[j].layout() != Layout::dense || p->numeric[j].rank() != p->shape.size())
                    fail(ErrorCode::shape_mismatch, f.name, "numeric field must be dense with the record outer shape");
                for (std::size_t a = 0; a < p->shape.size(); ++a) if (p->numeric[j].extent(a) != p->shape[a])
                    fail(ErrorCode::shape_mismatch, f.name, "numeric field outer shape differs");
                if (p->numeric[j].native_type() != visit_numeric(f.dtype, []<class T> { return detail::native_type_of<T>(); }))
                    fail(ErrorCode::dtype_mismatch, f.name, "numeric field requires its canonical fixed-width native type");
                if (content) p->numeric[j].validate();
            }
            if (content && f.dtype == DType::boolean) {
                auto b = numeric_descriptor(p, j);
                for (std::size_t i = 0; i < n; ++i) {
                    auto q = i; std::ptrdiff_t k = 0;
                    for (std::size_t a = b.dimensions.size; a > 0; --a) {
                        const auto m = b.dimensions.extent(a - 1);
                        k += std::ptrdiff_t(q % m)*b.dimensions.stride(a - 1); q /= m;
                    }
                    if (std::to_integer<unsigned>(*(static_cast<const std::byte*>(b.data) + k)) > 1)
                        fail(ErrorCode::invalid_argument, f.name, "boolean members require canonical zero/one bytes");
                }
            }
        } else if (!p->interleaved) {
            if (p->strings[j].shape() != p->shape) fail(ErrorCode::shape_mismatch, f.name, "text field outer shape differs");
            if (content) p->strings[j].validate();
        } else if (content) for (std::size_t i = 0; i < n; ++i)
            if (!utf8(text_at(*p, j, i))) fail(ErrorCode::invalid_argument, f.name, "text field is not valid UTF-8");
    }
}
struct Allocation {
    std::byte* data = nullptr;
    std::size_t alignment = 1;
    std::vector<char> text;
    ~Allocation() { if (data) ::operator delete(data, std::align_val_t(alignment)); }
};
std::shared_ptr<Allocation> allocate_records(const Shape& shape, const RecordLayout& layout, std::size_t text_bytes) {
    const auto n = count(shape), bytes = mul(n, layout.item_size);
    auto out = std::make_shared<Allocation>();
    out->alignment = std::max(layout.alignment, alignof(std::max_align_t)); out->text.resize(text_bytes);
    if (!bytes) return out;
    out->data = ::new(::operator new(bytes, std::align_val_t(out->alignment))) std::byte[bytes];
    for (const auto& f : layout.fields) {
        if (f.kind == RecordFieldKind::numeric) visit_numeric(f.dtype, [&]<class T> {
            for (std::size_t i = 0; i < n; ++i)
                ::new(static_cast<void*>(out->data + i*layout.item_size + f.offset)) T;
        });
        else if (f.kind == RecordFieldKind::utf8_ref)
            for (std::size_t i = 0; i < n; ++i)
                ::new(static_cast<void*>(out->data + i*layout.item_size + f.offset)) Utf8Ref{};
        else for (std::size_t i = 0; i < n; ++i) std::memset(out->data + i*layout.item_size + f.offset, 0, f.width);
    }
    return out;
}
void copy_members(const Descriptor& source, std::byte* destination, std::size_t destination_stride, std::size_t n) {
    if (!n) return;
    const auto* data = static_cast<const std::byte*>(source.data);
    const auto rank = source.dimensions.size;
    if (rank <= 1) {
        for (std::size_t i = 0; i < n; ++i)
            std::memcpy(destination + i*destination_stride, data + std::ptrdiff_t(i)*source.first_stride, source.width);
        return;
    }
    const auto* shape = source.dimensions.large ? source.dimensions.large->shape.data() : source.dimensions.shape;
    const auto* strides = source.dimensions.large ? source.dimensions.large->strides.data() : source.dimensions.strides;
    std::vector<std::size_t> coordinate(rank, 0);
    std::ptrdiff_t from = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::memcpy(destination + i*destination_stride, data + from, source.width);
        if (i + 1 == n) break;
        for (std::size_t a = rank; a > 0; --a) {
            if (++coordinate[a - 1] < shape[a - 1]) { from += strides[a - 1]; break; }
            from -= std::ptrdiff_t(shape[a - 1] - 1)*strides[a - 1]; coordinate[a - 1] = 0;
        }
    }
}

}

RecordArray::RecordArray() = default;
RecordArray::RecordArray(std::shared_ptr<detail::RecordImpl> p) : impl_(std::move(p)) {}
RecordArray::~RecordArray() = default;
RecordArray::RecordArray(RecordArray&&) noexcept = default;
RecordArray& RecordArray::operator=(RecordArray&&) noexcept = default;
RecordArray RecordArray::with_shape(Shape shape) {
    (void)count(shape); auto p = std::make_shared<detail::RecordImpl>(); p->shape = std::move(shape);
    return RecordArray(std::move(p));
}
RecordArrayView::RecordArrayView() = default;
RecordArrayView::RecordArrayView(std::shared_ptr<View> p) : impl_(std::move(p)) {}
RecordArrayView RecordArray::view() const {
    if (!impl_) return {};
    auto p = std::make_shared<View>();
    p->shape = impl_->shape; p->item_size = impl_->item_size; p->alignment = impl_->alignment; p->strides = impl_->strides;
    p->data = impl_->data; p->interleaved = impl_->interleaved; p->owner = impl_->owner;
    p->fields = impl_->fields;
    if (!impl_->interleaved) {
        p->numeric.resize(p->fields.size()); p->strings.resize(p->fields.size());
        for (std::size_t j = 0; j < p->fields.size(); ++j)
            if (p->fields[j].kind == RecordFieldKind::numeric) p->numeric[j] = impl_->numeric[j].view();
            else p->strings[j] = impl_->strings[j].view();
    }
    return RecordArrayView(std::move(p));
}
ArrayView RecordArray::array(std::string_view name) const {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "array requires fieldwise records");
    const auto j = find(impl_->fields, name);
    if (impl_->fields[j].kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not numerical");
    return impl_->numeric[j].view();
}
StringArrayView RecordArray::string_array(std::string_view name) const {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "string_array requires fieldwise records");
    const auto j = find(impl_->fields, name);
    if (impl_->fields[j].kind != RecordFieldKind::string) fail(ErrorCode::dtype_mismatch, name, "field is not packed text");
    return impl_->strings[j].view();
}
RecordTextView RecordArray::text(std::string_view name) const {
    if (!impl_) fail(ErrorCode::invalid_argument, {}, "empty record owner");
    const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j];
    if (f.kind == RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not text");
    auto p = std::make_shared<View>();
    p->shape = impl_->shape; p->strides = impl_->strides; p->data = impl_->data;
    p->item_size = impl_->item_size; p->alignment = impl_->alignment;
    p->interleaved = impl_->interleaved; p->owner = impl_->owner; p->fields.push_back(f);
    if (!p->interleaved) p->strings.push_back(impl_->strings[j].view());
    return RecordTextView(std::move(p), 0);
}
Descriptor RecordArray::member_descriptor(std::string_view name, DType type, bool ref) const {
    if (!impl_) fail(ErrorCode::invalid_argument, {}, "empty record owner");
    const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j];
    if (ref ? f.kind != RecordFieldKind::utf8_ref : f.kind != RecordFieldKind::numeric || f.dtype != type)
        fail(ErrorCode::dtype_mismatch, name, "member dtype differs from requested canonical type");
    if (impl_->interleaved) {
        auto* b = static_cast<std::byte*>(impl_->data);
        return descriptor(impl_->owner, b ? b + f.offset : nullptr, impl_->shape, impl_->strides, width(f));
    }
    auto v = impl_->numeric[j].view();
    auto b = v.dense_descriptor(v.dtype(), v.native_type(), true);
    Descriptor d; d.owner = std::move(b.owner); d.data = b.data; d.width = dtype_size(type);
    d.dimensions = std::move(b.dimensions);
    bool empty = false;
    for (std::size_t a = 0; a < d.dimensions.size; ++a) empty = empty || d.dimensions.extent(a) == 0;
    for (std::size_t a = 0; a < d.dimensions.size; ++a) {
        const auto stride = !empty && d.dimensions.extent(a) > 1
            ? detail::checked_byte_stride(d.dimensions.stride(a), d.width) : 0;
        if (d.dimensions.large) d.dimensions.large->strides[a] = stride;
        else d.dimensions.strides[a] = stride;
    }
    d.first_stride = d.dimensions.size ? d.dimensions.stride(0) : 0;
    return d;
}
bool RecordArrayView::interleaved() const { require(impl_); return impl_->interleaved; }
std::size_t RecordArrayView::rank() const noexcept { return impl_ ? impl_->shape.size() : 0; }
std::size_t RecordArrayView::size() const { require(impl_); return count(impl_->shape); }
std::size_t RecordArrayView::extent(std::size_t a) const {
    require(impl_); if (a >= rank()) fail(ErrorCode::out_of_bounds, {}, "axis out of bounds"); return impl_->shape[a];
}
Shape RecordArrayView::shape() const { require(impl_); return impl_->shape; }
std::size_t RecordArrayView::n_fields() const { require(impl_); return impl_->fields.size(); }
std::vector<std::string> RecordArrayView::field_names() const {
    require(impl_); std::vector<std::string> names; names.reserve(impl_->fields.size());
    for (const auto& f : impl_->fields) names.push_back(f.name);
    return names;
}
RecordField RecordArrayView::field(std::string_view name) const {
    require(impl_); return impl_->fields[find(impl_->fields, name)];
}
RecordLayout RecordArrayView::record_layout() const {
    require(impl_); if (!impl_->interleaved) fail(ErrorCode::unsupported_layout, {}, "fieldwise records have no interleaved layout");
    return {impl_->item_size, impl_->alignment, impl_->fields};
}
ArrayView RecordArrayView::array(std::string_view name) const {
    require(impl_); const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j];
    if (f.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not numerical");
    if (impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "interleaved members require a byte-strided member view or explicit copy_array");
    return impl_->numeric[j];
}
StringArrayView RecordArrayView::string_array(std::string_view name) const {
    require(impl_); const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j];
    if (f.kind == RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not text");
    if (impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "interleaved text requires explicit copy_strings");
    return impl_->strings[j];
}
RecordMemberView<const std::byte> RecordArrayView::numeric_bytes(std::string_view name) const {
    require(impl_); return RecordMemberView<const std::byte>(numeric_descriptor(impl_, find(impl_->fields, name)));
}
RecordMemberView<const std::byte> RecordArrayView::record_bytes() const {
    require(impl_); if (!impl_->interleaved) fail(ErrorCode::unsupported_layout, {}, "fieldwise records have no record bytes");
    return RecordMemberView<const std::byte>(descriptor(impl_, impl_->data, impl_->shape, impl_->strides, impl_->item_size));
}
Descriptor RecordArrayView::member_descriptor(std::string_view name, DType type, bool ref) const {
    require(impl_); const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j];
    if (ref) {
        if (f.kind != RecordFieldKind::utf8_ref) fail(ErrorCode::dtype_mismatch, name, "field is not an interleaved Utf8Ref");
        auto* b = static_cast<std::byte*>(impl_->data);
        auto* p = b ? std::launder(reinterpret_cast<Utf8Ref*>(b + f.offset)) : nullptr;
        return descriptor(impl_, p, impl_->shape, impl_->strides, sizeof(Utf8Ref));
    }
    if (f.kind != RecordFieldKind::numeric || f.dtype != type) fail(ErrorCode::dtype_mismatch, name, "member dtype differs from requested canonical type");
    auto d = numeric_descriptor(impl_, j);
    if (d.data) d.data = visit_numeric(type, [&]<class T>() -> void* { return std::launder(static_cast<T*>(d.data)); });
    return d;
}
RecordTextView::RecordTextView(std::shared_ptr<View> p, std::size_t j) : impl_(std::move(p)), field_(j) {}
std::size_t RecordTextView::size() const { require(impl_); return count(impl_->shape); }
std::string_view RecordTextView::string(std::size_t i) const {
    require(impl_); if (i >= count(impl_->shape)) fail(ErrorCode::out_of_bounds, impl_->fields[field_].name, "text index out of bounds");
    return text_at(*impl_, field_, i);
}
RecordTextView RecordArrayView::text(std::string_view name) const {
    require(impl_); const auto j = find(impl_->fields, name);
    if (impl_->fields[j].kind == RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not text");
    return RecordTextView(impl_, j);
}
std::string_view RecordArrayView::string(std::string_view name, std::size_t i) const { return text(name).string(i); }
void RecordArrayView::validate() const { validate_view(impl_, true); }
void RecordArray::validate() const { view().validate(); }

void RecordArray::set(std::string name, Array value, Validation validation) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "set requires fieldwise records");
    check_name(name);
    RecordField f{std::move(name), RecordFieldKind::numeric, value.dtype(), 0, dtype_size(value.dtype()), value.serialization_byte_order()};
    auto p = std::make_shared<View>(); p->shape = impl_->shape;
    p->fields.push_back(f); p->numeric.push_back(value.view()); p->strings.emplace_back();
    validate_view(p, validation == Validation::full);
    for (std::size_t j = 0; j < impl_->fields.size(); ++j) if (impl_->fields[j].name == f.name) {
        impl_->fields[j] = std::move(f); impl_->numeric[j] = std::move(value); impl_->strings[j] = {}; return;
    }
    reserve_field(*impl_);
    impl_->fields.push_back(std::move(f)); impl_->numeric.push_back(std::move(value)); impl_->strings.emplace_back();
}
void RecordArray::set(std::string name, StringArray value, Validation validation) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "set requires fieldwise records");
    check_name(name); if (validation == Validation::full) value.validate();
    if (value.rank() != impl_->shape.size()) fail(ErrorCode::shape_mismatch, name, "text field outer rank differs");
    for (std::size_t a = 0; a < value.rank(); ++a)
        if (value.extent(a) != impl_->shape[a]) fail(ErrorCode::shape_mismatch, name, "text field outer shape differs");
    RecordField f; f.name = std::move(name); f.kind = RecordFieldKind::string;
    for (std::size_t j = 0; j < impl_->fields.size(); ++j) if (impl_->fields[j].name == f.name) {
        impl_->fields[j] = std::move(f); impl_->strings[j] = std::move(value); impl_->numeric[j] = {}; return;
    }
    reserve_field(*impl_);
    impl_->fields.push_back(std::move(f)); impl_->strings.push_back(std::move(value)); impl_->numeric.emplace_back();
}
void RecordArray::erase(std::string_view name) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "erase requires fieldwise records");
    for (std::size_t j = 0; j < impl_->fields.size(); ++j) if (impl_->fields[j].name == name) {
        impl_->fields.erase(impl_->fields.begin() + j);
        impl_->numeric.erase(impl_->numeric.begin() + j); impl_->strings.erase(impl_->strings.begin() + j); return;
    }
}
void RecordArray::set_field_serialization_byte_order(std::string_view name, ByteOrder order) {
    if (!impl_) fail(ErrorCode::invalid_argument, {}, "empty record owner");
    const auto j = find(impl_->fields, name); auto& f = impl_->fields[j];
    if (f.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "byte order applies to numeric fields");
    f.byte_order = order;
    if (!impl_->interleaved) impl_->numeric[j].set_serialization_byte_order(order);
}
ArrayEdit RecordArray::edit_array(std::string_view name) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "edit_array requires fieldwise records");
    const auto j = find(impl_->fields, name);
    if (impl_->fields[j].kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not numeric");
    return impl_->numeric[j].edit();
}
StringArrayEdit RecordArray::edit_strings(std::string_view name) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "edit_strings requires fieldwise records");
    const auto j = find(impl_->fields, name);
    if (impl_->fields[j].kind != RecordFieldKind::string) fail(ErrorCode::dtype_mismatch, name, "field is not packed text");
    return impl_->strings[j].edit();
}
Descriptor RecordArray::edit_descriptor(std::string_view name, DType type, bool ref) {
    if (!impl_) fail(ErrorCode::invalid_argument, {}, "empty record owner");
    if (impl_->interleaved) {
        if (!impl_->writable) fail(ErrorCode::invalid_argument, name, "record storage is read-only");
    } else { (void)edit_array(name); }
    return member_descriptor(name, type, ref);
}
RecordMemberView<Utf8Ref> RecordArray::unsafe_edit_text_refs(std::string_view name) {
    return RecordMemberView<Utf8Ref>(edit_descriptor(name, DType::uint8, true));
}
RecordMemberView<std::byte> RecordArray::unsafe_edit_record_bytes() {
    if (!impl_ || !impl_->interleaved) fail(ErrorCode::unsupported_layout, {}, "fieldwise records have no record bytes");
    if (!impl_->writable) fail(ErrorCode::invalid_argument, {}, "record storage is read-only");
    return RecordMemberView<std::byte>(descriptor(impl_->owner, impl_->data, impl_->shape, impl_->strides, impl_->item_size));
}

RecordArray RecordArray::wrap(void* data, Shape shape, RecordLayout layout, Strides strides,
                            std::shared_ptr<const void> owner, bool writable, Validation validation) {
    check_layout(layout);
    if (strides.empty()) strides = contiguous_strides(shape, layout.item_size);
    check_buffer(shape, layout.item_size, layout.alignment, strides, data);
    auto p = std::make_shared<detail::RecordImpl>();
    p->shape = std::move(shape); p->item_size = layout.item_size; p->alignment = layout.alignment; p->strides = std::move(strides);
    p->data = data; p->owner = std::move(owner); p->writable = writable; p->interleaved = true;
    p->fields.reserve(layout.fields.size());
    p->fields = std::move(layout.fields);
    RecordArray out(std::move(p)); if (validation == Validation::full) out.validate(); return out;
}
RecordArray RecordArray::borrow_interleaved(void* data, Shape shape, RecordLayout layout, Strides strides, Validation validation) {
    return wrap(data, std::move(shape), std::move(layout), std::move(strides), {}, true, validation);
}
RecordArray RecordArray::borrow_interleaved(const void* data, Shape shape, RecordLayout layout, Strides strides, Validation validation) {
    return wrap(const_cast<void*>(data), std::move(shape), std::move(layout), std::move(strides), {}, false, validation);
}
RecordArray RecordArray::share_interleaved(void* data, Shape shape, RecordLayout layout, Strides strides,
                                       std::shared_ptr<const void> owner, Validation validation) {
    if (!owner.use_count()) fail(ErrorCode::invalid_argument, {}, "shared record owner is empty");
    return wrap(data, std::move(shape), std::move(layout), std::move(strides), std::move(owner), true, validation);
}
RecordArray RecordArray::share_interleaved(const void* data, Shape shape, RecordLayout layout, Strides strides,
                                       std::shared_ptr<const void> owner, Validation validation) {
    if (!owner.use_count()) fail(ErrorCode::invalid_argument, {}, "shared record owner is empty");
    return wrap(const_cast<void*>(data), std::move(shape), std::move(layout), std::move(strides), std::move(owner), false, validation);
}
RecordArray RecordArray::adopt_interleaved(void* data, Shape shape, RecordLayout layout, Strides strides,
                                       BufferDeleter deleter, void* context, Validation validation) {
    if (!deleter) fail(ErrorCode::invalid_argument, {}, "record deleter is null");
    std::shared_ptr<const void> owner(data, [deleter, context](const void* p) { deleter(const_cast<void*>(p), context); });
    return wrap(data, std::move(shape), std::move(layout), std::move(strides), std::move(owner), true, validation);
}
RecordArray RecordArray::empty_interleaved(Shape shape, RecordLayout layout) {
    check_layout(layout);
    auto allocation = allocate_records(shape, layout, 0);
    return wrap(allocation->data, std::move(shape), std::move(layout), {}, allocation, true, Validation::none);
}

Array RecordArrayView::copy_array(std::string_view name) const {
    require(impl_); const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j];
    if (f.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not numerical");
    if (!impl_->interleaved) {
        auto out = impl_->numeric[j].copy(); out.set_serialization_byte_order(f.byte_order); return out;
    }
    auto b = numeric_descriptor(impl_, j); const auto n = size();
    auto out = visit_numeric(f.dtype, [&]<class T> {
        auto a = Array::empty<T>(impl_->shape);
        copy_members(b, a.edit().bytes().data(), sizeof(T), n);
        return a;
    });
    out.set_serialization_byte_order(f.byte_order); return out;
}
StringArray RecordArrayView::copy_strings(std::string_view name) const {
    require(impl_); const auto j = find(impl_->fields, name), n = size();
    if (impl_->fields[j].kind == RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not text");
    std::vector<std::uint64_t> offsets(n + 1, 0);
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < n; ++i) { bytes = add(bytes, text_at(*impl_, j, i).size()); offsets[i + 1] = bytes; }
    std::vector<char> data(bytes);
    for (std::size_t i = 0; i < n; ++i) {
        auto s = text_at(*impl_, j, i);
        if (!s.empty()) std::memcpy(data.data() + offsets[i], s.data(), s.size());
    }
    return StringArray::from_packed(shape(), std::move(data), std::move(offsets));
}

RecordArray RecordArrayView::copy_fields() const {
    validate(); auto out = RecordArray::with_shape(shape());
    for (const auto& f : impl_->fields) {
        if (f.kind == RecordFieldKind::numeric) out.set(f.name, copy_array(f.name));
        else out.set(f.name, copy_strings(f.name));
    }
    return out;
}
RecordArray RecordArrayView::copy_interleaved(RecordLayout layout) const {
    validate(); check_layout(layout);
    if (layout.fields.size() != n_fields()) fail(ErrorCode::shape_mismatch, {}, "layout field count differs");
    std::size_t bytes = 0; const auto n = size();
    for (std::size_t j = 0; j < n_fields(); ++j) {
        auto& f = layout.fields[j]; const auto& g = impl_->fields[j];
        if (f.name != g.name || ((f.kind == RecordFieldKind::numeric) != (g.kind == RecordFieldKind::numeric)) ||
            (f.kind == RecordFieldKind::numeric && f.dtype != g.dtype))
            fail(ErrorCode::dtype_mismatch, f.name, "layout must preserve ordered field names and scalar types");
        if (f.kind == RecordFieldKind::numeric) f.byte_order = g.byte_order;
        else {
            auto s = text(f.name);
            for (std::size_t i = 0; i < n; ++i) {
                auto value = s.string(i);
                if (f.kind == RecordFieldKind::utf8_ref) bytes = add(bytes, value.size());
                else if (value.size() > f.width || (!value.empty() && !value.back()))
                    fail(ErrorCode::shape_mismatch, f.name, "text does not fit fixed UTF-8 storage without truncation");
            }
        }
    }
    auto allocation = allocate_records(impl_->shape, layout, bytes);
    auto out = RecordArray::wrap(allocation->data, impl_->shape, std::move(layout), {}, allocation, true, Validation::none);
    std::size_t k = 0;
    for (std::size_t j = 0; j < out.impl_->fields.size(); ++j) {
        const auto& f = out.impl_->fields[j];
        if (f.kind == RecordFieldKind::numeric) {
            auto b = numeric_descriptor(impl_, j);
            if (n) copy_members(b, static_cast<std::byte*>(out.impl_->data) + f.offset, out.impl_->item_size, n);
        } else for (std::size_t i = 0; i < n; ++i) {
            const auto value = text_at(*impl_, j, i);
            auto* p = static_cast<std::byte*>(out.impl_->data) + i*out.impl_->item_size + f.offset;
            if (f.kind == RecordFieldKind::utf8_ref) {
                if (!value.empty()) std::memcpy(allocation->text.data() + k, value.data(), value.size());
                *reinterpret_cast<Utf8Ref*>(p) = {value.empty() ? nullptr : allocation->text.data() + k, value.size()}; k += value.size();
            } else if (!value.empty()) std::memcpy(p, value.data(), value.size());
        }
    }
    return out;
}
RecordArray RecordArrayView::copy() const { return interleaved() ? copy_interleaved(record_layout()) : copy_fields(); }
RecordArray RecordArray::copy() const { return view().copy(); }
RecordArray RecordArray::copy_fields() const { return view().copy_fields(); }
RecordArray RecordArray::copy_interleaved(RecordLayout layout) const { return view().copy_interleaved(std::move(layout)); }
}
