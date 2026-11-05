module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

module anndatapp.core;
import :records;

namespace anndatapp {
namespace detail {
struct RecordFieldOwner {
    RecordField info;
    std::optional<Array> numeric;
    std::optional<StringArray> strings;
};
struct RecordFieldView {
    RecordField info;
    ArrayView numeric;
    StringArrayView strings;
};
struct RecordImpl {
    std::vector<std::size_t> shape;
    std::vector<RecordFieldOwner> fields;
    RecordLayout layout;
    std::vector<std::ptrdiff_t> strides;
    void* data = nullptr;
    bool interleaved = false, writable = true;
    std::shared_ptr<const void> owner;
};
struct RecordViewImpl {
    std::vector<std::size_t> shape;
    std::vector<RecordFieldView> fields;
    RecordLayout layout;
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
    for (std::size_t i = 0; i < fields.size(); ++i) if (fields[i].info.name == name) return i;
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
void check_buffer(const Shape& shape, const RecordLayout& layout, const Strides& strides, const void* data) {
    const auto n = count(shape);
    if (strides.size() != shape.size()) fail(ErrorCode::shape_mismatch, {}, "record strides do not cover the outer rank");
    if (!n) return;
    if (!data || reinterpret_cast<std::uintptr_t>(data) % layout.alignment)
        fail(ErrorCode::invalid_argument, {}, "record pointer is null or misaligned");
    std::ptrdiff_t low = 0, high = 0;
    for (std::size_t a = 0; a < shape.size(); ++a) {
        if (shape[a] <= 1) continue;
        if (strides[a] % std::ptrdiff_t(layout.alignment))
            fail(ErrorCode::invalid_argument, {}, "record stride does not preserve alignment");
        std::ptrdiff_t k;
        if (__builtin_mul_overflow(std::ptrdiff_t(shape[a] - 1), strides[a], &k) ||
            (k < 0 ? __builtin_add_overflow(low, k, &low) : __builtin_add_overflow(high, k, &high)))
            fail(ErrorCode::invalid_argument, {}, "record address range overflows");
    }
    if (high > PTRDIFF_MAX - std::ptrdiff_t(layout.item_size) ||
        (low < 0 && high > PTRDIFF_MAX + low - std::ptrdiff_t(layout.item_size)))
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
    if (f.info.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, f.info.name, "field is not numerical");
    if (p->interleaved) {
        auto* b = static_cast<std::byte*>(p->data);
        return descriptor(p, b ? b + f.info.offset : nullptr, p->shape, p->strides, width(f.info));
    }
    auto b = f.numeric.bytes(); Strides strides(b.rank());
    for (std::size_t a = 0; a < b.rank(); ++a) strides[a] = b.stride_bytes(a);
    return descriptor(p, const_cast<std::byte*>(b.data()), p->shape, strides, b.element_size());
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
    if (!p.interleaved) return f.strings.string(i);
    auto* b = static_cast<const std::byte*>(p.data) + offset(p.shape, p.strides, i) + f.info.offset;
    if (f.info.kind == RecordFieldKind::utf8_ref) {
        Utf8Ref s; std::memcpy(&s, b, sizeof s);
        if ((!s.data && s.size) || s.size > std::size_t(PTRDIFF_MAX))
            fail(ErrorCode::invalid_argument, f.info.name, "invalid text pointer/length");
        return s.size ? std::string_view(s.data, s.size) : std::string_view{};
    }
    auto* s = reinterpret_cast<const char*>(b); auto n = f.info.width;
    while (n && !s[n - 1]) --n;
    return {s, n};
}
void validate_view(const std::shared_ptr<View>& p, bool content) {
    require(p); const auto n = count(p->shape);
    if (p->interleaved) {
        auto layout = p->layout; check_layout(layout); check_buffer(p->shape, layout, p->strides, p->data);
    }
    for (std::size_t j = 0; j < p->fields.size(); ++j) {
        const auto& f = p->fields[j]; check_name(f.info.name);
        for (std::size_t k = 0; k < j; ++k) if (p->fields[k].info.name == f.info.name)
            fail(ErrorCode::invalid_argument, f.info.name, "duplicate field name");
        if (f.info.kind == RecordFieldKind::numeric) {
            if (!p->interleaved) {
                if (f.numeric.layout() != Layout::dense || f.numeric.rank() != p->shape.size())
                    fail(ErrorCode::shape_mismatch, f.info.name, "numeric field must be dense with the record outer shape");
                for (std::size_t a = 0; a < p->shape.size(); ++a) if (f.numeric.extent(a) != p->shape[a])
                    fail(ErrorCode::shape_mismatch, f.info.name, "numeric field outer shape differs");
                if (f.numeric.native_type() != visit_numeric(f.info.dtype, []<class T> { return detail::native_type_of<T>(); }))
                    fail(ErrorCode::dtype_mismatch, f.info.name, "numeric field requires its canonical fixed-width native type");
                if (content) f.numeric.validate();
            }
            if (content && f.info.dtype == DType::boolean) {
                RecordMemberView<const std::byte> b(numeric_descriptor(p, j));
                for (std::size_t i = 0; i < n; ++i) if (std::to_integer<unsigned>(*b.flat_ptr(i)) > 1)
                    fail(ErrorCode::invalid_argument, f.info.name, "boolean members require canonical zero/one bytes");
            }
        } else if (!p->interleaved) {
            if (f.strings.shape() != p->shape) fail(ErrorCode::shape_mismatch, f.info.name, "text field outer shape differs");
            if (content) f.strings.validate();
        } else if (content) for (std::size_t i = 0; i < n; ++i)
            if (!utf8(text_at(*p, j, i))) fail(ErrorCode::invalid_argument, f.info.name, "text field is not valid UTF-8");
    }
}
struct Allocation {
    std::byte* data = nullptr;
    std::size_t alignment = 1;
    std::vector<char> text;
    ~Allocation() { if (data) ::operator delete(data, std::align_val_t(alignment)); }
};
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
    p->shape = impl_->shape; p->layout = impl_->layout; p->strides = impl_->strides;
    p->data = impl_->data; p->interleaved = impl_->interleaved; p->owner = impl_->owner;
    for (const auto& f : impl_->fields) p->fields.push_back({f.info, f.numeric ? f.numeric->view() : ArrayView{},
                                                           f.strings ? f.strings->view() : StringArrayView{}});
    return RecordArrayView(std::move(p));
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
    for (const auto& f : impl_->fields) names.push_back(f.info.name);
    return names;
}
RecordField RecordArrayView::field(std::string_view name) const {
    require(impl_); return impl_->fields[find(impl_->fields, name)].info;
}
RecordLayout RecordArrayView::record_layout() const {
    require(impl_); if (!impl_->interleaved) fail(ErrorCode::unsupported_layout, {}, "fieldwise records have no interleaved layout");
    return impl_->layout;
}
ArrayView RecordArrayView::array(std::string_view name) const {
    require(impl_); const auto& f = impl_->fields[find(impl_->fields, name)];
    if (f.info.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not numerical");
    if (impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "interleaved members require a byte-strided member view or explicit copy_array");
    return f.numeric;
}
StringArrayView RecordArrayView::string_array(std::string_view name) const {
    require(impl_); const auto& f = impl_->fields[find(impl_->fields, name)];
    if (f.info.kind == RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not text");
    if (impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "interleaved text requires explicit copy_strings");
    return f.strings;
}
RecordMemberView<const std::byte> RecordArrayView::numeric_bytes(std::string_view name) const {
    require(impl_); return RecordMemberView<const std::byte>(numeric_descriptor(impl_, find(impl_->fields, name)));
}
RecordMemberView<const std::byte> RecordArrayView::record_bytes() const {
    require(impl_); if (!impl_->interleaved) fail(ErrorCode::unsupported_layout, {}, "fieldwise records have no record bytes");
    return RecordMemberView<const std::byte>(descriptor(impl_, impl_->data, impl_->shape, impl_->strides, impl_->layout.item_size));
}
Descriptor RecordArrayView::member_descriptor(std::string_view name, DType type, bool ref) const {
    require(impl_); const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j].info;
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
    require(impl_); if (i >= count(impl_->shape)) fail(ErrorCode::out_of_bounds, impl_->fields[field_].info.name, "text index out of bounds");
    return text_at(*impl_, field_, i);
}
RecordTextView RecordArrayView::text(std::string_view name) const {
    require(impl_); const auto j = find(impl_->fields, name);
    if (impl_->fields[j].info.kind == RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not text");
    return RecordTextView(impl_, j);
}
std::string_view RecordArrayView::string(std::string_view name, std::size_t i) const { return text(name).string(i); }
void RecordArrayView::validate() const { validate_view(impl_, true); }
void RecordArray::validate() const { view().validate(); }

void RecordArray::set(std::string name, Array value, Validation validation) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "set requires a fieldwise record owner");
    check_name(name);
    detail::RecordFieldOwner f; f.info = {std::move(name), RecordFieldKind::numeric, value.dtype(), 0,
                                        dtype_size(value.dtype()), value.serialization_byte_order()};
    f.numeric.emplace(std::move(value));
    auto p = std::make_shared<View>(); p->shape = impl_->shape; p->fields.push_back({f.info, f.numeric->view(), {}});
    validate_view(p, validation == Validation::full);
    for (auto& old : impl_->fields) if (old.info.name == f.info.name) { old = std::move(f); return; }
    impl_->fields.push_back(std::move(f));
}
void RecordArray::set(std::string name, StringArray value, Validation validation) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "set requires a fieldwise record owner");
    check_name(name); if (validation == Validation::full) value.validate();
    if (value.view().shape() != impl_->shape) fail(ErrorCode::shape_mismatch, name, "text field outer shape differs");
    detail::RecordFieldOwner f; f.info.name = std::move(name); f.info.kind = RecordFieldKind::string;
    f.strings.emplace(std::move(value));
    for (auto& old : impl_->fields) if (old.info.name == f.info.name) { old = std::move(f); return; }
    impl_->fields.push_back(std::move(f));
}
void RecordArray::erase(std::string_view name) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "erase requires fieldwise records");
    for (auto i = impl_->fields.begin(); i != impl_->fields.end(); ++i) if (i->info.name == name) { impl_->fields.erase(i); return; }
}
void RecordArray::set_field_serialization_byte_order(std::string_view name, ByteOrder order) {
    if (!impl_) fail(ErrorCode::invalid_argument, {}, "empty record owner");
    const auto j = find(impl_->fields, name); auto& f = impl_->fields[j];
    if (f.info.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "byte order applies to numeric fields");
    f.info.byte_order = order;
    if (impl_->interleaved) impl_->layout.fields[j].byte_order = order;
    else f.numeric->set_serialization_byte_order(order);
}
ArrayEdit RecordArray::edit_array(std::string_view name) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "edit_array requires fieldwise records");
    auto& f = impl_->fields[find(impl_->fields, name)];
    if (!f.numeric) fail(ErrorCode::dtype_mismatch, name, "field is not numeric"); return f.numeric->edit();
}
StringArrayEdit RecordArray::edit_strings(std::string_view name) {
    if (!impl_ || impl_->interleaved) fail(ErrorCode::unsupported_layout, name, "edit_strings requires fieldwise records");
    auto& f = impl_->fields[find(impl_->fields, name)];
    if (!f.strings) fail(ErrorCode::dtype_mismatch, name, "field is not packed text"); return f.strings->edit();
}
Descriptor RecordArray::edit_descriptor(std::string_view name, DType type, bool ref) {
    if (!impl_) fail(ErrorCode::invalid_argument, {}, "empty record owner");
    if (impl_->interleaved) {
        if (!impl_->writable) fail(ErrorCode::invalid_argument, name, "record storage is read-only");
    } else { (void)edit_array(name); }
    return view().member_descriptor(name, type, ref);
}
RecordMemberView<Utf8Ref> RecordArray::unsafe_edit_text_refs(std::string_view name) {
    return RecordMemberView<Utf8Ref>(edit_descriptor(name, DType::uint8, true));
}
RecordMemberView<std::byte> RecordArray::unsafe_edit_record_bytes() {
    if (!impl_ || !impl_->interleaved) fail(ErrorCode::unsupported_layout, {}, "fieldwise records have no record bytes");
    if (!impl_->writable) fail(ErrorCode::invalid_argument, {}, "record storage is read-only");
    auto p = view().impl_; return RecordMemberView<std::byte>(descriptor(p, p->data, p->shape, p->strides, p->layout.item_size));
}

RecordArray RecordArray::wrap(void* data, Shape shape, RecordLayout layout, Strides strides,
                            std::shared_ptr<const void> owner, bool writable, Validation validation) {
    check_layout(layout);
    if (strides.empty()) strides = contiguous_strides(shape, layout.item_size);
    check_buffer(shape, layout, strides, data);
    auto p = std::make_shared<detail::RecordImpl>();
    p->shape = std::move(shape); p->layout = std::move(layout); p->strides = std::move(strides);
    p->data = data; p->owner = std::move(owner); p->writable = writable; p->interleaved = true;
    for (const auto& f : p->layout.fields) p->fields.push_back({f, {}, {}});
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
    check_layout(layout); const auto n = count(shape), bytes = mul(n, layout.item_size);
    auto allocation = std::make_shared<Allocation>(); allocation->alignment = std::max(layout.alignment, alignof(std::max_align_t));
    if (bytes) {
        auto* raw = ::operator new(bytes, std::align_val_t(allocation->alignment));
        allocation->data = ::new(raw) std::byte[bytes];
        for (const auto& f : layout.fields) for (std::size_t i = 0; i < n; ++i) {
            auto* p = allocation->data + i*layout.item_size + f.offset;
            if (f.kind == RecordFieldKind::numeric) visit_numeric(f.dtype, [&]<class T> { ::new(static_cast<void*>(p)) T; });
            else if (f.kind == RecordFieldKind::utf8_ref) ::new(static_cast<void*>(p)) Utf8Ref{};
            else std::memset(p, 0, f.width);
        }
    }
    return wrap(allocation->data, std::move(shape), std::move(layout), {}, allocation, true, Validation::none);
}

Array RecordArrayView::copy_array(std::string_view name) const {
    require(impl_); const auto j = find(impl_->fields, name); const auto& f = impl_->fields[j].info;
    if (f.kind != RecordFieldKind::numeric) fail(ErrorCode::dtype_mismatch, name, "field is not numerical");
    if (!impl_->interleaved) {
        auto out = impl_->fields[j].numeric.copy(); out.set_serialization_byte_order(f.byte_order); return out;
    }
    auto b = numeric_bytes(name); const auto n = size();
    auto out = visit_numeric(f.dtype, [&]<class T> {
        auto a = Array::empty<T>(impl_->shape); auto* dst = a.edit().template dense<T>().data();
        for (std::size_t i = 0; i < n; ++i) std::memcpy(dst + i, b.flat_ptr(i), sizeof(T));
        return a;
    });
    out.set_serialization_byte_order(f.byte_order); return out;
}
StringArray RecordArrayView::copy_strings(std::string_view name) const {
    auto v = text(name); std::vector<char> bytes; std::vector<std::uint64_t> offsets{0}; offsets.reserve(size() + 1);
    for (std::size_t i = 0; i < v.size(); ++i) {
        auto s = v.string(i); (void)add(bytes.size(), s.size());
        if (!s.empty()) bytes.insert(bytes.end(), s.data(), s.data() + s.size());
        offsets.push_back(bytes.size());
    }
    return StringArray::from_packed(shape(), std::move(bytes), std::move(offsets));
}
RecordArray RecordArrayView::copy_fields() const {
    validate(); auto out = RecordArray::with_shape(shape());
    for (const auto& f : impl_->fields) {
        if (f.info.kind == RecordFieldKind::numeric) out.set(f.info.name, copy_array(f.info.name));
        else out.set(f.info.name, copy_strings(f.info.name));
    }
    return out;
}
RecordArray RecordArrayView::copy_interleaved(RecordLayout layout) const {
    validate(); check_layout(layout);
    if (layout.fields.size() != n_fields()) fail(ErrorCode::shape_mismatch, {}, "layout field count differs");
    std::size_t bytes = 0; const auto n = size();
    for (std::size_t j = 0; j < n_fields(); ++j) {
        auto& f = layout.fields[j]; const auto& g = impl_->fields[j].info;
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
    auto out = RecordArray::empty_interleaved(shape(), std::move(layout));
    auto allocation = std::make_shared<Allocation>(); allocation->text.resize(bytes);
    // Keep the native record allocation and the final, non-growing text arena.
    struct Owners { std::shared_ptr<const void> records; std::shared_ptr<Allocation> text; };
    auto owners = std::make_shared<Owners>(Owners{out.impl_->owner, allocation}); out.impl_->owner = owners;
    std::size_t k = 0;
    for (const auto& f : out.impl_->layout.fields) {
        if (f.kind == RecordFieldKind::numeric) {
            auto b = numeric_bytes(f.name);
            for (std::size_t i = 0; i < n; ++i)
                std::memcpy(static_cast<std::byte*>(out.impl_->data) + i*out.impl_->layout.item_size + f.offset, b.flat_ptr(i), f.width);
        } else {
            auto s = text(f.name);
            for (std::size_t i = 0; i < n; ++i) {
                auto value = s.string(i); auto* p = static_cast<std::byte*>(out.impl_->data) + i*out.impl_->layout.item_size + f.offset;
                if (f.kind == RecordFieldKind::utf8_ref) {
                    if (value.size()) std::memcpy(allocation->text.data() + k, value.data(), value.size());
                    *reinterpret_cast<Utf8Ref*>(p) = {value.size() ? allocation->text.data() + k : nullptr, value.size()}; k += value.size();
                } else if (value.size()) std::memcpy(p, value.data(), value.size());
            }
        }
    }
    return out;
}
RecordArray RecordArrayView::copy() const { return interleaved() ? copy_interleaved(record_layout()) : copy_fields(); }
RecordArray RecordArray::copy() const { return view().copy(); }
RecordArray RecordArray::copy_fields() const { return view().copy_fields(); }
RecordArray RecordArray::copy_interleaved(RecordLayout layout) const { return view().copy_interleaved(std::move(layout)); }
}
