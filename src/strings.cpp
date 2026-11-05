module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

module anndatapp.core;
import :strings;

namespace anndatapp {
namespace detail {
struct StringArrayImpl {
    std::vector<std::size_t> shape;
    const char* bytes{};
    std::size_t byte_size{};
    const std::uint64_t* offsets{};
    bool writable{};
    std::shared_ptr<void> owner;
};
}

namespace {

[[noreturn]] void string_fail(ErrorCode code, std::string message) {
    throw Error(code, "string-array", std::move(message));
}

std::size_t element_count(const std::vector<std::size_t>& shape) {
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) return 0;
    std::size_t count = 1;
    for (const auto extent : shape) {
        if (extent && count > std::numeric_limits<std::size_t>::max() / extent)
            string_fail(ErrorCode::invalid_argument, "shape product overflows size_t");
        count *= extent;
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(std::uint64_t) - 1)
        string_fail(ErrorCode::invalid_argument, "offset count exceeds addressable range");
    return count;
}

bool valid_utf8(const char* text, std::size_t size) noexcept {
    auto continuation = [](unsigned char c) { return c >= 0x80 && c <= 0xbf; };
    std::size_t i = 0;
    while (size - i >= sizeof(std::uint64_t)) {
        std::uint64_t word;
        std::memcpy(&word, text + i, sizeof(word));
        if (word & 0x8080808080808080ULL) break;
        i += sizeof(word);
    }
    while (i < size) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c <= 0x7f) { ++i; continue; }
        if (c >= 0xc2 && c <= 0xdf) {
            if (i + 1 >= size || !continuation(static_cast<unsigned char>(text[i + 1]))) return false;
            i += 2;
            continue;
        }
        if (c >= 0xe0 && c <= 0xef) {
            if (i + 2 >= size) return false;
            const auto second = static_cast<unsigned char>(text[i + 1]);
            const auto third = static_cast<unsigned char>(text[i + 2]);
            if (!continuation(third)) return false;
            if (c == 0xe0) { if (second < 0xa0 || second > 0xbf) return false; }
            else if (c == 0xed) { if (second < 0x80 || second > 0x9f) return false; }
            else if (!continuation(second)) return false;
            i += 3;
            continue;
        }
        if (c >= 0xf0 && c <= 0xf4) {
            if (i + 3 >= size) return false;
            const auto second = static_cast<unsigned char>(text[i + 1]);
            if (!continuation(static_cast<unsigned char>(text[i + 2])) ||
                !continuation(static_cast<unsigned char>(text[i + 3]))) return false;
            if (c == 0xf0) { if (second < 0x90 || second > 0xbf) return false; }
            else if (c == 0xf4) { if (second < 0x80 || second > 0x8f) return false; }
            else if (!continuation(second)) return false;
            i += 4;
            continue;
        }
        return false;
    }
    return true;
}

std::size_t selected_extent(const detail::StringArrayImpl& impl,
                            const std::shared_ptr<const std::vector<Axis>>& selection,
                            std::size_t axis) {
    if (axis >= impl.shape.size()) string_fail(ErrorCode::out_of_bounds, "axis is out of bounds");
    return selection ? (*selection)[axis].extent(impl.shape[axis]) : impl.shape[axis];
}

std::size_t view_size(const detail::StringArrayImpl& impl,
                      const std::shared_ptr<const std::vector<Axis>>& selection) {
    if (!selection) return element_count(impl.shape);
    for (std::size_t a = 0; a < impl.shape.size(); ++a)
        if (selected_extent(impl, selection, a) == 0) return 0;
    std::size_t count = 1;
    for (std::size_t a = 0; a < impl.shape.size(); ++a) {
        const auto n = selected_extent(impl, selection, a);
        if (count > std::numeric_limits<std::size_t>::max() / n)
            string_fail(ErrorCode::invalid_argument, "selected shape product overflows");
        count *= n;
    }
    return count;
}

std::size_t source_index(const std::shared_ptr<detail::StringArrayImpl>& impl,
                         const std::shared_ptr<const std::vector<Axis>>& selection,
                         std::size_t flat_index) {
    const auto count = view_size(*impl, selection);
    if (flat_index >= count) string_fail(ErrorCode::out_of_bounds, "string index is out of bounds");
    if (!selection) return flat_index;
    std::size_t result = 0, stride = 1;
    for (std::size_t a = impl->shape.size(); a > 0; --a) {
        const auto axis = a - 1;
        const auto n = selected_extent(*impl, selection, axis);
        const auto coordinate = flat_index % n;
        flat_index /= n;
        result += (*selection)[axis].index(coordinate) * stride;
        stride *= impl->shape[axis];
    }
    return result;
}

void validate_impl(const detail::StringArrayImpl& impl) {
    const auto count = element_count(impl.shape);
    if (!impl.offsets) string_fail(ErrorCode::invalid_argument, "offset buffer is null");
    if (reinterpret_cast<std::uintptr_t>(impl.offsets) % alignof(std::uint64_t))
        string_fail(ErrorCode::invalid_argument, "offset buffer is not aligned");
    if (impl.byte_size && !impl.bytes) string_fail(ErrorCode::invalid_argument, "nonempty byte buffer is null");
    if (impl.byte_size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        string_fail(ErrorCode::invalid_argument, "byte buffer exceeds addressable range");
    if (impl.offsets[0] != 0) string_fail(ErrorCode::invalid_argument, "first offset must be zero");
    std::uint64_t previous = 0;
    for (std::size_t i = 1; i <= count; ++i) {
        const auto next = impl.offsets[i];
        if (next < previous || next > impl.byte_size)
            string_fail(ErrorCode::invalid_argument, "offsets are not monotonic or exceed the byte buffer");
        const auto length = static_cast<std::size_t>(next - previous);
        if (length && !valid_utf8(impl.bytes + previous, length))
            string_fail(ErrorCode::invalid_argument, "string is not valid UTF-8");
        previous = next;
    }
    if (previous != impl.byte_size) string_fail(ErrorCode::invalid_argument, "final offset differs from byte buffer size");
}

struct PackedBuffers {
    std::vector<char> bytes;
    std::vector<std::uint64_t> offsets;
};

struct AdoptedBuffers {
    char* bytes{};
    std::uint64_t* offsets{};
    StringBufferDeleter deleter{};
    void* context{};
    ~AdoptedBuffers() { deleter(bytes, offsets, context); }
};

} // namespace

StringArrayView::StringArrayView() = default;
StringArrayView::StringArrayView(std::shared_ptr<detail::StringArrayImpl> impl,
    std::shared_ptr<const std::vector<Axis>> selection)
    : impl_(std::move(impl)), selection_(std::move(selection)) {}
std::size_t StringArrayView::rank() const noexcept { return impl_ ? impl_->shape.size() : 0; }
std::size_t StringArrayView::size() const { return impl_ ? view_size(*impl_, selection_) : 0; }
std::size_t StringArrayView::extent(std::size_t axis) const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    return selected_extent(*impl_, selection_, axis);
}
std::vector<std::size_t> StringArrayView::shape() const {
    std::vector<std::size_t> result;
    result.reserve(rank());
    for (std::size_t a = 0; a < rank(); ++a) result.push_back(extent(a));
    return result;
}
std::size_t StringArrayView::byte_size() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    if (!selection_) return impl_->byte_size;
    std::size_t result = 0;
    for (std::size_t i = 0, n = size(); i < n; ++i) {
        const auto length = string(i).size();
        if (length > std::numeric_limits<std::size_t>::max() - result)
            string_fail(ErrorCode::invalid_argument, "selected byte size overflows");
        result += length;
    }
    return result;
}
std::size_t StringArrayView::storage_byte_size() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    return impl_->byte_size;
}
std::string_view StringArrayView::string(std::size_t flat_index) const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    const auto original = source_index(impl_, selection_, flat_index);
    const auto begin = impl_->offsets[original];
    const auto end = impl_->offsets[original + 1];
    if (end == begin) return {};
    return {impl_->bytes + begin, static_cast<std::size_t>(end - begin)};
}
const char* StringArrayView::storage_bytes() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    return impl_->bytes;
}
const std::uint64_t* StringArrayView::storage_offsets() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    return impl_->offsets;
}
const char* StringArrayView::bytes() const {
    if (selection_) string_fail(ErrorCode::non_contiguous, "selected string array requires copy() for compact packed access");
    return storage_bytes();
}
const std::uint64_t* StringArrayView::offsets() const {
    if (selection_) string_fail(ErrorCode::non_contiguous, "selected string array requires copy() for compact packed access");
    return storage_offsets();
}
StringArrayView StringArrayView::select(const Axis* axes, std::size_t n) const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    if (n != rank()) string_fail(ErrorCode::shape_mismatch, "selector count differs from rank");
    if (n && !axes) string_fail(ErrorCode::invalid_argument, "selector pointer is null");
    bool any = false;
    auto out = std::make_shared<std::vector<Axis>>(n);
    for (std::size_t a = 0; a < n; ++a) {
        auto x = detail::compose_axis(selection_ ? (*selection_)[a] : Axis::all(), axes[a], impl_->shape[a]);
        // Packed access remains valid for an identity selection, including a
        // gathered identity. Affine identity checks do not traverse the axis.
        bool same = x.extent(impl_->shape[a]) == impl_->shape[a];
        if (same && x.is_gather()) {
            for (std::size_t i = 0; same && i < impl_->shape[a]; ++i) same = x.index(i) == i;
        } else if (same && !x.is_all()) {
            same = impl_->shape[a] == 0 || (x.first() == 0 &&
                (impl_->shape[a] == 1 || (!x.reverse() && x.step() == 1)));
        }
        (*out)[a] = same ? Axis::all() : std::move(x);
        any = any || !(*out)[a].is_all();
    }
    return {impl_, any ? std::move(out) : nullptr};
}
StringArray StringArrayView::copy() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    if (!selection_) {
        std::vector<char> bytes(impl_->byte_size);
        if (impl_->byte_size) std::memcpy(bytes.data(), impl_->bytes, impl_->byte_size);
        std::vector<std::uint64_t> offsets(size() + 1);
        std::memcpy(offsets.data(), impl_->offsets, offsets.size() * sizeof(std::uint64_t));
        return StringArray::from_packed(impl_->shape, std::move(bytes), std::move(offsets), Validation::none);
    }
    std::vector<std::string> values;
    values.reserve(size());
    for (std::size_t i = 0; i < size(); ++i) values.emplace_back(string(i));
    return StringArray::from_strings(shape(), std::move(values));
}
void StringArrayView::validate() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "view is empty or moved from");
    validate_impl(*impl_);
}

StringArrayEdit::StringArrayEdit(std::shared_ptr<detail::StringArrayImpl> impl) : impl_(std::move(impl)) {}
std::size_t StringArrayEdit::size() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "edit descriptor is empty");
    return element_count(impl_->shape);
}
std::size_t StringArrayEdit::byte_size() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "edit descriptor is empty");
    return impl_->byte_size;
}
char* StringArrayEdit::bytes() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "edit descriptor is empty");
    if (!impl_->writable) string_fail(ErrorCode::invalid_argument, "backing storage is read only");
    return const_cast<char*>(impl_->bytes);
}
const std::uint64_t* StringArrayEdit::offsets() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "edit descriptor is empty");
    return impl_->offsets;
}
std::uint64_t* StringArrayEdit::unsafe_offsets() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "edit descriptor is empty");
    if (!impl_->writable) string_fail(ErrorCode::invalid_argument, "backing storage is read only");
    return const_cast<std::uint64_t*>(impl_->offsets);
}

StringArray::StringArray() = default;
StringArray::StringArray(std::shared_ptr<detail::StringArrayImpl> impl) : impl_(std::move(impl)) {}
StringArray::~StringArray() = default;
StringArray::StringArray(StringArray&&) noexcept = default;
StringArray& StringArray::operator=(StringArray&&) noexcept = default;
StringArray StringArray::from_strings(std::vector<std::size_t> shape,
                                      std::vector<std::string> values) {
    const auto count = element_count(shape);
    if (values.size() != count)
        string_fail(ErrorCode::shape_mismatch, "string count differs from shape product");
    std::size_t total = 0;
    for (const auto& value : values) {
        if (value.size() > std::numeric_limits<std::size_t>::max() - total)
            string_fail(ErrorCode::invalid_argument, "packed UTF-8 byte size overflows size_t");
        total += value.size();
    }
    std::vector<char> bytes;
    bytes.reserve(total);
    std::vector<std::uint64_t> offsets;
    offsets.reserve(count + 1);
    offsets.push_back(0);
    for (const auto& value : values) {
        bytes.insert(bytes.end(), value.begin(), value.end());
        if (bytes.size() > std::numeric_limits<std::uint64_t>::max())
            string_fail(ErrorCode::invalid_argument, "UTF-8 byte size exceeds uint64 offsets");
        offsets.push_back(static_cast<std::uint64_t>(bytes.size()));
    }
    return from_packed(std::move(shape), std::move(bytes), std::move(offsets));
}
StringArray StringArray::from_strings(std::initializer_list<std::size_t> shape,
                                      std::initializer_list<std::string_view> values) {
    std::vector<std::string> copied;
    copied.reserve(values.size());
    for (const auto value : values) copied.emplace_back(value);
    return from_strings(std::vector<std::size_t>(shape), std::move(copied));
}
StringArray StringArray::from_packed(std::vector<std::size_t> shape,
                                     std::vector<char> bytes,
                                     std::vector<std::uint64_t> offsets,
                                     Validation validation) {
    const auto count = element_count(shape);
    if (offsets.size() != count + 1)
        string_fail(ErrorCode::shape_mismatch, "offset count differs from shape product plus one");
    auto impl = std::make_shared<detail::StringArrayImpl>();
    impl->shape = std::move(shape);
    auto p = std::make_shared<PackedBuffers>(PackedBuffers{std::move(bytes), std::move(offsets)});
    impl->bytes = p->bytes.empty() ? nullptr : p->bytes.data();
    impl->byte_size = p->bytes.size();
    impl->offsets = p->offsets.data();
    impl->owner = std::move(p);
    impl->writable = true;
    if (validation == Validation::full) validate_impl(*impl);
    return StringArray(std::move(impl));
}
StringArray StringArray::borrow_packed(const char* bytes, std::size_t byte_size,
                                      const std::uint64_t* offsets,
                                      std::vector<std::size_t> shape,
                                      Validation validation) {
    (void)element_count(shape);
    auto impl = std::make_shared<detail::StringArrayImpl>();
    impl->shape = std::move(shape);
    impl->bytes = bytes;
    impl->byte_size = byte_size;
    impl->offsets = offsets;
    impl->writable = false;
    if (byte_size && !bytes) string_fail(ErrorCode::invalid_argument, "nonempty byte buffer is null");
    if (!offsets) string_fail(ErrorCode::invalid_argument, "offset buffer is null");
    if (validation == Validation::full) validate_impl(*impl);
    else if (byte_size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        string_fail(ErrorCode::invalid_argument, "byte buffer exceeds addressable range");
    return StringArray(std::move(impl));
}
StringArray StringArray::borrow_packed(char* bytes, std::size_t byte_size,
                                      std::uint64_t* offsets, std::vector<std::size_t> shape,
                                      Validation validation) {
    auto result = borrow_packed(static_cast<const char*>(bytes), byte_size,
        static_cast<const std::uint64_t*>(offsets), std::move(shape), validation);
    result.impl_->writable = true;
    return result;
}
StringArray StringArray::share_packed(const char* bytes, std::size_t byte_size,
                                     const std::uint64_t* offsets,
                                     std::vector<std::size_t> shape,
                                     std::shared_ptr<const void> owner,
                                     Validation validation) {
    if (owner.use_count() == 0) string_fail(ErrorCode::invalid_argument, "retained owner is empty");
    auto result = borrow_packed(bytes, byte_size, offsets, std::move(shape), validation);
    auto* token = const_cast<void*>(owner.get());
    result.impl_->owner = std::shared_ptr<void>(std::move(owner), token);
    return result;
}
StringArray StringArray::adopt_packed(char* bytes, std::size_t byte_size,
                                     std::uint64_t* offsets,
                                     std::vector<std::size_t> shape,
                                     StringBufferDeleter deleter, void* context,
                                     Validation validation) {
    if (!deleter) string_fail(ErrorCode::invalid_argument, "buffer deleter is null");
    auto allocation = std::make_shared<AdoptedBuffers>();
    allocation->bytes = bytes;
    allocation->offsets = offsets;
    allocation->deleter = deleter;
    allocation->context = context;
    auto result = borrow_packed(bytes, byte_size, offsets, std::move(shape), validation);
    result.impl_->owner = std::move(allocation);
    result.impl_->writable = true;
    return result;
}
StringArray StringArray::share_packed(char* bytes, std::size_t byte_size,
                                     std::uint64_t* offsets, std::vector<std::size_t> shape,
                                     std::shared_ptr<const void> owner, Validation validation) {
    auto result = share_packed(static_cast<const char*>(bytes), byte_size,
        static_cast<const std::uint64_t*>(offsets), std::move(shape), std::move(owner), validation);
    result.impl_->writable = true;
    return result;
}
std::size_t StringArray::rank() const noexcept { return impl_ ? impl_->shape.size() : 0; }
std::size_t StringArray::size() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "array is empty or moved from");
    return element_count(impl_->shape);
}
std::size_t StringArray::extent(std::size_t axis) const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "array is empty or moved from");
    if (axis >= rank()) string_fail(ErrorCode::out_of_bounds, "axis is out of bounds");
    return impl_->shape[axis];
}
StringArrayView StringArray::view() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "array is empty or moved from");
    return StringArrayView(impl_, {});
}
StringArrayEdit StringArray::edit() {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "array is empty or moved from");
    if (!impl_->writable) string_fail(ErrorCode::invalid_argument, "backing storage is read only");
    return StringArrayEdit(impl_);
}
StringArray StringArray::copy() const { return view().copy(); }
void StringArray::validate() const {
    if (!impl_) string_fail(ErrorCode::invalid_argument, "array is empty or moved from");
    validate_impl(*impl_);
}

} // namespace anndatapp
