module;
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

export module anndatapp.core:records;
export import :arrays;
export import :strings;

export namespace anndatapp {

struct Utf8Ref { const char* data; std::size_t size; };
enum class RecordFieldKind { numeric, string, fixed_utf8, utf8_ref };
struct RecordField {
    std::string name;
    RecordFieldKind kind = RecordFieldKind::numeric;
    DType dtype = DType::float64;
    std::size_t offset = 0, width = 0;
    ByteOrder byte_order = ByteOrder::native;
};
struct RecordLayout {
    std::size_t item_size = 0, alignment = 1;
    std::vector<RecordField> fields;
};

namespace detail {
struct RecordImpl;
struct RecordViewImpl;
struct RecordDescriptor {
    std::shared_ptr<const void> owner;
    void* data = nullptr;
    DenseDimensions dimensions;
    std::ptrdiff_t first_stride = 0;
    std::size_t width = 0;
};
template<class T> constexpr DType record_dtype() {
    using U = std::remove_cv_t<T>;
    static_assert(std::is_same_v<U, bool> || std::is_same_v<U, std::int8_t> ||
        std::is_same_v<U, std::uint8_t> || std::is_same_v<U, std::int16_t> ||
        std::is_same_v<U, std::uint16_t> || std::is_same_v<U, std::int32_t> ||
        std::is_same_v<U, std::uint32_t> || std::is_same_v<U, std::int64_t> ||
        std::is_same_v<U, std::uint64_t> || std::is_same_v<U, float> ||
        std::is_same_v<U, double>, "record members require a canonical fixed-width numerical type");
    return dtype_of<U>();
}
}

// Retained native members. Strides are bytes, not T elements. Each unchecked
// ptr(i)/operator[](i) is the one-dimensional path (rank zero permits i=0).
// flat_ptr(i) explicitly unravels arbitrary-rank coordinates. data() is only
// the first member, rather than a T[] across records.
// No enclosing C++ struct object or T[] across record members is promised.
template<class T> class RecordMemberView {
public:
    RecordMemberView() = default;
    explicit RecordMemberView(detail::RecordDescriptor d) : d_(std::move(d)) {}
    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(d_.data); }
    [[nodiscard]] std::size_t rank() const noexcept { return d_.dimensions.size; }
    [[nodiscard]] std::size_t item_size() const noexcept { return d_.width; }
    [[nodiscard]] std::size_t extent(std::size_t a) const { return d_.dimensions.extent(a); }
    [[nodiscard]] std::ptrdiff_t stride_bytes(std::size_t a) const { return d_.dimensions.stride(a); }
    [[nodiscard]] T* ptr(std::size_t i) const noexcept {
        using Byte = std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;
        auto* p = reinterpret_cast<T*>(static_cast<Byte*>(d_.data) + std::ptrdiff_t(i)*d_.first_stride);
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::byte>) return p;
        else return std::launder(p);
    }
    [[nodiscard]] T* flat_ptr(std::size_t i) const noexcept {
        std::ptrdiff_t k = 0;
        for (std::size_t a = rank(); a > 0; --a) {
            const auto n = d_.dimensions.extent(a - 1);
            k += std::ptrdiff_t(i % n)*d_.dimensions.stride(a - 1); i /= n;
        }
        using Byte = std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;
        auto* p = reinterpret_cast<T*>(static_cast<Byte*>(d_.data) + k);
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::byte>) return p;
        else return std::launder(p);
    }
    [[nodiscard]] T& operator[](std::size_t i) const noexcept { return *ptr(i); }
private:
    detail::RecordDescriptor d_;
};

class RecordArray;
class RecordTextView {
public:
    RecordTextView() = default;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::string_view string(std::size_t flat_index) const;
private:
    std::shared_ptr<detail::RecordViewImpl> impl_;
    std::size_t field_ = 0;
    RecordTextView(std::shared_ptr<detail::RecordViewImpl>, std::size_t);
    friend class RecordArrayView;
};
class RecordArrayView {
public:
    RecordArrayView();
    [[nodiscard]] bool interleaved() const;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t extent(std::size_t axis) const;
    [[nodiscard]] std::vector<std::size_t> shape() const;
    [[nodiscard]] std::size_t n_fields() const;
    [[nodiscard]] std::vector<std::string> field_names() const;
    [[nodiscard]] RecordField field(std::string_view name) const;
    [[nodiscard]] RecordLayout record_layout() const;
    // Direct fieldwise owners; interleaved fields require explicit copy_*().
    [[nodiscard]] ArrayView array(std::string_view name) const;
    [[nodiscard]] StringArrayView string_array(std::string_view name) const;
    [[nodiscard]] RecordMemberView<const std::byte> numeric_bytes(std::string_view name) const;
    [[nodiscard]] RecordMemberView<const std::byte> record_bytes() const;
    [[nodiscard]] RecordTextView text(std::string_view name) const;
    [[nodiscard]] std::string_view string(std::string_view name, std::size_t flat_index) const;
    template<class T> [[nodiscard]] RecordMemberView<const T> member(std::string_view name) const {
        static_assert(!std::is_volatile_v<T>);
        if constexpr (std::is_same_v<std::remove_cv_t<T>, Utf8Ref>)
            return RecordMemberView<const T>(member_descriptor(name, DType::uint8, true));
        else return RecordMemberView<const T>(member_descriptor(name, detail::record_dtype<T>(), false));
    }
    [[nodiscard]] Array copy_array(std::string_view name) const;
    [[nodiscard]] StringArray copy_strings(std::string_view name) const;
    [[nodiscard]] RecordArray copy_fields() const;
    [[nodiscard]] RecordArray copy_interleaved(RecordLayout layout) const;
    [[nodiscard]] RecordArray copy() const;
    void validate() const;
private:
    std::shared_ptr<detail::RecordViewImpl> impl_;
    explicit RecordArrayView(std::shared_ptr<detail::RecordViewImpl>);
    detail::RecordDescriptor member_descriptor(std::string_view, DType, bool) const;
    friend class RecordArray;
};

class RecordArray {
public:
    RecordArray();
    ~RecordArray();
    RecordArray(RecordArray&&) noexcept;
    RecordArray& operator=(RecordArray&&) noexcept;
    RecordArray(const RecordArray&) = delete;
    RecordArray& operator=(const RecordArray&) = delete;
    [[nodiscard]] static RecordArray with_shape(std::vector<std::size_t> shape);
    // Constructs supported members, never an arbitrary Row. Numerical values
    // are uninitialized; fixed text is empty and Utf8Ref is {nullptr, 0}.
    [[nodiscard]] static RecordArray empty_interleaved(std::vector<std::size_t> shape, RecordLayout layout);
    // External storage contains live native members of the declared canonical
    // types. data is the first logical record; strides are bytes. Callers supply
    // the complete reachable buffer, including negative strides. Validation
    // cannot establish external object lifetime or pointed-text readability.
    [[nodiscard]] static RecordArray borrow_interleaved(void* data, std::vector<std::size_t> shape,
        RecordLayout layout, std::vector<std::ptrdiff_t> strides = {}, Validation validation = Validation::full);
    [[nodiscard]] static RecordArray borrow_interleaved(const void* data, std::vector<std::size_t> shape,
        RecordLayout layout, std::vector<std::ptrdiff_t> strides = {}, Validation validation = Validation::full);
    // The token/deleter retains/releases both record storage AND pointed text.
    [[nodiscard]] static RecordArray share_interleaved(void* data, std::vector<std::size_t> shape,
        RecordLayout layout, std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner,
        Validation validation = Validation::full);
    [[nodiscard]] static RecordArray share_interleaved(const void* data, std::vector<std::size_t> shape,
        RecordLayout layout, std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner,
        Validation validation = Validation::full);
    [[nodiscard]] static RecordArray adopt_interleaved(void* data, std::vector<std::size_t> shape,
        RecordLayout layout, std::vector<std::ptrdiff_t> strides, BufferDeleter deleter,
        void* context = nullptr, Validation validation = Validation::full);
    void set(std::string name, Array field, Validation validation = Validation::full);
    void set(std::string name, StringArray field, Validation validation = Validation::full);
    void erase(std::string_view name);
    void set_field_serialization_byte_order(std::string_view name, ByteOrder order);
    [[nodiscard]] RecordArrayView view() const;
    [[nodiscard]] RecordArray copy() const;
    [[nodiscard]] RecordArray copy_fields() const;
    [[nodiscard]] RecordArray copy_interleaved(RecordLayout layout) const;
    [[nodiscard]] ArrayEdit edit_array(std::string_view name);
    [[nodiscard]] StringArrayEdit edit_strings(std::string_view name);
    template<class T> [[nodiscard]] RecordMemberView<T> edit_member(std::string_view name) {
        static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>);
        return RecordMemberView<T>(edit_descriptor(name, detail::record_dtype<T>(), false));
    }
    // Rebinding a text reference transfers no ownership. Its pointer/length and
    // lifetime become the caller's responsibility; old aliases observe edits.
    [[nodiscard]] RecordMemberView<Utf8Ref> unsafe_edit_text_refs(std::string_view name);
    [[nodiscard]] RecordMemberView<std::byte> unsafe_edit_record_bytes();
    void validate() const;
private:
    std::shared_ptr<detail::RecordImpl> impl_;
    explicit RecordArray(std::shared_ptr<detail::RecordImpl>);
    static RecordArray wrap(void*, std::vector<std::size_t>, RecordLayout,
        std::vector<std::ptrdiff_t>, std::shared_ptr<const void>, bool, Validation);
    detail::RecordDescriptor edit_descriptor(std::string_view, DType, bool);
    friend class RecordArrayView;
};
}
