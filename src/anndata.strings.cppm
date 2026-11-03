module;
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

export module anndata.core:strings;
export import :arrays;

export namespace anndata {

namespace detail {
struct StringArrayImpl;
}

class StringArray;

// A shared descriptor over packed UTF-8 data. Selection is lazy and maps view
// coordinates back to the original allocation. storage_bytes()/storage_offsets()
// always expose that retained backing allocation, even for a selected view.
class StringArrayView {
public:
    StringArrayView();
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t extent(std::size_t axis) const;
    [[nodiscard]] std::vector<std::size_t> shape() const;
    [[nodiscard]] std::size_t byte_size() const;
    [[nodiscard]] std::size_t storage_byte_size() const;
    [[nodiscard]] std::string_view string(std::size_t flat_index) const;
    [[nodiscard]] const char* storage_bytes() const;
    [[nodiscard]] const std::uint64_t* storage_offsets() const;
    // Logical packed access requires an unselected view. storage_* explicitly
    // accesses the original backing allocation even when selection is lazy.
    [[nodiscard]] const char* bytes() const;
    [[nodiscard]] const std::uint64_t* offsets() const;
    [[nodiscard]] StringArrayView select(const Axis* axes, std::size_t rank) const;
    [[nodiscard]] StringArrayView select(std::initializer_list<Axis> axes) const {
        return select(axes.begin(), axes.size());
    }
    [[nodiscard]] StringArray copy() const;
    void validate() const;
private:
    std::shared_ptr<detail::StringArrayImpl> impl_;
    std::shared_ptr<const std::vector<Axis>> selection_;
    StringArrayView(std::shared_ptr<detail::StringArrayImpl>,
                    std::shared_ptr<const std::vector<Axis>>);
    friend class StringArray;
};

// Explicit mutable access to both packed buffers. Writable string views are
// intentionally unchecked after acquisition; call validate() after structural
// edits if offsets or UTF-8 bytes change.
class StringArrayEdit {
public:
    StringArrayEdit() = default;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t byte_size() const;
    [[nodiscard]] char* bytes() const;
    [[nodiscard]] const std::uint64_t* offsets() const;
    [[nodiscard]] std::uint64_t* unsafe_offsets() const;
private:
    std::shared_ptr<detail::StringArrayImpl> impl_;
    explicit StringArrayEdit(std::shared_ptr<detail::StringArrayImpl>);
    friend class StringArray;
};

using StringBufferDeleter = void (*)(char* bytes, std::uint64_t* offsets,
                                    void* context);

class StringArray {
public:
    StringArray();
    ~StringArray();
    StringArray(StringArray&&) noexcept;
    StringArray& operator=(StringArray&&) noexcept;
    StringArray(const StringArray&) = delete;
    StringArray& operator=(const StringArray&) = delete;

    [[nodiscard]] static StringArray from_strings(std::vector<std::size_t> shape,
                                                   std::vector<std::string> values);
    [[nodiscard]] static StringArray from_strings(
        std::initializer_list<std::size_t> shape,
        std::initializer_list<std::string_view> values);
    [[nodiscard]] static StringArray from_packed(std::vector<std::size_t> shape,
                                                  std::vector<char> bytes,
                                                  std::vector<std::uint64_t> offsets,
                                                  Validation validation = Validation::full);
    [[nodiscard]] static StringArray borrow_packed(
        const char* bytes, std::size_t byte_size, const std::uint64_t* offsets,
        std::vector<std::size_t> shape, Validation validation = Validation::full);
    [[nodiscard]] static StringArray borrow_packed(
        char* bytes, std::size_t byte_size, std::uint64_t* offsets,
        std::vector<std::size_t> shape, Validation validation = Validation::full);
    [[nodiscard]] static StringArray share_packed(
        const char* bytes, std::size_t byte_size, const std::uint64_t* offsets,
        std::vector<std::size_t> shape, std::shared_ptr<const void> owner,
        Validation validation = Validation::full);
    [[nodiscard]] static StringArray adopt_packed(
        char* bytes, std::size_t byte_size, std::uint64_t* offsets,
        std::vector<std::size_t> shape, StringBufferDeleter deleter,
        void* context = nullptr, Validation validation = Validation::full);
    [[nodiscard]] static StringArray share_packed(
        char* bytes, std::size_t byte_size, std::uint64_t* offsets,
        std::vector<std::size_t> shape, std::shared_ptr<const void> owner,
        Validation validation = Validation::full);

    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t extent(std::size_t axis) const;
    [[nodiscard]] StringArrayView view() const;
    [[nodiscard]] StringArrayEdit edit();
    [[nodiscard]] StringArray copy() const;
    void validate() const;
private:
    std::shared_ptr<detail::StringArrayImpl> impl_;
    explicit StringArray(std::shared_ptr<detail::StringArrayImpl>);
    friend class StringArrayView;
};

} // namespace anndata
