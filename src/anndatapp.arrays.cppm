module;
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

export module anndatapp.core:arrays;

export namespace anndatapp {

enum class DType { boolean, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64 };
enum class Layout { dense, csr, csc };
enum class Validation { full, none };
enum class ByteOrder { native, little, big };
enum class ErrorCode { invalid_argument, out_of_bounds, dtype_mismatch, unsupported_layout, shape_mismatch, invalid_sparse_structure, non_contiguous, io_error, unsupported_encoding, unsupported_version, unsupported_dtype, unavailable_filter };
enum class NativeType : std::uint8_t {
    boolean, character, signed_character, unsigned_character, wide_character,
    character8, character16, character32, short_integer, unsigned_short_integer,
    integer, unsigned_integer, long_integer, unsigned_long_integer,
    long_long_integer, unsigned_long_long_integer, single, double_precision
};

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string path, std::string message);
    [[nodiscard]] ErrorCode code() const noexcept;
    [[nodiscard]] const std::string& path() const noexcept;
private:
    ErrorCode code_;
    std::string path_;
};

class Axis;
namespace detail {
Axis compose_axis(const Axis& parent, const Axis& child, std::size_t source_extent);
Axis gather_axis(std::vector<std::size_t>&& positions);
}

// An axis selector owns its gather positions. index() is unchecked; use it
// only after selection has validated the mapping against its source extent.
class Axis {
public:
    Axis() = default;
    static Axis all() noexcept;
    static Axis stride(std::size_t first, std::size_t count,
                       std::size_t step = 1, bool reverse = false);
    static Axis gather(const std::size_t* positions, std::size_t count);
    static Axis gather(std::initializer_list<std::size_t> positions);
    [[nodiscard]] bool is_all() const noexcept { return all_; }
    [[nodiscard]] bool is_gather() const noexcept { return bool(positions_); }
    [[nodiscard]] std::size_t extent(std::size_t source_extent) const noexcept {
        return all_ ? source_extent : count_;
    }
    [[nodiscard]] std::size_t index(std::size_t i) const noexcept {
        return all_ ? i : positions_ ? (*positions_)[i]
            : reverse_ ? first_ - i * step_ : first_ + i * step_;
    }
    [[nodiscard]] std::size_t first() const noexcept { return first_; }
    [[nodiscard]] std::size_t step() const noexcept { return step_; }
    [[nodiscard]] bool reverse() const noexcept { return reverse_; }
private:
    bool all_ = true;
    bool reverse_ = false;
    std::size_t first_ = 0, count_ = 0, step_ = 1;
    std::shared_ptr<const std::vector<std::size_t>> positions_;
    friend Axis detail::compose_axis(const Axis&, const Axis&, std::size_t);
    friend Axis detail::gather_axis(std::vector<std::size_t>&&);
};

namespace detail {
struct State;
struct Selection;
using NativeType = ::anndatapp::NativeType;
struct LargeDimensions {
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
};
// Inline the common small ranks. Arbitrary ranks remain available via the
// shared overflow descriptor, without retaining the numerical allocation.
struct DenseDimensions {
    static constexpr std::size_t inline_rank = 4;
    std::size_t size = 0;
    std::size_t shape[inline_rank]{};
    std::ptrdiff_t strides[inline_rank]{};
    std::shared_ptr<LargeDimensions> large;
    [[nodiscard]] std::size_t extent(std::size_t axis) const noexcept {
        return large ? large->shape[axis] : shape[axis];
    }
    [[nodiscard]] std::ptrdiff_t stride(std::size_t axis) const noexcept {
        return large ? large->strides[axis] : strides[axis];
    }
};
struct DenseDescriptor {
    std::shared_ptr<const State> owner;
    void* data = nullptr;
    DenseDimensions dimensions;
    bool contiguous = false;
};
struct SparseDescriptor {
    std::shared_ptr<const State> owner;
    void* values = nullptr;
    void* indices = nullptr;
    void* offsets = nullptr;
    std::size_t rows = 0, columns = 0, nnz = 0;
    bool indices_writable = false, offsets_writable = false;
};
[[nodiscard]] std::ptrdiff_t checked_byte_stride(std::ptrdiff_t stride, std::size_t element_size);
template<class> inline constexpr bool unsupported_dtype = false;
template<class T> constexpr NativeType native_type_of() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, bool>) return NativeType::boolean;
    else if constexpr (std::is_same_v<U, char>) return NativeType::character;
    else if constexpr (std::is_same_v<U, signed char>) return NativeType::signed_character;
    else if constexpr (std::is_same_v<U, unsigned char>) return NativeType::unsigned_character;
    else if constexpr (std::is_same_v<U, wchar_t>) return NativeType::wide_character;
    else if constexpr (std::is_same_v<U, char8_t>) return NativeType::character8;
    else if constexpr (std::is_same_v<U, char16_t>) return NativeType::character16;
    else if constexpr (std::is_same_v<U, char32_t>) return NativeType::character32;
    else if constexpr (std::is_same_v<U, short>) return NativeType::short_integer;
    else if constexpr (std::is_same_v<U, unsigned short>) return NativeType::unsigned_short_integer;
    else if constexpr (std::is_same_v<U, int>) return NativeType::integer;
    else if constexpr (std::is_same_v<U, unsigned int>) return NativeType::unsigned_integer;
    else if constexpr (std::is_same_v<U, long>) return NativeType::long_integer;
    else if constexpr (std::is_same_v<U, unsigned long>) return NativeType::unsigned_long_integer;
    else if constexpr (std::is_same_v<U, long long>) return NativeType::long_long_integer;
    else if constexpr (std::is_same_v<U, unsigned long long>) return NativeType::unsigned_long_long_integer;
    else if constexpr (std::is_same_v<U, float>) return NativeType::single;
    else if constexpr (std::is_same_v<U, double>) return NativeType::double_precision;
    else static_assert(unsupported_dtype<U>, "unsupported anndatapp numerical type");
}
}

template<class T> constexpr DType dtype_of() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, bool>) return DType::boolean;
    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 1) return DType::int8;
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 1) return DType::uint8;
    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 2) return DType::int16;
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 2) return DType::uint16;
    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 4) return DType::int32;
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 4) return DType::uint32;
    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 8) return DType::int64;
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 8) return DType::uint64;
    else if constexpr (std::is_same_v<U, float>) return DType::float32;
    else if constexpr (std::is_same_v<U, double>) return DType::float64;
    else static_assert(detail::unsupported_dtype<U>, "unsupported anndatapp numerical type");
}

[[nodiscard]] std::size_t dtype_size(DType type) noexcept;

template<class T> class BorrowedDenseView {
public:
    BorrowedDenseView() = default;
    BorrowedDenseView(T* data, detail::DenseDimensions dimensions, bool contiguous)
        : data_(data), dimensions_(std::move(dimensions)), contiguous_(contiguous) {}
    [[nodiscard]] T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t rank() const noexcept { return dimensions_.size; }
    [[nodiscard]] std::size_t extent(std::size_t axis) const {
        if (axis >= rank()) throw Error(ErrorCode::out_of_bounds, "array", "axis out of bounds");
        return dimensions_.extent(axis);
    }
    [[nodiscard]] std::ptrdiff_t stride(std::size_t axis) const {
        if (axis >= rank()) throw Error(ErrorCode::out_of_bounds, "array", "axis out of bounds");
        return dimensions_.stride(axis);
    }
    [[nodiscard]] bool contiguous() const noexcept { return contiguous_; }
    [[nodiscard]] T* contiguous_data() const {
        if (!contiguous_) throw Error(ErrorCode::non_contiguous, "array", "array is not row-major contiguous");
        return data_;
    }
private:
    T* data_ = nullptr;
    detail::DenseDimensions dimensions_;
    bool contiguous_ = false;
};

template<class T> class DenseView {
public:
    DenseView() = default;
    explicit DenseView(detail::DenseDescriptor descriptor) : descriptor_(std::move(descriptor)) {}
    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(descriptor_.data); }
    [[nodiscard]] std::size_t rank() const noexcept { return descriptor_.dimensions.size; }
    [[nodiscard]] std::size_t extent(std::size_t axis) const {
        if (axis >= rank()) throw Error(ErrorCode::out_of_bounds, "array", "axis out of bounds");
        return descriptor_.dimensions.extent(axis);
    }
    [[nodiscard]] std::ptrdiff_t stride(std::size_t axis) const {
        if (axis >= rank()) throw Error(ErrorCode::out_of_bounds, "array", "axis out of bounds");
        return descriptor_.dimensions.stride(axis);
    }
    [[nodiscard]] bool contiguous() const noexcept { return descriptor_.contiguous; }
    [[nodiscard]] T* contiguous_data() const {
        if (!contiguous()) throw Error(ErrorCode::non_contiguous, "array", "array is not row-major contiguous");
        return data();
    }
    // Borrowed descriptors do not retain the underlying allocation.
    [[nodiscard]] BorrowedDenseView<T> borrow() const {
        return {data(), descriptor_.dimensions, descriptor_.contiguous};
    }
private:
    detail::DenseDescriptor descriptor_;
};

// A type-erased byte view aliases any native numerical type without violating
// C++ aliasing rules. Strides are in bytes; element_size() is the scalar width.
template<class Byte> class DenseBytesView {
public:
    DenseBytesView() = default;
    DenseBytesView(detail::DenseDescriptor descriptor, std::size_t width)
        : descriptor_(std::move(descriptor)), width_(width) {}
    [[nodiscard]] Byte* data() const noexcept { return static_cast<Byte*>(descriptor_.data); }
    [[nodiscard]] std::size_t rank() const noexcept { return descriptor_.dimensions.size; }
    [[nodiscard]] std::size_t extent(std::size_t axis) const {
        if (axis >= rank()) throw Error(ErrorCode::out_of_bounds, "array", "axis out of bounds");
        return descriptor_.dimensions.extent(axis);
    }
    [[nodiscard]] std::ptrdiff_t stride_bytes(std::size_t axis) const {
        if (axis >= rank()) throw Error(ErrorCode::out_of_bounds, "array", "axis out of bounds");
        return detail::checked_byte_stride(descriptor_.dimensions.stride(axis), width_);
    }
    [[nodiscard]] std::size_t element_size() const noexcept { return width_; }
    [[nodiscard]] bool contiguous() const noexcept { return descriptor_.contiguous; }
private:
    detail::DenseDescriptor descriptor_;
    std::size_t width_ = 0;
};

template<class T, class Index, class Offset> class BorrowedSparseView {
public:
    BorrowedSparseView() = default;
    BorrowedSparseView(T* values, const Index* indices, const Offset* offsets,
                       std::size_t rows, std::size_t columns, std::size_t nnz,
                       bool indices_writable, bool offsets_writable)
        : values_(values), indices_(indices), offsets_(offsets), rows_(rows), columns_(columns), nnz_(nnz),
          indices_writable_(indices_writable), offsets_writable_(offsets_writable) {}
    [[nodiscard]] T* values() const noexcept { return values_; }
    [[nodiscard]] const Index* indices() const noexcept { return indices_; }
    [[nodiscard]] const Offset* offsets() const noexcept { return offsets_; }
    [[nodiscard]] Index* unsafe_indices() const requires (!std::is_const_v<T>) {
        if (!indices_writable_) throw Error(ErrorCode::invalid_argument, "array", "sparse index buffer is read-only");
        return const_cast<Index*>(indices_);
    }
    [[nodiscard]] Offset* unsafe_offsets() const requires (!std::is_const_v<T>) {
        if (!offsets_writable_) throw Error(ErrorCode::invalid_argument, "array", "sparse offset buffer is read-only");
        return const_cast<Offset*>(offsets_);
    }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t nnz() const noexcept { return nnz_; }
private:
    T* values_ = nullptr;
    const Index* indices_ = nullptr;
    const Offset* offsets_ = nullptr;
    std::size_t rows_ = 0, columns_ = 0, nnz_ = 0;
    bool indices_writable_ = false, offsets_writable_ = false;
};

template<class T, class Index, class Offset> class SparseView {
public:
    SparseView() = default;
    explicit SparseView(detail::SparseDescriptor descriptor) : descriptor_(std::move(descriptor)) {}
    [[nodiscard]] T* values() const noexcept { return static_cast<T*>(descriptor_.values); }
    [[nodiscard]] const Index* indices() const noexcept { return static_cast<const Index*>(descriptor_.indices); }
    [[nodiscard]] const Offset* offsets() const noexcept { return static_cast<const Offset*>(descriptor_.offsets); }
    [[nodiscard]] Index* unsafe_indices() const requires (!std::is_const_v<T>) {
        if (!descriptor_.indices_writable) throw Error(ErrorCode::invalid_argument, "array", "sparse index buffer is read-only");
        return static_cast<Index*>(descriptor_.indices);
    }
    [[nodiscard]] Offset* unsafe_offsets() const requires (!std::is_const_v<T>) {
        if (!descriptor_.offsets_writable) throw Error(ErrorCode::invalid_argument, "array", "sparse offset buffer is read-only");
        return static_cast<Offset*>(descriptor_.offsets);
    }
    [[nodiscard]] std::size_t rows() const noexcept { return descriptor_.rows; }
    [[nodiscard]] std::size_t columns() const noexcept { return descriptor_.columns; }
    [[nodiscard]] std::size_t nnz() const noexcept { return descriptor_.nnz; }
    [[nodiscard]] BorrowedSparseView<T, Index, Offset> borrow() const {
        return {values(), indices(), offsets(), rows(), columns(), nnz(),
                descriptor_.indices_writable, descriptor_.offsets_writable};
    }
private:
    detail::SparseDescriptor descriptor_;
};

class Array;
class ArrayEdit;

// Runtime-typed sparse access keeps ownership and exposes the buffers without
// a value/index/offset template cross product. Structural edits are explicit.
template<class Byte> class SparseBytesView {
public:
    [[nodiscard]] Byte* values() const noexcept { return static_cast<Byte*>(descriptor_.values); }
    [[nodiscard]] const std::byte* indices() const noexcept { return static_cast<const std::byte*>(descriptor_.indices); }
    [[nodiscard]] const std::byte* offsets() const noexcept { return static_cast<const std::byte*>(descriptor_.offsets); }
    [[nodiscard]] std::size_t rows() const noexcept { return descriptor_.rows; }
    [[nodiscard]] std::size_t columns() const noexcept { return descriptor_.columns; }
    [[nodiscard]] std::size_t nnz() const noexcept { return descriptor_.nnz; }
    [[nodiscard]] std::size_t value_size() const noexcept { return value_size_; }
    [[nodiscard]] std::size_t index_size() const noexcept { return index_size_; }
    [[nodiscard]] std::size_t offset_size() const noexcept { return offset_size_; }
    [[nodiscard]] std::byte* unsafe_indices() const requires (!std::is_const_v<Byte>) {
        if (!descriptor_.indices_writable) throw Error(ErrorCode::invalid_argument, "array", "sparse indices are read-only");
        return static_cast<std::byte*>(descriptor_.indices);
    }
    [[nodiscard]] std::byte* unsafe_offsets() const requires (!std::is_const_v<Byte>) {
        if (!descriptor_.offsets_writable) throw Error(ErrorCode::invalid_argument, "array", "sparse offsets are read-only");
        return static_cast<std::byte*>(descriptor_.offsets);
    }
private:
    SparseBytesView(detail::SparseDescriptor descriptor, std::size_t value_size, std::size_t index_size, std::size_t offset_size)
        : descriptor_(std::move(descriptor)), value_size_(value_size), index_size_(index_size), offset_size_(offset_size) {}
    detail::SparseDescriptor descriptor_;
    std::size_t value_size_, index_size_, offset_size_;
    friend class ArrayView;
    friend class ArrayEdit;
};

class ArrayView {
public:
    ArrayView() = default;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t extent(std::size_t axis) const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] Layout layout() const;
    [[nodiscard]] NativeType native_type() const;
    [[nodiscard]] DType index_dtype() const;
    [[nodiscard]] DType offset_dtype() const;
    [[nodiscard]] ByteOrder serialization_byte_order() const;
    [[nodiscard]] ByteOrder index_serialization_byte_order() const;
    [[nodiscard]] ByteOrder offset_serialization_byte_order() const;
    [[nodiscard]] NativeType index_native_type() const;
    [[nodiscard]] NativeType offset_native_type() const;
    [[nodiscard]] ArrayView select(const Axis* axes, std::size_t rank) const;
    [[nodiscard]] ArrayView select(std::initializer_list<Axis> axes) const {
        return select(axes.begin(), axes.size());
    }
    [[nodiscard]] Array copy(Validation validation = Validation::full) const;
    // Dense numerical copy into existing storage; dtype and shape must match.
    // Buffers must not overlap. No numerical allocation or implicit conversion.
    void copy_into(const ArrayEdit& destination, Validation validation = Validation::full) const;
    [[nodiscard]] DenseBytesView<const std::byte> bytes() const;
    [[nodiscard]] SparseBytesView<const std::byte> sparse_bytes() const;
    template<class T> [[nodiscard]] DenseView<const T> dense() const {
        return DenseView<const T>(dense_descriptor(dtype_of<T>(), detail::native_type_of<T>(), true));
    }
    template<class T> [[nodiscard]] DenseView<const T> unsafe_dense() const {
        return DenseView<const T>(dense_descriptor(dtype_of<T>(), detail::native_type_of<T>(), false));
    }
    template<class T, class Index, class Offset> [[nodiscard]] SparseView<const T, Index, Offset> csr() const {
        return SparseView<const T, Index, Offset>(sparse_descriptor(Layout::csr, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>()));
    }
    template<class T, class Index, class Offset> [[nodiscard]] SparseView<const T, Index, Offset> csc() const {
        return SparseView<const T, Index, Offset>(sparse_descriptor(Layout::csc, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>()));
    }
    void validate() const;
private:
    friend class Array;
    friend class ArrayEdit;
    ArrayView(std::shared_ptr<detail::State> state, std::shared_ptr<const detail::Selection> selection = {});
    [[nodiscard]] detail::DenseDescriptor dense_descriptor(DType type, detail::NativeType native, bool checked) const;
    [[nodiscard]] detail::SparseDescriptor sparse_descriptor(Layout layout, DType type, DType index, DType offset, detail::NativeType native, detail::NativeType native_index, detail::NativeType native_offset) const;
    std::shared_ptr<detail::State> state_;
    std::shared_ptr<const detail::Selection> selection_;
};

class ArrayEdit {
public:
    ArrayEdit() = default;
    [[nodiscard]] ArrayView view() const;
    [[nodiscard]] DenseBytesView<std::byte> bytes() const;
    [[nodiscard]] SparseBytesView<std::byte> sparse_bytes() const;
    template<class T> [[nodiscard]] DenseView<T> dense() const {
        return DenseView<T>(view().dense_descriptor(dtype_of<T>(), detail::native_type_of<T>(), true));
    }
    template<class T> [[nodiscard]] DenseView<T> unsafe_dense() const {
        return DenseView<T>(view().dense_descriptor(dtype_of<T>(), detail::native_type_of<T>(), false));
    }
    template<class T, class Index, class Offset> [[nodiscard]] SparseView<T, Index, Offset> csr() const {
        return SparseView<T, Index, Offset>(view().sparse_descriptor(Layout::csr, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>()));
    }
    template<class T, class Index, class Offset> [[nodiscard]] SparseView<T, Index, Offset> csc() const {
        return SparseView<T, Index, Offset>(view().sparse_descriptor(Layout::csc, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>()));
    }
private:
    friend class Array;
    explicit ArrayEdit(std::shared_ptr<detail::State> state);
    std::shared_ptr<detail::State> state_;
};

// Callback is called once with the original data pointer and caller context.
using BufferDeleter = void (*)(void* data, void* context);

class Array {
public:
    Array() = default;
    ~Array();
    Array(Array&&) noexcept;
    Array& operator=(Array&&) noexcept;
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    template<class T> [[nodiscard]] static Array empty(std::initializer_list<std::size_t> shape) {
        return allocate_dense(dtype_of<T>(), detail::native_type_of<T>(), std::vector<std::size_t>(shape), false);
    }
    template<class T> [[nodiscard]] static Array zeros(std::initializer_list<std::size_t> shape) {
        return allocate_dense(dtype_of<T>(), detail::native_type_of<T>(), std::vector<std::size_t>(shape), true);
    }
    template<class T> [[nodiscard]] static Array empty(std::vector<std::size_t> shape) {
        return allocate_dense(dtype_of<T>(), detail::native_type_of<T>(), std::move(shape), false);
    }
    template<class T> [[nodiscard]] static Array zeros(std::vector<std::size_t> shape) {
        return allocate_dense(dtype_of<T>(), detail::native_type_of<T>(), std::move(shape), true);
    }
    template<class T, class Index, class Offset> [[nodiscard]] static Array csr(std::size_t rows, std::size_t columns, std::size_t nnz) {
        return allocate_sparse(Layout::csr, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>(), rows, columns, nnz);
    }
    template<class T, class Index, class Offset> [[nodiscard]] static Array csc(std::size_t rows, std::size_t columns, std::size_t nnz) {
        return allocate_sparse(Layout::csc, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>(), rows, columns, nnz);
    }
    [[nodiscard]] static Array from_sparse_buffers(Layout layout, Array values, Array indices, Array offsets,
        std::size_t rows, std::size_t columns, Validation validation = Validation::full);
    // Strides are in elements; data denotes the first logical element. Checked
    // wrapping validates representable offsets and alignment by default. With
    // Validation::none the caller guarantees valid metadata and buffers; the
    // explicit validate() operation remains available. The caller must supply
    // a buffer covering the offsets and keep borrowed storage alive.
    template<class T> [[nodiscard]] static Array borrow_dense(T* data, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides, Validation validation = Validation::full) {
        static_assert(!std::is_volatile_v<T>, "volatile buffers are not supported");
        return wrap_dense(dtype_of<T>(), detail::native_type_of<T>(), const_cast<std::remove_const_t<T>*>(data), std::move(shape), std::move(strides), {}, !std::is_const_v<T>, validation);
    }
    template<class T> [[nodiscard]] static Array share_dense(T* data, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides, std::shared_ptr<const void> owner, Validation validation = Validation::full) {
        static_assert(!std::is_volatile_v<T>, "volatile buffers are not supported");
        if (owner.use_count() == 0) throw Error(ErrorCode::invalid_argument, "array", "retained owner is empty");
        // The owner is a lifetime token; its pointee need not be the data pointer.
        auto* token = const_cast<void*>(owner.get());
        std::shared_ptr<void> retained(std::move(owner), token);
        return wrap_dense(dtype_of<T>(), detail::native_type_of<T>(), const_cast<std::remove_const_t<T>*>(data), std::move(shape), std::move(strides), std::move(retained), !std::is_const_v<T>, validation);
    }
    template<class T> [[nodiscard]] static Array adopt_dense(T* data, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides, BufferDeleter deleter, void* context = nullptr, Validation validation = Validation::full) {
        static_assert(!std::is_const_v<T> && !std::is_volatile_v<T>, "adopted buffers must be writable");
        if (!deleter) throw Error(ErrorCode::invalid_argument, "array", "deleter is null");
        std::shared_ptr<void> owner(data, [deleter, context](void* p) { deleter(p, context); });
        return wrap_dense(dtype_of<T>(), detail::native_type_of<T>(), data, std::move(shape), std::move(strides), std::move(owner), true, validation);
    }
    template<class T, class Index, class Offset> [[nodiscard]] static Array borrow_sparse(Layout layout, std::size_t rows, std::size_t columns, std::size_t nnz, T* data, Index* indices, Offset* offsets, Validation validation = Validation::full) {
        static_assert(!std::is_volatile_v<T> && !std::is_volatile_v<Index> && !std::is_volatile_v<Offset>, "volatile buffers are not supported");
        return wrap_sparse(layout, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>(), rows, columns, nnz,
                           const_cast<std::remove_const_t<T>*>(data), const_cast<std::remove_const_t<Index>*>(indices),
                           const_cast<std::remove_const_t<Offset>*>(offsets), {},
                           !std::is_const_v<T>, !std::is_const_v<Index>, !std::is_const_v<Offset>, validation);
    }
    template<class T, class Index, class Offset> [[nodiscard]] static Array share_sparse(Layout layout, std::size_t rows, std::size_t columns, std::size_t nnz, T* data, Index* indices, Offset* offsets, std::shared_ptr<const void> owner, Validation validation = Validation::full) {
        static_assert(!std::is_volatile_v<T> && !std::is_volatile_v<Index> && !std::is_volatile_v<Offset>, "volatile buffers are not supported");
        if (owner.use_count() == 0) throw Error(ErrorCode::invalid_argument, "array", "retained owner is empty");
        auto* token = const_cast<void*>(owner.get());
        std::shared_ptr<void> retained(std::move(owner), token);
        return wrap_sparse(layout, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>(), rows, columns, nnz,
                           const_cast<std::remove_const_t<T>*>(data), const_cast<std::remove_const_t<Index>*>(indices),
                           const_cast<std::remove_const_t<Offset>*>(offsets), std::move(retained),
                           !std::is_const_v<T>, !std::is_const_v<Index>, !std::is_const_v<Offset>, validation);
    }
    template<class T, class Index, class Offset> [[nodiscard]] static Array adopt_sparse(Layout layout, std::size_t rows, std::size_t columns, std::size_t nnz, T* data, Index* indices, Offset* offsets, BufferDeleter deleter, void* context = nullptr, Validation validation = Validation::full) {
        static_assert(!std::is_const_v<T> && !std::is_const_v<Index> && !std::is_const_v<Offset>, "adopted sparse buffers must be writable");
        if (!deleter) throw Error(ErrorCode::invalid_argument, "array", "deleter is null");
        std::shared_ptr<void> owner(data, [deleter, context](void* p) { deleter(p, context); });
        return wrap_sparse(layout, dtype_of<T>(), dtype_of<Index>(), dtype_of<Offset>(), detail::native_type_of<T>(), detail::native_type_of<Index>(), detail::native_type_of<Offset>(), rows, columns, nnz, data, indices, offsets, std::move(owner), true, true, true, validation);
    }
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t extent(std::size_t axis) const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] ByteOrder serialization_byte_order() const;
    [[nodiscard]] ByteOrder index_serialization_byte_order() const;
    [[nodiscard]] ByteOrder offset_serialization_byte_order() const;
    [[nodiscard]] Layout layout() const;
    [[nodiscard]] NativeType native_type() const;
    [[nodiscard]] ArrayView view() const;
    [[nodiscard]] ArrayEdit edit();
    [[nodiscard]] Array copy(Validation validation = Validation::full) const;
    void set_serialization_byte_order(ByteOrder);
    void set_index_serialization_byte_order(ByteOrder);
    void set_offset_serialization_byte_order(ByteOrder);
    void validate() const;
private:
    friend class ArrayView;
    explicit Array(std::shared_ptr<detail::State> state);
    static Array allocate_dense(DType type, detail::NativeType native, std::vector<std::size_t> shape, bool zero);
    static Array allocate_sparse(Layout layout, DType type, DType index, DType offset, detail::NativeType native, detail::NativeType native_index, detail::NativeType native_offset, std::size_t rows, std::size_t columns, std::size_t nnz);
    static Array wrap_dense(DType type, detail::NativeType native, void* data, std::vector<std::size_t> shape, std::vector<std::ptrdiff_t> strides, std::shared_ptr<void> owner, bool writable, Validation validation);
    static Array wrap_sparse(Layout layout, DType type, DType index, DType offset, detail::NativeType native, detail::NativeType native_index, detail::NativeType native_offset, std::size_t rows, std::size_t columns, std::size_t nnz, void* values, void* indices, void* offsets, std::shared_ptr<void> owner, bool writable, bool indices_writable, bool offsets_writable, Validation validation);
    std::shared_ptr<detail::State> state_;
};

} // namespace anndatapp
