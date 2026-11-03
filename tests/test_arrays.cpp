#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

import anndata.core;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template<class F>
void expect_error(anndata::ErrorCode code, F&& f, const char* message) {
    try {
        f();
        expect(false, message);
    } catch (const anndata::Error& e) {
        expect(e.code() == code, message);
    } catch (...) {
        expect(false, message);
    }
}

void count_delete(void* data, void* context) {
    auto* count = static_cast<int*>(context);
    ++*count;
    delete[] static_cast<float*>(data);
}

struct SparseAllocation {
    float* values;
    std::int32_t* indices;
    std::int64_t* offsets;
    int* release_count;
};

void delete_sparse_allocation(void*, void* context) {
    auto* allocation = static_cast<SparseAllocation*>(context);
    ++*allocation->release_count;
    delete[] allocation->values;
    delete[] allocation->indices;
    delete[] allocation->offsets;
}

void dense_storage_and_mutation() {
    static_assert(!std::is_copy_constructible_v<anndata::Array>);
    static_assert(std::is_move_constructible_v<anndata::Array>);

    auto a = anndata::Array::empty<float>({2, 3});
    expect(a.layout() == anndata::Layout::dense, "empty dense layout");
    expect(a.rank() == 2 && a.extent(0) == 2 && a.extent(1) == 3,
           "dense shape is retained");

    auto edit = a.edit();
    auto values = edit.dense<float>();
    expect(values.rank() == 2 && values.stride(0) == 3 && values.stride(1) == 1,
           "new dense arrays are row-major");
    expect(values.contiguous(), "new dense arrays expose contiguous storage");
    const auto original_pointer = values.data();
    for (int i = 0; i < 6; ++i) values.data()[i] = static_cast<float>(i + 1);
    expect(a.view().dense<float>().data() == original_pointer,
           "typed access keeps the underlying buffer pointer");
    expect(a.view().dense<float>().data()[5] == 6.0F,
           "mutable access aliases read access");

    auto copy = a.copy();
    expect(copy.view().dense<float>().data() != original_pointer,
           "copy allocates independent dense storage");
    copy.edit().dense<float>().data()[0] = -1.0F;
    expect(a.view().dense<float>().data()[0] == 1.0F,
           "copy edits do not mutate the source");

    auto zeros = anndata::Array::zeros<std::int16_t>({2, 2});
    const auto* zero_values = zeros.view().dense<std::int16_t>().data();
    expect(zero_values[0] == 0 && zero_values[1] == 0 &&
           zero_values[2] == 0 && zero_values[3] == 0,
           "zeros explicitly initializes values");
    expect_error(anndata::ErrorCode::dtype_mismatch,
                 [&] { (void)a.view().dense<double>(); },
                 "checked dense access rejects the wrong dtype");

    auto empty_axis = anndata::Array::empty<std::uint8_t>({3, 0, 2});
    expect(empty_axis.rank() == 3 && empty_axis.extent(1) == 0,
           "zero-sized dimensions are retained");
    empty_axis.validate();

    auto rank_five = anndata::Array::empty<std::uint32_t>({2, 2, 2, 2, 2});
    auto high_rank = rank_five.edit().dense<std::uint32_t>();
    expect(high_rank.rank() == 5 && high_rank.extent(0) == 2 &&
           high_rank.extent(4) == 2 && high_rank.stride(0) == 16 &&
           high_rank.stride(4) == 1,
           "rank-five dense metadata preserves all extents and strides");
    high_rank.data()[0] = 17;
    high_rank.data()[31] = 29;
    auto nonowning_high_rank = high_rank.borrow();
    expect(nonowning_high_rank.rank() == 5 &&
           nonowning_high_rank.extent(2) == 2 &&
           nonowning_high_rank.stride(1) == 8 &&
           nonowning_high_rank.data() == high_rank.data() &&
           nonowning_high_rank.data()[31] == 29,
           "borrowed high-rank view exposes pointer and shared shape metadata");
}

void native_type_identity_and_bytes() {
    // LP64 platforms (including the initial Linux target) give long and long
    // long the same width while keeping them distinct C++ native types.
    if (sizeof(long) == sizeof(long long)) {
        std::array<long long, 2> external{13, 29};
        auto borrowed = anndata::Array::borrow_dense(external.data(), {2}, {1});
        auto typed = borrowed.view().dense<long long>();
        expect(typed.data() == external.data() && typed.data()[1] == 29,
               "borrowed long-long storage accepts its native typed view");
        expect_error(anndata::ErrorCode::dtype_mismatch,
                     [&] { (void)borrowed.view().dense<long>(); },
                     "same-width long access cannot reinterpret long-long storage");

        auto read_bytes = borrowed.view().bytes();
        expect(read_bytes.data() == reinterpret_cast<const std::byte*>(external.data()) &&
               read_bytes.element_size() == sizeof(long long) &&
               read_bytes.extent(0) == 2 &&
               read_bytes.stride_bytes(0) == static_cast<std::ptrdiff_t>(sizeof(long long)),
               "read-only byte view preserves pointer, element width, extent, and byte stride");
        long long decoded = 0;
        std::memcpy(&decoded, read_bytes.data() + sizeof(long long), sizeof(decoded));
        expect(decoded == 29, "byte view supports memcpy access to native storage");

        auto writable_bytes = borrowed.edit().bytes();
        const long long replacement = 41;
        std::memcpy(writable_bytes.data() + sizeof(long long), &replacement, sizeof(replacement));
        expect(external[1] == replacement && borrowed.view().dense<long long>().data()[1] == replacement,
               "mutable byte view aliases the original native buffer");

        auto native_copy = borrowed.copy();
        auto copied_values = native_copy.view().dense<long long>();
        expect(copied_values.data() != external.data() && copied_values.data()[0] == 13 &&
               copied_values.data()[1] == replacement,
               "copy preserves long-long native identity and copies its values");
        expect_error(anndata::ErrorCode::dtype_mismatch,
                     [&] { (void)native_copy.view().dense<long>(); },
                     "copy retains native identity instead of only equal-width dtype");

        anndata::DenseBytesView<const std::byte> retained_bytes;
        {
            auto owned_copy = native_copy.copy();
            retained_bytes = owned_copy.view().bytes();
        }
        decoded = 0;
        std::memcpy(&decoded, retained_bytes.data() + sizeof(long long), sizeof(decoded));
        expect(decoded == replacement,
               "byte descriptor retains its allocation after the array owner is destroyed");

        auto sparse = anndata::Array::csr<float, long long, long long>(1, 2, 1);
        auto sparse_edit = sparse.edit().csr<float, long long, long long>();
        sparse_edit.unsafe_offsets()[0] = 0;
        sparse_edit.unsafe_offsets()[1] = 1;
        sparse_edit.unsafe_indices()[0] = 1;
        sparse_edit.values()[0] = 3.5F;
        sparse.validate();
        auto sparse_view = sparse.view().csr<float, long long, long long>();
        expect(sparse_view.indices()[0] == 1 && sparse_view.offsets()[1] == 1,
               "sparse access accepts long-long index and offset buffers");
        expect_error(anndata::ErrorCode::dtype_mismatch,
                     [&] { (void)sparse.view().csr<float, long, long long>(); },
                     "same-width long index access cannot reinterpret long-long indices");
        expect_error(anndata::ErrorCode::dtype_mismatch,
                     [&] { (void)sparse.view().csr<float, long long, long>(); },
                     "same-width long offset access cannot reinterpret long-long offsets");

        auto sparse_copy = sparse.copy();
        auto sparse_copy_view = sparse_copy.view().csr<float, long long, long long>();
        expect(sparse_copy_view.indices()[0] == 1 && sparse_copy_view.offsets()[1] == 1 &&
               sparse_copy_view.values()[0] == 3.5F,
               "sparse copy retains native index and offset type identity");
        expect_error(anndata::ErrorCode::dtype_mismatch,
                     [&] { (void)sparse_copy.view().csr<float, long, long long>(); },
                     "sparse copy retains long-long index identity");
        expect_error(anndata::ErrorCode::dtype_mismatch,
                     [&] { (void)sparse_copy.view().csr<float, long long, long>(); },
                     "sparse copy retains long-long offset identity");
    }
}

void checked_ranges_and_shared_owners() {
    const auto max_stride = std::numeric_limits<std::ptrdiff_t>::max();
    const auto min_stride = std::numeric_limits<std::ptrdiff_t>::min();
    std::array<std::int8_t, 1> byte_storage{0};

    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_dense(
                         byte_storage.data(), {3}, {max_stride / 2 + 1});
                 },
                 "dense borrowing rejects positive element-offset overflow");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_dense(
                         byte_storage.data(), {3}, {min_stride / 2 - 1});
                 },
                 "dense borrowing rejects negative element-offset overflow");

    std::array<std::int64_t, 1> wide_storage{0};
    const auto byte_overflow_stride =
        max_stride / static_cast<std::ptrdiff_t>(sizeof(std::int64_t)) + 1;
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_dense(
                         wide_storage.data(), {2}, {byte_overflow_stride});
                 },
                 "dense borrowing rejects byte-offset overflow after element offset fits");

    const auto axis_stride = max_stride / 2 + 1;
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_dense(
                         byte_storage.data(), {2, 2}, {axis_stride, axis_stride});
                 },
                 "dense borrowing rejects total offset overflow across axes");

    auto empty_extreme = anndata::Array::borrow_dense(
        static_cast<const std::int8_t*>(nullptr), {0, 2}, {min_stride, max_stride});
    empty_extreme.validate();
    auto empty_view = empty_extreme.view().dense<std::int8_t>();
    expect(empty_view.extent(0) == 0 && empty_view.extent(1) == 2 &&
           empty_view.stride(0) == min_stride && empty_view.stride(1) == max_stride,
           "empty dense arrays accept extreme strides without address arithmetic");

    const auto size_max = std::numeric_limits<std::size_t>::max();
    const auto offset_width = sizeof(std::int64_t);
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_sparse(
                         anndata::Layout::csr, size_max / offset_width, 0, 0,
                         static_cast<float*>(nullptr), static_cast<std::int32_t*>(nullptr),
                         static_cast<std::int64_t*>(nullptr));
                 },
                 "sparse borrowing rejects size_t overflow in the offsets byte span");

    const auto huge_value_count = size_max / sizeof(double) + 1;
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_sparse(
                         anndata::Layout::csr, 0, 0, huge_value_count,
                         static_cast<double*>(nullptr), static_cast<std::int32_t*>(nullptr),
                         static_cast<std::int32_t*>(nullptr));
                 },
                 "sparse borrowing rejects size_t overflow in value bytes before reading buffers");

    const auto huge_index_count = size_max / sizeof(std::int64_t) + 1;
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_sparse(
                         anndata::Layout::csr, 0, 0, huge_index_count,
                         static_cast<std::uint8_t*>(nullptr), static_cast<std::int64_t*>(nullptr),
                         static_cast<std::int32_t*>(nullptr));
                 },
                 "sparse borrowing rejects size_t overflow in index bytes before reading buffers");

    std::array<std::int32_t, 2> dense_external{5, 8};
    int dense_owner_releases = 0;
    bool dense_released_null = false;
    anndata::DenseView<const std::int32_t> retained_dense;
    {
        std::shared_ptr<const void> null_value_owner(
            nullptr, [&](const void* pointer) {
                dense_released_null = pointer == nullptr;
                ++dense_owner_releases;
            });
        expect(!null_value_owner && null_value_owner.use_count() == 1,
               "null-valued shared owner still has a live control block");
        auto shared = anndata::Array::share_dense(
            dense_external.data(), {2}, {1}, null_value_owner);
        null_value_owner.reset();
        retained_dense = shared.view().dense<std::int32_t>();
    }
    expect(dense_owner_releases == 0 && retained_dense.data() == dense_external.data(),
           "shared dense view retains a null-valued owner's control block");
    retained_dense = {};
    expect(dense_owner_releases == 1 && dense_released_null,
           "dense shared-owner deleter fires once after the final retained view");

    std::array<float, 1> sparse_values{2.5F};
    std::array<std::int32_t, 1> sparse_indices{0};
    std::array<std::int64_t, 2> sparse_offsets{0, 1};
    int sparse_owner_releases = 0;
    bool sparse_released_null = false;
    anndata::SparseView<const float, std::int32_t, std::int64_t> retained_sparse;
    {
        std::shared_ptr<const void> null_value_owner(
            nullptr, [&](const void* pointer) {
                sparse_released_null = pointer == nullptr;
                ++sparse_owner_releases;
            });
        auto shared = anndata::Array::share_sparse(
            anndata::Layout::csr, 1, 1, 1, sparse_values.data(), sparse_indices.data(),
            sparse_offsets.data(), null_value_owner);
        null_value_owner.reset();
        retained_sparse = shared.view().csr<float, std::int32_t, std::int64_t>();
    }
    expect(sparse_owner_releases == 0 && retained_sparse.values() == sparse_values.data(),
           "shared sparse view retains a null-valued owner's control block");
    retained_sparse = {};
    expect(sparse_owner_releases == 1 && sparse_released_null,
           "sparse shared-owner deleter fires once after the final retained view");

    std::shared_ptr<const void> truly_empty_owner;
    std::shared_ptr<const void> nonowning_alias(
        truly_empty_owner, static_cast<const void*>(dense_external.data()));
    expect(nonowning_alias.use_count() == 0 && nonowning_alias.get() == dense_external.data(),
           "aliasing a null control block can have a nonnull stored pointer");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::share_dense(
                         dense_external.data(), {2}, {1}, truly_empty_owner);
                 },
                 "share_dense rejects a truly empty owner");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::share_dense(
                         dense_external.data(), {2}, {1}, nonowning_alias);
                 },
                 "share_dense rejects a nonowning alias with a nonnull stored pointer");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::share_sparse(
                         anndata::Layout::csr, 1, 1, 1, sparse_values.data(),
                         sparse_indices.data(), sparse_offsets.data(), truly_empty_owner);
                 },
                 "share_sparse rejects a truly empty owner");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::share_sparse(
                         anndata::Layout::csr, 1, 1, 1, sparse_values.data(),
                         sparse_indices.data(), sparse_offsets.data(), nonowning_alias);
                 },
                 "share_sparse rejects a nonowning alias with a nonnull stored pointer");
}

void empty_and_broadcast_extreme_dimensions() {
    const auto size_max = std::numeric_limits<std::size_t>::max();
    const auto stride_max = std::numeric_limits<std::ptrdiff_t>::max();
    std::uint8_t byte = 73;

    auto borrowed_empty = anndata::Array::borrow_dense(
        &byte, {size_max, size_max, 0}, {stride_max, stride_max, stride_max});
    auto borrowed_view = borrowed_empty.view().dense<std::uint8_t>();
    expect(borrowed_view.rank() == 3 && borrowed_view.extent(0) == size_max &&
           borrowed_view.extent(1) == size_max && borrowed_view.extent(2) == 0 &&
           borrowed_view.stride(0) == stride_max && borrowed_view.stride(2) == stride_max,
           "zero extent after huge axes permits dense view acquisition without product overflow");

    auto borrowed_copy = borrowed_empty.copy();
    auto borrowed_copy_view = borrowed_copy.view().dense<std::uint8_t>();
    expect(borrowed_copy_view.extent(0) == size_max &&
           borrowed_copy_view.extent(1) == size_max && borrowed_copy_view.extent(2) == 0,
           "copy materializes an empty shape with huge leading dimensions");

    auto allocated_empty = anndata::Array::empty<std::uint8_t>({0, size_max, size_max});
    allocated_empty.validate();
    auto allocated_empty_view = allocated_empty.view().dense<std::uint8_t>();
    expect(allocated_empty_view.extent(0) == 0 &&
           allocated_empty_view.extent(1) == size_max &&
           allocated_empty_view.extent(2) == size_max,
           "empty allocation accepts a leading zero before huge trailing dimensions");

    const auto large_index = static_cast<std::size_t>(stride_max) + 1;
    const auto broadcast_extent = large_index + 1;
    auto broadcast = anndata::Array::borrow_dense(&byte, {broadcast_extent}, {0});
    using anndata::Axis;
    auto selected = broadcast.view().select({Axis::gather({large_index})});
    auto selected_view = selected.dense<std::uint8_t>();
    expect(selected_view.extent(0) == 1 && selected_view.stride(0) == 0 &&
           selected_view.data() == &byte && selected_view.data()[0] == 73,
           "zero-stride broadcast selection accepts an index beyond PTRDIFF_MAX");

    auto selected_copy = selected.copy();
    auto selected_copy_view = selected_copy.view().dense<std::uint8_t>();
    expect(selected_copy_view.extent(0) == 1 && selected_copy_view.data()[0] == 73,
           "zero-stride broadcast materialization handles a huge logical index");
}

void validation_opt_out_and_raw_sparse_copy() {
    std::array<float, 1> values{4.0F};
    std::array<std::int32_t, 1> indices{0};
    std::array<std::int64_t, 2> invalid_offsets{0, 2};

    auto unchecked_sparse = anndata::Array::borrow_sparse(
        anndata::Layout::csr, 1, 1, 1, values.data(), indices.data(),
        invalid_offsets.data(), anndata::Validation::none);
    auto unchecked_sparse_view =
        unchecked_sparse.view().csr<float, std::int32_t, std::int64_t>();
    expect(unchecked_sparse_view.values() == values.data() &&
           unchecked_sparse_view.indices() == indices.data() &&
           unchecked_sparse_view.offsets() == invalid_offsets.data(),
           "sparse Validation::none wraps invalid structure without changing buffers");
    expect_error(anndata::ErrorCode::invalid_sparse_structure,
                 [&] { unchecked_sparse.validate(); },
                 "explicit validation diagnoses invalid sparse structure after unchecked wrap");

    auto raw_copy = unchecked_sparse.copy();
    auto raw_copy_view = raw_copy.view().csr<float, std::int32_t, std::int64_t>();
    expect(raw_copy_view.values() != values.data() &&
           raw_copy_view.indices() != indices.data() &&
           raw_copy_view.offsets() != invalid_offsets.data() &&
           raw_copy_view.values()[0] == values[0] &&
           raw_copy_view.indices()[0] == indices[0] &&
           raw_copy_view.offsets()[0] == invalid_offsets[0] &&
           raw_copy_view.offsets()[1] == invalid_offsets[1],
           "unselected sparse copy preserves invalid raw buffers byte-for-byte");
    expect_error(anndata::ErrorCode::invalid_sparse_structure,
                 [&] { raw_copy.validate(); },
                 "explicit validation diagnoses invalid structure in an unselected copy");

    expect_error(anndata::ErrorCode::invalid_sparse_structure,
                 [&] {
                     (void)anndata::Array::borrow_sparse(
                         anndata::Layout::csr, 1, 1, 1, values.data(), indices.data(),
                         invalid_offsets.data());
                 },
                 "default sparse wrapping retains full structural validation");

    auto unchecked_dense = anndata::Array::borrow_dense(
        static_cast<std::int32_t*>(nullptr), {1}, {1}, anndata::Validation::none);
    expect(unchecked_dense.view().dense<std::int32_t>().data() == nullptr,
           "dense Validation::none accepts representable nonempty null metadata without dereference");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { unchecked_dense.validate(); },
                 "explicit validation diagnoses unchecked nonempty null dense storage");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] {
                     (void)anndata::Array::borrow_dense(
                         static_cast<std::int32_t*>(nullptr), {1}, {1});
                 },
                 "default dense wrapping retains full buffer validation");

    auto unchecked_readonly = anndata::Array::borrow_dense(
        static_cast<const std::int32_t*>(nullptr), {1}, {1}, anndata::Validation::none);
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)unchecked_readonly.edit(); },
                 "Validation::none does not bypass the read-only mutation guard");

    std::array<std::int32_t, 1> dense_external{17};
    int dense_owner_releases = 0;
    anndata::DenseView<const std::int32_t> retained_dense;
    {
        std::shared_ptr<const void> owner(
            static_cast<const void*>(dense_external.data()),
            [&](const void*) { ++dense_owner_releases; });
        auto shared = anndata::Array::share_dense(
            dense_external.data(), {1}, {1}, owner, anndata::Validation::none);
        owner.reset();
        retained_dense = shared.view().dense<std::int32_t>();
    }
    expect(dense_owner_releases == 0 && retained_dense.data() == dense_external.data(),
           "valid unchecked dense wrap preserves its pointer and retained owner");
    retained_dense = {};
    expect(dense_owner_releases == 1,
           "unchecked dense shared owner releases after the last retained descriptor");

    int dense_adopt_releases = 0;
    anndata::DenseView<const float> retained_adopted_dense;
    {
        auto* adopted_values = new float[1]{6.25F};
        auto adopted = anndata::Array::adopt_dense(
            adopted_values, {1}, {1}, count_delete, &dense_adopt_releases,
            anndata::Validation::none);
        retained_adopted_dense = adopted.view().dense<float>();
    }
    expect(dense_adopt_releases == 0 && retained_adopted_dense.data()[0] == 6.25F,
           "unchecked adopted dense storage remains alive through its retained descriptor");
    retained_adopted_dense = {};
    expect(dense_adopt_releases == 1,
           "unchecked adopted dense deleter runs after the last retained descriptor");

    std::array<float, 1> sparse_external_values{2.5F};
    std::array<std::int32_t, 1> sparse_external_indices{0};
    std::array<std::int64_t, 2> sparse_external_offsets{0, 1};
    int sparse_owner_releases = 0;
    anndata::SparseView<const float, std::int32_t, std::int64_t> retained_sparse;
    {
        std::shared_ptr<const void> owner(
            static_cast<const void*>(sparse_external_values.data()),
            [&](const void*) { ++sparse_owner_releases; });
        auto shared = anndata::Array::share_sparse(
            anndata::Layout::csr, 1, 1, 1, sparse_external_values.data(),
            sparse_external_indices.data(), sparse_external_offsets.data(), owner,
            anndata::Validation::none);
        owner.reset();
        retained_sparse = shared.view().csr<float, std::int32_t, std::int64_t>();
    }
    expect(sparse_owner_releases == 0 &&
           retained_sparse.values() == sparse_external_values.data() &&
           retained_sparse.indices() == sparse_external_indices.data() &&
           retained_sparse.offsets() == sparse_external_offsets.data(),
           "valid unchecked sparse wrap preserves its pointers and retained owner");
    retained_sparse = {};
    expect(sparse_owner_releases == 1,
           "unchecked sparse shared owner releases after the last retained descriptor");

    int sparse_adopt_releases = 0;
    SparseAllocation allocation{
        new float[1]{8.5F}, new std::int32_t[1]{0}, new std::int64_t[2]{0, 1},
        &sparse_adopt_releases
    };
    anndata::SparseView<const float, std::int32_t, std::int64_t> retained_adopted_sparse;
    {
        auto adopted = anndata::Array::adopt_sparse(
            anndata::Layout::csr, 1, 1, 1, allocation.values, allocation.indices,
            allocation.offsets, delete_sparse_allocation, &allocation,
            anndata::Validation::none);
        retained_adopted_sparse = adopted.view().csr<float, std::int32_t, std::int64_t>();
    }
    expect(sparse_adopt_releases == 0 && retained_adopted_sparse.values()[0] == 8.5F &&
           retained_adopted_sparse.indices()[0] == 0 && retained_adopted_sparse.offsets()[1] == 1,
           "unchecked adopted sparse buffers remain alive through their retained descriptor");
    retained_adopted_sparse = {};
    expect(sparse_adopt_releases == 1,
           "unchecked adopted sparse deleter runs after the last retained descriptor");
}

void retained_and_borrowed_memory() {
    int deleted = 0;
    anndata::DenseView<const float> retained;
    {
        auto* pointer = new float[4]{1, 2, 3, 4};
        auto array = anndata::Array::adopt_dense(
            pointer, {4}, {1}, count_delete, &deleted);
        retained = array.view().dense<float>();
        expect(retained.data() == pointer, "adopted pointer remains direct");
    }
    expect(deleted == 0 && retained.data()[3] == 4.0F,
           "typed access retains the adopted allocation");
    retained = {};
    expect(deleted == 1, "adopted allocation is deleted after the last view");

    std::array<std::int32_t, 4> source{10, 20, 30, 40};
    {
        auto borrowed = anndata::Array::borrow_dense(
            source.data(), {4}, {1});
        auto access = borrowed.edit().dense<std::int32_t>();
        expect(access.data() == source.data(), "borrowed access does not copy");
        access.data()[1] = 99;
    }
    expect(source[1] == 99, "borrowed mutable access aliases caller storage");

    auto reverse = anndata::Array::borrow_dense(
        source.data() + 3, {4}, {-1});
    reverse.validate();
    auto reversed = reverse.view().dense<std::int32_t>();
    expect(reversed.stride(0) == -1 && reversed.data()[0] == 40 &&
           reversed.data()[-1] == 30 && reversed.data()[-3] == 10,
           "negative element strides preserve the logical first element");
    expect_error(anndata::ErrorCode::non_contiguous,
                 [&] { (void)reversed.contiguous_data(); },
                 "contiguous access rejects a reversed view");

    auto readonly = anndata::Array::borrow_dense(
        static_cast<const std::int32_t*>(source.data()), {4}, {1});
    static_assert(std::is_same_v<decltype(readonly.view().dense<std::int32_t>().data()),
                                 const std::int32_t*>);
    auto borrow = readonly.view().dense<std::int32_t>().borrow();
    expect(borrow.data() == source.data() && borrow.contiguous_data() == source.data(),
           "explicit borrowed view exposes a non-owning direct pointer");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)readonly.edit(); },
                 "read-only borrowed storage cannot be edited");

    std::shared_ptr<void> owner(source.data(), [](void*) {});
    auto shared = anndata::Array::share_dense(
        source.data(), {4}, {1}, std::move(owner));
    expect(shared.view().dense<std::int32_t>().data() == source.data(),
           "shared external ownership retains the pointer");
}

void lazy_dense_selection() {
    auto a = anndata::Array::empty<std::int32_t>({3, 4});
    auto edit = a.edit().dense<std::int32_t>();
    for (std::int32_t i = 0; i < 12; ++i) edit.data()[i] = i;

    using anndata::Axis;
    auto selection = a.view().select({
        Axis::gather({2, 0}), Axis::gather({1, 1})
    });
    expect(selection.extent(0) == 2 && selection.extent(1) == 2,
           "gather selection shape follows selector lengths");
    auto affine = selection.dense<std::int32_t>();
    expect(affine.data()[0] == 9 && affine.stride(0) == -8 &&
           affine.stride(1) == 0 && !affine.contiguous(),
           "affine negative and repeated selections expose their true strides");
    auto non_affine = a.view().select({
        Axis::gather({2, 0, 2}), Axis::gather({1, 1})
    });
    expect_error(anndata::ErrorCode::non_contiguous,
                 [&] { (void)non_affine.dense<std::int32_t>(); },
                 "non-affine gathers require explicit materialization");

    auto gathered = selection.copy();
    expect(gathered.layout() == anndata::Layout::dense,
           "dense selection materializes densely");
    auto out = gathered.view().dense<std::int32_t>();
    expect(out.data()[0] == 9 && out.data()[1] == 9 &&
           out.data()[2] == 1 && out.data()[3] == 1,
           "materialization respects order, repeats, and axis mapping");

    auto all = a.view().select({Axis::all(), Axis::all()});
    expect(all.dense<std::int32_t>().data() == a.view().dense<std::int32_t>().data(),
           "all-axis selection retains the same allocation");

    auto contiguous_row = a.view().select({
        Axis::gather({1}), Axis::gather({0, 1, 2, 3})
    });
    auto row_access = contiguous_row.dense<std::int32_t>();
    expect(row_access.contiguous() && row_access.data() == edit.data() + 4,
           "a contiguous selection exposes its source buffer interval");
    auto row_copy = contiguous_row.copy();
    auto row_values = row_copy.view().dense<std::int32_t>();
    expect(row_values.data()[0] == 4 && row_values.data()[3] == 7,
           "contiguous selection copy preserves its buffer interval");

    auto empty = a.view().select({
        Axis::gather({}), Axis::gather({1, 2})
    });
    expect(Axis::gather(nullptr, 0).is_gather(), "empty gather retains its kind");
    auto materialized_empty = empty.copy();
    expect(materialized_empty.extent(0) == 0 && materialized_empty.extent(1) == 2,
           "empty gathers keep the nonempty axis and rank");

    auto line = anndata::Array::empty<std::int32_t>({10});
    for (int i = 0; i < 10; ++i) line.edit().dense<std::int32_t>().data()[i] = i;
    auto reverse = line.view().select({Axis::stride(9, 5, 2, true)});
    auto nested = reverse.select({Axis::stride(1, 2, 2)});
    auto nested_dense = nested.dense<std::int32_t>();
    expect(nested.extent(0) == 2 && nested_dense.data()[0] == 7 &&
           nested_dense.stride(0) == -4 && nested_dense.data()[nested_dense.stride(0)] == 3,
           "nested affine axes compose to the original allocation");
    auto repeated = nested.select({Axis::gather({1, 0, 1})}).copy();
    auto repeated_data = repeated.view().dense<std::int32_t>().data();
    expect(repeated_data[0] == 3 && repeated_data[1] == 7 && repeated_data[2] == 3,
           "gathers compose after reverse strides, preserving duplicates");
    std::size_t input[] = {8, 2, 6};
    auto retained = line.view().select({Axis::gather(input, 3)});
    input[0] = 0;
    auto retained_copy = retained.copy();
    expect(retained_copy.view().dense<std::int32_t>().data()[0] == 8,
           "gather copies caller positions before returning");
    auto composed = [] {
        auto source = anndata::Array::empty<std::int32_t>({10});
        auto values = source.edit().dense<std::int32_t>();
        for (int i = 0; i < 10; ++i) values.data()[i] = i;
        auto parent = source.view().select({Axis::gather({8, 2, 6})});
        return parent.select({Axis::gather({2, 0, 2})});
    }();
    auto composed_copy = composed.copy();
    auto composed_values = composed_copy.view().dense<std::int32_t>().data();
    expect(composed_values[0] == 6 && composed_values[1] == 8 && composed_values[2] == 6,
           "composed gathers retain their positions and source after local owners expire");
    auto normalized_axis = [] {
        std::vector<std::size_t> normalized{3, 7};
        return anndata::detail::gather_axis(std::move(normalized));
    }();
    expect(normalized_axis.index(0) == 3 && normalized_axis.index(1) == 7,
           "normalized gather owns independent retained positions");
    auto broadcast = line.view().select({Axis::stride(4, 3, 0)}).dense<std::int32_t>();
    expect(broadcast.data()[0] == 4 && broadcast.stride(0) == 0,
           "zero-step affine selector represents repeated coordinates");
    const auto max = std::numeric_limits<std::size_t>::max();
    auto zero = line.view().select({Axis::stride(max, 0, max, true)});
    auto one = line.view().select({Axis::stride(9, 1, max, true)});
    expect(zero.extent(0) == 0 && one.dense<std::int32_t>().data()[0] == 9,
           "empty and singleton strides ignore unused arithmetic");
    expect_error(anndata::ErrorCode::out_of_bounds,
                 [&] { (void)line.view().select({Axis::stride(1, 2, max)}); },
                 "overflowing stride endpoint is rejected");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)Axis::gather(nullptr, 1); }, "null nonempty gather is rejected");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)line.view().select(nullptr, 1); }, "null nonempty selectors are rejected");
}

void sparse_layout_and_validation() {
    auto csr = anndata::Array::csr<float, std::int32_t, std::int64_t>(2, 3, 4);
    {
        auto sparse = csr.edit().csr<float, std::int32_t, std::int64_t>();
        sparse.unsafe_offsets()[0] = 0;
        sparse.unsafe_offsets()[1] = 3;
        sparse.unsafe_offsets()[2] = 4;
        sparse.unsafe_indices()[0] = 2;
        sparse.unsafe_indices()[1] = 1;
        sparse.unsafe_indices()[2] = 1;
        sparse.unsafe_indices()[3] = 0;
        sparse.values()[0] = 5;
        sparse.values()[1] = 7;
        sparse.values()[2] = 8;
        sparse.values()[3] = 9;
    }
    csr.validate(); // Unsorted columns and duplicate entries are legal CSR storage.
    auto c = csr.view().csr<float, std::int32_t, std::int64_t>();
    expect(c.rows() == 2 && c.columns() == 3 && c.nnz() == 4,
           "CSR dimensions and nnz");
    expect(c.indices()[0] == 2 && c.indices()[1] == 1 && c.indices()[2] == 1,
           "CSR preserves unsorted duplicate indices");
    expect(c.offsets()[2] == 4 && c.values()[3] == 9,
           "CSR preserves independent offset width");
    expect_error(anndata::ErrorCode::dtype_mismatch,
                 [&] { (void)csr.view().csr<float, std::int64_t, std::int64_t>(); },
                 "checked CSR access checks the index width");

    auto csc = anndata::Array::csc<double, std::uint64_t, std::uint32_t>(2, 3, 2);
    {
        auto sparse = csc.edit().csc<double, std::uint64_t, std::uint32_t>();
        sparse.unsafe_offsets()[0] = 0;
        sparse.unsafe_offsets()[1] = 1;
        sparse.unsafe_offsets()[2] = 1;
        sparse.unsafe_offsets()[3] = 2;
        sparse.unsafe_indices()[0] = 1;
        sparse.unsafe_indices()[1] = 0;
        sparse.values()[0] = 2.5;
        sparse.values()[1] = 3.5;
    }
    csc.validate();
    auto csc_view = csc.view().csc<double, std::uint64_t, std::uint32_t>();
    expect(csc.layout() == anndata::Layout::csc && csc_view.rows() == 2 &&
           csc_view.columns() == 3 && csc_view.nnz() == 2,
           "CSC layout and dimensions");
    expect(csc_view.offsets()[3] == 2 && csc_view.indices()[1] == 0,
           "CSC preserves 64-bit indices and 32-bit offsets");

    auto preserved = csr.copy();
    auto p = preserved.view().csr<float, std::int32_t, std::int64_t>();
    expect(preserved.layout() == anndata::Layout::csr &&
           p.indices()[1] == 1 && p.indices()[2] == 1 && p.offsets()[2] == 4,
           "sparse copies preserve format, widths, and duplicate entries");
    expect(p.values() != c.values() && p.indices() != c.indices() &&
           p.offsets() != c.offsets() && p.values()[0] == c.values()[0] &&
           p.indices()[0] == c.indices()[0] && p.offsets()[2] == c.offsets()[2],
           "unselected CSR copy owns independent, faithfully copied buffers");

    auto csc_copy = csc.copy();
    auto csc_copy_view = csc_copy.view().csc<double, std::uint64_t, std::uint32_t>();
    expect(csc_copy.layout() == anndata::Layout::csc &&
           csc_copy_view.values() != csc_view.values() &&
           csc_copy_view.indices() != csc_view.indices() &&
           csc_copy_view.offsets() != csc_view.offsets() &&
           csc_copy_view.values()[1] == 3.5 && csc_copy_view.indices()[1] == 0 &&
           csc_copy_view.offsets()[3] == 2,
           "unselected CSC copy owns independent, faithfully copied buffers");

    {
        auto writable_structure = csr.edit().csr<float, std::int32_t, std::int64_t>();
        writable_structure.unsafe_offsets()[2] = 3;
    }
    expect_error(anndata::ErrorCode::invalid_sparse_structure,
                 [&] { csr.validate(); },
                 "sparse validation detects a broken terminal offset");

    auto empty = anndata::Array::csr<double, std::int32_t, std::int64_t>(3, 0, 0);
    auto empty_view = empty.view().csr<double, std::int32_t, std::int64_t>();
    expect(empty_view.nnz() == 0 && empty_view.offsets()[3] == 0,
           "empty CSR has an initialized all-zero offset array");
    empty.validate();

    std::array<float, 1> external_values{4.0F};
    std::array<std::int32_t, 1> external_indices{0};
    std::array<std::int64_t, 2> external_offsets{0, 1};
    auto borrowed_sparse = anndata::Array::borrow_sparse(
        anndata::Layout::csr, 1, 1, 1, external_values.data(),
        external_indices.data(), external_offsets.data());
    auto external_view = borrowed_sparse.view().csr<float, std::int32_t, std::int64_t>();
    auto nonowning = external_view.borrow();
    expect(nonowning.values() == external_values.data() &&
           nonowning.indices() == external_indices.data() &&
           nonowning.offsets() == external_offsets.data(),
           "sparse borrowed descriptor exposes all external buffers directly");
    borrowed_sparse.validate();

    auto readonly_indices = anndata::Array::borrow_sparse(
        anndata::Layout::csr, 1, 1, 1, external_values.data(),
        static_cast<const std::int32_t*>(external_indices.data()), external_offsets.data());
    auto mixed_edit = readonly_indices.edit().csr<float, std::int32_t, std::int64_t>();
    mixed_edit.values()[0] = 6.0F;
    expect(external_values[0] == 6.0F,
           "mutable sparse values remain writable with read-only indices");
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)mixed_edit.unsafe_indices(); },
                 "unsafe sparse index access rejects a read-only index buffer");
    mixed_edit.unsafe_offsets()[1] = 1;
    auto mixed_borrow = mixed_edit.borrow();
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)mixed_borrow.unsafe_indices(); },
                 "non-owning sparse descriptor preserves read-only index access");
    mixed_borrow.unsafe_offsets()[1] = 1;

    auto readonly_offsets = anndata::Array::borrow_sparse(
        anndata::Layout::csr, 1, 1, 1, external_values.data(), external_indices.data(),
        static_cast<const std::int64_t*>(external_offsets.data()));
    auto mixed_offsets_edit = readonly_offsets.edit().csr<float, std::int32_t, std::int64_t>();
    mixed_offsets_edit.values()[0] = 7.0F;
    mixed_offsets_edit.unsafe_indices()[0] = 0;
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)mixed_offsets_edit.unsafe_offsets(); },
                 "unsafe sparse offset access rejects a read-only offset buffer");
    auto mixed_offsets_borrow = mixed_offsets_edit.borrow();
    expect_error(anndata::ErrorCode::invalid_argument,
                 [&] { (void)mixed_offsets_borrow.unsafe_offsets(); },
                 "non-owning sparse descriptor preserves read-only offset access");
}

void compact_sparse_integer_widths() {
    auto signed_csr = anndata::Array::csr<float, std::int8_t, std::int16_t>(2, 3, 3);
    {
        auto sparse = signed_csr.edit().csr<float, std::int8_t, std::int16_t>();
        sparse.unsafe_offsets()[0] = 0;
        sparse.unsafe_offsets()[1] = 2;
        sparse.unsafe_offsets()[2] = 3;
        sparse.unsafe_indices()[0] = 2;
        sparse.unsafe_indices()[1] = 0;
        sparse.unsafe_indices()[2] = 1;
        sparse.values()[0] = 1;
        sparse.values()[1] = 2;
        sparse.values()[2] = 3;
    }
    signed_csr.validate();
    expect_error(anndata::ErrorCode::non_contiguous,
        [&] { (void)signed_csr.view().select({anndata::Axis::stride(0, 2), anndata::Axis::all()})
            .csr<float, std::int8_t, std::int16_t>(); },
        "explicit identity selection still requires sparse materialization");
    auto signed_csr_view = signed_csr.view().csr<float, std::int8_t, std::int16_t>();
    auto signed_csr_copy = signed_csr.copy();
    auto signed_csr_copy_view = signed_csr_copy.view().csr<float, std::int8_t, std::int16_t>();
    auto signed_csr_gather = signed_csr.view().select({
        anndata::Axis::gather({1, 0}), anndata::Axis::all()
    }).copy();
    auto signed_csr_gather_view = signed_csr_gather.view().csr<float, std::int8_t, std::int16_t>();
    expect(signed_csr_view.indices()[0] == 2 && signed_csr_view.offsets()[2] == 3 &&
           signed_csr_copy_view.indices()[0] == 2 && signed_csr_copy_view.offsets()[2] == 3 &&
           signed_csr_gather_view.indices()[0] == 1 && signed_csr_gather_view.offsets()[1] == 1,
           "signed 8-bit CSR indices and 16-bit offsets survive access, copy, and row gather");

    auto unsigned_csr = anndata::Array::csr<double, std::uint8_t, std::uint16_t>(2, 3, 3);
    {
        auto sparse = unsigned_csr.edit().csr<double, std::uint8_t, std::uint16_t>();
        sparse.unsafe_offsets()[0] = 0;
        sparse.unsafe_offsets()[1] = 2;
        sparse.unsafe_offsets()[2] = 3;
        sparse.unsafe_indices()[0] = 2;
        sparse.unsafe_indices()[1] = 0;
        sparse.unsafe_indices()[2] = 1;
        sparse.values()[0] = 4;
        sparse.values()[1] = 5;
        sparse.values()[2] = 6;
    }
    unsigned_csr.validate();
    auto unsigned_csr_view = unsigned_csr.view().csr<double, std::uint8_t, std::uint16_t>();
    auto unsigned_csr_copy = unsigned_csr.copy();
    auto unsigned_csr_copy_view = unsigned_csr_copy.view().csr<double, std::uint8_t, std::uint16_t>();
    expect(unsigned_csr_view.indices()[0] == 2 && unsigned_csr_view.offsets()[2] == 3 &&
           unsigned_csr_copy_view.indices()[2] == 1 && unsigned_csr_copy_view.offsets()[2] == 3,
           "unsigned 8-bit CSR indices and 16-bit offsets survive direct access and copy");

    auto signed_csc = anndata::Array::csc<float, std::int8_t, std::int16_t>(3, 2, 3);
    {
        auto sparse = signed_csc.edit().csc<float, std::int8_t, std::int16_t>();
        sparse.unsafe_offsets()[0] = 0;
        sparse.unsafe_offsets()[1] = 2;
        sparse.unsafe_offsets()[2] = 3;
        sparse.unsafe_indices()[0] = 2;
        sparse.unsafe_indices()[1] = 0;
        sparse.unsafe_indices()[2] = 1;
        sparse.values()[0] = 7;
        sparse.values()[1] = 8;
        sparse.values()[2] = 9;
    }
    signed_csc.validate();
    auto signed_csc_view = signed_csc.view().csc<float, std::int8_t, std::int16_t>();
    auto signed_csc_gather = signed_csc.view().select({
        anndata::Axis::all(), anndata::Axis::gather({1, 0})
    }).copy();
    auto signed_csc_gather_view = signed_csc_gather.view().csc<float, std::int8_t, std::int16_t>();
    expect(signed_csc_view.indices()[0] == 2 && signed_csc_view.offsets()[2] == 3 &&
           signed_csc_gather_view.indices()[0] == 1 && signed_csc_gather_view.offsets()[2] == 3,
           "signed 8-bit CSC indices and 16-bit offsets survive access and column gather");

    auto index_overflow = anndata::Array::csr<float, std::uint8_t, std::uint16_t>(1, 256, 1);
    {
        auto sparse = index_overflow.edit().csr<float, std::uint8_t, std::uint16_t>();
        sparse.unsafe_offsets()[0] = 0;
        sparse.unsafe_offsets()[1] = 1;
        sparse.unsafe_indices()[0] = 255;
        sparse.values()[0] = 1;
    }
    std::vector<std::size_t> repeated_column(257, 255);
    expect_error(anndata::ErrorCode::invalid_sparse_structure,
        [&] { (void)index_overflow.view().select({anndata::Axis::all(), anndata::Axis::gather(repeated_column.data(), repeated_column.size())}).copy(); },
        "gather rejects a remapped compact index that no longer fits uint8");

    auto offset_overflow = anndata::Array::csr<float, std::uint8_t, std::uint8_t>(1, 1, 1);
    {
        auto sparse = offset_overflow.edit().csr<float, std::uint8_t, std::uint8_t>();
        sparse.unsafe_offsets()[0] = 0;
        sparse.unsafe_offsets()[1] = 1;
        sparse.unsafe_indices()[0] = 0;
        sparse.values()[0] = 1;
    }
    std::vector<std::size_t> repeated_row(257, 0);
    expect_error(anndata::ErrorCode::invalid_sparse_structure,
        [&] { (void)offset_overflow.view().select({anndata::Axis::gather(repeated_row.data(), repeated_row.size()), anndata::Axis::all()}).copy(); },
        "gather rejects a compact offset that no longer fits uint8");
}

void sparse_copy_major_preflight() {
    using anndata::Axis;
    const auto max = std::numeric_limits<std::size_t>::max();
    const auto addressable = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    auto csr = anndata::Array::csr<float, std::uint8_t, std::uint64_t>(1, 1, 0);
    auto csc = anndata::Array::csc<float, std::uint8_t, std::uint8_t>(1, 1, 0);
    expect_error(anndata::ErrorCode::invalid_argument,
        [&] { (void)csr.view().select({Axis::stride(0, max, 0), Axis::all()}).copy(); },
        "CSR copy rejects major count plus one overflow before traversal");
    expect_error(anndata::ErrorCode::invalid_argument,
        [&] { (void)csc.view().select({Axis::all(), Axis::stride(0, max, 0)}).copy(); },
        "CSC copy rejects major count plus one overflow before traversal");
    expect_error(anndata::ErrorCode::invalid_argument,
        [&] { (void)csr.view().select({Axis::stride(0, addressable / sizeof(std::uint64_t), 0), Axis::all()}).copy(); },
        "CSR copy rejects an unaddressable stored offset span");
    expect_error(anndata::ErrorCode::invalid_argument,
        [&] { (void)csc.view().select({Axis::all(), Axis::stride(0, addressable / sizeof(std::size_t), 0)}).copy(); },
        "CSC copy rejects an unaddressable staging offset span");
}

} // namespace

int main() {
    dense_storage_and_mutation();
    native_type_identity_and_bytes();
    checked_ranges_and_shared_owners();
    empty_and_broadcast_extreme_dimensions();
    validation_opt_out_and_raw_sparse_copy();
    retained_and_borrowed_memory();
    lazy_dense_selection();
    sparse_layout_and_validation();
    compact_sparse_integer_widths();
    sparse_copy_major_preflight();
    if (failures != 0) {
        std::cerr << failures << " array assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "array tests passed\n";
    return EXIT_SUCCESS;
}
