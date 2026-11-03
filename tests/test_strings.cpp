#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

import anndata;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template<class F>
void expect_error(anndata::ErrorCode code, F&& operation, const char* message) {
    try {
        operation();
        expect(false, message);
    } catch (const anndata::Error& error) {
        if (error.code() != code) {
            std::cerr << "FAIL: " << message << " (wrong error code "
                      << static_cast<int>(error.code()) << ")\n";
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong exception: "
                  << error.what() << ")\n";
        ++failures;
    } catch (...) {
        expect(false, message);
    }
}

void packed_layout_and_copy() {
    using namespace anndata;
    auto values = StringArray::from_strings(
        {2, 3}, {"alpha", "β", "", "δelta", "epsilon", "z"});
    expect(values.rank() == 2 && values.size() == 6 &&
           values.extent(0) == 2 && values.extent(1) == 3,
           "StringArray records arbitrary multidimensional shape");
    auto read = values.view();
    auto edit = values.edit();
    expect(read.storage_bytes() == edit.bytes() && read.bytes() == edit.bytes() &&
           read.storage_offsets() == edit.offsets() && read.offsets() == edit.offsets(),
           "read and write descriptors expose the same packed buffers");
    expect(read.string(0) == "alpha" && read.string(1) == "β" &&
           read.string(2).empty() && read.string(3) == "δelta" &&
           read.string(4) == "epsilon" && read.string(5) == "z",
           "packed row-major array preserves every UTF-8 and empty string");
    expect(read.storage_offsets()[0] == 0 &&
           read.storage_offsets()[read.size()] == read.byte_size(),
           "offsets span the packed byte buffer");

    auto independent = values.copy();
    independent.edit().bytes()[0] = 'A';
    expect(independent.view().string(0) == "Alpha" &&
           values.view().string(0) == "alpha",
           "explicit copy owns independent packed buffers");
}

void lazy_selections() {
    using namespace anndata;
    auto values = StringArray::from_strings(
        {2, 3}, {"a", "b", "c", "d", "e", "f"});
    auto original = values.view();
    auto selected = original.select({
        std::vector<std::size_t>{1, 0, 1},
        std::vector<std::size_t>{2, 0}});
    expect(selected.shape() == std::vector<std::size_t>{3, 2} &&
           selected.string(0) == "f" && selected.string(1) == "d" &&
           selected.string(2) == "c" && selected.string(3) == "a" &&
           selected.string(4) == "f" && selected.string(5) == "d",
           "lazy multidimensional selection preserves order and duplicates");
    expect(selected.storage_bytes() == original.storage_bytes() &&
           selected.storage_offsets() == original.storage_offsets(),
           "selected views retain and expose original backing storage");
    expect_error(ErrorCode::non_contiguous,
        [&] { (void)selected.bytes(); },
        "selected view requires explicit materialization for compact byte access");
    expect_error(ErrorCode::non_contiguous,
        [&] { (void)selected.offsets(); },
        "selected view requires explicit materialization for compact offset access");

    auto materialized = selected.copy();
    expect(materialized.view().shape() == std::vector<std::size_t>{3, 2} &&
           materialized.view().string(0) == "f" &&
           materialized.view().string(5) == "d" &&
           materialized.view().storage_bytes() != original.storage_bytes(),
           "copy materializes selected strings into independent packed storage");

    auto empty = original.select({std::vector<std::size_t>{}, std::nullopt});
    expect(empty.shape() == std::vector<std::size_t>{0, 3} && empty.size() == 0 &&
           empty.copy().view().shape() == std::vector<std::size_t>{0, 3},
           "empty selections retain their dimensionality");
    expect_error(ErrorCode::out_of_bounds,
        [&] { (void)original.string(6); }, "flat string access checks bounds");
    expect_error(ErrorCode::out_of_bounds,
        [&] { (void)original.select({std::vector<std::size_t>{2}, std::nullopt}); },
        "selection checks axis bounds");
}

void external_buffers() {
    using namespace anndata;
    std::array<char, 4> local_bytes{'o', 'k', 'x', 'y'};
    std::array<std::uint64_t, 3> local_offsets{0, 2, 4};
    StringArrayView retained_view;
    {
        auto owner = std::make_shared<std::array<char, 4>>(local_bytes);
        auto offset_owner = std::make_shared<std::array<std::uint64_t, 3>>(local_offsets);
        struct Holder {
            std::shared_ptr<std::array<char, 4>> bytes;
            std::shared_ptr<std::array<std::uint64_t, 3>> offsets;
        };
        auto lifetime = std::make_shared<Holder>(Holder{owner, offset_owner});
        std::shared_ptr<const void> token = lifetime;
        auto data = StringArray::share_packed(
            owner->data(), owner->size(), offset_owner->data(), {2}, token);
        retained_view = data.view();
        owner.reset();
        offset_owner.reset();
        lifetime.reset();
        token.reset();
        expect(retained_view.string(0) == "ok" && retained_view.string(1) == "xy",
               "retained view keeps external byte and offset buffers alive");
    }
    expect(retained_view.string(0) == "ok",
           "retained view remains valid after the StringArray owner is destroyed");

    const char* read_only_bytes = local_bytes.data();
    const std::uint64_t* read_only_offsets = local_offsets.data();
    auto borrowed = StringArray::borrow_packed(
        read_only_bytes, local_bytes.size(), read_only_offsets, {2});
    expect(borrowed.view().string(1) == "xy",
           "borrowed buffers are directly readable while caller storage lives");
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)borrowed.edit(); },
        "borrowed const storage refuses mutable descriptor acquisition");
    auto mutable_borrow = StringArray::borrow_packed(
        local_bytes.data(), local_bytes.size(), local_offsets.data(), {2});
    expect(mutable_borrow.edit().bytes()[0] == 'o',
           "mutable borrowed buffers retain caller-controlled write access");

    // adopt_packed calls the deleter exactly once for the pair of buffers.
    struct AdoptCounter { int calls{}; } counter;
    auto* bytes = new char[2]{'h', 'i'};
    auto* offsets = new std::uint64_t[2]{0, 2};
    auto adopted = StringArray::adopt_packed(
        bytes, 2, offsets, {1},
        [](char* b, std::uint64_t* o, void* context) {
            ++static_cast<AdoptCounter*>(context)->calls;
            delete[] b;
            delete[] o;
        }, &counter);
    expect(adopted.view().string(0) == "hi" && adopted.edit().bytes()[0] == 'h',
           "adopted packed storage remains writable");
    adopted = StringArray{};
    expect(counter.calls == 1, "custom packed-buffer deleter runs once");
}

void validation_and_unsafe_editing() {
    using namespace anndata;
    auto invalid_utf8 = StringArray::from_packed(
        {1}, {static_cast<char>(0xc0), static_cast<char>(0xaf)}, {0, 2},
        Validation::none);
    expect_error(ErrorCode::invalid_argument,
        [&] { invalid_utf8.validate(); },
        "explicit validation finds malformed UTF-8 after an unchecked import");

    auto invalid_offsets = StringArray::from_packed(
        {1}, {'a', 'b'}, {0, 1}, Validation::none);
    expect_error(ErrorCode::invalid_argument,
        [&] { invalid_offsets.validate(); },
        "explicit validation checks packed offset coverage");

    auto unsafe = StringArray::from_strings({2}, {"ab", "cd"});
    unsafe.edit().unsafe_offsets()[1] = 1;
    unsafe.edit().unsafe_offsets()[2] = 4;
    unsafe.validate();
    expect(unsafe.view().string(0) == "a" && unsafe.view().string(1) == "bcd",
           "direct offset edits change string boundaries without a proxy layer");
}

void alignment_and_metadata() {
    using namespace anndata;
    auto data = AnnData::empty(3, 2);
    data.set_obsm("labels", StringArray::from_strings(
        {3, 2}, {"c0a", "c0b", "c1a", "c1b", "c2a", "c2b"}));
    data.set_varm("gene_labels", StringArray::from_strings({2}, {"g0", "g1"}));

    auto metadata = Element::mapping();
    metadata.set("names", Element(StringArray::from_strings({2}, {"μ", "β"})));
    metadata.set("signed", Element::scalar<std::int16_t>(-12));
    metadata.at("signed");
    auto signed_scalar = metadata.at("signed");
    expect(signed_scalar.scalar<std::int16_t>() == -12,
           "string arrays coexist with typed scalar nested metadata");
    data.set_uns("custom", std::move(metadata));

    // AnnData raw may be present with an independent var axis and no X.
    data.set_raw(Table::with_index({"r0", "r1"}, "raw_gene"));
    data.set_raw_varm("labels", StringArray::from_strings({2}, {"raw0", "raw1"}));
    expect(data.has_raw() && !data.has_raw_x(),
           "raw metadata can be present when raw X is absent");

    auto view = data.slice(indices{2, 0}, all);
    auto cell_strings = view.obsm("labels").string_array();
    expect(cell_strings.shape() == std::vector<std::size_t>{2, 2} &&
           cell_strings.string(0) == "c2a" && cell_strings.string(1) == "c2b" &&
           cell_strings.string(2) == "c0a",
           "obsm strings follow observation slicing order");
    expect(view.varm("gene_labels").string_array().string(1) == "g1",
           "varm strings remain aligned to variables");
    expect(view.has_raw() && !view.has_raw_x() && view.raw_var().size() == 2 &&
           view.raw_varm("labels").string_array().string(1) == "raw1",
           "raw variable axis and annotations survive observation slicing");

    auto copy = view.copy();
    expect(copy.obsm("labels").string_array().string(0) == "c2a" &&
           copy.has_raw() && !copy.has_raw_x() &&
           copy.raw_varm("labels").string_array().string(0) == "raw0",
           "explicit AnnData copy preserves selected strings and raw without X");
    copy.edit_obsm_string_array("labels").bytes()[0] = 'C';
    expect(copy.obsm("labels").string_array().string(0) == "C2a" &&
           data.obsm("labels").string_array().string(4) == "c2a",
           "copied AnnData string buffers are independent");

    auto nullable = Column::nullable_strings(
        {"", "x"}, {0, 1}, StringMissingValue::NaN);
    expect(nullable.view().string_missing_value() == StringMissingValue::NaN &&
           !nullable.view().valid(0),
           "nullable string columns retain the NA versus NaN sentinel");
    expect(nullable.copy().view().string_missing_value() == StringMissingValue::NaN,
           "nullable string copies preserve the sentinel");

    expect_error(ErrorCode::shape_mismatch,
        [&] { data.set_obsm("bad", StringArray::from_strings({2}, {"a", "b"})); },
        "obsm string arrays enforce first-axis alignment");
}

void serialization_byte_order_hints() {
    using namespace anndata;
    auto dense = Array::empty<float>({2, 2});
    dense.set_serialization_byte_order(ByteOrder::big);
    dense.edit().dense<float>().data()[0] = 2.5F;
    auto dense_copy = dense.view().select({std::vector<std::size_t>{1, 0}, std::nullopt}).copy();
    expect(dense_copy.serialization_byte_order() == ByteOrder::big,
           "dense selection copies preserve source file byte order");

    auto sparse = Array::csr<float, std::int32_t, std::int64_t>(3, 3, 3);
    auto access = sparse.edit().csr<float, std::int32_t, std::int64_t>();
    const float values[]{1, 2, 3};
    const std::int32_t indices[]{0, 2, 1};
    const std::int64_t offsets[]{0, 2, 2, 3};
    for (std::size_t i = 0; i < 3; ++i) {
        access.values()[i] = values[i];
        access.unsafe_indices()[i] = indices[i];
    }
    for (std::size_t i = 0; i < 4; ++i) access.unsafe_offsets()[i] = offsets[i];
    sparse.set_serialization_byte_order(ByteOrder::big);
    sparse.set_index_serialization_byte_order(ByteOrder::little);
    sparse.set_offset_serialization_byte_order(ByteOrder::big);
    auto same = sparse.copy();
    expect(same.serialization_byte_order() == ByteOrder::big &&
           same.index_serialization_byte_order() == ByteOrder::little &&
           same.offset_serialization_byte_order() == ByteOrder::big,
           "sparse copies preserve each buffer byte order independently");
    auto selected = sparse.view().select({
        std::vector<std::size_t>{2, 0}, std::vector<std::size_t>{1, 2}}).copy();
    expect(selected.layout() == Layout::csr && selected.extent(0) == 2 &&
           selected.extent(1) == 2 &&
           selected.serialization_byte_order() == ByteOrder::big &&
           selected.index_serialization_byte_order() == ByteOrder::little &&
           selected.offset_serialization_byte_order() == ByteOrder::big,
           "selected sparse copies keep layout and all file byte order hints");
    auto scalar = Element::scalar<std::uint16_t>(513);
    scalar.set_serialization_byte_order(ByteOrder::big);
    expect(scalar.copy().view().serialization_byte_order() == ByteOrder::big,
           "typed scalar element copies preserve source file byte order");
}

void erased_sparse_buffers() {
    using namespace anndata;
    auto values = Array::empty<float>({3});
    auto indices = Array::empty<std::int32_t>({3});
    auto offsets = Array::empty<std::int64_t>({4});
    const float input_values[]{1, 2, 3};
    const std::int32_t input_indices[]{0, 2, 1};
    const std::int64_t input_offsets[]{0, 2, 2, 3};
    auto* values_pointer = values.edit().dense<float>().data();
    auto* indices_pointer = indices.edit().dense<std::int32_t>().data();
    auto* offsets_pointer = offsets.edit().dense<std::int64_t>().data();
    for (std::size_t i = 0; i < 3; ++i) {
        values_pointer[i] = input_values[i];
        indices_pointer[i] = input_indices[i];
    }
    for (std::size_t i = 0; i < 4; ++i) offsets_pointer[i] = input_offsets[i];
    values.set_serialization_byte_order(ByteOrder::big);
    indices.set_serialization_byte_order(ByteOrder::little);
    offsets.set_serialization_byte_order(ByteOrder::big);

    auto sparse = Array::from_sparse_buffers(Layout::csr, std::move(values),
        std::move(indices), std::move(offsets), 3, 3);
    auto buffers = sparse.edit().sparse_bytes();
    expect(buffers.values() == reinterpret_cast<std::byte*>(values_pointer) &&
           buffers.indices() == reinterpret_cast<const std::byte*>(indices_pointer) &&
           buffers.offsets() == reinterpret_cast<const std::byte*>(offsets_pointer),
           "erased sparse composition retains component pointer identity");
    expect(sparse.serialization_byte_order() == ByteOrder::big &&
           sparse.index_serialization_byte_order() == ByteOrder::little &&
           sparse.offset_serialization_byte_order() == ByteOrder::big,
           "erased sparse composition retains per-component byte order");

    auto copied = sparse.copy();
    auto typed = copied.view().csr<float, std::int32_t, std::int64_t>();
    expect(typed.values()[2] == 3 && typed.indices()[1] == 2 &&
           typed.offsets()[3] == 3,
           "erased sparse components support direct typed access after copy");
    expect(copied.serialization_byte_order() == ByteOrder::big &&
           copied.index_serialization_byte_order() == ByteOrder::little &&
           copied.offset_serialization_byte_order() == ByteOrder::big,
           "erased sparse copies preserve per-component byte order");
}

void categorical_string_array_colors() {
    using namespace anndata;
    auto make_data = [] {
        auto data = AnnData::empty(4, 1);
        auto codes = Array::empty<std::int8_t>({4});
        auto code_values = codes.edit().dense<std::int8_t>();
        code_values.data()[0] = 2;
        code_values.data()[1] = 0;
        code_values.data()[2] = -1;
        code_values.data()[3] = 2;
        data.set_obs_column("group", Column::categorical(
            std::move(codes), Column::strings({"low", "mid", "high"})));
        return data;
    };

    auto one_dimensional = make_data();
    one_dimensional.set_uns("group_colors", Element(StringArray::from_strings(
        {3}, {"blue", "green", "red"})));
    auto one_slice = one_dimensional.slice(indices{0, 1, 2});
    auto one_colors = one_slice.uns().at("group_colors").string_array();
    expect(one_colors.shape() == std::vector<std::size_t>{2} &&
           one_colors.string(0) == "blue" && one_colors.string(1) == "red",
           "one-dimensional string color palettes follow pruned category order");

    auto palette = make_data();
    palette.set_uns("group_colors", Element(StringArray::from_strings(
        {3, 2}, {"low-x", "low-y", "mid-x", "mid-y", "high-x", "high-y"})));
    auto palette_slice = palette.slice(indices{0, 1, 2});
    auto palette_colors = palette_slice.uns().at("group_colors").string_array();
    expect(palette_colors.shape() == std::vector<std::size_t>{2, 2} &&
           palette_colors.string(0) == "low-x" &&
           palette_colors.string(1) == "low-y" &&
           palette_colors.string(2) == "high-x" &&
           palette_colors.string(3) == "high-y",
           "multidimensional string palettes slice their category axis lazily");

    auto scalar = make_data();
    scalar.set_uns("group_colors", Element(StringArray::from_strings(
        std::vector<std::size_t>{}, std::vector<std::string>{"single"})));
    auto scalar_slice = scalar.slice(indices{2, 0});
    auto promoted = scalar_slice.uns().at("group_colors").string_array();
    expect(promoted.shape() == std::vector<std::size_t>{1} &&
           promoted.string(0) == "single",
           "rank-zero string color follows upstream scalar promotion behavior");
}

} // namespace

int main() {
    packed_layout_and_copy();
    lazy_selections();
    external_buffers();
    validation_and_unsafe_editing();
    alignment_and_metadata();
    serialization_byte_order_hints();
    erased_sparse_buffers();
    categorical_string_array_colors();
    return failures == 0 ? 0 : 1;
}
