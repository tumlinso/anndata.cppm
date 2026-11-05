#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import anndatapp;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template<class F>
void expect_error(anndatapp::ErrorCode code, F&& operation, const char* message) {
    try {
        operation();
        expect(false, message);
    } catch (const anndatapp::Error& error) {
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
    using namespace anndatapp;
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
    using namespace anndatapp;
    auto values = StringArray::from_strings(
        {2, 3}, {"a", "b", "c", "d", "e", "f"});
    auto original = values.view();
    auto selected = original.select({
        Axis::gather({1, 0, 1}),
        Axis::gather({2, 0})});
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

    auto empty = original.select({Axis::gather({}), Axis::all()});
    expect(empty.shape() == std::vector<std::size_t>{0, 3} && empty.size() == 0 &&
           empty.copy().view().shape() == std::vector<std::size_t>{0, 3},
           "empty selections retain their dimensionality");
    expect_error(ErrorCode::out_of_bounds,
        [&] { (void)original.string(6); }, "flat string access checks bounds");
    expect_error(ErrorCode::out_of_bounds,
        [&] { (void)original.select({Axis::gather({2}), Axis::all()}); },
        "selection checks axis bounds");
}

void compact_string_axes() {
    using namespace anndatapp;
    StringArrayView retained;
    {
        auto a = StringArray::from_strings({2,3}, {"a","b","c","d","e","f"});
        const Axis axes[]{Axis::stride(1,2,1,true), Axis::stride(2,2,2,true)};
        auto v = a.view().select(axes, 2);
        expect(v.string(0) == "f" && v.string(1) == "d" &&
               v.string(2) == "c" && v.string(3) == "a",
               "string axes retain reverse affine mappings on every dimension");
        std::size_t rows[]{1,0,1};
        retained = v.select({Axis::gather(rows, 3), Axis::gather({1,0})});
        rows[0] = 0;
        a.edit().bytes()[0] = 'A';
        auto identity = a.view().select({Axis::gather({0,1}), Axis::stride(0,3)});
        expect(identity.bytes() == a.edit().bytes() && identity.offsets() == a.edit().offsets(),
               "identity string selection retains logical packed access");
        auto restored = a.view().select({Axis::stride(1,2,1,true), Axis::all()})
                                .select({Axis::stride(1,2,1,true), Axis::all()});
        expect(restored.bytes() == a.edit().bytes(),
               "nested inverse string ranges restore packed identity access");
        expect_error(ErrorCode::non_contiguous, [&] { (void)retained.bytes(); },
                     "nonidentity string mappings still require explicit compact copies");
    }
    expect(retained.shape() == std::vector<std::size_t>({3,2}) &&
           retained.string(0) == "A" && retained.string(1) == "c" &&
           retained.string(2) == "d" && retained.string(3) == "f" &&
           retained.string(4) == "A",
           "nested string gathers copy caller maps and retain bytes after owner destruction");
}

void external_buffers() {
    using namespace anndatapp;
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
            static_cast<const char*>(owner->data()), owner->size(),
            static_cast<const std::uint64_t*>(offset_owner->data()), {2}, token);
        retained_view = data.view();
        owner.reset();
        offset_owner.reset();
        lifetime.reset();
        token.reset();
        expect(retained_view.string(0) == "ok" && retained_view.string(1) == "xy",
               "retained view keeps external byte and offset buffers alive");
        expect_error(ErrorCode::invalid_argument, [&] { (void)data.edit(); },
                     "const shared packed buffers reject mutable acquisition");
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

void packed_storage_lifetimes() {
    using namespace anndatapp;
    StringArrayView retained;
    StringArrayEdit edit;
    StringArray independent;
    {
        std::vector<char> b{'a', 'b', char(0xce), char(0xb2)};
        std::vector<std::uint64_t> o{0, 2, 2, 4};
        const auto* bp = b.data();
        const auto* op = o.data();
        auto a = StringArray::from_packed({3}, std::move(b), std::move(o));
        retained = a.view(); edit = a.edit();
        expect(retained.bytes() == bp && retained.offsets() == op &&
               edit.bytes() == bp && edit.offsets() == op,
               "packed construction moves both buffers without changing their addresses");
        auto moved = std::move(a);
        independent = moved.copy();
        expect(independent.view().bytes() != bp && independent.view().offsets() != op,
               "packed deep copy independently owns both buffers");
    }
    edit.bytes()[0] = 'A';
    expect(retained.string(0) == "Ab" && retained.string(1).empty() &&
           retained.string(2) == "β" && independent.view().string(0) == "ab" &&
           independent.view().string(2) == "β",
           "packed views retain moved storage and independent copies after owner destruction");

    std::vector<char> b;
    b.reserve(8); // Empty owned bytes must remain a null public pointer.
    std::vector<std::uint64_t> o{0, 0, 0};
    const auto* op = o.data();
    auto empty = StringArray::from_packed({2}, std::move(b), std::move(o));
    expect(empty.view().bytes() == nullptr && empty.edit().bytes() == nullptr &&
           empty.view().offsets() == op && empty.view().string(1).empty() &&
           empty.view().select({Axis::gather({1, 0, 1})}).copy().view().byte_size() == 0,
           "empty owned UTF-8 retains offsets and exposes null bytes through selection and copy");
    auto scalar = StringArray::from_packed({}, {}, {0, 0});
    expect(scalar.rank() == 0 && scalar.view().string(0).empty(),
           "rank-zero empty UTF-8 remains readable without byte-pointer arithmetic");
    auto zero = StringArray::from_packed({2, 0}, {}, {0});
    expect_error(ErrorCode::out_of_bounds, [&] { (void)zero.view().string(0); },
                 "zero extents reject string access before mixed-radix division");

    const std::uint64_t ro[]{0, 0, 0};
    auto borrowed = StringArray::borrow_packed(static_cast<const char*>(nullptr), 0, ro, {2});
    expect(borrowed.view().string(0).empty(), "const null byte buffers represent empty borrowed UTF-8");
    expect_error(ErrorCode::invalid_argument, [&] { (void)borrowed.edit(); },
                 "const empty borrows retain read-only storage semantics");
    std::uint64_t rw[]{0, 0};
    auto writable = StringArray::borrow_packed(static_cast<char*>(nullptr), 0, rw, {1});
    expect(writable.edit().bytes() == nullptr && writable.edit().unsafe_offsets() == rw,
           "mutable empty borrows expose the caller's offset buffer");
}

void mutable_shared_storage() {
    using namespace anndatapp;
    struct Buffers { char b[2]{'o', 'k'}; std::uint64_t o[2]{0, 2}; };
    StringArrayView view;
    StringArrayEdit edit;
    std::weak_ptr<Buffers> lifetime;
    {
        auto p = std::make_shared<Buffers>();
        lifetime = p;
        auto a = StringArray::share_packed(p->b, 2, p->o, {1}, p);
        view = a.view(); edit = a.edit();
        expect(view.bytes() == p->b && view.offsets() == p->o,
               "mutable shared storage preserves both external pointers");
    }
    edit.bytes()[0] = 'O';
    expect(!lifetime.expired() && view.string(0) == "Ok",
           "retained mutable shared descriptors keep the external owner alive");
    edit = StringArrayEdit{}; view = StringArrayView{};
    expect(lifetime.expired(), "external shared storage is released with its last descriptor");
}

void adopted_storage_lifetimes() {
    using namespace anndatapp;
    int calls = 0;
    const auto release = [](char* b, std::uint64_t* o, void* context) {
        ++*static_cast<int*>(context); delete[] b; delete[] o;
    };
    StringArrayView retained;
    StringArrayEdit edit;
    {
        auto* b = new char[2]{'a', '\0'};
        auto* o = new std::uint64_t[2]{0, 2};
        auto a = StringArray::adopt_packed(b, 2, o, {1}, release, &calls);
        retained = a.view(); edit = a.edit();
        auto moved = std::move(a);
        expect(retained.bytes() == b && retained.offsets() == o &&
               retained.string(0) == std::string_view("a\0", 2),
               "adopted UTF-8 retains buffer identity and embedded null bytes");
    }
    expect(calls == 0, "retained read and edit views delay adopted-buffer deletion");
    edit.bytes()[0] = 'A';
    expect(retained.string(0) == std::string_view("A\0", 2),
           "retained adopted edits stay visible after owner destruction");
    retained = StringArrayView{};
    expect(calls == 0, "an edit descriptor alone retains adopted buffers");
    edit = StringArrayEdit{};
    expect(calls == 1, "the last retained descriptor deletes adopted buffers once");

    expect_error(ErrorCode::invalid_argument, [&] {
        (void)StringArray::adopt_packed(new char[2]{char(0xc0), char(0xaf)}, 2,
            new std::uint64_t[2]{0, 2}, {1}, release, &calls);
    }, "failed UTF-8 validation releases adopted buffers");
    expect(calls == 2, "adopted validation failure invokes the deleter once");
    expect_error(ErrorCode::invalid_argument, [&] {
        (void)StringArray::adopt_packed(new char[1]{'a'}, 1, nullptr,
                                      {1}, release, &calls);
    }, "failed packed-buffer acquisition releases adopted buffers");
    expect(calls == 3, "adopted acquisition failure invokes the deleter once");
}

void validation_and_unsafe_editing() {
    using namespace anndatapp;
    for (const std::size_t n : {0, 7, 8, 9, 15, 16}) {
        const std::string prefix(n, 'a');
        const std::vector<std::string> valid{
            prefix, prefix + "\xce\xb2", prefix + "\xe2\x82\xac",
            prefix + "\xf4\x8f\xbf\xbf", prefix + std::string("\0x", 2)};
        auto packed = StringArray::from_strings({valid.size()}, valid);
        packed.validate();
        for (std::size_t i = 0; i < valid.size(); ++i)
            expect(packed.view().string(i) == valid[i], "ASCII chunks preserve valid UTF-8 and embedded nulls");
        for (const auto suffix : {"\xc2", "\xe2\x82", "\xf0\x90\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80"})
            expect_error(ErrorCode::invalid_argument,
                [&] { (void)StringArray::from_strings({1}, {prefix + suffix}); },
                "UTF-8 validation rejects truncated or invalid sequences at chunk boundaries");
    }
    expect_error(ErrorCode::invalid_argument, [] {
        (void)StringArray::from_packed({2}, {'a','a','a','a','a','a','a',char(0xc2),char(0xa2)}, {0,8,9});
    }, "UTF-8 sequences cannot cross packed string boundaries");
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
    using namespace anndatapp;
    auto data = AnnData::empty(3, 2);
    data.set_obsm("labels", StringArray::from_strings(
        {3, 2}, {"c0a", "c0b", "c1a", "c1b", "c2a", "c2b"}));
    data.set_varm("gene_labels", StringArray::from_strings({2}, {"g0", "g1"}));

    auto metadata = Element::mapping();
    metadata.set("names", Element(StringArray::from_strings({2}, {"μ", "β"})));
    metadata.set("signed", Element::scalar<std::int16_t>(-12));
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
    using namespace anndatapp;
    auto dense = Array::empty<float>({2, 2});
    dense.set_serialization_byte_order(ByteOrder::big);
    dense.edit().dense<float>().data()[0] = 2.5F;
    auto dense_copy = dense.view().select({Axis::gather({1, 0}), Axis::all()}).copy();
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
        Axis::gather({2, 0}), Axis::gather({1, 2})}).copy();
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
    using namespace anndatapp;
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
    using namespace anndatapp;
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
    compact_string_axes();
    external_buffers();
    packed_storage_lifetimes();
    mutable_shared_storage();
    adopted_storage_lifetimes();
    validation_and_unsafe_editing();
    alignment_and_metadata();
    serialization_byte_order_hints();
    erased_sparse_buffers();
    categorical_string_array_colors();
    return failures == 0 ? 0 : 1;
}
