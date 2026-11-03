#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

import anndata;

namespace {

using namespace anndata;
int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

template<class F>
void expect_error(ErrorCode code, F&& operation, const char* message) {
    try { operation(); expect(false, message); }
    catch (const Error& error) {
        if (error.code() != code) {
            std::cerr << "FAIL: " << message << " (wrong error code "
                      << static_cast<int>(error.code()) << ")\n";
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong exception: " << error.what() << ")\n";
        ++failures;
    } catch (...) { expect(false, message); }
}

struct Row { std::int32_t id; float value; char label[8]; };
struct RefRow { std::int32_t id; Utf8Ref text; };

RecordLayout row_layout() {
    return {sizeof(Row), alignof(Row), {
        {"id", RecordFieldKind::numeric, DType::int32, offsetof(Row, id), sizeof(std::int32_t), ByteOrder::native},
        {"value", RecordFieldKind::numeric, DType::float32, offsetof(Row, value), sizeof(float), ByteOrder::native},
        {"label", RecordFieldKind::fixed_utf8, DType::uint8, offsetof(Row, label), 8, ByteOrder::native}
    }};
}

RecordLayout ref_layout() {
    return {sizeof(RefRow), alignof(RefRow), {
        {"id", RecordFieldKind::numeric, DType::int32, offsetof(RefRow, id), sizeof(std::int32_t), ByteOrder::native},
        {"text", RecordFieldKind::utf8_ref, DType::uint8, offsetof(RefRow, text), sizeof(Utf8Ref), ByteOrder::native}
    }};
}

void fieldwise_shape_order_and_copies() {
    static_assert(!std::is_copy_constructible_v<RecordArray>);
    static_assert(std::is_move_constructible_v<RecordArray>);

    auto records = RecordArray::with_shape({2, 2});
    auto counts = Array::empty<std::int32_t>({2, 2});
    auto count_edit = counts.edit().dense<std::int32_t>();
    auto* count_ptr = count_edit.data();
    for (std::size_t i = 0; i < 4; ++i) count_ptr[i] = std::int32_t(i + 1);
    records.set("count", std::move(counts));
    auto labels = StringArray::from_strings({2, 2}, {"a", "β", "", "delta"});
    const char* label_ptr = labels.view().storage_bytes();
    records.set("label", std::move(labels));

    auto view = records.view();
    expect(!view.interleaved() && view.rank() == 2 && view.size() == 4 &&
           view.shape() == std::vector<std::size_t>{2, 2} &&
           view.extent(0) == 2 && view.extent(1) == 2,
           "fieldwise records retain multidimensional shape and size");
    expect(view.n_fields() == 2 &&
           view.field_names() == std::vector<std::string>{"count", "label"},
           "record fields retain insertion order");
    expect(view.field("count").kind == RecordFieldKind::numeric &&
           view.field("count").dtype == DType::int32 &&
           view.field("label").kind == RecordFieldKind::string,
           "field descriptors distinguish numeric and string fields");
    auto count_view = view.array("count").dense<std::int32_t>();
    auto label_view = view.string_array("label");
    expect(count_view.data() == count_ptr && count_view.data()[3] == 4,
           "set retains the exact numeric allocation");
    expect(label_view.shape() == std::vector<std::size_t>{2, 2} &&
           label_view.storage_bytes() == label_ptr && view.string("label", 0) == "a" &&
           view.string("label", 1) == "β" &&
           view.string("label", 2).empty() && view.string("label", 3) == "delta",
           "fieldwise string access preserves UTF-8 and empty values");

    auto counts_copy = view.copy_array("count");
    auto copied_counts = counts_copy.view().dense<std::int32_t>();
    expect(copied_counts.data() != count_ptr && copied_counts.data()[0] == 1 &&
           copied_counts.data()[3] == 4,
           "copy_array owns an independent numeric field");
    auto labels_copy = view.copy_strings("label");
    auto label_bytes = labels_copy.edit().bytes();
    label_bytes[0] = 'A';
    expect(labels_copy.view().string(0) == "A" && view.string("label", 0) == "a",
           "copy_strings owns independent packed text");

    auto replacement = Array::empty<std::int32_t>({2, 2});
    auto replacement_values = replacement.edit().dense<std::int32_t>();
    auto* replacement_ptr = replacement_values.data();
    replacement_ptr[0] = 20; replacement_ptr[1] = 21;
    replacement_ptr[2] = 22; replacement_ptr[3] = 23;
    records.set("count", std::move(replacement));
    auto replaced = records.view();
    expect(replaced.field_names() == std::vector<std::string>{"count", "label"} &&
           replaced.array("count").dense<std::int32_t>().data() == replacement_ptr &&
           replaced.array("count").dense<std::int32_t>().data()[2] == 22,
           "replacing a field retains its new buffer and field position");

    auto clone = records.copy();
    auto clone_values = clone.view().array("count").dense<std::int32_t>();
    expect(clone.view().field_names() == replaced.field_names() &&
           clone_values.data() != replacement_ptr && clone_values.data()[3] == 23,
           "record copy preserves order and deep-copies numeric data");
    expect(clone.view().string_array("label").storage_bytes() != label_ptr,
           "record copy deep-copies packed text storage");
    clone.edit_array("count").dense<std::int32_t>().data()[0] = -7;
    clone.edit_strings("label").bytes()[0] = 'Z';
    expect(records.view().array("count").dense<std::int32_t>().data()[0] == 20 &&
           records.view().string("label", 0) == "a" && clone.view().string("label", 0) == "Z",
           "editing a record copy leaves numeric and string sources unchanged");

    auto replacement_labels = StringArray::from_strings({2, 2}, {"new", "x", "y", "z"});
    const char* replacement_label_ptr = replacement_labels.view().storage_bytes();
    records.set("label", std::move(replacement_labels));
    expect(records.view().field_names() == std::vector<std::string>{"count", "label"} &&
           records.view().string_array("label").storage_bytes() == replacement_label_ptr &&
           records.view().string("label", 0) == "new",
           "replacing a string field retains its new buffer and field position");
    expect(count_view.data()[0] == 1 && label_view.string(0) == "a",
           "previously acquired field views retain their old values after replacement");

    auto scalar = RecordArray::with_shape({});
    auto scalar_value = Array::empty<std::uint16_t>({});
    scalar_value.edit().dense<std::uint16_t>().data()[0] = 17;
    scalar.set("value", std::move(scalar_value));
    expect(scalar.view().rank() == 0 && scalar.view().size() == 1 &&
           scalar.view().array("value").dense<std::uint16_t>().data()[0] == 17,
           "rank-zero record shape contains one scalar record");

    auto empty = RecordArray::with_shape({2, 0, 3});
    empty.set("numeric", Array::empty<double>({2, 0, 3}));
    empty.set("text", StringArray::from_strings(
        std::vector<std::size_t>{2, 0, 3}, std::vector<std::string>{}));
    expect(empty.view().rank() == 3 && empty.view().size() == 0 &&
           empty.view().shape() == std::vector<std::size_t>{2, 0, 3},
           "zero-sized dimensions retain all axes");

    expect_error(ErrorCode::shape_mismatch,
        [&] { records.set("bad", Array::empty<float>({4})); },
        "numeric field insertion rejects a mismatched record shape");
    expect_error(ErrorCode::shape_mismatch,
        [&] { records.set("bad_text", StringArray::from_strings({4}, {"a", "b", "c", "d"})); },
        "string field insertion rejects a mismatched record shape");

    if (sizeof(long) == sizeof(long long) && !std::is_same_v<long, long long>) {
        auto wide = Array::empty<long long>({1});
        auto native = RecordArray::with_shape({1});
        expect_error(ErrorCode::dtype_mismatch,
            [&] { native.set("wide", std::move(wide)); },
            "record insertion rejects a noncanonical native array type");
    }

    auto slash = RecordArray::with_shape({1});
    slash.set("obs/region", StringArray::from_strings({1}, {"north"}));
    expect(slash.view().field_names() == std::vector<std::string>{"obs/region"} &&
           slash.view().string("obs/region", 0) == "north",
           "field names preserve slash-separated paths literally");

    Element element(std::move(records));
    expect(element.view().kind() == ElementKind::record_array &&
           element.view().record_array().shape() == std::vector<std::size_t>{2, 2},
           "Element stores and exposes record arrays");
}

void interleaved_layout_and_strides() {
    std::array<Row, 4> rows{};
    const char* labels[]{"abcdefgh", "ijklmnop", "QRSTUVWX", "12345678"};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        rows[i].id = std::int32_t(10 + i);
        rows[i].value = float(i) + 0.5F;
        std::memcpy(rows[i].label, labels[i], 8);
    }
    const std::vector<std::ptrdiff_t> strides{
        std::ptrdiff_t(2 * sizeof(Row)), std::ptrdiff_t(sizeof(Row))};
    auto records = RecordArray::borrow_interleaved(rows.data(), {2, 2}, row_layout(), strides);
    auto view = records.view();
    auto ids = view.member<std::int32_t>("id");
    auto values = view.member<float>("value");
    expect(view.interleaved() && view.shape() == std::vector<std::size_t>{2, 2} &&
           view.n_fields() == 3 && ids.rank() == 2 && ids.extent(0) == 2 &&
           ids.extent(1) == 2 && ids.stride_bytes(0) == std::ptrdiff_t(2 * sizeof(Row)) &&
           ids.stride_bytes(1) == std::ptrdiff_t(sizeof(Row)),
           "interleaved views retain explicit multidimensional byte strides");
    expect(ids.data() == &rows[0].id && ids.flat_ptr(0) == &rows[0].id &&
           ids.flat_ptr(3) == &rows[3].id && ids.flat_ptr(3)[0] == 13 &&
           values.flat_ptr(2)[0] == 2.5F &&
           view.string("label", 1) == "ijklmnop" &&
           view.field("label").offset == offsetof(Row, label),
           "typed and text access use declared offsets without treating rows as a T array");
    auto raw = view.record_bytes();
    expect(raw.data() == reinterpret_cast<const std::byte*>(rows.data()) &&
           raw.item_size() == sizeof(Row),
           "record byte view exposes the first record and item width");

    auto numeric_copy = view.copy_array("id");
    auto copied_ids = numeric_copy.view().dense<std::int32_t>();
    expect(numeric_copy.view().rank() == 2 && numeric_copy.view().extent(0) == 2 &&
           numeric_copy.view().extent(1) == 2 && copied_ids.data() != &rows[0].id &&
           copied_ids.data()[0] == 10 &&
           copied_ids.data()[3] == 13,
           "copy_array materializes a strided field into independent dense storage");
    auto fields = view.copy_fields();
    expect(!fields.view().interleaved() && fields.view().shape() == view.shape() &&
           fields.view().string("label", 3) == "12345678",
           "copy_fields converts all interleaved fields to fieldwise owners");
    auto copied = view.copy_interleaved(row_layout());
    expect(copied.view().interleaved() && copied.view().member<std::int32_t>("id").flat_ptr(2)[0] == 12 &&
           copied.view().record_bytes().data() != raw.data(),
           "copy_interleaved retains field values in independent record storage");
    auto same_layout_copy = view.copy();
    expect(same_layout_copy.view().interleaved() &&
           same_layout_copy.view().member<float>("value").flat_ptr(1)[0] == 1.5F,
           "copy preserves interleaved representation and values");

    auto reverse = RecordArray::borrow_interleaved(&rows[3], {4}, row_layout(),
        {-std::ptrdiff_t(sizeof(Row))});
    auto reverse_ids = reverse.view().member<std::int32_t>("id");
    expect(reverse_ids.rank() == 1 && reverse_ids.stride_bytes(0) == -std::ptrdiff_t(sizeof(Row)) &&
           reverse_ids[0] == 13 && reverse_ids[3] == 10,
           "negative byte strides traverse caller-owned live records in reverse");
    reverse.validate();

    auto allocated = RecordArray::empty_interleaved({2}, row_layout());
    auto edit = allocated.edit_member<std::int32_t>("id");
    edit[0] = 4; edit[1] = 9;
    expect(allocated.view().member<std::int32_t>("id")[1] == 9,
           "owned interleaved member edits alias its read view");
}

void interleaved_lifetimes_and_permissions() {
    RecordArrayView retained;
    RecordMemberView<const std::int32_t> retained_ids;
    {
        auto storage = std::make_shared<std::vector<Row>>(2);
        (*storage)[0].id = 5; (*storage)[1].id = 8;
        std::shared_ptr<const void> token = storage;
        auto shared = RecordArray::share_interleaved(storage->data(), {2}, row_layout(),
            {std::ptrdiff_t(sizeof(Row))}, token);
        retained = shared.view();
        retained_ids = retained.member<std::int32_t>("id");
        storage.reset(); token.reset();
    }
    expect(retained_ids[0] == 5 && retained_ids[1] == 8 && retained.shape() == std::vector<std::size_t>{2},
           "shared storage and retained views keep external records alive");

    struct DeleteState { int calls = 0; } state;
    RecordMemberView<const std::int32_t> adopted_ids;
    {
        auto* rows = new Row[1]{};
        rows[0].id = 44;
        auto adopted = RecordArray::adopt_interleaved(rows, {1}, row_layout(),
            {std::ptrdiff_t(sizeof(Row))},
            [](void* p, void* context) {
                ++static_cast<DeleteState*>(context)->calls;
                delete[] static_cast<Row*>(p);
            }, &state);
        adopted_ids = adopted.view().member<std::int32_t>("id");
        expect(state.calls == 0 && adopted_ids[0] == 44,
               "adopted storage remains owned while a member view is retained");
    }
    expect(state.calls == 0 && adopted_ids[0] == 44,
           "member views retain adopted storage after the record owner is destroyed");
    adopted_ids = {};
    expect(state.calls == 1, "adopted record deleter runs exactly once after the last view");

    std::array<Row, 1> rows{};
    rows[0].id = 12;
    auto readonly = RecordArray::borrow_interleaved(
        static_cast<const void*>(rows.data()), {1}, row_layout(), {std::ptrdiff_t(sizeof(Row))});
    expect(readonly.view().member<std::int32_t>("id")[0] == 12,
           "const external records remain readable");
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)readonly.edit_member<std::int32_t>("id"); },
        "const borrowed records reject typed editing");
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)readonly.unsafe_edit_record_bytes(); },
        "const borrowed records reject raw editing");

    auto writable = RecordArray::borrow_interleaved(
        rows.data(), {1}, row_layout(), {std::ptrdiff_t(sizeof(Row))});
    writable.edit_member<std::int32_t>("id")[0] = 21;
    expect(rows[0].id == 21, "mutable borrowing writes directly to caller storage");
}

void fixed_utf8_and_owned_utf8_refs() {
    auto short_text = RecordArray::with_shape({1});
    short_text.set("label", StringArray::from_strings({1}, {"abcdefgh"}));
    auto fixed = short_text.copy_interleaved({8, 1, {
        {"label", RecordFieldKind::fixed_utf8, DType::uint8, 0, 8, ByteOrder::native}
    }});
    expect(fixed.view().interleaved() && fixed.view().string("label", 0) == "abcdefgh",
           "fixed UTF-8 fields preserve exact-width valid text");

    std::array<char, 3> internal_nul{'a', '\0', 'b'};
    RecordLayout nul_layout{3, 1, {{"label", RecordFieldKind::fixed_utf8, DType::uint8, 0, 3}}};
    auto borrowed_nul = RecordArray::borrow_interleaved(internal_nul.data(), {1}, nul_layout);
    auto copied_nul = borrowed_nul.copy();
    auto nul_text = copied_nul.view().string("label", 0);
    expect(nul_text.size() == 3 && std::memcmp(nul_text.data(), internal_nul.data(), 3) == 0 &&
           copied_nul.view().record_bytes().data() != borrowed_nul.view().record_bytes().data(),
           "fixed text copy preserves internal NUL bytes in independent storage");
    auto trailing_nul = RecordArray::with_shape({1});
    trailing_nul.set("label", StringArray::from_strings({1}, std::vector<std::string>{std::string("a\0", 2)}));
    expect_error(ErrorCode::shape_mismatch,
        [&] { (void)trailing_nul.copy_interleaved(nul_layout); },
        "fixed text conversion rejects a logical trailing NUL that would be lost as padding");

    auto too_long = RecordArray::with_shape({1});
    too_long.set("label", StringArray::from_strings({1}, {"βx"}));
    expect_error(ErrorCode::shape_mismatch,
        [&] {
            (void)too_long.copy_interleaved({2, 1, {
                {"label", RecordFieldKind::fixed_utf8, DType::uint8, 0, 2, ByteOrder::native}
            }});
        }, "fixed UTF-8 conversion rejects truncation inside a multibyte string");

    auto fields = RecordArray::with_shape({3});
    auto ids = Array::empty<std::int32_t>({3});
    auto id_values = ids.edit().dense<std::int32_t>();
    id_values.data()[0] = 2; id_values.data()[1] = 4; id_values.data()[2] = 6;
    fields.set("id", std::move(ids));
    fields.set("text", StringArray::from_strings({3}, {"alpha", "", "βeta"}));
    const char* source_text = fields.view().string_array("text").storage_bytes();
    auto converted = fields.copy_interleaved(ref_layout());
    auto refs = converted.view().member<Utf8Ref>("text");
    expect(converted.view().string("text", 0) == "alpha" &&
           converted.view().string("text", 1).empty() && converted.view().string("text", 2) == "βeta" &&
           refs.flat_ptr(0)[0].data != source_text && refs.flat_ptr(2)[0].size == 5,
           "UTF-8 reference conversion owns copied text and preserves byte lengths");

    const char* first_text = refs.flat_ptr(0)[0].data;
    auto duplicate = converted.copy();
    auto duplicate_refs = duplicate.view().member<Utf8Ref>("text");
    expect(duplicate.view().string("text", 0) == "alpha" &&
           duplicate_refs.flat_ptr(0)[0].data != first_text &&
           duplicate_refs.flat_ptr(0)[0].data != source_text,
           "deep record copy retargets UTF-8 references into independent owned text");
    auto fields_copy = converted.copy_fields();
    expect(!fields_copy.view().interleaved() && fields_copy.view().string("text", 2) == "βeta",
           "copy_fields converts UTF-8 references to owned string arrays");

    RecordTextView retained_text;
    {
        auto owned = fields.copy_interleaved(ref_layout());
        retained_text = owned.view().text("text");
    }
    expect(retained_text.size() == 3 && retained_text.string(0) == "alpha" &&
           retained_text.string(2) == "βeta",
           "text views retain record and pointed-text owners after destruction");
}

void byte_order_copy_and_conversion() {
    auto source = RecordArray::with_shape({2});
    auto values = Array::empty<std::int32_t>({2});
    values.edit().dense<std::int32_t>().data()[0] = 7;
    values.edit().dense<std::int32_t>().data()[1] = 9;
    source.set("value", std::move(values));
    source.set_field_serialization_byte_order("value", ByteOrder::big);

    auto fields = source.copy_fields();
    expect(fields.view().field("value").byte_order == ByteOrder::big,
           "fieldwise copy preserves numeric serialization byte order");
    const RecordLayout layout{sizeof(std::int32_t), alignof(std::int32_t), {
        {"value", RecordFieldKind::numeric, DType::int32, 0,
         sizeof(std::int32_t), ByteOrder::big}
    }};
    auto interleaved = source.copy_interleaved(layout);
    expect(interleaved.view().field("value").byte_order == ByteOrder::big &&
           interleaved.view().member<std::int32_t>("value")[1] == 9,
           "interleaved conversion retains numeric byte order and values");
    auto roundtrip = interleaved.copy_fields();
    expect(roundtrip.view().field("value").byte_order == ByteOrder::big &&
           roundtrip.view().array("value").dense<std::int32_t>().data()[0] == 7,
           "copy_fields preserves numeric byte order after interleaved storage");
}

void unsafe_text_edits_and_layout_validation() {
    static char valid[] = "ok";
    auto refs = RecordArray::empty_interleaved({1}, ref_layout());
    refs.unsafe_edit_text_refs("text").flat_ptr(0)[0] = {valid, 2};
    expect(refs.view().string("text", 0) == "ok",
           "unsafe text-reference edits expose caller-supplied UTF-8 bytes");

    static char malformed[]{char(0xc0), char(0xaf)};
    auto invalid_utf8 = RecordArray::empty_interleaved({1}, ref_layout());
    invalid_utf8.unsafe_edit_text_refs("text").flat_ptr(0)[0] = {malformed, 2};
    expect_error(ErrorCode::invalid_argument, [&] { invalid_utf8.validate(); },
                 "validation rejects malformed UTF-8 behind an unsafe reference edit");

    auto null_ref = RecordArray::empty_interleaved({1}, ref_layout());
    null_ref.unsafe_edit_text_refs("text").flat_ptr(0)[0] = {nullptr, 1};
    expect_error(ErrorCode::invalid_argument, [&] { null_ref.validate(); },
                 "validation rejects a null UTF-8 pointer with nonzero length");

    auto raw_edit = RecordArray::empty_interleaved({1}, ref_layout());
    auto raw = raw_edit.unsafe_edit_record_bytes();
    const std::int32_t replacement = 73;
    std::memcpy(raw.data() + offsetof(RefRow, id), &replacement, sizeof(replacement));
    expect(raw_edit.view().member<std::int32_t>("id")[0] == replacement,
           "unsafe record-byte edits alias declared numeric members");
    const Utf8Ref bad_ref{nullptr, 1};
    std::memcpy(raw.data() + offsetof(RefRow, text), &bad_ref, sizeof(bad_ref));
    expect_error(ErrorCode::invalid_argument, [&] { raw_edit.validate(); },
                 "validation checks UTF-8 references changed through raw record bytes");

    const RecordLayout overlap{8, 4, {
        {"a", RecordFieldKind::numeric, DType::int32, 0, 4, ByteOrder::native},
        {"b", RecordFieldKind::numeric, DType::int32, 2, 4, ByteOrder::native}
    }};
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)RecordArray::empty_interleaved({1}, overlap); },
        "interleaved layout rejects overlapping fields");
    const RecordLayout misaligned{8, 4, {
        {"a", RecordFieldKind::numeric, DType::int32, 1, 4, ByteOrder::native}
    }};
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)RecordArray::empty_interleaved({1}, misaligned); },
        "interleaved layout rejects misaligned numeric members");
    const RecordLayout bad_alignment{8, 0, {}};
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)RecordArray::empty_interleaved({1}, bad_alignment); },
        "interleaved layout rejects zero record alignment");
    const RecordLayout overflowing_field{8, 4, {
        {"a", RecordFieldKind::numeric, DType::int32,
         std::numeric_limits<std::size_t>::max() - 1, 4, ByteOrder::native}
    }};
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)RecordArray::empty_interleaved({1}, overflowing_field); },
        "interleaved layout rejects field offset and width overflow");

    expect_error(ErrorCode::invalid_argument,
        [&] {
            (void)RecordArray::with_shape({std::numeric_limits<std::size_t>::max(), 2});
        }, "record shape rejects element-count overflow");
    std::array<Row, 1> row{};
    expect_error(ErrorCode::invalid_argument,
        [&] {
            (void)RecordArray::borrow_interleaved(row.data(), {2}, row_layout(),
                {std::numeric_limits<std::ptrdiff_t>::max()}, Validation::none);
        }, "external records reject overflowing signed byte strides");

    const RecordLayout bool_layout{sizeof(bool), alignof(bool), {
        {"flag", RecordFieldKind::numeric, DType::boolean, 0, sizeof(bool), ByteOrder::native}
    }};
    auto flags = RecordArray::empty_interleaved({1}, bool_layout);
    auto flag_bytes = flags.unsafe_edit_record_bytes();
    std::array<unsigned char, sizeof(bool)> invalid_bool{};
    invalid_bool[0] = 2;
    std::memcpy(flag_bytes.data(), invalid_bool.data(), invalid_bool.size());
    expect_error(ErrorCode::invalid_argument, [&] { flags.validate(); },
                 "validation rejects noncanonical boolean object bytes");
}

} // namespace

int main() {
    fieldwise_shape_order_and_copies();
    interleaved_layout_and_strides();
    interleaved_lifetimes_and_permissions();
    fixed_utf8_and_owned_utf8_refs();
    byte_order_copy_and_conversion();
    unsafe_text_edits_and_layout_validation();
    if (failures) {
        std::cerr << failures << " record assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "record tests passed\n";
    return EXIT_SUCCESS;
}
