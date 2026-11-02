#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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
void expect_error(anndata::ErrorCode code, std::string_view path,
                  F&& operation, const char* message) {
    try {
        operation();
        expect(false, message);
    } catch (const anndata::Error& e) {
        if (e.code() != code || e.path() != path) {
            std::cerr << "FAIL: " << message << " (got error code/path "
                      << static_cast<int>(e.code()) << "/" << e.path() << ")\n";
            ++failures;
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << message << " (got different exception: "
                  << e.what() << ")\n";
        ++failures;
    } catch (...) {
        expect(false, message);
    }
}

template<class T>
anndata::Array array_of(const std::vector<T>& values,
                        std::vector<std::size_t> shape) {
    auto result = anndata::Array::empty<T>(std::move(shape));
    auto view = result.edit().template dense<T>();
    for (std::size_t i = 0; i < values.size(); ++i) view.data()[i] = values[i];
    return result;
}

template<class T>
anndata::Array array_of(std::initializer_list<T> values,
                        std::vector<std::size_t> shape) {
    return array_of(std::vector<T>(values), std::move(shape));
}

template<class T>
std::vector<T> flattened(const anndata::ArrayView& view) {
    auto materialized = view.copy();
    const auto n = materialized.rank() == 0 ? std::size_t{1} : [&] {
        std::size_t count = 1;
        for (std::size_t axis = 0; axis < materialized.rank(); ++axis)
            count *= materialized.extent(axis);
        return count;
    }();
    auto values = materialized.view().template dense<T>();
    return std::vector<T>(values.data(), values.data() + n);
}

void owners_columns_and_tables() {
    static_assert(!std::is_copy_constructible_v<anndata::Column>);
    static_assert(std::is_move_constructible_v<anndata::Column>);
    static_assert(!std::is_copy_constructible_v<anndata::Table>);
    static_assert(std::is_move_constructible_v<anndata::Table>);
    static_assert(!std::is_copy_constructible_v<anndata::Element>);
    static_assert(std::is_move_constructible_v<anndata::Element>);
    static_assert(!std::is_copy_constructible_v<anndata::AnnData>);
    static_assert(std::is_move_constructible_v<anndata::AnnData>);

    auto numeric = anndata::Column::from_values<std::int32_t>(
        {12, 0, -4}, {1, 0, 1});
    const auto numeric_view = numeric.view();
    expect(numeric_view.kind() == anndata::ColumnKind::numeric &&
           numeric_view.nullable(), "numeric column records nullable storage");
    expect(numeric_view.valid(0) && !numeric_view.valid(1) &&
           numeric_view.valid(2), "numeric column validity follows its bitmap");
    auto numeric_values = numeric_view.values().dense<std::int32_t>();
    expect(numeric_values.data()[0] == 12 && numeric_values.data()[1] == 0 &&
           numeric_values.data()[2] == -4,
           "numeric column retains its native values and dtype");

    auto text = anndata::Column::strings(
        {"café", "", "β-cell"}, {1, 0, 1});
    const auto text_view = text.view();
    expect(text_view.kind() == anndata::ColumnKind::string &&
           text_view.nullable(), "UTF-8 string column retains nullable kind");
    expect(text_view.string(0) == "café" && text_view.string(1).empty() &&
           text_view.string(2) == "β-cell" && !text_view.valid(1),
           "UTF-8 bytes, empty strings, and null validity remain distinguishable");

    auto byte_codes = array_of<std::int8_t>({2, 0, -1, 2, 1}, {5});
    auto groups = anndata::Column::categorical(
        std::move(byte_codes), anndata::Column::strings({"low", "mid", "high"}), true);
    auto group_view = groups.view();
    expect(group_view.kind() == anndata::ColumnKind::categorical &&
           group_view.ordered() && group_view.nullable(),
           "categorical columns retain ordered and nullable properties");
    expect(group_view.codes().dtype() == anndata::DType::int8 &&
           group_view.code(0) == 2 && group_view.code(1) == 0 &&
           !group_view.valid(2) && group_view.code(2) == -1,
           "categorical codes retain signed byte width and missing code");
    expect(group_view.categories().string(0) == "low" &&
           group_view.categories().string(2) == "high",
           "string category dictionary preserves its declared order");

    auto selected_groups = group_view.select({0, 2, 4});
    expect(selected_groups.categories().size() == 2 &&
           selected_groups.categories().string(0) == "mid" &&
           selected_groups.categories().string(1) == "high" &&
           selected_groups.code(0) == 1 && selected_groups.code(1) == -1 &&
           selected_groups.code(2) == 0,
           "pruned category dictionaries preserve order and recode selected rows");
    auto retained_full_dictionary = group_view.select({0, 2, 4}, false);
    expect(retained_full_dictionary.categories().size() == 3 &&
           retained_full_dictionary.code(0) == 2,
           "categorical selection can retain the complete dictionary");

    auto numeric_codes = array_of<std::int16_t>({1, 0, -1}, {3});
    auto numeric_categories = anndata::Column::categorical(
        std::move(numeric_codes), anndata::Column::from_values<std::int32_t>({10, 20}), false);
    auto numeric_category_view = numeric_categories.view();
    auto dictionary_values = numeric_category_view.categories().values().dense<std::int32_t>();
    expect(!numeric_category_view.ordered() &&
           numeric_category_view.codes().dtype() == anndata::DType::int16 &&
           dictionary_values.data()[0] == 10 && dictionary_values.data()[1] == 20,
           "numeric dictionary and signed 16-bit codes retain their native types");

    auto table = anndata::Table::with_index(
        {"dup", "dup", "cell-2"}, "cell_id");
    table.set("zeta", anndata::Column::from_values<std::int64_t>({4, 5, 6}));
    table.set("alpha", anndata::Column::strings({"first", "second", "third"}));
    auto table_view = table.view();
    expect(table_view.size() == 3 && table_view.index_name() == "cell_id" &&
           table_view.names() == std::vector<std::string>({"dup", "dup", "cell-2"}),
           "table preserves index name and permits duplicate labels");
    expect(table_view.column_names() == std::vector<std::string>({"zeta", "alpha"}),
           "table returns columns in insertion order");
    expect(table_view.column("alpha").string(1) == "second" &&
           table_view.column("zeta").values().dense<std::int64_t>().data()[2] == 6,
           "table columns retain their typed contents");

    auto copied_table = table.copy();
    auto copied_values = copied_table.column("zeta").values().dense<std::int64_t>();
    auto original_values = table.column("zeta").values().dense<std::int64_t>();
    expect(copied_values.data() != original_values.data(),
           "explicit table copy owns independent numeric column storage");
    copied_table.edit_values("zeta").dense<std::int64_t>().data()[0] = 900;
    expect(table.column("zeta").values().dense<std::int64_t>().data()[0] == 4,
           "editing a copied table does not mutate its source");

    anndata::ColumnView retained_column;
    {
        auto temporary = anndata::Column::strings({"retained", "after-scope"});
        retained_column = temporary.view();
    }
    expect(retained_column.size() == 2 &&
           retained_column.string(1) == "after-scope",
           "column views retain their storage after the owner is destroyed");
}

void elements_and_deep_copy() {
    const auto i8 = anndata::Element::scalar<std::int8_t>(-7).view();
    const auto u8 = anndata::Element::scalar<std::uint8_t>(250).view();
    const auto i16 = anndata::Element::scalar<std::int16_t>(-1234).view();
    const auto u16 = anndata::Element::scalar<std::uint16_t>(60000).view();
    const auto i32 = anndata::Element::scalar<std::int32_t>(-123456).view();
    const auto u32 = anndata::Element::scalar<std::uint32_t>(4000000000U).view();
    const auto i64 = anndata::Element::scalar<std::int64_t>(-900000000000LL).view();
    const auto u64 = anndata::Element::scalar<std::uint64_t>(
        18000000000000000000ULL).view();
    const auto f32 = anndata::Element::scalar<float>(1.25F).view();
    const auto f64 = anndata::Element::scalar<double>(-2.5).view();
    const auto boolean = anndata::Element::scalar<bool>(true).view();
    expect(i8.dtype() == anndata::DType::int8 && i8.scalar<std::int8_t>() == -7 &&
           u8.dtype() == anndata::DType::uint8 && u8.scalar<std::uint8_t>() == 250,
           "element scalar API retains 8-bit signed and unsigned widths");
    expect(i16.dtype() == anndata::DType::int16 && i16.scalar<std::int16_t>() == -1234 &&
           u16.dtype() == anndata::DType::uint16 && u16.scalar<std::uint16_t>() == 60000,
           "element scalar API retains 16-bit signed and unsigned widths");
    expect(i32.dtype() == anndata::DType::int32 && i32.scalar<std::int32_t>() == -123456 &&
           u32.dtype() == anndata::DType::uint32 &&
           u32.scalar<std::uint32_t>() == 4000000000U,
           "element scalar API retains 32-bit signed and unsigned widths");
    expect(i64.dtype() == anndata::DType::int64 &&
           i64.scalar<std::int64_t>() == -900000000000LL &&
           u64.dtype() == anndata::DType::uint64 &&
           u64.scalar<std::uint64_t>() == 18000000000000000000ULL,
           "element scalar API retains 64-bit signed and unsigned widths");
    expect(f32.dtype() == anndata::DType::float32 && f32.scalar<float>() == 1.25F &&
           f64.dtype() == anndata::DType::float64 && f64.scalar<double>() == -2.5 &&
           boolean.dtype() == anndata::DType::boolean && boolean.scalar<bool>(),
           "element scalar API preserves floating widths and booleans");

    anndata::Element null_value;
    anndata::Element string_value("λ-value");
    anndata::Element array_value(array_of<std::int32_t>({3, 8}, {2}));
    anndata::Element column_value(anndata::Column::strings({"x", "y"}));
    auto nested_table = anndata::Table::with_index({"r0", "r1"}, "row");
    nested_table.set("value", anndata::Column::from_values<double>({0.5, 1.5}));
    anndata::Element table_value(std::move(nested_table));
    auto mapping_value = anndata::Element::mapping();
    mapping_value.set("null", std::move(null_value));
    mapping_value.set("string", std::move(string_value));
    mapping_value.set("array", std::move(array_value));
    mapping_value.set("column", std::move(column_value));
    mapping_value.set("table", std::move(table_value));
    mapping_value.set("scalar", anndata::Element::scalar<std::uint16_t>(512));
    const auto mapping_view = mapping_value.view();
    expect(mapping_view.kind() == anndata::ElementKind::mapping &&
           mapping_view.contains("table") &&
           mapping_view.keys() == std::vector<std::string>({
               "null", "string", "array", "column", "table", "scalar"}),
           "nested element mapping preserves keys and insertion order");
    expect(mapping_view.at("null").kind() == anndata::ElementKind::null &&
           mapping_view.at("string").string() == "λ-value" &&
           mapping_view.at("array").array().dense<std::int32_t>().data()[1] == 8 &&
           mapping_view.at("column").column().string(1) == "y" &&
           mapping_view.at("table").table().column("value").values()
               .dense<double>().data()[0] == 0.5 &&
           mapping_view.at("scalar").scalar<std::uint16_t>() == 512,
           "element mapping exposes each supported nested value kind");

    std::array<std::int32_t, 2> borrowed_payload{41, 42};
    auto data = anndata::AnnData::empty(1, 1);
    auto metadata = anndata::Element::mapping();
    auto payload = anndata::Element::mapping();
    payload.set("values", anndata::Element(anndata::Array::borrow_dense(
        borrowed_payload.data(), {2}, {1})));
    metadata.set("payload", std::move(payload));
    data.set_uns("metadata", std::move(metadata));
    auto source_payload = data.uns().at("metadata").at("payload").at("values").array();
    auto clone = data.copy();
    auto cloned_payload = clone.uns().at("metadata").at("payload").at("values").array();
    expect(source_payload.dense<std::int32_t>().data() == borrowed_payload.data() &&
           cloned_payload.dense<std::int32_t>().data() != borrowed_payload.data(),
           "AnnData copy deep-copies arrays nested in uns mappings");
    borrowed_payload[0] = 99;
    expect(source_payload.dense<std::int32_t>().data()[0] == 99 &&
           cloned_payload.dense<std::int32_t>().data()[0] == 41,
           "nested metadata copy remains independent when source storage changes");
}

void anndata_slots_alignment_and_aliasing() {
    static_assert(!std::is_copy_constructible_v<anndata::AnnData>);
    static_assert(std::is_move_constructible_v<anndata::AnnData>);

    auto data = anndata::AnnData::empty(3, 2);
    data.set_obs(anndata::Table::with_index({"c0", "c1", "c2"}, "cell"));
    data.set_var(anndata::Table::with_index({"g0", "g1"}, "gene"));
    data.set_x(array_of<std::int32_t>({0, 1, 2, 3, 4, 5}, {3, 2}));
    data.set_layer("counts", array_of<float>({1, 2, 3, 4, 5, 6}, {3, 2}));
    data.set_obsm("embedding", array_of<double>({0, 1, 2, 3, 4, 5}, {3, 2}));
    auto obsm_table = anndata::Table::with_index({"c0", "c1", "c2"}, "cell");
    obsm_table.set("cluster", anndata::Column::strings({"A", "B", "A"}));
    data.set_obsm("annotations", std::move(obsm_table));
    data.set_varm("loadings", array_of<float>({2, 3, 4, 5}, {2, 2}));
    data.set_obsp("neighbors", array_of<std::int32_t>(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
        {3, 3, 2}));
    data.set_varp("similarity", array_of<std::int32_t>(
        {0, 1, 2, 3, 4, 5, 6, 7}, {2, 2, 2}));
    auto uns = anndata::Element::mapping();
    uns.set("version", anndata::Element::scalar<std::uint16_t>(3));
    data.set_uns("metadata", std::move(uns));

    auto raw_var = anndata::Table::with_index({"rg0", "rg1", "rg2"}, "gene");
    raw_var.set("symbol", anndata::Column::strings({"rA", "rB", "rC"}));
    data.set_raw(array_of<float>({10, 11, 12, 20, 21, 22, 30, 31, 32}, {3, 3}),
                 std::move(raw_var));
    data.set_raw_varm("raw_loadings", array_of<std::int16_t>(
        {7, 8, 9, 10, 11, 12}, {3, 2}));
    data.validate();

    expect(data.n_obs() == 3 && data.n_vars() == 2 && data.has_x(),
           "AnnData retains axis sizes and X");
    expect(data.obs().index_name() == "cell" && data.var().index_name() == "gene" &&
           data.layer_names() == std::vector<std::string>({"counts"}) &&
           data.obsm_names() == std::vector<std::string>({"embedding", "annotations"}) &&
           data.varm_names() == std::vector<std::string>({"loadings"}) &&
           data.obsp_names() == std::vector<std::string>({"neighbors"}) &&
           data.varp_names() == std::vector<std::string>({"similarity"}),
           "standard AnnData slots preserve their names and axis indexes");
    expect(data.obsm("annotations").kind() == anndata::ElementKind::table &&
           data.obsm("annotations").table().names() == data.obs().names() &&
           data.obsm("annotations").table().column("cluster").string(1) == "B",
           "obsm accepts and retains an index-aligned table");
    expect(data.obsp("neighbors").rank() == 3 &&
           data.obsp("neighbors").extent(0) == 3 &&
           data.obsp("neighbors").extent(1) == 3 &&
           data.varp("similarity").extent(0) == 2 &&
           data.varp("similarity").extent(1) == 2,
           "pairwise slots align their first two dimensions and allow trailing axes");
    expect(data.has_raw() && data.raw_x().extent(0) == 3 &&
           data.raw_x().extent(1) == 3 && data.raw_var().size() == 3 &&
           data.raw_var().name(1) == "rg1" &&
           data.raw_x().dense<float>().data()[8] == 32 &&
           data.raw_varm("raw_loadings").array().dense<std::int16_t>().data()[5] == 12,
           "raw keeps its own variable axis, matrix values, and varm dimensions");

    auto copied = data.copy();
    expect(copied.x().dense<std::int32_t>().data() != data.x().dense<std::int32_t>().data(),
           "AnnData copy owns independent X storage");
    copied.edit_x().dense<std::int32_t>().data()[0] = -10;
    expect(data.x().dense<std::int32_t>().data()[0] == 0,
           "editing copied X does not mutate the original dataset");

    std::array<float, 4> external{1, 2, 3, 4};
    auto borrowed_data = anndata::AnnData::empty(2, 2);
    borrowed_data.set_x(anndata::Array::borrow_dense(external.data(), {2, 2}, {2, 1}));
    auto retained_x = borrowed_data.x();
    expect(retained_x.dense<float>().data() == external.data(),
           "AnnData preserves the caller-owned matrix pointer");
    borrowed_data.edit_x().dense<float>().data()[2] = 30;
    expect(external[2] == 30, "edits through AnnData alias caller-owned storage");
    external[1] = 20;
    expect(retained_x.dense<float>().data()[1] == 20,
           "writes through the original pointer are visible through retained views");

    auto old_x = data.x();
    const auto* old_x_pointer = old_x.dense<std::int32_t>().data();
    data.set_x(array_of<std::int32_t>({50, 51, 52, 53, 54, 55}, {3, 2}));
    expect(old_x.dense<std::int32_t>().data() == old_x_pointer &&
           old_x_pointer[0] == 0 && data.x().dense<std::int32_t>().data()[0] == 50,
           "replacing X leaves an existing view bound to the old slot value");
    auto old_layer = data.layer("counts");
    data.set_layer("counts", array_of<float>({9, 9, 9, 9, 9, 9}, {3, 2}));
    expect(old_layer.dense<float>().data()[0] == 1 &&
           data.layer("counts").dense<float>().data()[0] == 9,
           "replacing a named layer leaves prior views on the previous array");

    auto short_obs = anndata::Table::numbered(2);
    expect_error(anndata::ErrorCode::shape_mismatch, "obs",
                 [&] { data.set_obs(std::move(short_obs)); },
                 "observation table length mismatch reports its slot path");
    expect_error(anndata::ErrorCode::shape_mismatch, "X",
                 [&] { data.set_x(array_of<float>({1, 2, 3}, {3, 1})); },
                 "X shape mismatch reports its slot path");
    expect_error(anndata::ErrorCode::shape_mismatch, "layers/wrong",
                 [&] { data.set_layer("wrong", array_of<float>({1, 2, 3}, {3, 1})); },
                 "layer shape mismatch reports its slot path");
    expect_error(anndata::ErrorCode::shape_mismatch, "obsm/short",
                 [&] { data.set_obsm("short", array_of<float>({1, 2}, {2, 1})); },
                 "obsm first-axis mismatch reports its slot path");
    auto mismatched_obsm_table = anndata::Table::with_index({"c2", "c1", "c0"}, "cell");
    expect_error(anndata::ErrorCode::shape_mismatch, "obsm/misaligned",
                 [&] { data.set_obsm("misaligned", std::move(mismatched_obsm_table)); },
                 "obsm table index mismatch reports its slot path");
    expect_error(anndata::ErrorCode::shape_mismatch, "obsp/not_pairwise",
                 [&] { data.set_obsp("not_pairwise", array_of<float>(
                     {1, 2, 3, 4, 5, 6}, {3, 2, 1})); },
                 "obsp validates both aligned dimensions");
    expect_error(anndata::ErrorCode::shape_mismatch, "varp/not_pairwise",
                 [&] { data.set_varp("not_pairwise", array_of<float>(
                     {1, 2, 3, 4}, {2, 1, 2})); },
                 "varp validates both aligned dimensions");
    expect_error(anndata::ErrorCode::shape_mismatch, "raw/X",
                 [&] { data.set_raw(array_of<float>({1, 2, 3, 4}, {2, 2}),
                                    anndata::Table::numbered(2)); },
                 "raw matrix observation mismatch reports its slot path");
}

void slicing_and_category_colors() {
    auto data = anndata::AnnData::empty(5, 3);
    data.set_obs(anndata::Table::with_index(
        {"cell0", "cell1", "cell2", "cell3", "cell4"}, "cell"));
    data.set_var(anndata::Table::with_index({"g0", "g1", "g2"}, "gene"));
    data.set_x(array_of<std::int32_t>(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, {5, 3}));
    data.set_layer("shifted", array_of<std::int32_t>(
        {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114},
        {5, 3}));
    data.set_obsm("coordinates", array_of<std::int32_t>(
        {0, 1, 10, 11, 20, 21, 30, 31, 40, 41}, {5, 2}));
    auto group_codes = array_of<std::int8_t>({2, 0, -1, 2, 1}, {5});
    data.set_obs_column("group", anndata::Column::categorical(
        std::move(group_codes), anndata::Column::strings({"low", "mid", "high"}), true));
    auto colors = anndata::Element::mapping();
    colors.set("group_colors", anndata::Element(
        anndata::Column::strings({"blue", "green", "red"})));
    data.set_uns(std::move(colors));

    auto reverse = data.slice(anndata::range{std::nullopt, std::nullopt, -1});
    expect(reverse.n_obs() == 5 && reverse.obs().name(0) == "cell4" &&
           reverse.obs().name(4) == "cell0" &&
           flattened<std::int32_t>(reverse.x()) ==
               std::vector<std::int32_t>({12, 13, 14, 9, 10, 11, 6, 7, 8,
                                          3, 4, 5, 0, 1, 2}),
           "negative-step range slices reverse the observation axis and preserve values");

    auto indexed = data.slice(anndata::indices{-1, 1, 1}, anndata::indices{2, 0});
    expect(indexed.n_obs() == 3 && indexed.n_vars() == 2 &&
           indexed.obs().names() == std::vector<std::string>({"cell4", "cell1", "cell1"}) &&
           flattened<std::int32_t>(indexed.x()) ==
               std::vector<std::int32_t>({14, 12, 5, 3, 5, 3}),
           "positional index lists support negative indexes, repeats, and column reordering");

    auto masked = data.slice(anndata::mask{0, 1, 0, 1, 1});
    expect(masked.obs().names() ==
               std::vector<std::string>({"cell1", "cell3", "cell4"}) &&
           flattened<std::int32_t>(masked.x()) ==
               std::vector<std::int32_t>({3, 4, 5, 9, 10, 11, 12, 13, 14}),
           "boolean mask selects rows in source order");

    auto one_label = data.slice("cell3");
    expect(one_label.n_obs() == 1 && one_label.obs().name(0) == "cell3" &&
           flattened<std::int32_t>(one_label.x()) ==
               std::vector<std::int32_t>({9, 10, 11}),
           "single-label slicing resolves the observation index");
    auto label_list = data.slice(anndata::labels{"cell4", "cell0"},
                                 anndata::labels{"g2", "g0"});
    expect(label_list.obs().names() == std::vector<std::string>({"cell4", "cell0"}) &&
           label_list.var().names() == std::vector<std::string>({"g2", "g0"}) &&
           flattened<std::int32_t>(label_list.x()) ==
               std::vector<std::int32_t>({14, 12, 2, 0}),
           "label-list slicing preserves requested order on both axes");

    auto empty = data.slice(anndata::range{2, 2, 1});
    expect(empty.n_obs() == 0 && empty.n_vars() == 3 && empty.obs().size() == 0 &&
           empty.x().extent(0) == 0 && empty.x().extent(1) == 3 &&
           empty.layer("shifted").extent(0) == 0,
           "empty positional slices preserve ranks and unselected axis sizes");

    auto pruned = data.slice(anndata::indices{0, 2, 4});
    auto groups = pruned.obs().column("group");
    expect(groups.categories().size() == 2 &&
           groups.categories().string(0) == "mid" &&
           groups.categories().string(1) == "high" &&
           groups.code(0) == 1 && groups.code(1) == -1 && groups.code(2) == 0,
           "AnnData slicing prunes categorical dictionaries in original order");
    auto sliced_colors = pruned.uns().at("group_colors").column();
    expect(sliced_colors.size() == 2 && sliced_colors.string(0) == "green" &&
           sliced_colors.string(1) == "red",
           "uns string-column category colors follow the pruned dictionary");
    auto empty_categories = data.slice(anndata::indices{}).obs().column("group");
    expect(empty_categories.categories().size() == 0,
           "empty row selection prunes all unused categories");

    auto duplicate_labels = anndata::AnnData::empty(3, 1);
    duplicate_labels.set_obs(anndata::Table::with_index({"dup", "other", "dup"}, "cell"));
    duplicate_labels.set_x(array_of<std::int32_t>({10, 20, 30}, {3, 1}));
    auto matching_duplicates = duplicate_labels.slice("dup");
    expect(matching_duplicates.n_obs() == 2 &&
           matching_duplicates.obs().names() == std::vector<std::string>({"dup", "dup"}) &&
           flattened<std::int32_t>(matching_duplicates.x()) ==
               std::vector<std::int32_t>({10, 30}),
           "single-label lookup returns all matching duplicate index labels");
    expect_error(anndata::ErrorCode::invalid_argument, "obs/index",
                 [&] { (void)duplicate_labels.slice(anndata::labels{"dup"}); },
                 "label-list lookup rejects a non-unique axis with its path");
}

void retained_dataset_views_and_raw_slicing() {
    anndata::AnnDataView retained;
    anndata::ArrayView retained_x;
    anndata::TableView retained_obs;
    anndata::ElementView retained_metadata;
    {
        auto data = anndata::AnnData::empty(3, 2);
        data.set_obs(anndata::Table::with_index({"o0", "o1", "o2"}, "obs_id"));
        data.set_var(anndata::Table::with_index({"v0", "v1"}, "var_id"));
        data.set_x(array_of<std::int32_t>({1, 2, 3, 4, 5, 6}, {3, 2}));
        data.set_obs_column("quality", anndata::Column::from_values<float>({0.1F, 0.2F, 0.3F}));
        auto metadata = anndata::Element::mapping();
        metadata.set("label", anndata::Element("kept"));
        data.set_uns("metadata", std::move(metadata));
        retained = data.view();
        retained_x = data.x();
        retained_obs = data.obs();
        retained_metadata = data.uns().at("metadata").at("label");
    }
    expect(retained.n_obs() == 3 && retained.n_vars() == 2 &&
           retained.x().dense<std::int32_t>().data()[5] == 6 &&
           retained_x.dense<std::int32_t>().data()[2] == 3,
           "dataset and array views retain X after the AnnData owner is destroyed");
    expect(retained_obs.name(1) == "o1" &&
           retained_obs.column("quality").values().dense<float>().data()[2] == 0.3F &&
           retained_metadata.string() == "kept",
           "table, column, and nested metadata views outlive their AnnData owner");

    auto data = anndata::AnnData::empty(4, 2);
    data.set_obs(anndata::Table::with_index({"a", "b", "c", "d"}, "cell"));
    data.set_var(anndata::Table::with_index({"x", "y"}, "gene"));
    data.set_x(array_of<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2}));
    auto raw_var = anndata::Table::with_index({"r0", "r1", "r2"}, "gene");
    raw_var.set("symbol", anndata::Column::strings({"A", "B", "C"}));
    data.set_raw(array_of<float>({10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42},
                                 {4, 3}), std::move(raw_var));
    data.set_raw_varm("embedding", array_of<std::int16_t>(
        {1, 2, 3, 4, 5, 6}, {3, 2}));

    auto view = data.slice(anndata::indices{3, 1}, anndata::indices{1});
    auto raw = flattened<float>(view.raw_x());
    expect(view.n_obs() == 2 && view.n_vars() == 1 && view.raw_x().extent(0) == 2 &&
           view.raw_x().extent(1) == 3 && raw == std::vector<float>({40, 41, 42, 20, 21, 22}),
           "raw matrix follows observation selection while retaining its own variable axis");
    expect(view.raw_var().names() == std::vector<std::string>({"r0", "r1", "r2"}) &&
           view.raw_var().column("symbol").string(2) == "C" &&
           view.raw_varm("embedding").array().extent(0) == 3 &&
           view.raw_varm("embedding").array().dense<std::int16_t>().data()[5] == 6,
           "raw variable metadata and varm remain unsliced by current var selection");
}

void structural_snapshot_and_scalar_edges() {
    auto data = anndata::AnnData::empty(2, 1);
    data.set_obs(anndata::Table::with_index({"old-a", "old-b"}));
    auto annotation = anndata::Table::with_index({"old-a", "old-b"});
    annotation.set("value", anndata::Column::from_values<int>({1, 2}));
    data.set_obsm("frame", std::move(annotation));
    data.set_uns("label", anndata::Element("before"));
    auto before = data.view();
    data.set_obs(anndata::Table::with_index({"new-a", "new-b"}));
    data.set_uns("label", anndata::Element("after"));
    expect(before.obs().name(0) == "old-a" &&
           before.obsm("frame").table().name(0) == "old-a" &&
           before.uns().at("label").string() == "before",
           "dataset views retain index and metadata bindings through structural replacements");
    expect(data.obsm("frame").table().name(0) == "new-a",
           "replacing obs propagates its index to aligned table annotations");
    auto one = data.slice(0, anndata::all);
    expect(one.n_obs() == 1 && one.obs().name(0) == "new-a",
           "ordinary integer zero selects a row rather than a null string pointer");
    auto narrow = anndata::Element::scalar<std::int16_t>(-12);
    auto narrow_copy = narrow.copy();
    expect(narrow_copy.view().dtype() == anndata::DType::int16 &&
           narrow_copy.view().scalar<std::int16_t>() == -12,
           "narrow scalar width is preserved through explicit metadata copies");
    auto real = anndata::Element(0.5f);
    expect(real.view().dtype() == anndata::DType::float32 && real.view().scalar<float>() == 0.5f,
           "ordinary float scalar construction retains float32 width");
    auto null = anndata::Element(nullptr);
    expect(null.view().kind() == anndata::ElementKind::null, "nullptr constructs an explicit null element");
    expect_error(anndata::ErrorCode::dtype_mismatch, "element", [&] {
        (void)real.view().scalar<double>();
    }, "typed scalar reads reject implicit width conversion");
    anndata::AnnDataView empty;
    expect_error(anndata::ErrorCode::out_of_bounds, "layers/absent", [&] {
        (void)empty.layer("absent");
    }, "empty dataset views diagnose absent map keys");
    anndata::ColumnView empty_column;
    expect(empty_column.select({}).size() == 0, "empty column views permit an empty selection");
}

void raw_tables_and_empty_nullable_columns() {
    auto data = anndata::AnnData::empty(2, 1);
    data.set_x(array_of<float>({1, 2}, {2, 1}));
    data.set_obs_column("nullable", anndata::Column::from_values<std::int32_t>({1, 0}, {1, 0}));
    data.set_obs_column("text", anndata::Column::strings({"a", ""}, {1, 0}));
    data.set_raw(array_of<float>({10, 11, 12, 20, 21, 22}, {2, 3}),
                 anndata::Table::with_index({"r0", "r1", "r2"}));
    auto raw_table = anndata::Table::with_index({"r0", "r1", "r2"});
    raw_table.set("label", anndata::Column::strings({"A", "B", "C"}));
    data.set_raw_varm("annotations", std::move(raw_table));
    data.validate();
    expect(data.raw_varm("annotations").table().name(2) == "r2" &&
           data.raw_varm_names() == std::vector<std::string>({"annotations"}),
           "raw varm accepts a table aligned to its independent variable axis");
    auto selected = data.slice(anndata::indices{1}, anndata::all);
    expect(selected.raw_varm("annotations").table().size() == 3 &&
           selected.raw_varm("annotations").table().column("label").string(2) == "C",
           "observation slicing leaves raw variable-aligned tables intact");
    auto copied = selected.copy();
    expect(copied.raw_varm("annotations").table().column("label").string(1) == "B",
           "deep dataset copies preserve dataframe-valued raw varm");
    expect_error(anndata::ErrorCode::shape_mismatch, "raw/varm/bad", [&] {
        data.set_raw_varm("bad", anndata::Table::with_index({"r2", "r1", "r0"}));
    }, "raw variable table insertion checks label order, not only row count");
    auto empty = data.slice(anndata::indices{}, anndata::all).copy();
    expect(empty.obs().column("nullable").nullable() &&
           empty.obs().column("nullable").values().dtype() == anndata::DType::int32 &&
           empty.obs().column("text").nullable(),
           "empty dataset copies preserve numeric and string nullable markers");
    auto n = anndata::Column::nullable_numeric(anndata::Array::empty<bool>({0}), {});
    auto s = anndata::Column::nullable_strings({}, {});
    expect(n.view().nullable() && n.copy().view().nullable() &&
           s.view().nullable() && s.copy().view().nullable(),
           "explicit nullable factories represent zero-row numeric and string columns");
}

void reviewed_category_and_validation_regressions() {
    auto one = anndata::AnnData::empty(2, 1);
    one.set_obs_column("group", anndata::Column::categorical(
        array_of<std::int8_t>({0, 0}, {2}), anndata::Column::strings({"A"})));
    one.set_uns("group_colors", anndata::Element("red"));
    auto selected = one.slice(anndata::indices{1}, anndata::all);
    expect(selected.uns().at("group_colors").column().size() == 1 &&
           selected.uns().at("group_colors").column().string(0) == "red",
           "one-category scalar colors are promoted to a length-one string array");
    one.set_obs_column("group", anndata::Column::categorical(
        array_of<std::int8_t>({0, 1}, {2}), anndata::Column::strings({"A", "B"})));
    expect(one.slice(anndata::indices{}, anndata::all).uns().at("group_colors").column().size() == 1,
           "scalar colors remain length one for an empty category selection");
    one.set_uns("group_colors", anndata::Element(anndata::Column::strings({"red"})));
    expect(!one.slice(anndata::indices{0}, anndata::all).uns().contains("group_colors"),
           "non-scalar colors with mismatched category count are removed");
    one.set_uns("group_colors", anndata::Element::scalar<std::int16_t>(7));
    auto numeric_color = one.slice(anndata::indices{}, anndata::all).uns().at("group_colors").array();
    expect(numeric_color.rank() == 1 && numeric_color.extent(0) == 1 &&
           numeric_color.dense<std::int16_t>().data()[0] == 7,
           "typed numeric scalar colors preserve width and survive empty selection");
    auto scalar_array = anndata::Array::empty<float>({});
    scalar_array.edit().dense<float>().data()[0] = 0.5f;
    one.set_uns("group_colors", anndata::Element(std::move(scalar_array)));
    auto rank_zero_color = one.slice(anndata::indices{1}, anndata::all).uns().at("group_colors").array();
    expect(rank_zero_color.rank() == 1 && rank_zero_color.extent(0) == 1 &&
           rank_zero_color.dense<float>().data()[0] == 0.5f,
           "rank-zero numerical color arrays are promoted before category handling");
    one.set_uns("group_colors", anndata::Element(array_of<float>(
        {1, 0, 0, 1, 0, 0, 1, 1}, {2, 4})));
    auto rgba = one.slice(anndata::indices{0}, anndata::all).uns().at("group_colors").array();
    expect(rgba.rank() == 2 && rgba.extent(0) == 1 && rgba.extent(1) == 4 &&
           flattened<float>(rgba) == std::vector<float>({1, 0, 0, 1}),
           "color pruning preserves trailing dimensions of RGBA arrays");

    auto codes = array_of<std::int8_t>({0, 1}, {2});
    auto alias = codes.edit().dense<std::int8_t>();
    auto group = anndata::Column::categorical(std::move(codes), anndata::Column::strings({"A", "B"}));
    auto all = group.view().select_all();
    expect(all.codes().dense<std::int8_t>().data() == alias.data(),
           "unchanged category dictionaries preserve direct code-buffer access");
    auto pruned = group.view().select({0});
    expect(pruned.codes().dense<std::int8_t>().data() == alias.data(),
           "prefix-only category pruning needs no code-buffer copy");
    alias.data()[0] = 1;
    group.validate();
    expect_error(anndata::ErrorCode::invalid_argument, "categorical/codes", [&] { (void)pruned.valid(0); },
                 "stale pruned category validity rejects mutation outside its dictionary");
    expect_error(anndata::ErrorCode::invalid_argument, "categorical/codes", [&] { (void)pruned.code(0); },
                 "stale category codes do not silently become missing values");
    expect_error(anndata::ErrorCode::invalid_argument, "categorical/codes", [&] { (void)pruned.codes(); },
                 "stale category buffer acquisition rejects invalidated remaps");
    expect_error(anndata::ErrorCode::invalid_argument, "categorical/codes", [&] { (void)pruned.copy(); },
                 "stale category copies do not silently lose data");
    expect(group.view().select({0}).categories().string(0) == "B",
           "fresh category views reflect deliberate source-code mutation");

    auto dictionary = array_of<std::int32_t>({1, 2}, {2});
    auto dictionary_alias = dictionary.edit().dense<std::int32_t>();
    auto numeric_group = anndata::Column::categorical(array_of<std::int8_t>({0, 1}, {2}),
        anndata::Column::numeric(std::move(dictionary)));
    dictionary_alias.data()[1] = 1;
    expect_error(anndata::ErrorCode::invalid_argument, "categorical/categories", [&] { numeric_group.validate(); },
                 "explicit validation detects duplicate category dictionaries after raw writes");

    auto data = anndata::AnnData::empty(2, 1);
    auto counts = anndata::Array::csr<float, std::int32_t, std::int32_t>(2, 1, 1);
    auto sparse = counts.edit().csr<float, std::int32_t, std::int32_t>();
    sparse.values()[0] = 1; sparse.unsafe_indices()[0] = 0;
    sparse.unsafe_offsets()[0] = 0; sparse.unsafe_offsets()[1] = 1; sparse.unsafe_offsets()[2] = 1;
    data.set_layer("counts", std::move(counts));
    sparse.unsafe_offsets()[1] = 2;
    expect_error(anndata::ErrorCode::invalid_sparse_structure, "layers/counts", [&] { data.validate(); },
                 "dataset validation annotates numerical errors with the affected slot key");
}

} // namespace

int main() {
    owners_columns_and_tables();
    elements_and_deep_copy();
    anndata_slots_alignment_and_aliasing();
    slicing_and_category_colors();
    retained_dataset_views_and_raw_slicing();
    structural_snapshot_and_scalar_edges();
    raw_tables_and_empty_nullable_columns();
    reviewed_category_and_validation_regressions();
    if (failures != 0) {
        std::cerr << failures << " data test(s) failed\n";
        return 1;
    }
    std::cout << "all data tests passed\n";
    return 0;
}
