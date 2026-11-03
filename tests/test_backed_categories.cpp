#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import anndata;

namespace {
using namespace anndata;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<class T>
Array dense(std::initializer_list<std::size_t> shape,
            std::initializer_list<T> values) {
    auto result = Array::empty<T>(shape);
    auto output = result.edit().template dense<T>();
    std::size_t i = 0;
    for (const auto value : values) output.data()[i++] = value;
    std::size_t expected = 1;
    for (std::size_t axis = 0; axis < output.rank(); ++axis)
        expected *= output.extent(axis);
    require(i == expected, "fixture element count differs from shape");
    return result;
}

AnnData fixture() {
    auto data = AnnData::empty(3, 2);
    data.set_x(dense<float>({3, 2}, {1, 2, 3, 4, 5, 6}));

    auto obs = Table::with_index({"c0", "c1", "c2"}, "cell_id");
    obs.set("group", Column::categorical(
        dense<std::int8_t>({3}, {0, 1, -1}),
        Column::strings({"T", "B", "unused"}), true));
    obs.set("numeric_dictionary", Column::categorical(
        dense<std::int8_t>({3}, {0, 1, 1}),
        Column::nullable_numeric(dense<std::int16_t>({2}, {17, 29}), {1, 1}), true));
    obs.set("string_dictionary_na", Column::categorical(
        dense<std::int8_t>({3}, {1, 0, 0}),
        Column::nullable_strings({"alpha", ""}, {1, 1}, StringMissingValue::NA)));
    obs.set("string_dictionary_nan", Column::categorical(
        dense<std::int8_t>({3}, {0, 1, 1}),
        Column::nullable_strings({"first", "β"}, {1, 1}, StringMissingValue::NaN), true));
    obs.set("count", Column::nullable_numeric(
        dense<std::int16_t>({3}, {7, 0, -2}), {1, 0, 1}));
    obs.set("label_na", Column::nullable_strings(
        {"left", "", "right"}, {1, 0, 1}, StringMissingValue::NA));
    obs.set("label_nan", Column::nullable_strings(
        {"first", "", "last"}, {1, 0, 1}, StringMissingValue::NaN));
    data.set_obs(std::move(obs));
    data.set_var(Table::with_index({"g0", "g1"}, "gene_id"));

    data.set_layer("counts", dense<std::int32_t>({3, 2}, {10, 11, 20, 21, 30, 31}));
    data.set_obsp("graph", dense<float>({3, 3}, {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    }));
    data.set_uns("group_colors", Element(StringArray::from_strings(
        {3}, {"red", "blue", "green"})));

    // Raw has three variables while the main AnnData has two. This catches
    // accidental coupling between raw.var and a main-axis reindex.
    data.set_raw(dense<double>({3, 3}, {
        100, 101, 102,
        110, 111, 112,
        120, 121, 122
    }), Table::with_index({"raw0", "raw1", "raw2"}, "raw_gene_id"));
    return data;
}

template<class T>
void require_equal_dense(ArrayView actual, ArrayView expected,
                         std::string_view message) {
    require(actual.layout() == Layout::dense && expected.layout() == Layout::dense,
            message);
    require(actual.dtype() == dtype_of<T>() && expected.dtype() == dtype_of<T>(), message);
    require(actual.rank() == expected.rank(), message);
    for (std::size_t axis = 0; axis < actual.rank(); ++axis)
        require(actual.extent(axis) == expected.extent(axis), message);
    const auto left = actual.dense<T>();
    const auto right = expected.dense<T>();
    std::size_t size = 1;
    for (std::size_t axis = 0; axis < actual.rank(); ++axis)
        size *= actual.extent(axis);
    for (std::size_t i = 0; i < size; ++i)
        require(left.data()[i] == right.data()[i], message);
}

void require_equal_strings(StringArrayView actual, StringArrayView expected,
                           std::string_view message) {
    require(actual.shape() == expected.shape() && actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(actual.string(i) == expected.string(i), message);
}

void require_equal_strings(const BackedStringArray& actual, StringArrayView expected,
                           std::string_view message) {
    require(actual.shape() == expected.shape() && actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(actual.string(i) == expected.string(i), message);
}

void require_equal_category_dictionary(ColumnView actual, ColumnView expected,
                                       std::string_view message) {
    require(actual.kind() == expected.kind() && actual.size() == expected.size() &&
            actual.nullable() == expected.nullable(), message);
    if (actual.kind() == ColumnKind::string) {
        require(actual.string_missing_value() == expected.string_missing_value(), message);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require(actual.valid(i) == expected.valid(i), message);
            if (actual.valid(i)) require(actual.string(i) == expected.string(i), message);
        }
        return;
    }

    const auto actual_values = actual.values();
    const auto expected_values = expected.values();
    require(actual_values.dtype() == expected_values.dtype(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(actual.valid(i) == expected.valid(i), message);
    switch (actual_values.dtype()) {
    case DType::boolean: require_equal_dense<bool>(actual_values, expected_values, message); break;
    case DType::int8: require_equal_dense<std::int8_t>(actual_values, expected_values, message); break;
    case DType::uint8: require_equal_dense<std::uint8_t>(actual_values, expected_values, message); break;
    case DType::int16: require_equal_dense<std::int16_t>(actual_values, expected_values, message); break;
    case DType::uint16: require_equal_dense<std::uint16_t>(actual_values, expected_values, message); break;
    case DType::int32: require_equal_dense<std::int32_t>(actual_values, expected_values, message); break;
    case DType::uint32: require_equal_dense<std::uint32_t>(actual_values, expected_values, message); break;
    case DType::int64: require_equal_dense<std::int64_t>(actual_values, expected_values, message); break;
    case DType::uint64: require_equal_dense<std::uint64_t>(actual_values, expected_values, message); break;
    case DType::float32: require_equal_dense<float>(actual_values, expected_values, message); break;
    case DType::float64: require_equal_dense<double>(actual_values, expected_values, message); break;
    }
}

void require_equal_categorical(ColumnView actual, ColumnView expected,
                               std::string_view message) {
    require(actual.kind() == ColumnKind::categorical &&
            expected.kind() == ColumnKind::categorical &&
            actual.ordered() == expected.ordered(), message);
    const auto actual_categories = actual.categories();
    const auto expected_categories = expected.categories();
    require_equal_category_dictionary(actual_categories, expected_categories, message);
    require_equal_dense<std::int8_t>(actual.codes(), expected.codes(), message);
}

void require_equal_nullable_columns(const TableView& actual, const TableView& expected) {
    const auto actual_count = actual.column("count");
    const auto expected_count = expected.column("count");
    require(actual_count.kind() == expected_count.kind() &&
            actual_count.nullable() == expected_count.nullable(),
            "nullable numeric column metadata differs");
    for (std::size_t row = 0; row < actual_count.size(); ++row)
        require(actual_count.valid(row) == expected_count.valid(row),
                "nullable numeric validity differs");
    require_equal_dense<std::int16_t>(actual_count.values(), expected_count.values(),
                                      "nullable numeric values differ");

    for (const auto name : {"label_na", "label_nan"}) {
        const auto actual_label = actual.column(name);
        const auto expected_label = expected.column(name);
        require(actual_label.kind() == ColumnKind::string &&
                expected_label.kind() == ColumnKind::string &&
                actual_label.nullable() == expected_label.nullable() &&
                actual_label.string_missing_value() == expected_label.string_missing_value(),
                "nullable string metadata differs");
        for (std::size_t row = 0; row < actual_label.size(); ++row) {
            require(actual_label.valid(row) == expected_label.valid(row),
                    "nullable string validity differs");
            if (actual_label.valid(row))
                require(actual_label.string(row) == expected_label.string(row),
                        "nullable string values differ");
        }
    }
}

void require_equal_data(const AnnData& actual, const AnnData& expected) {
    require(actual.n_obs() == expected.n_obs() && actual.n_vars() == expected.n_vars(),
            "materialized backed dimensions differ");
    require_equal_dense<float>(actual.x(), expected.x(), "X values differ");
    require_equal_dense<std::int32_t>(actual.layer("counts"), expected.layer("counts"),
                                      "layer values differ");
    require_equal_dense<float>(actual.obsp("graph"), expected.obsp("graph"),
                               "pairwise values differ");
    for (const auto name : {"group", "numeric_dictionary", "string_dictionary_na",
                            "string_dictionary_nan"})
        require_equal_categorical(actual.obs().column(name), expected.obs().column(name),
                                  "categorical metadata, dictionary, or codes differ");
    require_equal_nullable_columns(actual.obs(), expected.obs());
    require_equal_strings(actual.uns().at("group_colors").string_array(),
                          expected.uns().at("group_colors").string_array(),
                          "categorical colors differ");

    require(actual.has_raw_x() && expected.has_raw_x(), "raw X is missing");
    require_equal_dense<double>(actual.raw_x(), expected.raw_x(), "raw X values differ");
    require(actual.raw_var().index_name() == expected.raw_var().index_name() &&
            actual.raw_var().names() == expected.raw_var().names(),
            "independent raw variable axis differs");
}

void require_equal_backed_view(const BackedAnnDataView& actual, const AnnData& expected) {
    require(actual.n_obs() == expected.n_obs() && actual.n_vars() == expected.n_vars(),
            "backed view dimensions differ");

    for (const auto name : {"group", "numeric_dictionary", "string_dictionary_na",
                            "string_dictionary_nan"}) {
        const auto actual_group = actual.obs().column(name);
        const auto expected_group = expected.obs().column(name);
        require(actual_group.kind() == ColumnKind::categorical &&
                actual_group.ordered() == expected_group.ordered(),
                "lazy categorical metadata differs");
        const auto actual_categories = actual_group.categories();
        const auto expected_categories = expected_group.categories();
        require(actual_categories.kind() == expected_categories.kind() &&
                actual_categories.nullable() == expected_categories.nullable(),
                "lazy category storage kind or nullable marker differs");
        auto actual_codes = actual_group.codes().read();
        require_equal_dense<std::int8_t>(actual_codes.view(), expected_group.codes(),
                                         "lazy categorical codes differ");
        auto materialized_group = actual_group.read();
        require_equal_categorical(materialized_group.view(), expected_group,
                                  "lazy categories differ in values, mask, or sentinel");
    }

    const auto actual_colors = actual.uns().at("group_colors").string_array();
    require_equal_strings(actual_colors, expected.uns().at("group_colors").string_array(),
                          "lazy categorical colors differ");

    auto read = actual.read();
    require_equal_data(read, expected);
    auto memory = actual.to_memory();
    require_equal_data(memory, expected);
}

void check_full_owner_without_slice(BackedAnnData& store, const AnnData& source) {
    auto expected = source.copy();
    require(expected.obs().column("group").categories().size() == 3,
            "source fixture did not retain its unused category");
    require(expected.uns().at("group_colors").string_array().size() == 3,
            "source fixture did not retain its complete color palette");
    auto view_read = store.view().read();
    require_equal_data(view_read, expected);
    auto owner_read = store.read();
    require_equal_data(owner_read, expected);
}

void check_slices(BackedAnnData& store, const AnnData& source) {
    auto check = [&](const BackedAnnDataView& actual, const AnnDataView& expected_view,
                     std::string_view name) {
        auto expected = expected_view.copy();
        try {
            require_equal_backed_view(actual, expected);
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(name) + ": " + error.what());
        }
    };

    check(store.slice(indices{2, 0}, all), source.slice(indices{2, 0}, all), "permuted rows");
    check(store.slice(indices{0}, all), source.slice(indices{0}, all), "single category row");
    check(store.slice(indices{1}, all), source.slice(indices{1}, all), "B-only row");
    check(store.slice(indices{}, all), source.slice(indices{}, all), "empty rows");
    check(store.slice(all, all), source.slice(all, all), "all selected rows");

    auto nested_actual = store.slice(indices{2, 0}, all).slice(indices{1, 0, 1}, all);
    auto nested_expected = source.slice(indices{2, 0}, all).slice(indices{1, 0, 1}, all);
    check(nested_actual, nested_expected, "nested repeated rows");

    const auto b_only = source.slice(indices{1}, all).copy();
    require(b_only.obs().column("group").code(0) == 0 &&
            b_only.obs().column("group").categories().string(0) == "B" &&
            b_only.uns().at("group_colors").string_array().string(0) == "blue",
            "in-memory B-only oracle does not demonstrate category recoding and color alignment");
}

template<class WriteEager, class Open, class WriteView>
void exercise_backend(const std::filesystem::path& run_directory,
                      std::string_view backend_name, std::string_view extension,
                      const AnnData& source, WriteEager write_eager,
                      Open open, WriteView write_view) {
    const auto source_path = run_directory / (std::string(backend_name) + "-source" +
                                               std::string(extension));
    const auto selected_path = run_directory / (std::string(backend_name) + "-selected" +
                                                 std::string(extension));
    write_eager(source_path, source);

    auto store = open(source_path, AccessMode::read_only);
    check_full_owner_without_slice(store, source);
    check_slices(store, source);

    auto expected_selected = source.slice(indices{2, 0}, all).copy();
    write_view(selected_path, store.slice(indices{2, 0}, all));
    auto selected = open(selected_path, AccessMode::read_only);
    require_equal_backed_view(selected.view(), expected_selected);
    selected.close();
    store.close();

    auto writable = open(source_path, AccessMode::read_write);
    auto edit = writable.edit_structure();
    edit.reindex(indices{2, 0}, all);
    edit.apply();
    writable.flush();
    writable.close();

    auto reindexed = open(source_path, AccessMode::read_only);
    require_equal_backed_view(reindexed.view(), expected_selected);
    reindexed.close();
}

#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5
void exercise_small_budget(const std::filesystem::path& run_directory) {
    constexpr std::size_t n = 32;
    auto source = AnnData::empty(n, 1);
    std::vector<std::string> names, categories;
    auto codes = Array::empty<std::int8_t>({n});
    auto values = codes.edit().dense<std::int8_t>();
    for (std::size_t i = 0; i < n; ++i) {
        names.push_back("c" + std::to_string(i));
        categories.push_back("v" + std::to_string(i));
        values.data()[i] = std::int8_t(i);
    }
    auto obs = Table::with_index(std::move(names));
    obs.set("group", Column::categorical(std::move(codes), Column::strings(std::move(categories)), true));
    source.set_obs(std::move(obs));
    const auto path = run_directory / "small-budget.h5ad";
    write_h5ad(path.string(), source);
    auto store = open_h5ad(path.string(), BackedOptions{.scratch_bytes = 256});
    auto selected = store.slice(range{std::nullopt, std::nullopt, -1}, all);
    auto nested = selected.slice(indices{31, 0, 15, 31}, all);
    auto expected = source.slice(range{std::nullopt, std::nullopt, -1}, all)
                          .slice(indices{31, 0, 15, 31}, all).copy();
    require_equal_categorical(nested.obs().column("group").read().view(),
                              expected.obs().column("group"), "small-budget nested category plan differs");
    try { (void)open_h5ad(path.string(), BackedOptions{.scratch_bytes = 63}); }
    catch (const Error& error) {
        require(error.code() == ErrorCode::invalid_argument, "small scratch budget reported wrong error");
        return;
    }
    throw std::runtime_error("scratch budget below 64 bytes was accepted");
}

void exercise_h5ad(const std::filesystem::path& run_directory, const AnnData& source) {
    exercise_backend(run_directory, "h5ad", ".h5ad", source,
        [](const auto& path, const AnnData& data) { write_h5ad(path.string(), data); },
        [](const auto& path, AccessMode mode) {
            return open_h5ad(path.string(), BackedOptions{.mode = mode});
        },
        [](const auto& path, const BackedAnnDataView& view) {
            write_h5ad(path.string(), view);
        });
}
#endif

#if defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
void exercise_zarr(const std::filesystem::path& run_directory, const AnnData& source,
                   ZarrFormat format, std::string_view format_name) {
    exercise_backend(run_directory, std::string("zarr-") + std::string(format_name), ".zarr",
                     source,
        [format](const auto& path, const AnnData& data) {
            ZarrWriteOptions options;
            options.format = format;
            options.compression = ZarrCompression::none;
            write_zarr(path.string(), data, options);
        },
        [](const auto& path, AccessMode mode) {
            return open_zarr(path.string(), BackedOptions{.mode = mode});
        },
        [format](const auto& path, const BackedAnnDataView& view) {
            ZarrWriteOptions options;
            options.format = format;
            options.compression = ZarrCompression::none;
            write_zarr(path.string(), view, options);
        });
}
#endif

} // namespace

int main(int argc, char** argv) try {
    const std::filesystem::path scratch = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "backed-categories-tests";
    std::filesystem::create_directories(scratch);
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto run_directory = scratch / ("run-" + std::to_string(nonce));
    std::filesystem::create_directories(run_directory);

    const auto source = fixture();
    bool exercised = false;
#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5
    exercise_h5ad(run_directory, source);
    exercise_small_budget(run_directory);
    exercised = true;
#endif
#if defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
    exercise_zarr(run_directory, source, ZarrFormat::v2, "v2");
    exercise_zarr(run_directory, source, ZarrFormat::v3, "v3");
    exercised = true;
#endif
    if (!exercised) std::cout << "backed category tests skipped: no storage backend enabled\n";
    else std::cout << "backed category tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "backed category tests failed: " << error.what() << '\n';
    return 1;
}
