#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(ANNDATA_TEST_HDF5)
#include <hdf5.h>
#include <sys/resource.h>
#endif

import anndata;

namespace {
using namespace anndata;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<class F>
void expect_error(ErrorCode code, F&& operation, std::string_view message) {
    try {
        operation();
    } catch (const Error& error) {
        if (error.code() != code)
            throw std::runtime_error(std::string(message) + ": unexpected error code");
        return;
    }
    throw std::runtime_error(std::string(message) + ": expected an error");
}

template<class T>
Array dense(std::initializer_list<std::size_t> shape,
            std::initializer_list<T> values) {
    auto result = Array::empty<T>(shape);
    auto output = result.edit().template dense<T>();
    std::size_t i = 0;
    for (const auto value : values) output.data()[i++] = value;
    std::size_t expected = 1;
    for (std::size_t axis = 0; axis < output.rank(); ++axis) expected *= output.extent(axis);
    require(i == expected, "fixture element count differs from its shape");
    return result;
}

template<class T>
Array csr(std::size_t rows, std::size_t columns,
          std::initializer_list<T> values,
          std::initializer_list<std::int32_t> indices,
          std::initializer_list<std::int64_t> offsets) {
    auto result = Array::csr<T, std::int32_t, std::int64_t>(rows, columns, values.size());
    auto output = result.edit().template csr<T, std::int32_t, std::int64_t>();
    std::size_t i = 0;
    for (const auto value : values) output.values()[i++] = value;
    i = 0;
    for (const auto value : indices) output.unsafe_indices()[i++] = value;
    i = 0;
    for (const auto value : offsets) output.unsafe_offsets()[i++] = value;
    result.validate();
    return result;
}

Array fixture_sparse() {
    return csr<std::int64_t>(3, 2, {1, 0, 2, 3, 4}, {1, 0, 0, 0, 1}, {0, 2, 4, 5});
}

AnnData fixture() {
    auto data = AnnData::empty(3, 2);
    data.set_x(dense<float>({3, 2}, {0, 1, 2, 3, 4, 5}));

    auto obs = Table::with_index({"cell-a", "cell-a", "cell-γ"}, "cell_id");
    obs.set("kind", Column::categorical(
        dense<std::int8_t>({3}, {0, 1, -1}), Column::strings({"T", "B", "unused"}), true));
    obs.set("kind_unsigned", Column::categorical(
        dense<std::uint8_t>({3}, {0, 1, 0}), Column::strings({"T", "B", "unused"}), false));
    obs.set("count", Column::nullable_numeric(
        dense<std::int16_t>({3}, {7, 0, -2}), {1, 0, 1}));
    obs.set("enabled", Column::nullable_numeric(
        dense<bool>({3}, {true, false, false}), {1, 0, 1}));
    obs.set("label", Column::nullable_strings({"α", "", "cell γ"}, {1, 0, 1}, StringMissingValue::NA));
    obs.set("label_nan", Column::nullable_strings({"left", "", "right"}, {1, 0, 1}, StringMissingValue::NaN));
    obs.set("unsigned", Column::from_values<std::uint8_t>({1, 255, 7}));
    data.set_obs(std::move(obs));

    auto var = Table::with_index({"gene-1", "gene-2"}, "gene_id");
    var.set("active", Column::from_values<bool>({true, false}));
    data.set_var(std::move(var));

    data.set_layer("counts", fixture_sparse());
    data.set_layer("empty", csr<std::int32_t>(3, 2, {}, {}, {0, 0, 0, 0}));
    data.set_layer("dense_layer", dense<std::uint16_t>({3, 2}, {0, 1, 2, 3, 4, 5}));
    data.set_obsm("embedding", dense<double>({3, 2}, {0, 1, 2, 3, 4, 5}));
    data.set_obsm("labels", StringArray::from_strings({3, 2}, {"a", "β", "", "d", "e", "f"}));
    auto table = Table::with_index({"cell-a", "cell-a", "cell-γ"});
    table.set("score", Column::from_values<double>({0.25, 0.5, 0.75}));
    data.set_obsm("table", std::move(table));
    data.set_varm("embedding", dense<std::int16_t>({2, 2}, {1, 2, 3, 4}));

    auto graph = Array::csr<float, std::int32_t, std::int32_t>(3, 3, 5);
    auto gv = graph.edit().csr<float, std::int32_t, std::int32_t>();
    const float graph_values[]{1, 0.5f, 1, 0.5f, 1};
    const std::int32_t graph_indices[]{0, 2, 1, 0, 2};
    const std::int32_t graph_offsets[]{0, 2, 3, 5};
    for (std::size_t i = 0; i < 5; ++i) {
        gv.values()[i] = graph_values[i];
        gv.unsafe_indices()[i] = graph_indices[i];
    }
    for (std::size_t i = 0; i < 4; ++i) gv.unsafe_offsets()[i] = graph_offsets[i];
    data.set_obsp("connectivities", std::move(graph));
    data.set_obsp("rank3", dense<float>({3, 3, 2}, {
        0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17}));

    auto similarity = Array::csc<double, std::int32_t, std::int32_t>(2, 2, 2);
    auto sv = similarity.edit().csc<double, std::int32_t, std::int32_t>();
    const double similarity_values[]{1, 1};
    const std::int32_t similarity_indices[]{0, 1};
    const std::int32_t similarity_offsets[]{0, 1, 2};
    for (std::size_t i = 0; i < 2; ++i) {
        sv.values()[i] = similarity_values[i];
        sv.unsafe_indices()[i] = similarity_indices[i];
    }
    for (std::size_t i = 0; i < 3; ++i) sv.unsafe_offsets()[i] = similarity_offsets[i];
    data.set_varp("similarity", std::move(similarity));

    auto nested = Element::mapping();
    nested.set("description", Element("nested α"));
    nested.set("null", Element(nullptr));
    nested.set("threshold", Element::scalar<float>(0.5f));
    auto dtypes = Element::mapping();
    dtypes.set("u64", Element(dense<std::uint64_t>(
        {2}, {std::uint64_t{1} << 40, (std::uint64_t{1} << 63) + 1})));
    data.set_uns("nested", std::move(nested));
    data.set_uns("dtypes", std::move(dtypes));
    data.set_uns("zero_numeric_array", Element(dense<std::int16_t>({}, {23})));
    data.set_uns("zero_string_array", Element(StringArray::from_strings({}, {"zero λ"})));
    data.set_uns("matrix_text", Element(StringArray::from_strings({2, 2}, {"μ", "", "g2", "gene β"})));

    auto wide = Array::csr<double, std::int32_t, std::int32_t>(1, 300, 1);
    auto wv = wide.edit().csr<double, std::int32_t, std::int32_t>();
    wv.values()[0] = 9.5;
    wv.unsafe_indices()[0] = 299;
    wv.unsafe_offsets()[0] = 0;
    wv.unsafe_offsets()[1] = 1;
    data.set_uns("wide_sparse", Element(std::move(wide)));

    data.set_raw(data.x().copy(), data.var().copy());
    auto raw_labels = Table::with_index({"gene-1", "gene-2"}, "gene_id");
    raw_labels.set("label", Column::strings({"raw α", "raw β"}));
    data.set_raw_varm("label_table", std::move(raw_labels));
    return data;
}

constexpr std::size_t rectangle_obs = 37;
constexpr std::size_t rectangle_vars = 69;
constexpr std::size_t rectangle_depth = 3;

float rectangle_value(std::size_t row, std::size_t column) {
    return static_cast<float>(row * 1000 + column);
}

float rectangle_cube_value(std::size_t row, std::size_t column, std::size_t depth) {
    return static_cast<float>((row * 1000 + column) * 4 + depth);
}

AnnData rectangular_fixture() {
    auto data = AnnData::empty(rectangle_obs, rectangle_vars);
    auto x = Array::empty<float>({rectangle_obs, rectangle_vars});
    auto x_values = x.edit().dense<float>();
    for (std::size_t row = 0; row < rectangle_obs; ++row)
        for (std::size_t column = 0; column < rectangle_vars; ++column)
            x_values.data()[row * rectangle_vars + column] = rectangle_value(row, column);
    data.set_x(std::move(x));

    std::vector<std::string> obs_names(rectangle_obs);
    std::vector<std::string> var_names(rectangle_vars);
    for (std::size_t i = 0; i < rectangle_obs; ++i) obs_names[i] = "rect-cell-" + std::to_string(i);
    for (std::size_t i = 0; i < rectangle_vars; ++i) var_names[i] = "rect-gene-" + std::to_string(i);
    data.set_obs(Table::with_index(std::move(obs_names), "cell_id"));
    data.set_var(Table::with_index(std::move(var_names), "gene_id"));

    auto cube = Array::empty<float>({rectangle_obs, rectangle_vars, rectangle_depth});
    auto cube_values = cube.edit().dense<float>();
    for (std::size_t row = 0; row < rectangle_obs; ++row)
        for (std::size_t column = 0; column < rectangle_vars; ++column)
            for (std::size_t depth = 0; depth < rectangle_depth; ++depth)
                cube_values.data()[(row * rectangle_vars + column) * rectangle_depth + depth] =
                    rectangle_cube_value(row, column, depth);
    data.set_uns("rectangular_cube", Element(std::move(cube)));
    data.set_raw(data.x().copy(), data.var().copy());
    return data;
}

void check_dense(ArrayView view, std::initializer_list<float> expected) {
    require(view.layout() == Layout::dense && view.dtype() == DType::float32,
            "expected float32 dense view");
    auto values = view.dense<float>();
    require(values.rank() == 2, "expected matrix rank two");
    require(values.extent(0) * values.extent(1) == expected.size(), "matrix element count differs");
    std::size_t i = 0;
    for (const auto value : expected) require(values.data()[i++] == value, "matrix values differ");
}

BackedAnnData open_store(std::string_view backend, const std::filesystem::path& path,
                         AccessMode mode = AccessMode::read_only,
                         std::size_t scratch_bytes = 64u * 1024u * 1024u) {
    BackedOptions options;
    options.mode = mode;
    options.scratch_bytes = scratch_bytes;
#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5
    if (backend == "h5ad") return open_h5ad(path.string(), options);
#endif
#if defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
    if (backend == "zarr") return open_zarr(path.string(), options);
#endif
    if (backend != "h5ad" && backend != "zarr")
        throw std::runtime_error("unknown backed test backend: " + std::string(backend));
#if !defined(ANNDATA_HAS_HDF5) || !ANNDATA_HAS_HDF5
    if (backend == "h5ad") throw std::runtime_error("HDF5 backend is disabled");
#endif
#if !defined(ANNDATA_HAS_ZARR) || !ANNDATA_HAS_ZARR
    if (backend == "zarr") throw std::runtime_error("Zarr backend is disabled");
#endif
    throw std::runtime_error("requested backed test backend is unavailable");
}

void check_rectangle(ArrayView view, std::size_t row_start, std::size_t column_start,
                     std::size_t rows, std::size_t columns) {
    require(view.layout() == Layout::dense && view.dtype() == DType::float32,
            "rectangular selection is not float32 dense data");
    const auto values = view.dense<float>();
    require(values.rank() == 2 && values.extent(0) == rows && values.extent(1) == columns,
            "rectangular selection shape differs");
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            require(values.data()[row * columns + column] ==
                        rectangle_value(row_start + row, column_start + column),
                    "rectangular selection values differ");
}

void check_cube(ArrayView view, std::size_t row_start, std::size_t column_start,
                std::size_t rows, std::size_t columns, std::size_t depth_start,
                std::size_t depth) {
    require(view.layout() == Layout::dense && view.dtype() == DType::float32,
            "rectangular rank-three selection is not float32 dense data");
    const auto values = view.dense<float>();
    require(values.rank() == 3 && values.extent(0) == rows && values.extent(1) == columns &&
                values.extent(2) == depth,
            "rectangular rank-three selection shape differs");
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            for (std::size_t layer = 0; layer < depth; ++layer) {
                const auto index = (row * columns + column) * depth + layer;
                require(values.data()[index] == rectangle_cube_value(
                            row_start + row, column_start + column, depth_start + layer),
                        "rectangular rank-three selection values differ");
            }
}

void check_read_contract(const std::filesystem::path& path, std::string_view backend) {
    auto store = open_store(backend, path);
    require(!store.writable(), "read-only open unexpectedly grants writes");
    require(store.n_obs() == 3 && store.n_vars() == 2, "backed axes differ");
    require(store.view().x().shape() == std::vector<std::size_t>({3, 2}), "lazy X shape differs");
    require(store.obs().index_name() == "cell_id" && store.var().index_name() == "gene_id",
            "lazy dataframe index names differ");
    require(store.obs().name(0) == "cell-a" && store.var().name(1) == "gene-2",
            "lazy axis point reads differ");
    auto selected_names = store.obs().select(range{0, 3, 2}).names();
    require(selected_names == std::vector<std::string>({"cell-a", "cell-γ"}),
            "lazy axis range reads differ");

    // A single-label selector deliberately returns all matching duplicate names.
    auto repeated = store.slice(Selection("cell-a"), all);
    require(repeated.n_obs() == 2, "singular label selection lost duplicate rows");
    check_dense(repeated.x().read().view(), {0, 1, 2, 3});
    expect_error(ErrorCode::invalid_argument,
        [&] { (void)store.slice(Selection(labels{"cell-a"}), all); },
        "label-list selection must reject a duplicate index");

    auto x = store.x();
    auto prepared = x.prepare_read();
    const auto required = prepared.requirements();
    require(required.layout == Layout::dense && required.dtype == DType::float32 &&
            required.shape == std::vector<std::size_t>({3, 2}),
            "dense prepared-read requirements differ");
    std::array<float, 6> caller{99, 99, 99, 99, 99, 99};
    auto buffer = DenseBuffer::from(caller.data(), {3, 2});
    auto* const original = buffer.data;
    prepared.read_into(buffer);
    require(buffer.data == original && caller == std::array<float, 6>{0, 1, 2, 3, 4, 5},
            "read_into did not fill the supplied caller buffer directly");

    std::array<float, 8> padded{99, 99, -777, 99, 99, -777, 99, 99};
    auto strided = DenseBuffer::from(padded.data(), {3, 2},
        {static_cast<std::ptrdiff_t>(3 * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))});
    x.read_into(strided);
    require(padded[0] == 0 && padded[1] == 1 && padded[3] == 2 && padded[4] == 3 &&
            padded[6] == 4 && padded[7] == 5 && padded[2] == -777 && padded[5] == -777,
            "strided caller buffer changed padding or misplaced selected values");

    std::array<float, 6> reversed{};
    auto negative_stride = DenseBuffer::from(reversed.data() + 4, {3, 2},
        {-static_cast<std::ptrdiff_t>(2 * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))});
    x.read_into(negative_stride);
    require(reversed == std::array<float, 6>{4, 5, 2, 3, 0, 1},
            "negative-stride caller buffer did not preserve logical row order");

    std::array<float, 4> range_values{};
    auto range_x = x.select({range{0, 3, 2}, range{0, 2, 1}});
    range_x.read_into(DenseBuffer::from(range_values.data(), {2, 2}));
    require(range_values == std::array<float, 4>{0, 1, 4, 5}, "compact range selection differs");
    expect_error(ErrorCode::dtype_mismatch,
        [&] {
            std::array<double, 6> wrong{};
            x.read_into(DenseBuffer::from(wrong.data(), {3, 2}));
        }, "dense read_into must reject dtype conversion");

    const auto sparse = store.layer("counts");
    const auto sparse_read = sparse.prepare_read();
    const auto sparse_req = sparse_read.requirements();
    require(sparse_req.layout == Layout::csr && sparse_req.dtype == DType::int64 &&
            sparse_req.index_dtype == DType::int32 && sparse_req.offset_dtype == DType::int64 &&
            sparse_req.shape == std::vector<std::size_t>({3, 2}) && sparse_req.nnz == 5,
            "prepared sparse representation differs");
    std::array<std::int64_t, 5> values{};
    std::array<std::int32_t, 5> indices{};
    std::array<std::int64_t, 4> offsets{};
    sparse_read.read_into(SparseBuffer{values.data(), indices.data(), offsets.data(),
        DType::int64, DType::int32, DType::int64, values.size(), indices.size(), offsets.size()});
    require(values == std::array<std::int64_t, 5>{1, 0, 2, 3, 4} &&
            indices == std::array<std::int32_t, 5>{1, 0, 0, 0, 1} &&
            offsets == std::array<std::int64_t, 4>{0, 2, 4, 5},
            "prepared sparse read lost duplicates, explicit zeros, or stored widths");

    const auto empty_sparse = store.layer("empty").prepare_read();
    const auto empty_req = empty_sparse.requirements();
    require(empty_req.nnz == 0 && empty_req.shape == std::vector<std::size_t>({3, 2}),
            "empty sparse requirements differ");
    std::array<std::uint16_t, 4> empty_offsets{99, 99, 99, 99};
    empty_sparse.read_into(SparseBuffer{nullptr, nullptr, empty_offsets.data(), DType::int32,
        DType::uint8, DType::uint16, 0, 0, empty_offsets.size()});
    require(empty_offsets == std::array<std::uint16_t, 4>{0, 0, 0, 0},
            "empty sparse read did not write zero offsets");

    const auto wide = store.uns().at("wide_sparse").array();
    const auto wide_read = wide.prepare_read();
    const auto wide_req = wide_read.requirements();
    require(wide_req.nnz == 1 && wide_req.shape == std::vector<std::size_t>({1, 300}),
            "wide sparse read requirements differ");
    double wide_value = 0;
    std::uint8_t narrow_index = 0;
    std::uint8_t narrow_offset[2]{};
    expect_error(ErrorCode::invalid_sparse_structure,
        [&] {
            wide_read.read_into(SparseBuffer{&wide_value, &narrow_index, narrow_offset,
                DType::float64, DType::uint8, DType::uint8, 1, 1, 2});
        }, "sparse output must reject an overflowing narrow index");
    std::uint16_t wide_index = 0;
    std::uint32_t wide_offsets[2]{};
    wide_read.read_into(SparseBuffer{&wide_value, &wide_index, wide_offsets,
        DType::float64, DType::uint16, DType::uint32, 1, 1, 2});
    require(wide_value == 9.5 && wide_index == 299 && wide_offsets[0] == 0 && wide_offsets[1] == 1,
            "explicitly widened sparse output differs");

    auto eager = store.read();
    check_dense(eager.x(), {0, 1, 2, 3, 4, 5});
    auto cat = eager.obs().column("kind");
    require(cat.kind() == ColumnKind::categorical && cat.ordered() && cat.code(2) == -1 &&
            cat.categories().size() == 3 && cat.categories().string(2) == "unused",
            "backed categorical metadata differs");
    auto nullable = eager.obs().column("count");
    require(nullable.nullable() && !nullable.valid(1) &&
            nullable.values().dense<std::int16_t>().data()[0] == 7,
            "backed nullable integer metadata differs");
    require(eager.obs().column("label").string_missing_value() == StringMissingValue::NA &&
            eager.obs().column("label_nan").string_missing_value() == StringMissingValue::NaN,
            "backed nullable string sentinels differ");
    require(eager.has_raw_x() && eager.raw_var().size() == 2 &&
            eager.raw_varm("label_table").table().column("label").string(0) == "raw α",
            "backed raw slots differ");
    auto numbers = eager.uns().at("dtypes").at("u64").array().dense<std::uint64_t>();
    require(numbers.data()[1] == (std::uint64_t{1} << 63) + 1,
            "large unsigned values changed during backed eager read");

    expect_error(ErrorCode::invalid_argument,
        [&] { x.write(dense<float>({3, 2}, {0, 1, 2, 3, 4, 5}).view()); },
        "read-only backed array must refuse a numerical write");
    auto readonly_view = store.view();
    expect_error(ErrorCode::invalid_argument,
        [&] { readonly_view.x().write(dense<float>({3, 2}, {0, 1, 2, 3, 4, 5}).view()); },
        "retained read-only data view must refuse a numerical write");

    // Array and view wrappers keep the backend session alive after the owning
    // BackedAnnData value leaves scope.
    BackedArray retained;
    {
        auto owner = open_store(backend, path);
        retained = owner.view().x();
    }
    check_dense(retained.read().view(), {0, 1, 2, 3, 4, 5});

    auto stale_after_close = store.x();
    store.close();
    expect_error(ErrorCode::invalid_argument, [&] { (void)stale_after_close.read(); },
        "retained array handle must become stale when the session closes");
}

#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5
void update_and_reindex_h5ad(const std::filesystem::path& source,
                            const std::filesystem::path& output) {
    write_h5ad(source.string(), fixture());
    std::filesystem::copy_file(source, output, std::filesystem::copy_options::overwrite_existing);
    auto store = open_h5ad(output.string(), BackedOptions{.mode = AccessMode::read_write});
    auto replaced = store.x();
    auto selected = store.x().select({indices{0}, indices{1}});
    auto patch = dense<float>({1, 1}, {11});
    selected.write(patch.view());
    require(store.x().read().view().dense<float>().data()[1] == 11,
            "direct numerical update is not visible through the retained store");

    auto edit = store.edit_structure();
    edit.reindex(indices{2, 0}, indices{1});
    edit.apply();
    expect_error(ErrorCode::invalid_argument, [&] { (void)replaced.read(); },
        "array handle affected by reindex must be stale");
    require(store.n_obs() == 2 && store.n_vars() == 1, "reindex axes differ");
    auto current = store.x().read();
    check_dense(current.view(), {5, 11});
    require(store.obs().names() == std::vector<std::string>({"cell-γ", "cell-a"}) &&
            store.var().names() == std::vector<std::string>({"gene-2"}),
            "reindexed table order differs");
    require(store.obsp("connectivities").shape() == std::vector<std::size_t>({2, 2}) &&
            store.varp("similarity").shape() == std::vector<std::size_t>({1, 1}),
            "reindex did not transform both pairwise axes");
    auto rank3_owner = store.obsp("rank3").read();
    const auto rank3 = rank3_owner.view().dense<float>();
    const std::array<float, 8> rank3_values{
        rank3.data()[0], rank3.data()[1], rank3.data()[2], rank3.data()[3],
        rank3.data()[4], rank3.data()[5], rank3.data()[6], rank3.data()[7]};
    require(rank3.rank() == 3 && rank3.extent(0) == 2 && rank3.extent(1) == 2 &&
            rank3.extent(2) == 2 && rank3_values == std::array<float, 8>{16, 17, 12, 13, 4, 5, 0, 1},
            "reindex did not transform the first two axes of rank-three pairwise data");
    check_dense(store.view().raw_x().read().view(), {4, 5, 0, 1});
    // The raw variable axis is independent of the main variable axis.
    require(store.view().raw_var().size() == 2, "reindex changed the raw variable axis");
    auto untouched = store.uns();
    require(untouched.at("nested").at("description").read().view().string() == "nested α",
            "unaffected metadata handle was invalidated");
    store.flush();
    store.close();

    auto reopened = open_h5ad(output.string());
    check_dense(reopened.x().read().view(), {5, 11});
    require(reopened.n_obs() == 2 && reopened.n_vars() == 1,
            "flushed structural update did not survive reopen");
}

void test_alignment_preflight(const std::filesystem::path& path) {
    write_h5ad(path.string(), fixture());
    auto store = open_h5ad(path.string(), BackedOptions{.mode = AccessMode::read_write});
    auto invalid = store.edit_structure();
    invalid.set_obs(Table::with_index({"new-0", "new-1", "new-2", "new-3"}, "cell_id"));
    expect_error(ErrorCode::shape_mismatch, [&] { invalid.apply(); },
        "axis growth without aligned-slot replacements must fail preflight");
    require(store.n_obs() == 3, "failed preflight changed the observation count");
    check_dense(store.x().read().view(), {0, 1, 2, 3, 4, 5});
    require(store.layer("counts").prepare_read().requirements().nnz == 5,
            "failed preflight changed a sparse layer");

    auto malformed = store.edit_structure();
    malformed.set_obs(Table::with_index({"bad-0", "bad-1", "bad-2", "bad-3"}, "cell_id"));
    malformed.set("/X", Element(dense<float>({3, 2}, {20, 21, 22, 23, 24, 25})));
    expect_error(ErrorCode::shape_mismatch, [&] { malformed.apply(); },
        "malformed replacement must fail before an axis or slot is mutated");
    require(store.n_obs() == 3, "malformed replacement changed the observation count");
    check_dense(store.x().read().view(), {0, 1, 2, 3, 4, 5});

    auto valid = store.edit_structure();
    valid.set_obs(Table::with_index({"new-0", "new-1", "new-2", "new-3"}, "cell_id"));
    valid.set("/X", Element(dense<float>({4, 2}, {10, 11, 12, 13, 14, 15, 16, 17})));
    valid.set("/layers/counts", Element(csr<std::int64_t>(
        4, 2, {5, 6}, {0, 1}, {0, 1, 1, 2, 2})));
    valid.erase("/layers/dense_layer");
    valid.erase("/layers/empty");
    valid.set("/obsm/embedding", Element(dense<double>(
        {4, 2}, {0, 1, 2, 3, 4, 5, 6, 7})));
    valid.set("/obsm/labels", Element(StringArray::from_strings(
        {4, 2}, {"p", "q", "r", "s", "t", "u", "v", "w"})));
    valid.erase("/obsm/table");
    valid.set("/obsp/connectivities", Element(dense<float>(
        {4, 4}, {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1})));
    valid.erase("/obsp/rank3");
    valid.set("/raw/X", Element(dense<float>({4, 2}, {10, 11, 12, 13, 14, 15, 16, 17})));
    valid.apply();
    require(store.n_obs() == 4 && store.x().extent(0) == 4,
            "valid axis growth with replacements did not commit");
    require(store.view().layer_names() == std::vector<std::string>({"counts"}) &&
            store.view().obsm_names() == std::vector<std::string>({"embedding", "labels"}),
            "structural deletions did not remove aligned paths");
    require(store.obsp("connectivities").shape() == std::vector<std::size_t>({4, 4}) &&
            store.view().raw_x().shape() == std::vector<std::size_t>({4, 2}),
            "structural replacements did not align pairwise/raw matrices");
    require(store.view().obsp_names() == std::vector<std::string>({"connectivities"}),
            "structural deletion did not remove the rank-three pairwise path");
    store.flush();
}

void test_same_store_swap(const std::filesystem::path& path) {
    auto data = AnnData::empty(2, 2);
    data.set_x(dense<float>({2, 2}, {0, 1, 2, 3}));
    data.set_obs(Table::with_index({"c0", "c1"}));
    data.set_var(Table::with_index({"g0", "g1"}));
    data.set_layer("left", dense<float>({2, 2}, {10, 11, 12, 13}));
    data.set_layer("right", dense<float>({2, 2}, {20, 21, 22, 23}));
    write_h5ad(path.string(), data);

    auto store = open_h5ad(path.string(), BackedOptions{.mode = AccessMode::read_write});
    auto left = store.layer("left");
    auto right = store.layer("right");
    auto edit = store.edit_structure();
    edit.set("/layers/left", right);
    edit.set("/layers/right", left);
    edit.apply();
    auto new_left = store.layer("left").read().view();
    auto new_right = store.layer("right").read().view();
    auto l = new_left.dense<float>();
    auto r = new_right.dense<float>();
    require(l.data()[0] == 20 && l.data()[3] == 23 &&
            r.data()[0] == 10 && r.data()[3] == 13,
            "same-store structural swap followed replaced paths instead of source snapshots");
}

void test_direct_column_order(const std::filesystem::path& h5ad_path,
                              const std::filesystem::path& zarr_path) {
    write_h5ad(h5ad_path.string(), fixture());
    auto h5_store = open_h5ad(h5ad_path.string(), BackedOptions{.mode = AccessMode::read_write});
    auto expected = h5_store.obs().column_names();
    h5_store.set("/obs/extra", Element(Column::from_values<std::int16_t>({17, 18, 19})));
    expected.push_back("extra");
    require(h5_store.obs().column_names() == expected,
            "direct HDF5 dataframe-column insertion did not append column order");
    h5_store.erase("/obs/count");
    expected.erase(std::find(expected.begin(), expected.end(), "count"));
    require(h5_store.obs().column_names() == expected,
            "direct HDF5 dataframe-column deletion did not update column order");
    require(h5_store.obs().column("extra").read().view().values().dense<std::int16_t>().data()[2] == 19,
            "direct HDF5 dataframe-column insertion lost values");
    h5_store.flush();
    h5_store.close();
    auto h5_reopened = open_h5ad(h5ad_path.string());
    require(h5_reopened.obs().column_names() == expected,
            "HDF5 dataframe-column order did not survive reopen");

#if defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
    ZarrWriteOptions options;
    options.format = ZarrFormat::v3;
    write_zarr(zarr_path.string(), fixture(), options);
    auto zarr_store = open_zarr(zarr_path.string(), BackedOptions{.mode = AccessMode::read_write});
    auto zarr_expected = zarr_store.obs().column_names();
    zarr_store.set("/obs/extra", Element(Column::from_values<std::int16_t>({17, 18, 19})));
    zarr_expected.push_back("extra");
    require(zarr_store.obs().column_names() == zarr_expected,
            "direct Zarr dataframe-column insertion did not append column order");
    zarr_store.erase("/obs/count");
    zarr_expected.erase(std::find(zarr_expected.begin(), zarr_expected.end(), "count"));
    require(zarr_store.obs().column_names() == zarr_expected,
            "direct Zarr dataframe-column deletion did not update column order");
    zarr_store.flush();
    zarr_store.close();
    auto zarr_reopened = open_zarr(zarr_path.string());
    require(zarr_reopened.obs().column_names() == zarr_expected,
            "Zarr dataframe-column order did not survive reopen");
#else
    (void)zarr_path;
#endif
}
#endif

#if defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
void test_zarr_native(const std::filesystem::path& root, ZarrFormat format) {
    const auto path = root / (format == ZarrFormat::v2 ? "native-v2.zarr" : "native-v3.zarr");
    ZarrWriteOptions options;
    options.format = format;
    write_zarr(path.string(), fixture(), options);
    auto store = open_zarr(path.string(), BackedOptions{.mode = AccessMode::read_write});
    require(store.n_obs() == 3 && store.n_vars() == 2,
            "native Zarr fixture axes differ");
    check_dense(store.x().read().view(), {0, 1, 2, 3, 4, 5});
    std::array<float, 2> block{};
    store.x().select({indices{2}, range{0, 2, 1}}).read_into(
        DenseBuffer::from(block.data(), {1, 2}));
    require(block == std::array<float, 2>{4, 5}, "Zarr selected block read differs");

    auto selection = store.x().select({indices{0}, indices{1}});
    selection.write(dense<float>({1, 1}, {31}).view());
    store.flush();
    store.close();
    auto reopened = open_zarr(path.string());
    check_dense(reopened.x().read().view(), {0, 31, 2, 3, 4, 5});
    require(reopened.uns().at("nested").at("description").read().view().string() == "nested α",
            "Zarr flush/reopen lost nested metadata");
}

void check_rectangular_transfers(BackedAnnData& store) {
    constexpr std::size_t row_start = 1;
    constexpr std::size_t column_start = 2;
    constexpr std::size_t rows = rectangle_obs - 2;
    constexpr std::size_t columns = rectangle_vars - 4;
    const auto selection = std::vector<Selection>{
        range{row_start, rectangle_obs - 1, 1},
        range{column_start, rectangle_vars - 2, 1}};

    const auto selected = store.x().select(selection);
    check_rectangle(selected.read().view(), row_start, column_start, rows, columns);
    std::vector<float> caller(rows * columns, -1);
    selected.read_into(DenseBuffer::from(caller.data(), {rows, columns}));
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            require(caller[row * columns + column] ==
                        rectangle_value(row + row_start, column + column_start),
                    "rectangular read_into values differ");

    const auto cube = store.uns().at("rectangular_cube").array().select({
        range{row_start, rectangle_obs - 1, 1},
        range{column_start, rectangle_vars - 2, 1},
        range{0, rectangle_depth, 1}});
    std::vector<float> cube_caller(rows * columns * rectangle_depth, -1);
    cube.read_into(DenseBuffer::from(cube_caller.data(), {rows, columns, rectangle_depth}));
    auto cube_copy = Array::empty<float>({rows, columns, rectangle_depth});
    auto cube_copy_values = cube_copy.edit().dense<float>();
    std::copy(cube_caller.begin(), cube_caller.end(), cube_copy_values.data());
    check_cube(cube_copy.view(), row_start, column_start, rows, columns, 0, rectangle_depth);
}

void reindex_rectangular_store(BackedAnnData& store) {
    constexpr std::size_t row_start = 1;
    constexpr std::size_t column_start = 2;
    constexpr std::size_t rows = rectangle_obs - 2;
    constexpr std::size_t columns = rectangle_vars - 4;
    auto edit = store.edit_structure();
    edit.reindex(range{row_start, rectangle_obs - 1, 1},
                 range{column_start, rectangle_vars - 2, 1});
    edit.apply();
    require(store.n_obs() == rows && store.n_vars() == columns,
            "rectangular reindex changed axis lengths incorrectly");
    check_rectangle(store.x().read().view(), row_start, column_start, rows, columns);
    check_cube(store.uns().at("rectangular_cube").array().read().view(),
               0, 0, rectangle_obs, rectangle_vars, 0, rectangle_depth);
    store.flush();
}

void check_rectangular_export(BackedAnnData& exported) {
    check_rectangle(exported.x().read().view(), 1, 2, rectangle_obs - 2, rectangle_vars - 4);
    check_rectangle(exported.view().raw_x().read().view(), 1, 0, rectangle_obs - 2, rectangle_vars);
    check_cube(exported.uns().at("rectangular_cube").array().read().view(),
               0, 0, rectangle_obs, rectangle_vars, 0, rectangle_depth);
}

void test_rectangular_zarr_batches(const std::filesystem::path& root, ZarrFormat format) {
    constexpr std::size_t scratch = 16u * 1024u;
    const auto suffix = format == ZarrFormat::v2 ? "v2" : "v3";
    const auto source_path = root / (std::string("rectangular-source-") + suffix + ".zarr");
    const auto export_path = root / (std::string("rectangular-export-") + suffix + ".zarr");
    ZarrWriteOptions write_options;
    write_options.format = format;
    write_options.scratch_bytes = scratch;
    write_zarr(source_path.string(), rectangular_fixture(), write_options);

    auto source = open_store("zarr", source_path, AccessMode::read_write, scratch);
    check_rectangular_transfers(source);
    const auto rectangle = source.view().slice(
        range{1, rectangle_obs - 1, 1}, range{2, rectangle_vars - 2, 1});
    write_zarr(export_path.string(), rectangle, write_options);
    auto exported = open_store("zarr", export_path, AccessMode::read_only, scratch);
    check_rectangular_export(exported);

    reindex_rectangular_store(source);
    source.close();
    auto reopened = open_store("zarr", source_path, AccessMode::read_only, scratch);
    check_rectangle(reopened.x().read().view(), 1, 2, rectangle_obs - 2, rectangle_vars - 4);
}

void test_zarr_bool_canonicalization(const std::filesystem::path& root, ZarrFormat format) {
    const auto path = root / (format == ZarrFormat::v2 ? "bool-byte-v2.zarr" : "bool-byte-v3.zarr");
    auto data = AnnData::empty(2, 2);
    data.set_x(dense<bool>({2, 2}, {false, false, false, false}));
    data.set_obs(Table::with_index({"c0", "c1"}));
    data.set_var(Table::with_index({"g0", "g1"}));
    ZarrWriteOptions write_options;
    write_options.format = format;
    write_options.compression = ZarrCompression::none;
    write_zarr(path.string(), data, write_options);

    const auto chunk = format == ZarrFormat::v2 ? path / "X" / "0.0" : path / "X" / "c" / "0" / "0";
    std::fstream bytes(chunk, std::ios::in | std::ios::out | std::ios::binary);
    require(bytes.good(), "could not open uncompressed Zarr bool chunk");
    bytes.seekp(2);
    const char noncanonical_true = 2;
    bytes.write(&noncanonical_true, 1);
    bytes.close();
    require(std::filesystem::file_size(chunk) >= 3, "uncompressed Zarr bool chunk is too short");

    auto full = open_zarr(path.string());
    auto selected = full.x().select({indices{1}, indices{0}}).read();
    require(selected.view().bytes().data()[0] == std::byte{1},
            "full-validation selected Zarr bool was not canonicalized to byte one");

    BackedOptions unchecked_options;
    unchecked_options.validation = Validation::none;
    auto unchecked = open_zarr(path.string(), unchecked_options);
    auto raw = unchecked.x().read();
    require(raw.view().bytes().data()[2] == std::byte{2},
            "Validation::none changed the stored Zarr bool byte");
}
#endif

#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5 && defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
void test_rectangular_h5ad_batches(const std::filesystem::path& root) {
    constexpr std::size_t scratch = 16u * 1024u;
    const auto source_path = root / "rectangular-source.h5ad";
    const auto export_path = root / "rectangular-export-from-h5ad.zarr";
    write_h5ad(source_path.string(), rectangular_fixture());

    auto source = open_store("h5ad", source_path, AccessMode::read_write, scratch);
    check_rectangular_transfers(source);
    const auto rectangle = source.view().slice(
        range{1, rectangle_obs - 1, 1}, range{2, rectangle_vars - 2, 1});
    ZarrWriteOptions options;
    options.format = ZarrFormat::v3;
    options.scratch_bytes = scratch;
    write_zarr(export_path.string(), rectangle, options);
    auto exported = open_store("zarr", export_path, AccessMode::read_only, scratch);
    check_rectangular_export(exported);

    reindex_rectangular_store(source);
    source.close();
    auto reopened = open_store("h5ad", source_path, AccessMode::read_only, scratch);
    check_rectangle(reopened.x().read().view(), 1, 2, rectangle_obs - 2, rectangle_vars - 4);
}
#endif

#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5 && defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
void test_cross_backend_export(const std::filesystem::path& source,
                               const std::filesystem::path& target) {
    write_h5ad(source.string(), fixture());
    auto backed = open_h5ad(source.string());
    ZarrWriteOptions options;
    options.format = ZarrFormat::v3;
    options.scratch_bytes = 64u * 1024u;
    write_zarr(target.string(), backed.view(), options);
    auto reopened = open_zarr(target.string());
    check_dense(reopened.x().read().view(), {0, 1, 2, 3, 4, 5});
    require(reopened.layer("counts").prepare_read().requirements().nnz == 5,
            "streamed cross-backend export changed sparse storage");
}
#endif

#if defined(ANNDATA_TEST_HDF5) && defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5
void h5_check(herr_t result, std::string_view operation) {
    if (result < 0) throw std::runtime_error("HDF5 test fixture failed: " + std::string(operation));
}

void h5_string_attr(hid_t object, const char* name, const char* value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    h5_check(type < 0 ? -1 : 0, "copy string type");
    h5_check(H5Tset_size(type, H5T_VARIABLE), "set variable string attribute");
    h5_check(H5Tset_cset(type, H5T_CSET_UTF8), "set UTF-8 string attribute");
    hid_t space = H5Screate(H5S_SCALAR);
    h5_check(space < 0 ? -1 : 0, "create string attribute space");
    hid_t attr = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    h5_check(attr < 0 ? -1 : 0, "create string attribute");
    const char* pointer = value;
    h5_check(H5Awrite(attr, type, &pointer), "write string attribute");
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
}

hid_t make_chunked_dataset(hid_t file, const char* path, hid_t type,
                           std::vector<hsize_t> shape, std::vector<hsize_t> chunk,
                           const void* fill) {
    hid_t space = H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr);
    h5_check(space < 0 ? -1 : 0, "create huge dataspace");
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    h5_check(dcpl < 0 ? -1 : 0, "create huge dataset properties");
    h5_check(H5Pset_chunk(dcpl, static_cast<int>(chunk.size()), chunk.data()), "set huge dataset chunks");
    h5_check(H5Pset_fill_value(dcpl, type, fill), "set fill value");
    h5_check(H5Pset_fill_time(dcpl, H5D_FILL_TIME_IFSET), "set fill-time policy");
    h5_check(H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_INCR), "set incremental allocation");
    hid_t dataset = H5Dcreate2(file, path, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    h5_check(dataset < 0 ? -1 : 0, "create huge dataset");
    H5Pclose(dcpl);
    H5Sclose(space);
    return dataset;
}

void replace_string_index(hid_t file, const char* path, std::size_t length) {
    h5_check(H5Ldelete(file, path, H5P_DEFAULT), "replace axis index link");
    hid_t type = H5Tcopy(H5T_C_S1);
    h5_check(type < 0 ? -1 : 0, "copy axis string type");
    h5_check(H5Tset_size(type, 16), "set axis string width");
    h5_check(H5Tset_cset(type, H5T_CSET_UTF8), "set axis string encoding");
    std::array<char, 16> fill{};
    auto dataset = make_chunked_dataset(file, path, type,
        {static_cast<hsize_t>(length)}, {1024}, fill.data());
    h5_string_attr(dataset, "encoding-type", "string-array");
    h5_string_attr(dataset, "encoding-version", "0.2.0");
    H5Dclose(dataset);
    H5Tclose(type);
}

void make_huge_fill_h5ad(const std::filesystem::path& path) {
    auto data = AnnData::empty(1, 1);
    data.set_obs(Table::with_index({"c0"}, "cell_id"));
    data.set_var(Table::with_index({"g0"}, "gene_id"));
    data.set_x(dense<float>({1, 1}, {0}));
    write_h5ad(path.string(), data);

    constexpr std::size_t extent = 8192;
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    h5_check(file < 0 ? -1 : 0, "open fixture for sparse allocation");
    h5_check(H5Ldelete(file, "/X", H5P_DEFAULT), "replace fixture X");
    const float zero = 0;
    auto x = make_chunked_dataset(file, "/X", H5T_IEEE_F32LE,
        {extent, extent}, {256, 256}, &zero);
    h5_string_attr(x, "encoding-type", "array");
    h5_string_attr(x, "encoding-version", "0.2.0");
    require(H5Dget_storage_size(x) == 0, "huge fill X unexpectedly owns physical chunks");
    H5Dclose(x);
    replace_string_index(file, "/obs/cell_id", extent);
    replace_string_index(file, "/var/gene_id", extent);
    h5_check(H5Fflush(file, H5F_SCOPE_GLOBAL), "flush fill-only file");
    H5Fclose(file);
    require(std::filesystem::file_size(path) < 1024u * 1024u,
            "logical fill fixture allocated unexpected on-disk payload");
}

long max_rss_kib() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
    return usage.ru_maxrss;
}

void test_bounded_fill_reads(const std::filesystem::path& root) {
    const auto path = root / "huge-fill.h5ad";
    make_huge_fill_h5ad(path);
    BackedOptions options;
    options.scratch_bytes = 64u * 1024u;
    options.decoded_cache_bytes = 0;
    auto store = open_h5ad(path.string(), options);
    require(store.n_obs() == 8192 && store.n_vars() == 8192 &&
            store.x().shape() == std::vector<std::size_t>({8192, 8192}),
            "huge logical fill axes differ");
    std::array<float, 4> values{9, 9, 9, 9};
    const auto before = max_rss_kib();
    store.x().select({range{1000, 1002, 1}, range{2000, 2002, 1}}).read_into(
        DenseBuffer::from(values.data(), {2, 2}));
    const auto after = max_rss_kib();
    require(values == std::array<float, 4>{0, 0, 0, 0}, "fill-valued region read differs");
    require(after - before < 96 * 1024,
            "small backed block read materialized the 256 MiB logical matrix");
    store.close();

    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    h5_check(file < 0 ? -1 : 0, "reopen fill-only file");
    hid_t x = H5Dopen2(file, "/X", H5P_DEFAULT);
    h5_check(x < 0 ? -1 : 0, "reopen fill-only X");
    require(H5Dget_storage_size(x) == 0, "selected read allocated physical fill chunks");
    H5Dclose(x);
    H5Fclose(file);
}

void test_hdf5_mapping(const std::filesystem::path& path) {
    static_assert(std::is_same_v<decltype(std::declval<MappedArray>().unsafe_dense<float>()), const float*>);
    auto store = open_h5ad(path.string());
    auto mapping = store.x().map_readonly();
    require(mapping.shape() == std::vector<std::size_t>({3, 2}) &&
            mapping.byte_size() == 6 * sizeof(float),
            "eligible HDF5 mapping shape or byte count differs");
    const auto* values = mapping.unsafe_dense<float>();
    require(values[0] == 0 && values[5] == 5, "typed const mapping differs on Linux/Clang");
    store.close();
    require(mapping.unsafe_data<float>()[3] == 3,
            "mapped bytes did not retain their read-only mapping lifetime after close");
}
#endif
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected a scratch directory");
        const std::filesystem::path root(argv[1]);
        std::filesystem::create_directories(root);

#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5
        const auto native_h5ad = root / "native.h5ad";
        write_h5ad(native_h5ad.string(), fixture());
        check_read_contract(native_h5ad, "h5ad");
#if defined(ANNDATA_TEST_HDF5) && defined(__linux__) && defined(__clang__)
        test_hdf5_mapping(native_h5ad);
#endif
        update_and_reindex_h5ad(root / "source.h5ad", root / "update-reindex.h5ad");
        test_alignment_preflight(root / "alignment.h5ad");
        test_same_store_swap(root / "same-store-swap.h5ad");
        test_direct_column_order(root / "column-order.h5ad", root / "column-order.zarr");
#endif

#if defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
        test_zarr_native(root, ZarrFormat::v2);
        test_zarr_native(root, ZarrFormat::v3);
        test_rectangular_zarr_batches(root, ZarrFormat::v2);
        test_rectangular_zarr_batches(root, ZarrFormat::v3);
        test_zarr_bool_canonicalization(root, ZarrFormat::v2);
        test_zarr_bool_canonicalization(root, ZarrFormat::v3);
#endif
#if defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5 && defined(ANNDATA_HAS_ZARR) && ANNDATA_HAS_ZARR
        test_rectangular_h5ad_batches(root);
        test_cross_backend_export(root / "export-source.h5ad", root / "export-target.zarr");
#endif

#if defined(ANNDATA_TEST_HDF5) && defined(ANNDATA_HAS_HDF5) && ANNDATA_HAS_HDF5
        test_bounded_fill_reads(root);
#endif
        std::cout << "native backed H5AD/Zarr tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
