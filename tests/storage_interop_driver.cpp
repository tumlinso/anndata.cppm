#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

std::string_view code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::invalid_argument: return "invalid_argument";
        case ErrorCode::out_of_bounds: return "out_of_bounds";
        case ErrorCode::dtype_mismatch: return "dtype_mismatch";
        case ErrorCode::unsupported_layout: return "unsupported_layout";
        case ErrorCode::shape_mismatch: return "shape_mismatch";
        case ErrorCode::invalid_sparse_structure: return "invalid_sparse_structure";
        case ErrorCode::non_contiguous: return "non_contiguous";
        case ErrorCode::io_error: return "io_error";
        case ErrorCode::unsupported_encoding: return "unsupported_encoding";
        case ErrorCode::unsupported_version: return "unsupported_version";
        case ErrorCode::unsupported_dtype: return "unsupported_dtype";
        case ErrorCode::unavailable_filter: return "unavailable_filter";
    }
    return "unknown";
}

template<class T>
Array dense(std::initializer_list<std::size_t> shape,
            std::initializer_list<T> values) {
    auto result = Array::empty<T>(shape);
    auto output = result.edit().template dense<T>();
    std::size_t i = 0;
    for (const auto value : values) output.data()[i++] = value;
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < output.rank(); ++axis) count *= output.extent(axis);
    require(i == count, "fixture element count differs from its shape");
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

    data.set_layer("counts", csr<std::int64_t>(3, 2,
        {1, 0, 2, 3, 4}, {1, 0, 0, 0, 1}, {0, 2, 4, 5}));
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
    data.set_uns("nested", std::move(nested));
    auto dtypes = Element::mapping();
    dtypes.set("u64", Element(dense<std::uint64_t>(
        {2}, {std::uint64_t{1} << 40, (std::uint64_t{1} << 63) + 1})));
    data.set_uns("dtypes", std::move(dtypes));
    data.set_uns("zero_numeric_array", Element(dense<std::int16_t>({}, {23})));
    data.set_uns("zero_string_array", Element(StringArray::from_strings({}, {"zero λ"})));
    data.set_uns("matrix_text", Element(StringArray::from_strings({2, 2}, {"μ", "", "g2", "gene β"})));
    data.set_raw(data.x().copy(), data.var().copy());
    auto raw_labels = Table::with_index({"gene-1", "gene-2"}, "gene_id");
    raw_labels.set("label", Column::strings({"raw α", "raw β"}));
    data.set_raw_varm("label_table", std::move(raw_labels));
    return data;
}

BackedAnnData open(std::string_view backend, const std::filesystem::path& path,
                   AccessMode mode = AccessMode::read_only,
                   std::size_t scratch_bytes = 64u * 1024u * 1024u) {
    BackedOptions options;
    options.mode = mode;
    options.scratch_bytes = scratch_bytes;
    if (backend == "h5ad") return open_h5ad(path.string(), options);
    if (backend == "zarr") return open_zarr(path.string(), options);
    throw std::runtime_error("backend must be h5ad or zarr");
}

std::uint64_t index_value(const std::byte* source, DType type) {
    switch (type) {
        case DType::int8: { std::int8_t v; std::memcpy(&v, source, sizeof v); require(v >= 0, "negative sparse index"); return v; }
        case DType::uint8: { std::uint8_t v; std::memcpy(&v, source, sizeof v); return v; }
        case DType::int16: { std::int16_t v; std::memcpy(&v, source, sizeof v); require(v >= 0, "negative sparse index"); return v; }
        case DType::uint16: { std::uint16_t v; std::memcpy(&v, source, sizeof v); return v; }
        case DType::int32: { std::int32_t v; std::memcpy(&v, source, sizeof v); require(v >= 0, "negative sparse index"); return v; }
        case DType::uint32: { std::uint32_t v; std::memcpy(&v, source, sizeof v); return v; }
        case DType::int64: { std::int64_t v; std::memcpy(&v, source, sizeof v); require(v >= 0, "negative sparse index"); return v; }
        case DType::uint64: { std::uint64_t v; std::memcpy(&v, source, sizeof v); return v; }
        default: throw std::runtime_error("sparse index is not integer typed");
    }
}

void check_sparse(const BackedArray& array) {
    const auto prepared = array.prepare_read();
    const auto r = prepared.requirements();
    require(r.layout == Layout::csr && r.dtype == DType::int64 && r.nnz == 5 &&
            r.shape == std::vector<std::size_t>({3, 2}), "sparse counts requirements differ");
    std::vector<std::int64_t> values(r.nnz);
    std::vector<std::byte> indices(r.nnz * dtype_size(r.index_dtype));
    std::vector<std::byte> offsets((r.shape[0] + 1) * dtype_size(r.offset_dtype));
    prepared.read_into(SparseBuffer{values.data(), indices.data(), offsets.data(), r.dtype,
        r.index_dtype, r.offset_dtype, values.size(), values.size(), r.shape[0] + 1});
    require(values == std::vector<std::int64_t>({1, 0, 2, 3, 4}),
            "sparse counts values differ");
    const std::uint64_t expected_indices[]{1, 0, 0, 0, 1};
    const std::uint64_t expected_offsets[]{0, 2, 4, 5};
    for (std::size_t i = 0; i < values.size(); ++i)
        require(index_value(indices.data() + i * dtype_size(r.index_dtype), r.index_dtype) == expected_indices[i],
                "sparse counts index buffer differs");
    for (std::size_t i = 0; i < r.shape[0] + 1; ++i)
        require(index_value(offsets.data() + i * dtype_size(r.offset_dtype), r.offset_dtype) == expected_offsets[i],
                "sparse counts offset buffer differs");
}

void check_initial(const BackedAnnDataView& data) {
    require(data.n_obs() == 3 && data.n_vars() == 2 && data.has_x(), "backed axes or X presence differs");
    require(data.obs().index_name() == "cell_id" && data.var().index_name() == "gene_id",
            "dataframe index names differ");
    require(data.obs().names() == std::vector<std::string>({"cell-a", "cell-a", "cell-γ"}) &&
            data.var().names() == std::vector<std::string>({"gene-1", "gene-2"}),
            "axis labels differ");
    std::array<float, 6> x{};
    data.x().read_into(DenseBuffer::from(x.data(), {3, 2}));
    require(x == std::array<float, 6>{0, 1, 2, 3, 4, 5}, "X values differ");

    const auto kind = data.obs().column("kind");
    require(kind.kind() == ColumnKind::categorical && kind.ordered() &&
            kind.codes().dtype() == DType::int8,
            "ordered signed categorical metadata differs");
    const auto kind_values = kind.read().view();
    require(kind_values.code(0) == 0 && kind_values.code(1) == 1 && kind_values.code(2) == -1 &&
            kind_values.categories().string(2) == "unused",
            "ordered categorical codes or category order differ");
    const auto unsigned_kind = data.obs().column("kind_unsigned");
    require(unsigned_kind.codes().dtype() == DType::uint8 && !unsigned_kind.ordered(),
            "unsigned categorical storage width/order flag differs");
    const auto count = data.obs().column("count").read().view();
    require(count.nullable() && count.valid(0) && !count.valid(1) && count.valid(2) &&
            count.values().dense<std::int16_t>().data()[2] == -2,
            "nullable integer values or mask differ");
    const auto enabled = data.obs().column("enabled").read().view();
    require(enabled.nullable() && !enabled.valid(1) &&
            enabled.values().dense<bool>().data()[0],
            "nullable boolean values or mask differ");
    const auto label = data.obs().column("label").read().view();
    const auto label_nan = data.obs().column("label_nan").read().view();
    require(label.string_missing_value() == StringMissingValue::NA && !label.valid(1) &&
            label.string(0) == "α" && label_nan.string_missing_value() == StringMissingValue::NaN &&
            !label_nan.valid(1) && label_nan.string(2) == "right",
            "nullable UTF-8 string values or NA/NaN sentinels differ");
    require(data.obs().column("unsigned").values().dtype() == DType::uint8,
            "unsigned numeric metadata dtype differs");

    check_sparse(data.layer("counts"));
    require(data.layer("dense_layer").dtype() == DType::uint16,
            "dense layer dtype differs");
    const auto labels = data.obsm("labels").string_array();
    require(labels.shape() == std::vector<std::size_t>({3, 2}) &&
            labels.string(1) == "β" && labels.string(5) == "f",
            "multidimensional UTF-8 observation metadata differs");
    require(data.obsm("table").table().column("score").read().view().values().dense<double>().data()[2] == 0.75,
            "aligned observation dataframe differs");
    require(data.varm("embedding").array().read().view().dense<std::int16_t>().data()[3] == 4,
            "variable-metadata array differs");
    require(data.obsp("connectivities").shape() == std::vector<std::size_t>({3, 3}) &&
            data.varp("similarity").shape() == std::vector<std::size_t>({2, 2}),
            "pairwise matrix shapes differ");
    require(data.has_raw() && data.has_raw_x() && data.raw_var().size() == 2 &&
            data.raw_varm("label_table").table().column("label").read().view().string(1) == "raw β",
            "raw slot shape or metadata differs");

    const auto uns = data.uns();
    require(uns.at("nested").at("description").read().view().string() == "nested α" &&
            uns.at("nested").at("null").kind() == ElementKind::null &&
            uns.at("nested").at("threshold").read().view().scalar<float>() == 0.5f,
            "nested scalar/null metadata differs");
    const auto u64 = uns.at("dtypes").at("u64").array().read().view().dense<std::uint64_t>();
    require(u64.data()[1] == (std::uint64_t{1} << 63) + 1,
            "uint64 metadata was narrowed");
}

void check_mutated(const BackedAnnDataView& data) {
    require(data.n_obs() == 2 && data.n_vars() == 1, "mutated axes differ");
    std::array<float, 2> x{};
    data.x().read_into(DenseBuffer::from(x.data(), {2, 1}));
    require(x == std::array<float, 2>{5, 11}, "mutated/reindexed X differs");
    require(data.obs().names() == std::vector<std::string>({"cell-γ", "cell-a"}) &&
            data.var().names() == std::vector<std::string>({"gene-2"}),
            "mutated axis order differs");
    const auto graph = data.obsp("connectivities").read();
    const auto graph_bytes = graph.view().sparse_bytes();
    require(graph_bytes.rows() == 2 && graph_bytes.columns() == 2,
            "reindexed observation pairwise matrix differs");
    require(data.obsm("embedding").array().shape() == std::vector<std::size_t>({2, 2}) &&
            data.obsm("labels").string_array().shape() == std::vector<std::size_t>({2, 2}),
            "reindexed multidimensional metadata differs");
    require(data.raw_x().shape() == std::vector<std::size_t>({2, 2}) &&
            data.raw_var().size() == 2,
            "raw row alignment or independent raw variable axis differs");
}

void inspect(std::string_view backend, const std::filesystem::path& path) {
    auto store = open(backend, path);
    check_initial(store.view());
    auto repeated = store.slice(Selection("cell-a"), all);
    require(repeated.n_obs() == 2, "singular label selector failed to include duplicate names");
    auto block = store.x().select({range{0, 3, 2}, all});
    require(block.shape() == std::vector<std::size_t>({2, 2}), "compact range selection shape differs");
}

void mutate(std::string_view backend, const std::filesystem::path& path) {
    auto store = open(backend, path, AccessMode::read_write);
    auto stale = store.x();
    auto one_cell = store.x().select({indices{0}, indices{1}});
    one_cell.write(dense<float>({1, 1}, {11}).view());
    auto structure = store.edit_structure();
    structure.reindex(indices{2, 0}, indices{1});
    structure.apply();
    try {
        (void)stale.read();
    } catch (const Error& error) {
        require(error.code() == ErrorCode::invalid_argument, "replaced X handle has wrong stale error");
        store.flush();
        store.close();
        auto reopened = open(backend, path);
        check_mutated(reopened.view());
        return;
    }
    throw std::runtime_error("X handle remained usable after structural replacement");
}

void write_file(std::string_view backend, const std::filesystem::path& path) {
    auto data = fixture();
    if (backend == "h5ad") {
        write_h5ad(path.string(), data);
    } else {
        ZarrWriteOptions options;
        options.format = backend == "zarr2" ? ZarrFormat::v2 : ZarrFormat::v3;
        write_zarr(path.string(), data, options);
    }
}

void export_file(std::string_view source_backend, const std::filesystem::path& source_path,
                 std::string_view target_backend, const std::filesystem::path& target_path) {
    auto source = open(source_backend, source_path);
    if (target_backend == "h5ad") {
        write_h5ad(target_path.string(), source.view());
    } else {
        ZarrWriteOptions options;
        options.format = target_backend == "zarr2" ? ZarrFormat::v2 : ZarrFormat::v3;
        options.scratch_bytes = 64u * 1024u * 1024u;
        write_zarr(target_path.string(), source.view(), options);
    }
}

constexpr std::size_t rectangle_scratch_bytes = 16u * 1024u;

BackedAnnDataView select_boundary_rectangle(const BackedAnnData& source) {
    require(source.n_obs() > 2 && source.n_vars() > 4,
            "boundary rectangle requires at least three observations and five variables");
    // Exercise a forward rectangle with nonzero starts: obs [1, n_obs-1), var [2, n_vars-2).
    return source.view().slice(range{1, source.n_obs() - 1, 1},
                               range{2, source.n_vars() - 2, 1});
}

void export_rectangle(std::string_view source_backend, const std::filesystem::path& source_path,
                      std::string_view target_backend, const std::filesystem::path& target_path) {
    auto source = open(source_backend, source_path, AccessMode::read_only, rectangle_scratch_bytes);
    const auto rectangle = select_boundary_rectangle(source);
    if (target_backend == "h5ad") {
        write_h5ad(target_path.string(), rectangle);
    } else if (target_backend == "zarr2" || target_backend == "zarr3") {
        ZarrWriteOptions options;
        options.format = target_backend == "zarr2" ? ZarrFormat::v2 : ZarrFormat::v3;
        options.scratch_bytes = rectangle_scratch_bytes;
        write_zarr(target_path.string(), rectangle, options);
    } else {
        throw std::runtime_error("rectangle export target must be h5ad, zarr2, or zarr3");
    }
}

void reindex_rectangle(std::string_view backend, const std::filesystem::path& path) {
    auto store = open(backend, path, AccessMode::read_write, rectangle_scratch_bytes);
    (void)select_boundary_rectangle(store);
    auto structure = store.edit_structure();
    structure.reindex(range{1, store.n_obs() - 1, 1},
                      range{2, store.n_vars() - 2, 1});
    structure.apply();
    store.flush();
    store.close();
}

void read_file(std::string_view backend, const std::filesystem::path& path) {
    auto source = open(backend, path);
    (void)source.read();
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("missing command");
        const std::string_view command(argv[1]);
        if (command == "inspect") {
            if (argc != 4) throw std::runtime_error("inspect requires backend and path");
            inspect(argv[2], argv[3]);
        } else if (command == "mutate") {
            if (argc != 4) throw std::runtime_error("mutate requires backend and path");
            mutate(argv[2], argv[3]);
        } else if (command == "write") {
            if (argc != 4) throw std::runtime_error("write requires target format and path");
            write_file(argv[2], argv[3]);
        } else if (command == "export") {
            if (argc != 6) throw std::runtime_error("export requires source backend/path and target format/path");
            export_file(argv[2], argv[3], argv[4], argv[5]);
        } else if (command == "export-rectangle") {
            if (argc != 6) throw std::runtime_error("export-rectangle requires source backend/path and target format/path");
            export_rectangle(argv[2], argv[3], argv[4], argv[5]);
        } else if (command == "reindex-rectangle") {
            if (argc != 4) throw std::runtime_error("reindex-rectangle requires backend and path");
            reindex_rectangle(argv[2], argv[3]);
        } else if (command == "read") {
            if (argc != 4) throw std::runtime_error("read requires backend and path");
            read_file(argv[2], argv[3]);
        } else {
            throw std::runtime_error("unknown storage driver command");
        }
        return 0;
    } catch (const Error& error) {
        std::cerr << "anndata error[" << code_name(error.code()) << "] ";
        if (!error.path().empty()) std::cerr << error.path() << ": ";
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
