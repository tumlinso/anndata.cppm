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

using namespace anndata;

namespace {
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

ErrorCode parse_code(std::string_view code) {
    for (ErrorCode value : {ErrorCode::invalid_argument, ErrorCode::out_of_bounds,
                            ErrorCode::dtype_mismatch, ErrorCode::unsupported_layout,
                            ErrorCode::shape_mismatch, ErrorCode::invalid_sparse_structure,
                            ErrorCode::non_contiguous, ErrorCode::io_error,
                            ErrorCode::unsupported_encoding, ErrorCode::unsupported_version,
                            ErrorCode::unsupported_dtype, ErrorCode::unavailable_filter})
        if (code_name(value) == code) return value;
    throw std::runtime_error("unknown expected error code");
}

template<class T>
Array vector_array(std::initializer_list<T> values) {
    auto result = Array::empty<T>({values.size()});
    auto view = result.edit().template dense<T>();
    std::size_t i = 0;
    for (T value : values) view.data()[i++] = value;
    return result;
}

template<class T>
Array matrix_array(std::size_t rows, std::size_t columns,
                   std::initializer_list<T> values) {
    auto result = Array::empty<T>({rows, columns});
    auto view = result.edit().template dense<T>();
    std::size_t i = 0;
    for (T value : values) view.data()[i++] = value;
    return result;
}

Array sparse_fixture() {
    auto result = Array::csr<std::int64_t, std::int32_t, std::int64_t>(3, 2, 5);
    auto view = result.edit().csr<std::int64_t, std::int32_t, std::int64_t>();
    const std::int64_t values[]{1, 0, 2, 3, 4};
    const std::int32_t indices[]{1, 0, 0, 0, 1};
    const std::int64_t offsets[]{0, 2, 4, 5};
    for (std::size_t i = 0; i < 5; ++i) {
        view.values()[i] = values[i];
        view.unsafe_indices()[i] = indices[i];
    }
    for (std::size_t i = 0; i < 4; ++i) view.unsafe_offsets()[i] = offsets[i];
    result.validate();
    return result;
}

Table label_table(std::initializer_list<std::string_view> labels) {
    auto table = Table::with_index({"gene-1", "gene-2"}, "gene_id");
    table.set("label", Column::strings(labels));
    return table;
}

AnnData fixture(bool big_endian_x = false) {
    auto result = AnnData::empty(3, 2);
    auto x = matrix_array<float>(3, 2, {0, 1, 2, 3, 4, 5});
    if (big_endian_x) x.set_serialization_byte_order(ByteOrder::big);
    result.set_x(std::move(x));

    auto obs = Table::with_index({"cell-a", "cell-a", "cell-γ"}, "cell_id");
    auto categories = Column::strings({"T", "B", "unused"});
    obs.set("kind", Column::categorical(vector_array<std::int8_t>({0, 1, -1}),
                                          std::move(categories), true));
    obs.set("kind_unsigned", Column::categorical(
        vector_array<std::uint8_t>({0, 1, 0}), Column::strings({"T", "B", "unused"}), false));
    obs.set("count", Column::nullable_numeric(
        vector_array<std::int16_t>({7, 0, -2}), {1, 0, 1}));
    obs.set("enabled", Column::nullable_numeric(
        vector_array<bool>({true, false, false}), {1, 0, 1}));
    obs.set("label", Column::nullable_strings(
        {"α", "", "cell γ"}, {1, 0, 1}, StringMissingValue::NA));
    obs.set("label_nan", Column::nullable_strings(
        {"left", "", "right"}, {1, 0, 1}, StringMissingValue::NaN));
    obs.set("unsigned", Column::from_values<std::uint8_t>({1, 255, 7}));
    result.set_obs(std::move(obs));

    auto var = Table::with_index({"gene-1", "gene-2"}, "gene_id");
    var.set("active", Column::from_values<bool>({true, false}));
    result.set_var(std::move(var));

    result.set_layer("counts", sparse_fixture());
    result.set_layer("dense_layer", matrix_array<std::uint16_t>(3, 2, {0, 1, 2, 3, 4, 5}));
    result.set_obsm("embedding", matrix_array<double>(3, 2, {0, 1, 2, 3, 4, 5}));
    result.set_obsm("labels", StringArray::from_strings(
        {3, 2}, {"a", "β", "", "d", "e", "f"}));
    auto obs_table = Table::with_index({"cell-a", "cell-a", "cell-γ"});
    obs_table.set("score", Column::from_values<double>({0.25, 0.5, 0.75}));
    result.set_obsm("table", std::move(obs_table));
    result.set_varm("embedding", matrix_array<std::int16_t>(2, 2, {1, 2, 3, 4}));

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
    result.set_obsp("connectivities", std::move(graph));

    auto varp = Array::csc<double, std::int32_t, std::int32_t>(2, 2, 2);
    auto vp = varp.edit().csc<double, std::int32_t, std::int32_t>();
    const double vp_values[]{1, 1};
    const std::int32_t vp_indices[]{0, 1};
    const std::int32_t vp_offsets[]{0, 1, 2};
    for (std::size_t i = 0; i < 2; ++i) {
        vp.values()[i] = vp_values[i];
        vp.unsafe_indices()[i] = vp_indices[i];
    }
    for (std::size_t i = 0; i < 3; ++i) vp.unsafe_offsets()[i] = vp_offsets[i];
    result.set_varp("similarity", std::move(varp));

    auto dtypes = Element::mapping();
    dtypes.set("bool", Element(vector_array<bool>({true, false})));
    dtypes.set("i8", Element(vector_array<std::int8_t>({-2, 3})));
    dtypes.set("u8", Element(vector_array<std::uint8_t>({2, 250})));
    dtypes.set("i16", Element(vector_array<std::int16_t>({-300, 1000})));
    dtypes.set("u16", Element(vector_array<std::uint16_t>({1000, 60000})));
    dtypes.set("i32", Element(vector_array<std::int32_t>({-100000, 200000})));
    dtypes.set("u32", Element(vector_array<std::uint32_t>({100000, 4000000000U})));
    dtypes.set("i64", Element(vector_array<std::int64_t>({-(std::int64_t{1} << 40), std::int64_t{1} << 40})));
    dtypes.set("u64", Element(vector_array<std::uint64_t>({std::uint64_t{1} << 40, (std::uint64_t{1} << 63) + 1})));
    dtypes.set("f32", Element(vector_array<float>({0.5f, 1.25f})));
    dtypes.set("f64", Element(vector_array<double>({3.141592653589793, -1.5})));

    auto nested = Element::mapping();
    nested.set("description", Element("nested α"));
    nested.set("null", Element(nullptr));
    nested.set("threshold", Element::scalar<float>(0.5f));
    auto uns = Element::mapping();
    uns.set("nested", std::move(nested));
    auto zero_numeric = Array::empty<std::int16_t>({});
    zero_numeric.edit().dense<std::int16_t>().data()[0] = 23;
    uns.set("zero_numeric_array", Element(std::move(zero_numeric)));
    uns.set("zero_string_array", Element(StringArray::from_strings({}, {"zero λ"})));
    uns.set("matrix_text", Element(StringArray::from_strings(
        {2, 2}, {"μ", "", "g2", "gene β"})));
    uns.set("dtypes", std::move(dtypes));
    result.set_uns(std::move(uns));

    result.set_raw(result.x().copy(), result.var().copy());
    result.set_raw_varm("label_table", label_table({"raw α", "raw β"}));
    return result;
}

template<class T>
void expect_dense(ArrayView array, std::initializer_list<T> expected) {
    require(array.layout() == Layout::dense, "expected dense array");
    auto view = array.dense<T>();
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < view.rank(); ++axis) count *= view.extent(axis);
    require(count == expected.size(), "dense element count differs");
    std::size_t i = 0;
    for (T value : expected) require(view.data()[i++] == value, "dense value differs");
}

void verify_fixture(const AnnData& data, bool foreign_endian) {
    require(data.n_obs() == 3 && data.n_vars() == 2, "shape differs");
    expect_dense<float>(data.x(), {0, 1, 2, 3, 4, 5});
    require(data.x().serialization_byte_order() ==
            (foreign_endian ? ByteOrder::big : ByteOrder::native), "X byte order hint differs");

    auto obs = data.obs();
    require(obs.index_name() == "cell_id", "obs index name differs");
    require(obs.name(0) == "cell-a" && obs.name(1) == "cell-a" && obs.name(2) == "cell-γ",
            "obs index values differ");
    auto kind = obs.column("kind");
    require(kind.kind() == ColumnKind::categorical && kind.ordered(), "categorical metadata differs");
    require(kind.categories().size() == 3 && kind.categories().string(2) == "unused",
            "categorical dictionary differs");
    require(kind.code(0) == 0 && kind.code(1) == 1 && kind.code(2) == -1,
            "categorical codes differ");
    auto kind_unsigned = obs.column("kind_unsigned");
    require(kind_unsigned.codes().dtype() == DType::uint8 &&
            kind_unsigned.code(0) == 0 && kind_unsigned.code(1) == 1 &&
            kind_unsigned.code(2) == 0,
            "unsigned categorical code width or values differ");

    auto count = obs.column("count");
    require(count.nullable() && count.valid(0) && !count.valid(1) && count.valid(2),
            "nullable integer mask differs");
    expect_dense<std::int16_t>(count.values(), {7, 0, -2});
    auto enabled = obs.column("enabled");
    require(enabled.nullable() && !enabled.valid(1), "nullable boolean mask differs");
    auto label = obs.column("label");
    require(label.valid(0) && !label.valid(1) && label.valid(2), "nullable string mask differs");
    require(label.string_missing_value() == StringMissingValue::NA,
            "nullable string NA sentinel differs");
    require(label.string(0) == "α" && label.string(2) == "cell γ", "UTF-8 column differs");
    auto label_nan = obs.column("label_nan");
    require(label_nan.string_missing_value() == StringMissingValue::NaN &&
            !label_nan.valid(1), "nullable string NaN sentinel differs");

    auto counts_view = data.layer("counts");
    require(counts_view.layout() == Layout::csr && counts_view.extent(0) == 3 &&
            counts_view.extent(1) == 2, "sparse layer layout/shape differs");
    auto counts = counts_view.csr<std::int64_t, std::int32_t, std::int32_t>();
    const std::int64_t data_expected[]{1, 0, 2, 3, 4};
    const std::int32_t indices_expected[]{1, 0, 0, 0, 1};
    const std::int32_t offsets_expected[]{0, 2, 4, 5};
    for (std::size_t i = 0; i < 5; ++i) {
        require(counts.values()[i] == data_expected[i], "sparse values differ");
        require(counts.indices()[i] == indices_expected[i], "sparse indices differ");
    }
    for (std::size_t i = 0; i < 4; ++i)
        require(counts.offsets()[i] == offsets_expected[i], "sparse offsets differ");
    require(counts_view.serialization_byte_order() == ByteOrder::native ||
            counts_view.serialization_byte_order() == ByteOrder::little,
            "sparse data byte order differs");

    auto labels = data.obsm("labels").string_array();
    require(labels.rank() == 2 && labels.extent(0) == 3 && labels.extent(1) == 2,
            "obsm string array shape differs");
    require(labels.string(0) == "a" && labels.string(1) == "β" &&
            labels.string(2).empty() && labels.string(5) == "f", "obsm strings differ");
    auto matrix_text = data.uns().at("matrix_text").string_array();
    require(matrix_text.rank() == 2 && matrix_text.extent(0) == 2 &&
            matrix_text.extent(1) == 2 && matrix_text.string(3) == "gene β",
            "nested string array differs");
    auto raw = data.raw_x();
    expect_dense<float>(raw, {0, 1, 2, 3, 4, 5});
    require(data.has_raw_x() && data.raw_var().size() == 2, "raw axis differs");
    require(data.raw_varm("label_table").table().column("label").string(0) == "raw α",
            "raw varm differs");
    require(data.uns().at("nested").at("null").kind() == ElementKind::null,
            "null scalar differs");
    require(data.uns().at("nested").at("threshold").scalar<float>() == 0.5f,
            "numeric scalar differs");
    require(data.uns().at("zero_numeric_array").kind() == ElementKind::signed_integer &&
            data.uns().at("zero_numeric_array").scalar<std::int16_t>() == 23,
            "Python zero-dimensional numeric array scalarization differs");
    require(data.uns().at("zero_string_array").kind() == ElementKind::string &&
            data.uns().at("zero_string_array").string() == "zero λ",
            "Python zero-dimensional string array scalarization differs");

    auto unsigned_values = data.uns().at("dtypes").at("u64").array();
    require(unsigned_values.dtype() == DType::uint64, "uint64 dtype differs");
    expect_dense<std::uint64_t>(unsigned_values, {std::uint64_t{1} << 40, (std::uint64_t{1} << 63) + 1});
    const auto dtypes = data.uns().at("dtypes");
    expect_dense<bool>(dtypes.at("bool").array(), {true, false});
    expect_dense<std::int8_t>(dtypes.at("i8").array(), {-2, 3});
    expect_dense<std::uint8_t>(dtypes.at("u8").array(), {2, 250});
    expect_dense<std::int16_t>(dtypes.at("i16").array(), {-300, 1000});
    expect_dense<std::uint16_t>(dtypes.at("u16").array(), {1000, 60000});
    expect_dense<std::int32_t>(dtypes.at("i32").array(), {-100000, 200000});
    expect_dense<std::uint32_t>(dtypes.at("u32").array(), {100000, 4000000000U});
    expect_dense<std::int64_t>(dtypes.at("i64").array(), {-(std::int64_t{1} << 40), std::int64_t{1} << 40});
    expect_dense<float>(dtypes.at("f32").array(), {0.5f, 1.25f});
    expect_dense<double>(dtypes.at("f64").array(), {3.141592653589793, -1.5});
}

void write_fixture(const std::filesystem::path& path, bool gzip) {
    auto data = fixture(gzip);
    WriteOptions options;
    if (gzip) options.compression = Compression::gzip;
    write_h5ad(path.string(), data, options);
    auto reopened = read_h5ad(path.string());
    const auto zero_numeric = reopened.uns().at("zero_numeric_array");
    require(zero_numeric.kind() == ElementKind::array &&
            zero_numeric.array().rank() == 0 &&
            zero_numeric.array().dense<std::int16_t>().data()[0] == 23,
            "rank-zero numeric Array must retain array kind and value");
    const auto zero_strings = reopened.uns().at("zero_string_array");
    require(zero_strings.kind() == ElementKind::string_array &&
            zero_strings.string_array().rank() == 0 &&
            zero_strings.string_array().string(0) == "zero λ",
            "rank-zero StringArray must retain string-array kind and value");
}

void verify_byte_order(const std::filesystem::path& input,
                       const std::filesystem::path& preserve,
                       const std::filesystem::path& native) {
    auto data = read_h5ad(input.string());
    require(data.x().serialization_byte_order() == ByteOrder::big,
            "dense X byte-order hint differs");
    expect_dense<float>(data.x(), {0, 1, 2, 3, 4, 5});

    auto counts = data.layer("counts");
    require(counts.serialization_byte_order() == ByteOrder::big &&
            counts.index_serialization_byte_order() == ByteOrder::big &&
            counts.offset_serialization_byte_order() == ByteOrder::big,
            "sparse data/index/offset byte-order hints differ");
    auto sparse = counts.csr<std::int64_t, std::int32_t, std::int32_t>();
    const std::int64_t expected_data[]{1, 0, 2, 3, 4};
    const std::int32_t expected_indices[]{1, 0, 0, 0, 1};
    const std::int32_t expected_offsets[]{0, 2, 4, 5};
    for (std::size_t i = 0; i < 5; ++i) {
        require(sparse.values()[i] == expected_data[i], "foreign-endian sparse values differ");
        require(sparse.indices()[i] == expected_indices[i], "foreign-endian sparse indices differ");
    }
    for (std::size_t i = 0; i < 4; ++i)
        require(sparse.offsets()[i] == expected_offsets[i], "foreign-endian sparse offsets differ");

    const auto threshold = data.uns().at("nested").at("threshold");
    require(threshold.serialization_byte_order() == ByteOrder::big &&
            threshold.scalar<float>() == 0.5f,
            "numeric scalar must load natively and retain its foreign-endian hint");
    write_h5ad(preserve.string(), data);
    WriteOptions native_options;
    native_options.byte_order = FileByteOrder::native;
    write_h5ad(native.string(), data, native_options);
}

void expect_read_error(const std::filesystem::path& input, ErrorCode expected,
                       std::string_view path_fragment) {
    try {
        (void)read_h5ad(input.string());
    } catch (const Error& error) {
        require(error.code() == expected,
                "read failed with a different error code than expected");
        require(error.path().find(path_fragment) != std::string_view::npos,
                "read error did not identify the expected storage path");
        return;
    }
    throw std::runtime_error("read unexpectedly succeeded instead of returning expected error");
}

void normalize_malformed_boolean(const std::filesystem::path& input,
                                 const std::filesystem::path& output) {
    auto data = read_h5ad(input.string());
    const auto enabled = data.obs().column("enabled");
    require(enabled.valid(0) && enabled.values().dense<bool>().data()[0],
            "nonzero malformed HDF5 enum value must canonicalize to true");
    write_h5ad(output.string(), data);
}

void verify_compact_indices(const std::filesystem::path& input,
                            const std::filesystem::path& output) {
    auto data = read_h5ad(input.string());
    const auto& uns = data.uns();
    auto small = uns.at("compact8").array();
    require(small.layout() == Layout::csr && small.index_dtype() == DType::uint8 &&
            small.offset_dtype() == DType::int32,
            "compact uint8 sparse index width differs");
    auto small_sparse = small.csr<double, std::uint8_t, std::int32_t>();
    require(small_sparse.values()[0] == 3.0 && small_sparse.indices()[0] == 1 &&
            small_sparse.offsets()[2] == 1,
            "compact uint8 sparse buffers differ");
    auto medium = uns.at("compact16").array();
    require(medium.layout() == Layout::csr && medium.index_dtype() == DType::uint16 &&
            medium.offset_dtype() == DType::int32,
            "compact uint16 sparse index width differs");
    auto medium_sparse = medium.csr<double, std::uint16_t, std::int32_t>();
    require(medium_sparse.values()[0] == 5.0 && medium_sparse.indices()[0] == 299 &&
            medium_sparse.offsets()[1] == 1,
            "compact uint16 sparse buffers differ");
    write_h5ad(output.string(), data);
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("missing command");
        const std::string_view command(argv[1]);
        if (command == "verify") {
            if (argc != 4) throw std::runtime_error("verify requires input and output paths");
            auto data = read_h5ad(argv[2]);
            verify_fixture(data, true);
            write_h5ad(argv[3], data);
        } else if (command == "copy") {
            if (argc != 4) throw std::runtime_error("copy requires input and output paths");
            auto data = read_h5ad(argv[2]);
            write_h5ad(argv[3], data);
        } else if (command == "read-none") {
            if (argc != 3) throw std::runtime_error("read-none requires an input path");
            ReadOptions options;
            options.validation = Validation::none;
            (void)read_h5ad(argv[2], options);
        } else if (command == "expect-read-error") {
            if (argc != 5)
                throw std::runtime_error("expect-read-error requires file, code, and path fragment");
            expect_read_error(argv[2], parse_code(argv[3]), argv[4]);
        } else if (command == "read") {
            if (argc != 3) throw std::runtime_error("read requires an input path");
            (void)read_h5ad(argv[2]);
        } else if (command == "write" || command == "write-gzip") {
            if (argc != 3) throw std::runtime_error("write requires an output path");
            write_fixture(argv[2], command == "write-gzip");
        } else if (command == "byte-order") {
            if (argc != 5)
                throw std::runtime_error("byte-order requires input, preserve, and native paths");
            verify_byte_order(argv[2], argv[3], argv[4]);
        } else if (command == "normalize-bool") {
            if (argc != 4) throw std::runtime_error("normalize-bool requires input and output paths");
            normalize_malformed_boolean(argv[2], argv[3]);
        } else if (command == "compact-indices") {
            if (argc != 4) throw std::runtime_error("compact-indices requires input and output paths");
            verify_compact_indices(argv[2], argv[3]);
        } else if (command == "write-no-x-raw") {
            if (argc != 3) throw std::runtime_error("write-no-x-raw requires an output path");
            auto data = AnnData::empty(1, 1);
            data.set_raw(Table::with_index({"raw-gene"}, "gene_id"));
            write_h5ad(argv[2], data);
            auto reopened = read_h5ad(argv[2]);
            require(!reopened.has_x() && reopened.has_raw() && !reopened.has_raw_x(),
                    "raw without X did not survive a native read/write cycle");
        } else if (command == "write-invalid-sparse") {
            if (argc != 4)
                throw std::runtime_error("write-invalid-sparse requires output path and full|none");
            auto data = AnnData::empty(2, 2);
            auto sparse = Array::csr<float, std::int32_t, std::int32_t>(2, 2, 1);
            auto view = sparse.edit().csr<float, std::int32_t, std::int32_t>();
            view.values()[0] = 1.0f;
            view.unsafe_indices()[0] = 0;
            view.unsafe_offsets()[0] = 0;
            view.unsafe_offsets()[1] = 1;
            view.unsafe_offsets()[2] = 1;
            data.set_x(std::move(sparse));
            data.edit_x().csr<float, std::int32_t, std::int32_t>().unsafe_indices()[0] = 9;
            WriteOptions options;
            if (std::string_view(argv[3]) == "none") options.validation = Validation::none;
            else if (std::string_view(argv[3]) != "full")
                throw std::runtime_error("validation must be full or none");
            write_h5ad(argv[2], data, options);
        } else if (command == "exclusive") {
            if (argc != 3) throw std::runtime_error("exclusive requires an output path");
            auto data = fixture();
            WriteOptions options;
            options.mode = WriteMode::exclusive;
            write_h5ad(argv[2], data, options);
        } else if (command == "write-bad-key") {
            if (argc != 3) throw std::runtime_error("write-bad-key requires an output path");
            auto data = AnnData::empty(0, 0);
            data.set_uns("bad/key", Element(std::int64_t{1}));
            write_h5ad(argv[2], data);
        } else if (command == "write-nul-key") {
            if (argc != 3) throw std::runtime_error("write-nul-key requires an output path");
            auto data = AnnData::empty(0, 0);
            data.set_uns(std::string("bad\0key", 7), Element(std::int64_t{1}));
            write_h5ad(argv[2], data);
        } else if (command == "write-nul-string") {
            if (argc != 3) throw std::runtime_error("write-nul-string requires an output path");
            auto data = AnnData::empty(0, 0);
            data.set_uns("bad", Element(std::string("bad\0string", 10)));
            write_h5ad(argv[2], data);
        } else if (command == "write-bad-index") {
            if (argc != 3) throw std::runtime_error("write-bad-index requires an output path");
            auto data = AnnData::empty(2, 1);
            auto obs = Table::with_index({"a", "b"}, "cell_id");
            obs.set("cell_id", Column::strings({"x", "y"}));
            data.set_obs(std::move(obs));
            data.set_var(Table::with_index({"gene"}, "gene_id"));
            write_h5ad(argv[2], data);
        } else if (command == "write-bad-index-name") {
            if (argc != 3) throw std::runtime_error("write-bad-index-name requires an output path");
            auto data = AnnData::empty(1, 1);
            data.set_obs(Table::with_index({"cell"}, "_index"));
            write_h5ad(argv[2], data);
        } else {
            throw std::runtime_error("unknown command");
        }
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
