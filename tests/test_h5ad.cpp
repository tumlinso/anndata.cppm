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

import anndatapp;

namespace {
void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<class T>
anndatapp::Array dense(std::initializer_list<std::size_t> shape,
                     std::initializer_list<T> values) {
    auto result = anndatapp::Array::empty<T>(shape);
    auto access = result.edit().template dense<T>();
    std::size_t i = 0;
    for (T value : values) access.data()[i++] = value;
    return result;
}

anndatapp::AnnData fixture() {
    using namespace anndatapp;
    auto data = AnnData::empty(3, 2);
    data.set_x(dense<float>({3, 2}, {1, 2, 3, 4, 5, 6}));

    auto obs = Table::with_index({"c0", "c1", "c2"}, "cell_id");
    obs.set("label", Column::nullable_strings(
        {"α", "", "γ"}, {1, 0, 1}, StringMissingValue::NA));
    obs.set("group", Column::categorical(
        dense<std::int8_t>({3}, {0, -1, 1}), Column::strings({"one", "two", "unused"}), true));
    data.set_obs(std::move(obs));
    data.set_var(Table::with_index({"g0", "g1"}, "gene_id"));

    auto sparse = Array::csc<float, std::int32_t, std::int64_t>(3, 2, 3);
    auto sv = sparse.edit().csc<float, std::int32_t, std::int64_t>();
    const float values[]{1, 0, 4};
    const std::int32_t indices[]{0, 2, 1};
    const std::int64_t offsets[]{0, 2, 3};
    for (std::size_t i = 0; i < 3; ++i) {
        sv.values()[i] = values[i];
        sv.unsafe_indices()[i] = indices[i];
    }
    for (std::size_t i = 0; i < 3; ++i) sv.unsafe_offsets()[i] = offsets[i];
    sparse.validate();
    data.set_layer("sparse", std::move(sparse));
    data.set_obsm("coords", dense<double>({3, 2}, {0, .5, 1, 1.5, 2, 2.5}));
    data.set_varm("loadings", dense<float>({2, 1}, {.25f, .75f}));
    data.set_obsp("graph", dense<float>({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}));
    data.set_varp("pair", dense<double>({2, 2}, {1, 0, 0, 1}));

    auto info = Element::mapping();
    info.set("none", Element(nullptr));
    info.set("title", Element("UTF-8 μ"));
    info.set("number", Element::scalar<std::uint16_t>(257));
    auto zero_numeric = Array::empty<std::int16_t>({});
    zero_numeric.edit().dense<std::int16_t>().data()[0] = 23;
    info.set("zero_numeric_array", Element(std::move(zero_numeric)));
    info.set("zero_string_array", Element(StringArray::from_strings({}, {"zero λ"})));
    info.set("strings", Element(StringArray::from_strings(
        {2}, {"plain", "λ"})));
    data.set_uns("info", std::move(info));
    data.set_raw(data.x().copy(), data.var().copy());
    return data;
}

void check(const anndatapp::AnnData& data) {
    using namespace anndatapp;
    require(data.n_obs() == 3 && data.n_vars() == 2, "round-trip shape differs");
    require(data.x().dtype() == DType::float32, "X dtype differs");
    auto x = data.x().dense<float>();
    const float expected[]{1, 2, 3, 4, 5, 6};
    for (std::size_t i = 0; i < 6; ++i) require(x.data()[i] == expected[i], "X values differ");
    require(data.obs().index_name() == "cell_id", "obs index name differs");
    require(data.obs().column("label").string(0) == "α", "UTF-8 string differs");
    require(!data.obs().column("label").valid(1), "nullable string mask differs");
    auto group = data.obs().column("group");
    require(group.ordered() && group.categories().size() == 3 && group.code(1) == -1,
            "categorical data differs");
    auto sparse = data.layer("sparse");
    require(sparse.layout() == Layout::csc && sparse.index_dtype() == DType::int32 &&
            sparse.offset_dtype() == DType::int64, "sparse orientation or widths differ");
    auto sv = sparse.csc<float, std::int32_t, std::int64_t>();
    const float values[]{1, 0, 4};
    for (std::size_t i = 0; i < 3; ++i) require(sv.values()[i] == values[i], "sparse values differ");
    auto strings = data.uns().at("info").at("strings").string_array();
    require(strings.extent(0) == 2 && strings.string(1) == "λ", "nested string array differs");
    require(data.uns().at("info").at("number").scalar<std::uint16_t>() == 257,
            "unsigned scalar width/value differs");
    auto zero_numeric = data.uns().at("info").at("zero_numeric_array");
    require(zero_numeric.kind() == ElementKind::array && zero_numeric.array().rank() == 0 &&
            zero_numeric.array().dense<std::int16_t>().data()[0] == 23,
            "rank-zero numeric array kind, shape, or value differs");
    auto zero_strings = data.uns().at("info").at("zero_string_array");
    require(zero_strings.kind() == ElementKind::string_array &&
            zero_strings.string_array().rank() == 0 &&
            zero_strings.string_array().string(0) == "zero λ",
            "rank-zero string-array kind, shape, or value differs");
    require(data.has_raw_x(), "raw X missing");
}

void check_trusted_validation(const std::filesystem::path& path) {
    using namespace anndatapp;
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

    bool full_write_rejected = false;
    try { write_h5ad(path.string(), data); }
    catch (const Error& error) {
        full_write_rejected = error.code() == ErrorCode::invalid_sparse_structure &&
                              error.path().find("/X") != std::string_view::npos;
    }
    require(full_write_rejected, "full-validation write must reject malformed sparse indices at /X");

    WriteOptions trusted;
    trusted.validation = Validation::none;
    write_h5ad(path.string(), data, trusted);
    bool full_read_rejected = false;
    try { (void)read_h5ad(path.string()); }
    catch (const Error& error) {
        full_read_rejected = error.code() == ErrorCode::invalid_sparse_structure &&
                             error.path().find("/X") != std::string_view::npos;
    }
    require(full_read_rejected, "full-validation read must reject malformed sparse indices at /X");

    ReadOptions trusted_read;
    trusted_read.validation = Validation::none;
    auto bypassed = read_h5ad(path.string(), trusted_read);
    require(bypassed.x().layout() == Layout::csr,
            "Validation::none must permit trusted sparse data without changing its layout");
}

void check_reserved_index_name(const std::filesystem::path& path) {
    using namespace anndatapp;
    auto data = AnnData::empty(1, 1);
    data.set_obs(Table::with_index({"cell"}, "_index"));
    bool rejected = false;
    try { write_h5ad(path.string(), data); }
    catch (const Error& error) {
        rejected = error.code() == ErrorCode::invalid_argument &&
                   error.path().find("/obs") != std::string_view::npos;
    }
    require(rejected, "explicit dataframe index name _index must fail without lossy round-trip");
}
void check_tiled_strides(const std::filesystem::path& path) {
    using namespace anndatapp;
    constexpr std::size_t n = 1048576/sizeof(std::int32_t)+3;
    std::vector<std::int32_t> values(4*n);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = std::int32_t(i);
    auto reversed = Array::borrow_dense(values.data()+n-1,{2,2,n},
        {std::ptrdiff_t(2*n),std::ptrdiff_t(n),-1});
    reversed.set_serialization_byte_order(ByteOrder::big);
    auto data = AnnData::empty(0,0); data.set_uns("reversed",Element(std::move(reversed)));
    write_h5ad(path.string(),data);
    auto output = read_h5ad(path.string());
    auto array = output.uns().at("reversed").array();
    require(array.serialization_byte_order()==ByteOrder::big,"tiled write lost byte order");
    auto bytes = array.dense<std::int32_t>();
    for (std::size_t row = 0; row < 4; ++row) for (std::size_t j = 0; j < n; ++j)
        require(bytes.data()[row*n+j]==std::int32_t(row*n+n-1-j),"rank-three signed-stride tile differs");
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected a scratch directory");
        const std::filesystem::path root(argv[1]);
        std::filesystem::create_directories(root);
        const auto path = root / "simple.h5ad";
        auto data = fixture();
        anndatapp::write_h5ad(path.string(), data);
        auto reopened = anndatapp::read_h5ad(path.string());
        check(reopened);

        const auto gzip_path = root / "gzip.h5ad";
        anndatapp::WriteOptions gzip;
        gzip.compression = anndatapp::Compression::gzip;
        anndatapp::write_h5ad(gzip_path.string(), data.view(), gzip);
        check(anndatapp::read_h5ad(gzip_path.string()));

        anndatapp::WriteOptions exclusive;
        exclusive.mode = anndatapp::WriteMode::exclusive;
        bool rejected_existing = false;
        try { anndatapp::write_h5ad(path.string(), data, exclusive); }
        catch (const anndatapp::Error& error) {
            rejected_existing = error.code() == anndatapp::ErrorCode::io_error;
        }
        require(rejected_existing, "exclusive mode must reject an existing file");

        const auto view_path = root / "selected-view.h5ad";
        auto selected = data.slice(anndatapp::indices{2, 0}, anndatapp::indices{1});
        anndatapp::write_h5ad(view_path.string(), selected);
        auto view_read = anndatapp::read_h5ad(view_path.string());
        require(view_read.n_obs() == 2 && view_read.n_vars() == 1,
                "selected view shape/order differs");
        require(view_read.obs().name(0) == "c2" && view_read.obs().name(1) == "c0" &&
                view_read.var().name(0) == "g1", "selected view axes differ");
        auto selected_x = view_read.x().dense<float>();
        require(selected_x.data()[0] == 6 && selected_x.data()[1] == 2,
                "selected dense view values differ");
        require(view_read.obsp("graph").extent(0) == 2 &&
                view_read.obsp("graph").extent(1) == 2,
                "selected pairwise view shape differs");

        bool failed_missing_path = false;
        try { (void)anndatapp::read_h5ad((root / "absent.h5ad").string()); }
        catch (const anndatapp::Error& error) {
            failed_missing_path = error.code() == anndatapp::ErrorCode::io_error;
        }
        require(failed_missing_path, "missing input must be an I/O error");

        auto zero_rows = anndatapp::AnnData::empty(0, 2);
        zero_rows.set_x(anndatapp::Array::empty<float>({0, 2}));
        zero_rows.set_obs(anndatapp::Table::with_index(std::vector<std::string>{}, "cell_id"));
        zero_rows.set_var(anndatapp::Table::with_index({"g0", "g1"}, "gene_id"));
        const auto empty_path = root / "empty.h5ad";
        anndatapp::write_h5ad(empty_path.string(), zero_rows);
        auto empty_read = anndatapp::read_h5ad(empty_path.string());
        require(empty_read.n_obs() == 0 && empty_read.n_vars() == 2 && empty_read.has_x(),
                "zero-sized axis or X presence differs");
        require(empty_read.x().extent(0) == 0 && empty_read.x().extent(1) == 2,
                "zero-sized dimensions differ");

        auto without_x = anndatapp::AnnData::empty(1, 1);
        without_x.set_obs(anndatapp::Table::with_index({"cell"}, "cell_id"));
        without_x.set_var(anndatapp::Table::with_index({"gene"}, "gene_id"));
        without_x.set_raw(anndatapp::Table::with_index({"raw-gene"}, "gene_id"));
        const auto no_x_path = root / "no-x-raw.h5ad";
        anndatapp::write_h5ad(no_x_path.string(), without_x);
        auto no_x_read = anndatapp::read_h5ad(no_x_path.string());
        require(!no_x_read.has_x() && no_x_read.has_raw() && !no_x_read.has_raw_x(),
                "optional X and raw without X differ");

        check_trusted_validation(root / "trusted-invalid-sparse.h5ad");
        check_reserved_index_name(root / "reserved-index.h5ad");
        check_tiled_strides(root / "tiled-strides.h5ad");
        std::cout << "native H5AD round-trip passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
