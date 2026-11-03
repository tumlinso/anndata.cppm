#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
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

template<class F>
void require_error(F&& operation, ErrorCode expected, std::string_view message) {
    bool matched = false;
    try {
        operation();
    } catch (const Error& error) {
        matched = error.code() == expected;
    }
    require(matched, message);
}

template<class T>
Array dense(std::initializer_list<std::size_t> shape, std::initializer_list<T> values) {
    auto result = Array::empty<T>(shape);
    auto out = result.edit().template dense<T>();
    std::size_t i = 0;
    for (T value : values) out.data()[i++] = value;
    require(i == out.extent(0) * (out.rank() > 1 ? out.extent(1) : 1),
            "fixture value count differs from shape");
    return result;
}

AnnData fixture() {
    auto data = AnnData::empty(4, 3);
    data.set_x(dense<float>({4, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    }));

    auto sparse = Array::csr<float, std::int32_t, std::int64_t>(4, 3, 7);
    auto values = sparse.edit().csr<float, std::int32_t, std::int64_t>();
    const float sparse_values[]{2, 3, 4, 5, 6, 7, 8};
    const std::int32_t sparse_indices[]{1, 0, 2, 1, 2, 0, 2};
    const std::int64_t sparse_offsets[]{0, 1, 3, 5, 7};
    for (std::size_t i = 0; i < 7; ++i) {
        values.values()[i] = sparse_values[i];
        values.unsafe_indices()[i] = sparse_indices[i];
    }
    for (std::size_t i = 0; i < 5; ++i) values.unsafe_offsets()[i] = sparse_offsets[i];
    sparse.validate();
    data.set_layer("sparse", std::move(sparse));

    auto obs = Table::with_index({"c0", "c1", "c2", "c3"}, "cell_id");
    obs.set("label", Column::strings({"α", "β", "γ", "δ"}));
    obs.set("group", Column::categorical(
        dense<std::int8_t>({4}, {0, 1, 0, 1}), Column::strings({"control", "treated"}), true));
    data.set_obs(std::move(obs));
    data.set_var(Table::with_index({"g0", "g1", "g2"}, "gene_id"));

    auto info = Element::mapping();
    info.set("caption", Element("backed UTF-8 μ"));
    info.set("strings", Element(StringArray::from_strings({2}, {"first λ", "second β"})));
    data.set_uns("info", std::move(info));
    data.set_uns("keep", Element("untouched"));
    return data;
}

void check_metadata(const BackedAnnData& backed) {
    require(backed.is_open() && !backed.writable(), "default backed session must be open and read-only");
    require(backed.n_obs() == 4 && backed.n_vars() == 3 && backed.has_x(),
            "backed axis metadata differs");
    require(backed.view().layer_names() == std::vector<std::string>{"sparse"},
            "backed layer names differ");

    auto obs = backed.obs();
    require(obs.size() == 4 && obs.index_name() == "cell_id" && obs.name(2) == "c2",
            "backed observation index metadata differs");
    auto labels = obs.column("label").read().view();
    require(labels.kind() == ColumnKind::string && labels.string(1) == "β",
            "backed string column differs");

    auto group = obs.column("group");
    require(group.kind() == ColumnKind::categorical && group.ordered() && group.categories().size() == 2,
            "backed categorical metadata differs");
    auto categories = group.categories().read().view();
    require(categories.kind() == ColumnKind::string && categories.string(1) == "treated",
            "backed categorical labels differ");
    auto codes = group.codes().read();
    auto code_view = codes.view().dense<std::int8_t>();
    require(code_view.data()[0] == 0 && code_view.data()[1] == 1 && code_view.data()[3] == 1,
            "backed categorical codes differ");

    auto caption = backed.uns().at("info").at("caption").read();
    require(caption.view().string() == "backed UTF-8 μ",
            "backed nested uns string differs");
    auto strings = backed.uns().at("info").at("strings").string_array().read();
    require(strings.view().string(1) == "second β",
            "backed nested string-array metadata differs");
}

void check_dense_reads(const BackedAnnData& backed) {
    auto selected = backed.slice(indices{3, 1}, indices{2, 0});
    auto x = selected.x();
    require(x.shape() == std::vector<std::size_t>{2, 2}, "gathered dense shape differs");

    auto eager = x.read();
    auto eager_values = eager.view().dense<float>();
    const float expected[]{12, 10, 6, 4};
    for (std::size_t i = 0; i < 4; ++i)
        require(eager_values.data()[i] == expected[i], "gathered dense read differs");

    float contiguous[4]{};
    x.read_into(DenseBuffer::from(contiguous, {2, 2}));
    for (std::size_t i = 0; i < 4; ++i)
        require(contiguous[i] == expected[i], "gathered dense read_into differs");

    float strided[7];
    for (float& value : strided) value = -1;
    x.read_into(DenseBuffer::from(strided, {2, 2},
                                  {static_cast<std::ptrdiff_t>(4 * sizeof(float)),
                                   static_cast<std::ptrdiff_t>(2 * sizeof(float))}));
    require(strided[0] == 12 && strided[2] == 10 && strided[4] == 6 && strided[6] == 4,
            "strided dense output values differ");
    require(strided[1] == -1 && strided[3] == -1 && strided[5] == -1,
            "strided dense output overwrote padding");
}

void check_sparse_read_into(const BackedAnnData& backed) {
    auto selected = backed.slice(indices{3, 1}, all).layer("sparse");
    const auto requirements = selected.prepare_read().requirements();
    require(requirements.layout == Layout::csr && requirements.shape == std::vector<std::size_t>{2, 3} &&
            requirements.nnz == 4 && requirements.index_dtype == DType::int32 &&
            requirements.offset_dtype == DType::int64,
            "selected sparse read requirements differ");

    float values[4]{};
    std::int32_t indices_out[4]{};
    std::int64_t offsets[3]{};
    SparseBuffer destination{values, indices_out, offsets, requirements.dtype,
        requirements.index_dtype, requirements.offset_dtype, 4, 4, 3};
    selected.read_into(destination);
    const float expected_values[]{7, 8, 3, 4};
    const std::int32_t expected_indices[]{0, 2, 0, 2};
    const std::int64_t expected_offsets[]{0, 2, 4};
    for (std::size_t i = 0; i < 4; ++i) {
        require(values[i] == expected_values[i], "selected sparse values differ");
        require(indices_out[i] == expected_indices[i], "selected sparse indices differ");
    }
    for (std::size_t i = 0; i < 3; ++i)
        require(offsets[i] == expected_offsets[i], "selected sparse offsets differ");
}

void check_mapping_and_close(const std::filesystem::path& path) {
    auto backed = open_h5ad(path.string());
    auto mapped = backed.x().map_readonly();
    require(mapped.dtype() == DType::float32 && mapped.shape() == std::vector<std::size_t>{4, 3},
            "mapped dense metadata differs");
    const auto* values = mapped.unsafe_data<float>();
    require(values[0] == 1 && values[11] == 12, "mapped native bytes differ");

    auto selected_handle = backed.slice(indices{3, 1}, all).x();
    backed.close();
    require(!backed.is_open(), "backed session did not close");
    require(mapped.unsafe_data<float>()[7] == 8,
            "read-only mapping did not retain its bytes after session close");
    require_error([&] { (void)selected_handle.read(); }, ErrorCode::invalid_argument,
                  "selected handle must reject access after session close");
}

void check_mapping_file_identity(const std::filesystem::path& root) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = root / ("mapping-identity-" + std::to_string(nonce));
    std::filesystem::create_directories(directory);
    const auto original_path = directory / "original.h5ad";
    const auto replacement_path = directory / "replacement.h5ad";
    const auto moved_path = directory / "moved-original.h5ad";

    auto original_data = fixture();
    auto replacement_data = fixture();
    replacement_data.edit_x().dense<float>().data()[0] = 99;
    write_h5ad(original_path.string(), original_data);
    write_h5ad(replacement_path.string(), replacement_data);

    auto backed = open_h5ad(original_path.string());
    auto mapped = backed.x().map_readonly();
    require(mapped.unsafe_data<float>()[0] == 1,
            "identity fixture did not map the original file contents");

    std::filesystem::rename(original_path, moved_path);
    std::filesystem::rename(replacement_path, original_path);
    auto session_read = backed.x().read();
    require(session_read.view().dense<float>().data()[0] == 1,
            "open backed session followed the replaced pathname");
    backed.close();
    require(mapped.unsafe_data<float>()[0] == 1,
            "mapping followed the replaced pathname or expired on session close");
}

void check_mapping_rejections(const std::filesystem::path& gzip_path,
                              const std::filesystem::path& foreign_endian_path) {
    auto gzip = open_h5ad(gzip_path.string());
    require_error([&] { (void)gzip.x().map_readonly(); }, ErrorCode::unsupported_layout,
                  "gzip-compressed X must not be memory mapped");
    gzip.close();

    auto foreign = open_h5ad(foreign_endian_path.string());
    require_error([&] { (void)foreign.x().map_readonly(); }, ErrorCode::unsupported_layout,
                  "foreign-endian X must not be memory mapped");
    foreign.close();
}

void check_readwrite(const std::filesystem::path& path) {
    auto replacement = dense<float>({2, 3}, {30, 31, 32, 10, 11, 12});
    {
        auto read_only = open_h5ad(path.string());
        require_error([&] { read_only.x().write(replacement.view()); }, ErrorCode::invalid_argument,
                      "read-only backed X write must fail");
        read_only.close();
    }

    BackedOptions options;
    options.mode = AccessMode::read_write;
    auto writable = open_h5ad(path.string(), options);
    require(writable.writable(), "read-write option was not retained");
    writable.x().select({indices{3, 1}, all}).write(replacement.view());
    writable.set("/uns/added", Element("new value"));
    writable.flush();
    writable.close();

    auto reopened = open_h5ad(path.string());
    auto all_x = reopened.x().read();
    auto values = all_x.view().dense<float>();
    const float expected[]{1, 2, 3, 10, 11, 12, 7, 8, 9, 30, 31, 32};
    for (std::size_t i = 0; i < 12; ++i)
        require(values.data()[i] == expected[i], "selected read-write X update differs");
    auto keep = reopened.uns().at("keep").read();
    auto added = reopened.uns().at("added").read();
    require(keep.view().string() == "untouched" && added.view().string() == "new value",
            "structural uns update did not preserve existing elements");
    reopened.close();
}

void check_bounded_strings(const std::filesystem::path& root) {
    std::vector<std::string> names;
    for (std::size_t i = 0; i < 65; ++i)
        // A longer value fits alone but forces partial-batch allocation retries.
        names.push_back("cell-" + std::to_string(i) + "-" + std::string(i == 17 ? 120 : i % 47, 'x'));
    auto data = AnnData::empty(names.size(), 0);
    data.set_obs(Table::with_index(names, "cell_id"));
    data.set_uns("oversized", Element(std::string(4096, 'y')));
    const auto path = root / "bounded-strings.h5ad";
    write_h5ad(path.string(), data);

    BackedOptions options;
    options.scratch_bytes = 512;
    auto backed = open_h5ad(path.string(), options);
    require(backed.obs().names() == names,
            "byte-bounded string batches lost values when shrinking prefixes");
    require_error([&] { (void)backed.uns().at("oversized").read(); },
                  ErrorCode::invalid_argument, "oversized scalar string exceeded its staging budget");
    require(backed.obs().name(3) == names[3],
            "oversized string failure left the session unusable");
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected a scratch directory");
        const std::filesystem::path root(argv[1]);
        std::filesystem::create_directories(root);

        // Exercise the public HDF5 capability query without assuming a runtime
        // thread-safety configuration that is controlled by the linked library.
        (void)hdf5_library_threadsafe();

        const auto path = root / "backed.h5ad";
        const auto gzip_path = root / "backed-gzip.h5ad";
        const auto foreign_path = root / "backed-foreign-endian.h5ad";
        auto data = fixture();
        write_h5ad(path.string(), data);

        WriteOptions gzip;
        gzip.compression = Compression::gzip;
        write_h5ad(gzip_path.string(), data, gzip);

        WriteOptions foreign_endian;
        foreign_endian.byte_order = std::endian::native == std::endian::little
            ? FileByteOrder::big : FileByteOrder::little;
        write_h5ad(foreign_path.string(), data, foreign_endian);

        {
            auto backed = open_h5ad(path.string());
            check_metadata(backed);
            check_dense_reads(backed);
            check_sparse_read_into(backed);
            backed.close();
        }

        check_mapping_and_close(path);
        check_mapping_file_identity(root);
        check_mapping_rejections(gzip_path, foreign_path);
        check_readwrite(path);
        check_bounded_strings(root);
        std::cout << "native backed H5AD checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
