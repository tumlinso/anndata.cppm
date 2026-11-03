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

void require(bool ok, std::string_view message) {
    if (!ok) throw std::runtime_error(std::string(message));
}

template<class F>
void require_error(F&& operation, ErrorCode code, std::string_view detail,
                   std::string_view message) {
    try {
        operation();
    } catch (const Error& error) {
        require(error.code() == code && std::string_view(error.what()).find(detail) != std::string_view::npos,
                message);
        return;
    }
    throw std::runtime_error(std::string(message));
}

template<class T>
Array dense(std::vector<std::size_t> shape, std::vector<T> values) {
    auto result = Array::empty<T>(shape);
    auto out = result.edit().template dense<T>();
    std::size_t n = 1;
    for (auto extent : shape) n *= extent;
    require(values.size() == n, "dense fixture size differs from its shape");
    for (std::size_t i = 0; i < n; ++i) out.data()[i] = values[i];
    return result;
}

RecordArray records(std::vector<std::size_t> shape, std::int32_t first,
                    bool oversized_text = false) {
    const auto field_shape = shape;
    std::size_t n = 1;
    for (auto extent : shape) n *= extent;
    std::vector<std::int32_t> ids(n);
    std::vector<bool> enabled(n);
    std::vector<float> scores(n);
    std::vector<std::string> labels(n);
    constexpr std::string_view names[]{"α", "βeta", "γ", "δelta"};
    for (std::size_t i = 0; i < n; ++i) {
        ids[i] = first + std::int32_t(i);
        enabled[i] = i % 2 == 0;
        scores[i] = float(first) + 0.25f * float(i);
        labels[i] = oversized_text ? std::string(2048, 'x') : std::string(names[i % 4]);
    }

    auto result = RecordArray::with_shape(std::move(shape));
    result.set("id", dense<std::int32_t>(field_shape, std::move(ids)));
    result.set("enabled", dense<bool>(field_shape, std::move(enabled)));
    result.set("score", dense<float>(field_shape, std::move(scores)));
    result.set("label", StringArray::from_strings(field_shape, std::move(labels)));
    result.set_field_serialization_byte_order("id", ByteOrder::big);
    return result;
}

AnnData fixture() {
    auto data = AnnData::empty(2, 2);
    data.set_x(dense<float>({2, 2}, {1, 2, 3, 4}));

    auto nested = Element::mapping();
    nested.set("record", Element(records({2, 3}, 10)));
    nested.set("empty", Element(records({0, 2}, 0)));
    nested.set("cube", Element(records({2, 1, 2}, 30)));
    nested.set("scalar", Element(records({}, 77)));
    nested.set("oversized", Element(records({1}, 0, true)));
    auto bundle = Element::mapping();
    bundle.set("nested", std::move(nested));
    data.set_uns("record_bundle", std::move(bundle));
    return data;
}

void check_schema_and_values(const RecordArrayView& actual, std::int32_t first) {
    require(actual.shape() == std::vector<std::size_t>{2, 3} && actual.size() == 6,
            "record outer shape differs");
    require(actual.field_names() == std::vector<std::string>{"id", "enabled", "score", "label"},
            "record field order differs");
    require(actual.field("id").dtype == DType::int32 &&
            actual.field("id").byte_order == ByteOrder::big &&
            actual.field("enabled").dtype == DType::boolean &&
            actual.field("score").dtype == DType::float32,
            "record field dtype or byte order differs");

    auto ids = actual.array("id").dense<std::int32_t>();
    auto enabled = actual.array("enabled").dense<bool>();
    auto score = actual.array("score").dense<float>();
    require(ids.data()[0] == first && ids.data()[5] == first + 5 &&
            enabled.data()[0] && !enabled.data()[1] && enabled.data()[4] &&
            score.data()[0] == float(first) && score.data()[5] == float(first) + 1.25f,
            "complete record numeric values differ");
    require(actual.string("label", 0) == "α" && actual.string("label", 1) == "βeta" &&
            actual.string("label", 5) == "βeta",
            "complete record Unicode values differ");
}

void check_backed(const std::filesystem::path& source, const std::filesystem::path& output) {
    BackedOptions options;
    options.mode = AccessMode::read_write;
    auto store = open_h5ad(source.string(), options);
    auto record = store.uns().at("record_bundle").at("nested").at("record");
    require(record.kind() == ElementKind::record_array &&
            record.field_names() == std::vector<std::string>{"id", "enabled", "score", "label"},
            "backed record kind or ordered schema differs");
    require_error([&] { (void)record.read_field("missing"); }, ErrorCode::out_of_bounds,
                  "record field 'missing' does not exist", "missing field diagnostic differs");

    ArrayView retained_ids;
    StringArrayView retained_labels;
    {
        auto field = record.read_field("id");
        require(field.view().kind() == ElementKind::array && field.view().array().dtype() == DType::int32 &&
                field.view().array().serialization_byte_order() == ByteOrder::big,
                "backed int32 field metadata differs");
        retained_ids = field.view().array();
    }
    {
        auto field = record.read_field("label");
        require(field.view().kind() == ElementKind::string_array,
                "backed text field is not a string array");
        retained_labels = field.view().string_array();
    }
    {
        auto full = record.read();
        require(full.view().kind() == ElementKind::record_array,
                "full backed record read lost its kind");
        check_schema_and_values(full.view().record_array(), 10);
    }

    {
        auto empty = store.uns().at("record_bundle").at("nested").at("empty").read();
        auto view = empty.view().record_array();
        require(view.shape() == std::vector<std::size_t>{0, 2} && view.size() == 0 &&
                view.field_names() == std::vector<std::string>{"id", "enabled", "score", "label"} &&
                view.array("id").extent(0) == 0 && view.string_array("label").size() == 0,
                "empty multidimensional record differs");
    }
    {
        auto cube = store.uns().at("record_bundle").at("nested").at("cube");
        auto values = cube.read_field("id");
        auto field = values.view().array();
        require(field.rank() == 3 && field.extent(0) == 2 && field.extent(1) == 1 &&
                field.extent(2) == 2 && field.dense<std::int32_t>().data()[3] == 33,
                "rank-three record shape or field values differ");
    }
    {
        auto scalar = store.uns().at("record_bundle").at("nested").at("scalar");
        auto id = scalar.read_field("id");
        auto label = scalar.read_field("label");
        const auto ids = id.view().array();
        const auto labels = label.view().string_array();
        require(ids.rank() == 0 && ids.dense<std::int32_t>().data()[0] == 77 &&
                ids.serialization_byte_order() == ByteOrder::big && labels.rank() == 0 &&
                labels.size() == 1 && labels.string(0) == "α",
                "rank-zero backed numeric or VLEN text field differs");
        auto full = scalar.read().view().record_array();
        require(full.rank() == 0 && full.size() == 1 && full.array("id").dense<std::int32_t>().data()[0] == 77 &&
                full.string("label", 0) == "α",
                "rank-zero full backed record differs");
    }

    {
        BackedOptions bounded;
        bounded.scratch_bytes = 512;
        auto small = open_h5ad(source.string(), bounded);
        auto long_text = small.uns().at("record_bundle").at("nested").at("oversized");
        require_error([&] { (void)long_text.read_field("label"); }, ErrorCode::invalid_argument,
                      "one string exceeds the decoded staging budget",
                      "oversized record string ignored its budget");
        small.close();
    }

    auto replacement = Element(records({2, 3}, 100));
    auto edit = store.edit_structure();
    edit.set("/uns/record_bundle/nested/record", std::move(replacement));
    edit.apply();
    require_error([&] { (void)record.read(); }, ErrorCode::invalid_argument,
                  "stale", "replaced backed record descriptor remained usable");

    auto current = store.uns().at("record_bundle").at("nested").at("record").read();
    check_schema_and_values(current.view().record_array(), 100);
    write_h5ad(output.string(), store);
    store.close();

    require(retained_ids.dense<std::int32_t>().data()[0] == 10 &&
            retained_ids.dense<std::int32_t>().data()[5] == 15 &&
            retained_labels.string(0) == "α" && retained_labels.string(5) == "βeta",
            "materialized field views did not survive element and session close");
}

void check_export(const std::filesystem::path& output) {
    auto actual = read_h5ad(output.string());
    require(actual.n_obs() == 2 && actual.n_vars() == 2 && actual.has_x(),
            "backed export changed AnnData dimensions or X presence");
    auto x = actual.x().dense<float>();
    const float expected[]{1, 2, 3, 4};
    for (std::size_t i = 0; i < 4; ++i)
        require(x.data()[i] == expected[i], "backed export changed X");

    const auto nested = actual.uns().at("record_bundle").at("nested");
    check_schema_and_values(nested.at("record").record_array(), 100);
    const auto empty = nested.at("empty").record_array();
    require(empty.shape() == std::vector<std::size_t>{0, 2},
            "backed export changed the empty record shape");
    const auto cube = nested.at("cube").record_array();
    require(cube.shape() == std::vector<std::size_t>{2, 1, 2},
            "backed export changed the rank-three record shape");
    require(cube.array("id").dense<std::int32_t>().data()[3] == 33,
            "backed export changed the rank-three record values");
    const auto scalar = nested.at("scalar").record_array();
    require(scalar.rank() == 0 && scalar.size() == 1 &&
            scalar.array("id").dense<std::int32_t>().data()[0] == 77 &&
            scalar.string("label", 0) == "α",
            "backed export changed rank-zero numeric or VLEN text values");
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected a scratch directory");
        const std::filesystem::path root(argv[1]);
        std::filesystem::create_directories(root);
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto run = root / ("backed-records-" + std::to_string(nonce));
        std::filesystem::create_directories(run);
        const auto source = run / "source.h5ad";
        const auto output = run / "export.h5ad";

        auto data = fixture();
        write_h5ad(source.string(), data);
        check_backed(source, output);
        check_export(output);
        std::cout << "native backed record checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
