#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

import anndatapp;

using namespace anndatapp;

namespace {
template<class T>
Array dense(std::initializer_list<std::size_t> shape, std::initializer_list<T> values) {
    auto a = Array::empty<T>(shape);
    auto x = a.edit().template dense<T>();
    std::size_t i = 0;
    for (auto value : values) x.data()[i++] = value;
    return a;
}

AnnData fixture() {
    auto a = AnnData::empty(3, 2);
    a.set_x(dense<float>({3, 2}, {0, 1, 2, 3, 4, 5}));
    auto obs = Table::with_index({"cell-a", "cell-b", "cell-c"}, "cell_id");
    obs.set("cell_type", Column::categorical(dense<std::int8_t>({3}, {0, 1, 0}),
        Column::strings({"T", "B", "unused"}), true));
    obs.set("count", Column::from_values<std::int32_t>({1, 0, 3}, {1, 0, 1}));
    obs.set("flag", Column::from_values<bool>({true, false, false}, {1, 0, 1}));
    obs.set("note", Column::strings({"α", "", "γ"}, {1, 0, 1}));
    a.set_obs(std::move(obs));
    a.set_var(Table::with_index({"g1", "g2"}, "gene_id"));
    auto counts = Array::csr<std::int32_t, std::int32_t, std::int32_t>(3, 2, 3);
    auto c = counts.edit().csr<std::int32_t, std::int32_t, std::int32_t>();
    const std::int32_t values[]{1, 2, 3}, indices[]{0, 1, 0};
    const std::int32_t offsets[]{0, 1, 2, 3};
    for (int i = 0; i < 3; ++i) { c.values()[i] = values[i]; c.unsafe_indices()[i] = indices[i]; }
    for (int i = 0; i < 4; ++i) c.unsafe_offsets()[i] = offsets[i];
    counts.validate();
    a.set_layer("counts", std::move(counts));
    a.set_obsm("embedding", dense<double>({3, 2}, {0, 1, 2, 3, 4, 5}));
    a.set_obsp("connectivities", dense<float>({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}));
    auto nested = Element::mapping();
    nested.set("kind", Element("fixture"));
    nested.set("threshold", Element::scalar<float>(0.5f));
    a.set_uns("nested", std::move(nested));
    a.set_uns("cell_type_colors", Element(Column::strings({"red", "blue", "green"})));
    auto var_table = Table::with_index({"g1", "g2"}, "gene_id");
    var_table.set("label", Column::strings({"gene A", "gene B"}));
    a.set_varm("table", var_table.copy());
    a.set_raw(a.x().copy(), a.var().copy());
    a.set_raw_varm("table", std::move(var_table));
    return a;
}

void json_string(std::string_view s) {
    std::cout << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') std::cout << '\\' << c;
        else if (c < 32) std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        else std::cout << c;
    }
    std::cout << '"';
}

template<class F>
void list(std::size_t n, F f) {
    std::cout << '[';
    for (std::size_t i = 0; i < n; ++i) { if (i) std::cout << ','; f(i); }
    std::cout << ']';
}

template<class T>
void matrix(ArrayView a) {
    auto copy = a.copy();
    auto v = copy.view().template dense<T>();
    list(v.extent(0), [&](auto r) {
        list(v.extent(1), [&](auto c) { std::cout << v.data()[r * v.extent(1) + c]; });
    });
}

void emit(const AnnDataView& a) {
    std::cout << "{\"shape\":[" << a.n_obs() << ',' << a.n_vars() << "],\"X\":";
    matrix<float>(a.x());
    std::cout << ",\"obs_names\":";
    list(a.n_obs(), [&](auto i) { json_string(a.obs().name(i)); });
    std::cout << ",\"var_names\":";
    list(a.n_vars(), [&](auto i) { json_string(a.var().name(i)); });
    auto cat = a.obs_column("cell_type");
    std::cout << ",\"categories\":";
    list(cat.categories().size(), [&](auto i) { json_string(cat.categories().string(i)); });
    std::cout << ",\"codes\":";
    list(cat.size(), [&](auto i) { std::cout << cat.code(i); });
    std::cout << ",\"ordered\":" << (cat.ordered() ? "true" : "false");
    std::cout << ",\"count\":";
    auto count = a.obs_column("count");
    auto count_copy = count.copy();
    auto counts = count_copy.view().values().dense<std::int32_t>();
    list(count.size(), [&](auto i) { if (count.valid(i)) std::cout << counts.data()[i]; else std::cout << "null"; });
    std::cout << ",\"flag\":";
    auto flag = a.obs().column("flag");
    auto flag_copy = flag.copy();
    auto flags = flag_copy.view().values().dense<bool>();
    list(flag.size(), [&](auto i) { if (flag.valid(i)) std::cout << (flags.data()[i] ? "true" : "false"); else std::cout << "null"; });
    std::cout << ",\"note\":";
    auto note = a.obs().column("note");
    list(note.size(), [&](auto i) { if (note.valid(i)) json_string(note.string(i)); else std::cout << "null"; });
    std::cout << ",\"colors\":";
    auto colors = a.uns().at("cell_type_colors").column();
    list(colors.size(), [&](auto i) { json_string(colors.string(i)); });
    std::cout << ",\"raw_X\":";
    matrix<float>(a.raw_x());
    std::cout << ",\"raw_var_names\":";
    list(a.raw_var().size(), [&](auto i) { json_string(a.raw_var().name(i)); });
    std::cout << ",\"varm_labels\":";
    auto var_labels = a.varm("table").table().column("label");
    list(var_labels.size(), [&](auto i) { json_string(var_labels.string(i)); });
    std::cout << ",\"raw_varm_labels\":";
    auto raw_labels = a.raw_varm("table").table().column("label");
    list(raw_labels.size(), [&](auto i) { json_string(raw_labels.string(i)); });
    std::cout << ",\"embedding\":";
    matrix<double>(a.obsm("embedding").array());
    std::cout << ",\"connectivities\":";
    matrix<float>(a.obsp("connectivities"));
    auto sparse = a.layer("counts").copy();
    auto sv = sparse.view().csr<std::int32_t, std::int32_t, std::int32_t>();
    std::cout << ",\"counts_data\":";
    list(sv.nnz(), [&](auto i) { std::cout << sv.values()[i]; });
    std::cout << ",\"counts_indices\":";
    list(sv.nnz(), [&](auto i) { std::cout << sv.indices()[i]; });
    std::cout << ",\"counts_indptr\":";
    list(sv.rows() + 1, [&](auto i) { std::cout << sv.offsets()[i]; });
    std::cout << ",\"index_name\":"; json_string(a.obs().index_name());
    std::cout << ",\"column_names\":";
    auto names = a.obs().column_names();
    list(names.size(), [&](auto i) { json_string(names[i]); });
    std::cout << ",\"threshold_dtype\":";
    json_string(a.uns().at("nested").at("threshold").dtype() == DType::float32 ? "float32" : "unexpected");
    std::cout << ",\"threshold\":" << a.uns().at("nested").at("threshold").scalar<float>() << "}\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected a scenario name");
        auto a = fixture();
        const std::string_view s = argv[1];
        if (s == "base") emit(a.view());
        else if (s == "all") emit(a.slice(all, all));
        else if (s == "subset") emit(a.slice(indices{2, 0}, indices{1}));
        else if (s == "repeat") emit(a.slice(indices{1, 1, 0}, all));
        else if (s == "reverse") emit(a.slice(range{.step = -1}, range{.step = -1}));
        else if (s == "range_nested") emit(a.slice(range{0, 3, 2}, range{.step = -1})
            .slice(range{.step = -1}, range{0, 2, 1}));
        else if (s == "gather_range") emit(a.slice(indices{2, 0, 2}, all)
            .slice(range{0, 3, 2}, range{.step = -1}));
        else if (s == "range_gather") emit(a.slice(range{.step = -1}, all)
            .slice(indices{2, 0, 2}, indices{1, 0}));
        else if (s == "range_empty_nested") emit(a.slice(range{.step = -1}, all)
            .slice(range{1, 1, 1}, all));
        else if (s == "empty") emit(a.slice(indices{}, all));
        else if (s == "empty_vars") emit(a.slice(all, indices{}));
        else if (s == "mask") emit(a.slice(mask{1, 0, 1}, labels{"g2"}));
        else if (s == "labels") emit(a.slice(labels{"cell-c", "cell-a"}, labels{"g2", "g1"}));
        else if (s == "singleton") emit(a.slice(std::ptrdiff_t{-1}, std::ptrdiff_t{0}));
        else if (s == "nested") emit(a.slice(indices{2, 0}, indices{1}).slice(indices{1}, all));
        else if (s == "scalar_colors" || s == "scalar_colors_empty") {
            a.set_obs_column("cell_type", Column::categorical(dense<std::int8_t>({3}, {0, 0, 0}),
                Column::strings({"T"}), true));
            a.set_uns("cell_type_colors", Element("red"));
            if (s == "scalar_colors") emit(a.slice(indices{2, 0}, all));
            else emit(a.slice(indices{}, all));
        }
        else throw std::runtime_error("unknown scenario");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
