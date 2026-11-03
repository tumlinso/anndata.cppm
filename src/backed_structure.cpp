module;
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

module anndata.backed;
import :storage;

namespace anndata {
namespace detail {
using StructureSource = std::variant<Element, BackedArray, BackedElement, BackedTable, BackedColumn>;
struct StructureEditImpl {
    explicit StructureEditImpl(std::shared_ptr<BackedDataset> dataset_)
        : dataset(std::move(dataset_)), observed_revision(dataset->session->dataset_revision) {}
    std::shared_ptr<BackedDataset> dataset;
    std::size_t observed_revision = 0;
    // nullopt means erase; an Element may itself have ElementKind::null.
    std::map<std::string, std::optional<StructureSource>, std::less<>> changes;
    std::optional<std::pair<::anndata::Selection, ::anndata::Selection>> selection;
    bool consumed = false;
};
}
namespace {
using backed_detail::Attributes;
using backed_detail::Backend;
using backed_detail::Session;
using detail::StructureSource;
using detail::StructureEditImpl;

[[noreturn]] void fail(ErrorCode code, std::string_view path, const std::string& message) {
    throw Error(code, std::string(path), message);
}
void usable(const StructureEditImpl* edit) {
    if (!edit || edit->consumed || !edit->dataset || !edit->dataset->session)
        fail(ErrorCode::invalid_argument, "structure", "structural edit is empty or already applied");
    edit->dataset->session->check_writable();
    if (edit->observed_revision != edit->dataset->session->dataset_revision)
        fail(ErrorCode::invalid_argument, "structure", "structural edit is stale after another structural mutation");
}
bool starts_child(std::string_view path, std::string_view parent) {
    return path.size() > parent.size() && path.starts_with(parent) && path[parent.size()] == '/';
}
std::vector<std::string> components(std::string_view path) {
    if (path.size() < 2 || path.front() != '/' || path.back() == '/')
        fail(ErrorCode::invalid_argument, path, "expected an absolute non-root element path");
    std::vector<std::string> parts;
    std::size_t start = 1;
    while (start < path.size()) {
        auto end = path.find('/', start);
        if (end == std::string_view::npos) end = path.size();
        auto part = path.substr(start, end - start);
        if (part.empty() || part == "." || part == ".." || part.find('\0') != std::string_view::npos)
            fail(ErrorCode::invalid_argument, path, "invalid element path component");
        parts.emplace_back(part);
        start = end + 1;
    }
    return parts;
}
bool is_table_column(std::string_view p) {
    auto c = components(p);
    return (c.size() == 2 && (c[0] == "obs" || c[0] == "var")) ||
           (c.size() == 3 && c[0] == "raw" && c[1] == "var");
}
std::string parent_path(std::string_view path) {
    auto p = path.rfind('/');
    return p == 0 ? "/" : std::string(path.substr(0, p));
}
std::string last_component(std::string_view path) {
    return std::string(path.substr(path.rfind('/') + 1));
}
void check_path(std::string_view path, bool erase) {
    const auto c = components(path);
    bool ok = false;
    if (c.size() == 1)
        ok = c[0] == "X" || c[0] == "obs" || c[0] == "var" || c[0] == "uns" || (erase && c[0] == "raw");
    else if (c[0] == "uns") ok = true;
    else if (c.size() == 2)
        ok = c[0] == "layers" || c[0] == "obsm" || c[0] == "varm" || c[0] == "obsp" ||
             c[0] == "varp" || c[0] == "obs" || c[0] == "var" ||
             (c[0] == "raw" && (c[1] == "X" || c[1] == "var"));
    else if (c.size() == 3) ok = (c[0] == "raw" && (c[1] == "var" || c[1] == "varm")) ||
                                  c[0] == "obsm" || c[0] == "varm";
    else if (c.size() == 4 && c[0] == "raw" && c[1] == "varm") ok = true;
    if (!ok) fail(ErrorCode::invalid_argument, path, "path is not an editable AnnData element");
    if (erase && (path == "/obs" || path == "/var" || path == "/uns" || path == "/raw/var"))
        fail(ErrorCode::invalid_argument, path, "required element cannot be erased; erase raw as a whole if appropriate");
    if (is_table_column(path) && c.back() == "_index")
        fail(ErrorCode::invalid_argument, path, "the reserved dataframe index cannot be edited as a column");
}
bool is_dataframe_column(Session& session, std::string_view path) {
    if (is_table_column(path)) return true;
    const auto parent = parent_path(path);
    if (parent == "/") return false;
    if (starts_child(parent, "/raw")) {
        const auto raw = session.backend->inspect("/raw");
        if (!raw.exists || backed_detail::encoding(raw, "/raw") == "null") return false;
    }
    const auto info = session.backend->inspect(parent);
    return info.exists && info.group && backed_detail::encoding(info, parent) == "dataframe";
}
void put(StructureEditImpl& edit, std::string path, std::optional<StructureSource> value) {
    check_path(path, !value.has_value());
    for (const auto& [other, ignored] : edit.changes) {
        (void)ignored;
        if (other != path && (starts_child(path, other) || starts_child(other, path)))
            fail(ErrorCode::invalid_argument, path, "a batch cannot replace an element and also edit its descendants");
    }
    edit.changes.insert_or_assign(std::move(path), std::move(value));
}
const std::optional<StructureSource>* change_for(const StructureEditImpl& edit, std::string_view path) {
    auto it = edit.changes.find(path);
    return it == edit.changes.end() ? nullptr : &it->second;
}
bool is_erased(const StructureEditImpl& edit, std::string_view path) {
    auto change = change_for(edit, path);
    return change && !change->has_value();
}
std::vector<std::size_t> array_shape(ArrayView array) {
    std::vector<std::size_t> result;
    result.reserve(array.rank());
    for (std::size_t i = 0; i < array.rank(); ++i) result.push_back(array.extent(i));
    return result;
}
struct Shape {
    ElementKind kind;
    std::vector<std::size_t> dims;
};
Shape value_shape(ElementView value, std::string_view path) {
    switch (value.kind()) {
    case ElementKind::array: return {ElementKind::array, array_shape(value.array())};
    case ElementKind::string_array: return {ElementKind::string_array, value.string_array().shape()};
    case ElementKind::column: return {ElementKind::column, {value.column().size()}};
    case ElementKind::table: return {ElementKind::table, {value.table().size()}};
    default: fail(ErrorCode::unsupported_encoding, path, "element is not an aligned array, column, or dataframe");
    }
}
Shape source_shape(const StructureSource& source, std::string_view path) {
    return std::visit([&](const auto& value) -> Shape {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Element>) return value_shape(value.view(), path);
        else if constexpr (std::is_same_v<T, BackedArray>) return {ElementKind::array, value.shape()};
        else if constexpr (std::is_same_v<T, BackedTable>) return {ElementKind::table, {value.size()}};
        else if constexpr (std::is_same_v<T, BackedColumn>) return {ElementKind::column, {value.size()}};
        else {
            switch (value.kind()) {
            case ElementKind::array: return {ElementKind::array, value.array().shape()};
            case ElementKind::string_array: return {ElementKind::string_array, value.string_array().shape()};
            case ElementKind::column: return {ElementKind::column, {value.column().size()}};
            case ElementKind::table: return {ElementKind::table, {value.table().size()}};
            default: fail(ErrorCode::unsupported_encoding, path, "element is not an aligned array, column, or dataframe");
            }
        }
    }, source);
}
ElementKind source_kind(const StructureSource& source) {
    return std::visit([](const auto& value) -> ElementKind {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Element>) return value.view().kind();
        else if constexpr (std::is_same_v<T, BackedArray>) return ElementKind::array;
        else if constexpr (std::is_same_v<T, BackedTable>) return ElementKind::table;
        else if constexpr (std::is_same_v<T, BackedColumn>) return ElementKind::column;
        else return value.kind();
    }, source);
}
void validate_source(const StructureSource& source, std::string_view path) {
    std::visit([&](const auto& value) {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Element>) {
            value.validate();
            backed_detail::validate_serializable(value.view(), path);
        } else {
            // This checks the retained handle without materializing its data.
            // copy_node validates selected payload as it streams to the staging node.
            (void)source_kind(source);
        }
    }, source);
}
Shape stored_shape(Session& session, std::string_view path) {
    const auto info = session.backend->inspect(path);
    if (!info.exists) fail(ErrorCode::out_of_bounds, path, "element does not exist");
    const auto code = backed_detail::encoding(info, path);
    ElementKind kind;
    if (code == "array" || code == "csr_matrix" || code == "csc_matrix") kind = ElementKind::array;
    else if (code == "string-array") kind = ElementKind::string_array;
    else if (code == "dataframe") kind = ElementKind::table;
    else if (code == "categorical" || code == "nullable-integer" || code == "nullable-boolean" || code == "nullable-string-array")
        kind = ElementKind::column;
    else fail(ErrorCode::unsupported_encoding, path, "unsupported aligned element encoding '" + code + "'");
    return {kind, backed_detail::logical_shape(session, path)};
}
Shape final_shape(const StructureEditImpl& edit, Session& session, std::string_view path,
                  std::size_t selected_obs, std::size_t selected_var) {
    if (auto change = change_for(edit, path)) {
        if (!*change) fail(ErrorCode::invalid_argument, path, "internal attempt to inspect a removed element");
        return source_shape(**change, path);
    }
    auto shape = stored_shape(session, path);
    if (edit.selection) {
        const auto no = selected_obs, nv = selected_var;
        auto c = components(path);
        // A selected view's dimensions alone can hide an oversized or short
        // original slot. Check the physical alignment before substituting the
        // selected dimensions, while the batch is still read-only.
        const auto old_obs = edit.dataset->obs.size;
        const auto old_var = edit.dataset->var.size;
        const bool observes = c[0] == "X" || c[0] == "layers" || c[0] == "obsm" || c[0] == "obsp" ||
                              (c[0] == "raw" && c.size() == 2 && c[1] == "X");
        const bool variables = c[0] == "varm" || c[0] == "varp";
        if ((observes || variables) && (shape.dims.empty() || shape.dims[0] != (observes ? old_obs : old_var)))
            fail(ErrorCode::shape_mismatch, path, "stored element is not aligned with its original first axis");
        if (c[0] == "X" || c[0] == "layers" || c[0] == "obsp" || c[0] == "varp") {
            const auto expected = c[0] == "obsp" ? old_obs : old_var;
            if (shape.dims.size() < 2 || shape.dims[1] != expected)
                fail(ErrorCode::shape_mismatch, path, "stored element is not aligned with its original second axis");
        }
        if (!shape.dims.empty()) {
            if (c[0] == "X" || c[0] == "layers" || c[0] == "obsm" || c[0] == "obsp" ||
                (c[0] == "raw" && c.size() == 2 && c[1] == "X")) shape.dims[0] = no;
            else if (c[0] == "varm" || c[0] == "varp") shape.dims[0] = nv;
            if (shape.dims.size() > 1) {
                if (c[0] == "X" || c[0] == "layers" || c[0] == "varp") shape.dims[1] = nv;
                else if (c[0] == "obsp") shape.dims[1] = no;
            }
        }
    }
    return shape;
}
void matrix_shape(const Shape& shape, std::size_t rows, std::size_t cols, std::string_view path) {
    if (shape.kind != ElementKind::array || shape.dims.size() != 2 || shape.dims[0] != rows || shape.dims[1] != cols)
        fail(ErrorCode::shape_mismatch, path, "matrix must have both final aligned axis lengths");
}
void pairwise_shape(const Shape& shape, std::size_t length, std::string_view path) {
    if (shape.kind != ElementKind::array || shape.dims.size() < 2 ||
        shape.dims[0] != length || shape.dims[1] != length)
        fail(ErrorCode::shape_mismatch, path, "pairwise array must have both final aligned axis lengths");
}
void aligned_shape(const Shape& shape, std::size_t length, std::string_view path) {
    if ((shape.kind != ElementKind::array && shape.kind != ElementKind::string_array && shape.kind != ElementKind::table) ||
        shape.dims.empty() || shape.dims[0] != length)
        fail(ErrorCode::shape_mismatch, path, "aligned element has the wrong first dimension");
}
std::vector<std::string> names_with_changes(Backend& backend, const StructureEditImpl& edit, std::string_view base) {
    std::set<std::string, std::less<>> names;
    bool may_exist = true;
    if (starts_child(base, "/raw")) {
        auto raw = backend.inspect("/raw");
        may_exist = raw.exists && backed_detail::encoding(raw, "/raw") != "null";
    }
    if (may_exist) {
        if (auto info = backend.inspect(base); info.exists)
            for (auto& name : backend.children(base)) names.insert(std::move(name));
    }
    for (const auto& [path, value] : edit.changes) {
        if (starts_child(path, base) && parent_path(path) == base) {
            const auto key = last_component(path);
            if (value) names.insert(key); else names.erase(key);
        }
    }
    return {names.begin(), names.end()};
}
bool physical_present(Backend& backend, std::string_view path) {
    const auto info = backend.inspect(path);
    return info.exists && backed_detail::encoding(info, path) != "null";
}
struct Index {
    std::size_t size;
    std::variant<BackedTable, TableView> table;
};
Index index_of(BackedTable t) {
    return {t.size(), std::move(t)};
}
Index index_of(TableView t) {
    return {t.size(), std::move(t)};
}
Index index_of_source(const StructureSource& source, std::string_view path) {
    return std::visit([&](const auto& value) -> Index {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Element>) {
            if (value.view().kind() == ElementKind::table) return index_of(value.view().table());
        } else if constexpr (std::is_same_v<T, BackedTable>) return index_of(value);
        else if constexpr (std::is_same_v<T, BackedElement>) {
            if (value.kind() == ElementKind::table) return index_of(value.table());
        }
        fail(ErrorCode::shape_mismatch, path, "axis replacement must be a dataframe");
    }, source);
}
Index axis_index(const StructureEditImpl& edit, const BackedAnnDataView& view, std::string_view path) {
    if (auto p = change_for(edit, path)) {
        if (!*p || source_kind(**p) != ElementKind::table)
            fail(ErrorCode::shape_mismatch, path, "axis replacement must be a dataframe");
        return index_of_source(**p, path);
    }
    if (path == "/obs") return index_of(view.obs());
    if (path == "/var") return index_of(view.var());
    return index_of(view.raw_var());
}
void compare_indexes(const Index& left, const Index& right, std::string_view path) {
    if (left.size != right.size) fail(ErrorCode::shape_mismatch, path, "dataframe index length differs from its aligned axis");
    try {
        if (auto* l = std::get_if<BackedTable>(&left.table)) {
            if (auto* r = std::get_if<BackedTable>(&right.table))
                backed_detail::compare_table_indexes(*r, *l);
            else backed_detail::compare_table_indexes(*l, std::get<TableView>(right.table));
            return;
        }
        if (auto* r = std::get_if<BackedTable>(&right.table)) {
            backed_detail::compare_table_indexes(*r, std::get<TableView>(left.table));
            return;
        }
    } catch (const Error& error) {
        throw Error(error.code(), std::string(path), error.what());
    }
    const auto& l = std::get<TableView>(left.table);
    const auto& r = std::get<TableView>(right.table);
    for (std::size_t i = 0; i < left.size; ++i)
        if (l.name(i) != r.name(i)) fail(ErrorCode::shape_mismatch, path, "dataframe index differs from its aligned axis");
}
void check_aligned_table(const StructureEditImpl& edit, const BackedAnnDataView& selected,
                         std::string_view path, const Index& index) {
    if (auto change = change_for(edit, path)) {
        if (!*change || source_kind(**change) != ElementKind::table) return;
        compare_indexes(index, index_of_source(**change, path), path);
        return;
    }
    const auto key = last_component(path);
    BackedElement item;
    if (starts_child(path, "/obsm")) item = selected.obsm(key);
    else if (starts_child(path, "/varm")) item = selected.varm(key);
    else item = selected.raw_varm(key);
    if (item.kind() == ElementKind::table) compare_indexes(index, index_of(item.table()), path);
}
void validate_column_changes(const StructureEditImpl& edit, const BackedTable& source,
                             Session& session, std::string_view table_path, std::size_t target_size) {
    auto columns = source.column_names();
    const auto original_shape = backed_detail::logical_shape(session, table_path);
    if (original_shape.size() != 1)
        fail(ErrorCode::shape_mismatch, table_path, "dataframe index is not one-dimensional");
    std::set<std::string, std::less<>> names(columns.begin(), columns.end());
    for (const auto& [path, value] : edit.changes) {
        if (!starts_child(path, table_path) || parent_path(path) != table_path) continue;
        const auto key = last_component(path);
        if (!value) {
            if (!names.erase(key)) fail(ErrorCode::out_of_bounds, path, "column does not exist");
        } else {
            if (source_kind(*value) != ElementKind::column)
                fail(ErrorCode::unsupported_encoding, path, "dataframe replacement requires a Column element");
            if (source_shape(*value, path).dims[0] != target_size)
                fail(ErrorCode::shape_mismatch, path, "column length differs from its final dataframe axis");
            if (!source.index_name().empty() && key == source.index_name()) {
                auto input = std::get_if<Element>(&*value);
                if (!input)
                    fail(ErrorCode::invalid_argument, path, "use an in-memory string column when replacing the dataframe index alias");
                auto column = input->view().column();
                if (column.kind() != ColumnKind::string || column.nullable())
                    fail(ErrorCode::invalid_argument, path, "column sharing the dataframe index name must contain the index strings");
                for (std::size_t row = 0; row < target_size; ++row)
                    if (column.string(row) != source.name(row))
                        fail(ErrorCode::invalid_argument, path, "column sharing the dataframe index name must match its index");
            }
            names.insert(key);
        }
    }
    for (const auto& key : names) {
        const auto path = backed_detail::child(table_path, key);
        if (change_for(edit, path)) continue;
        const auto physical_shape = backed_detail::logical_shape(session, path);
        if (physical_shape.size() != 1 || physical_shape[0] != original_shape[0] ||
            source.column(key).size() != target_size)
            fail(ErrorCode::shape_mismatch, path, "existing column length differs from its final dataframe axis");
    }
}
BackedTable nested_table(const BackedAnnDataView& selected, std::string_view path) {
    const auto parts = components(path);
    if (parts.size() == 2 && parts[0] == "obsm") return selected.obsm(parts[1]).table();
    if (parts.size() == 2 && parts[0] == "varm") return selected.varm(parts[1]).table();
    if (parts.size() == 3 && parts[0] == "raw" && parts[1] == "varm") return selected.raw_varm(parts[2]).table();
    if (!parts.empty() && parts[0] == "uns") {
        auto item = selected.uns();
        for (std::size_t i = 1; i < parts.size(); ++i) item = item.at(parts[i]);
        return item.table();
    }
    fail(ErrorCode::invalid_argument, path, "unsupported dataframe location");
}
void validate_parent(Backend& backend, std::string_view path) {
    const auto parent = parent_path(path);
    if (parent == "/") return;
    if (starts_child(parent, "/raw")) {
        auto raw = backend.inspect("/raw");
        if (!raw.exists || backed_detail::encoding(raw, "/raw") == "null") {
            if (parent == "/raw/varm" || parent == "/raw/var") return;
            fail(ErrorCode::out_of_bounds, parent, "parent mapping does not exist");
        }
    }
    const auto info = backend.inspect(parent);
    if (!info.exists) {
        // Conventional slot mappings, and a new raw group, may be created.
        if (parent == "/layers" || parent == "/obsm" || parent == "/varm" || parent == "/obsp" ||
            parent == "/varp" || parent == "/raw" || parent == "/raw/varm" || parent == "/uns") return;
        fail(ErrorCode::out_of_bounds, parent, "parent mapping does not exist");
    }
    const auto code = backed_detail::encoding(info, parent);
    if (code != "dict" && code != "dataframe" && code != "raw" && !(parent == "/raw" && code == "null"))
        fail(ErrorCode::invalid_argument, parent, "parent is not a mapping or dataframe");
}
struct Preflight {
    BackedAnnDataView source;
    BackedAnnDataView selected;
    Index obs, var;
    std::optional<Index> raw_var;
    bool reindex_obs = false, reindex_var = false, raw = false;
};
Preflight preflight(StructureEditImpl& edit) {
    usable(&edit);
    auto& session = *edit.dataset->session;
    auto& backend = *session.backend;
    auto source = backed_detail::root_view(edit.dataset);
    auto selected = edit.selection ? source.slice(edit.selection->first, edit.selection->second) : source;
    std::set<std::string, std::less<>> nested_dataframes;
    for (const auto& [path, value] : edit.changes) {
        validate_parent(backend, path);
        const bool column = is_dataframe_column(session, path);
        const auto parts = components(path);
        if (column && parts.back() == "_index")
            fail(ErrorCode::invalid_argument, path, "the reserved dataframe index cannot be edited as a column");
        if (column && !is_table_column(path)) nested_dataframes.insert(parent_path(path));
        if (!column && ((parts.size() == 3 && (parts[0] == "obsm" || parts[0] == "varm")) ||
                        (parts.size() == 4 && parts[0] == "raw" && parts[1] == "varm")))
            fail(ErrorCode::invalid_argument, path, "parent is not a dataframe");
        if (value) {
            validate_source(*value, path);
        } else if (!column && !backend.inspect(path).exists) {
            fail(ErrorCode::out_of_bounds, path, "element to erase does not exist");
        }
    }
    const bool raw_present = source.has_raw() && !is_erased(edit, "/raw");
    const bool creates_raw = !source.has_raw() && change_for(edit, "/raw/var") && !is_erased(edit, "/raw/var");
    for (const auto& [path, value] : edit.changes) {
        if (starts_child(path, "/raw") && !raw_present && !creates_raw)
            fail(ErrorCode::invalid_argument, path, "raw must have a variable table; supply /raw/var when creating raw");
        if (path == "/uns" && value && source_kind(*value) != ElementKind::mapping)
            fail(ErrorCode::unsupported_encoding, path, "uns must be a mapping");
    }
    auto obs = axis_index(edit, selected, "/obs");
    auto var = axis_index(edit, selected, "/var");
    std::optional<Index> raw_var;
    if (raw_present || creates_raw) raw_var = axis_index(edit, selected, "/raw/var");
    const auto no = obs.size, nv = var.size;
    if (!change_for(edit, "/obs")) validate_column_changes(edit, selected.obs(), session, "/obs", no);
    if (!change_for(edit, "/var")) validate_column_changes(edit, selected.var(), session, "/var", nv);
    if (raw_present && !change_for(edit, "/raw/var")) validate_column_changes(edit, selected.raw_var(), session, "/raw/var", raw_var->size);
    for (const auto& table_path : nested_dataframes) {
        auto table = nested_table(selected, table_path);
        validate_column_changes(edit, table, session, table_path, table.size());
    }
    const bool x = !is_erased(edit, "/X") && (change_for(edit, "/X") || physical_present(backend, "/X"));
    if (x) matrix_shape(final_shape(edit, session, "/X", selected.n_obs(), selected.n_vars()), no, nv, "/X");
    for (auto base : {"/layers", "/obsm", "/varm", "/obsp", "/varp"}) {
        for (const auto& key : names_with_changes(backend, edit, base)) {
            const auto path = backed_detail::child(base, key);
            auto shape = final_shape(edit, session, path, selected.n_obs(), selected.n_vars());
            if (std::string_view(base) == "/layers") matrix_shape(shape, no, nv, path);
            else if (std::string_view(base) == "/obsp") pairwise_shape(shape, no, path);
            else if (std::string_view(base) == "/varp") pairwise_shape(shape, nv, path);
            else {
                const auto& index = std::string_view(base) == "/obsm" ? obs : var;
                aligned_shape(shape, index.size, path);
                if (shape.kind == ElementKind::table) check_aligned_table(edit, selected, path, index);
            }
        }
    }
    if (raw_var) {
        const bool raw_x = !is_erased(edit, "/raw/X") && (change_for(edit, "/raw/X") || (raw_present && physical_present(backend, "/raw/X")));
        if (raw_x) matrix_shape(final_shape(edit, session, "/raw/X", selected.n_obs(), selected.n_vars()), no, raw_var->size, "/raw/X");
        for (const auto& key : names_with_changes(backend, edit, "/raw/varm")) {
            const auto path = backed_detail::child("/raw/varm", key);
            auto shape = final_shape(edit, session, path, selected.n_obs(), selected.n_vars());
            aligned_shape(shape, raw_var->size, path);
            if (shape.kind == ElementKind::table) check_aligned_table(edit, selected, path, *raw_var);
        }
    }
    return {std::move(source), std::move(selected), std::move(obs), std::move(var), std::move(raw_var),
            edit.selection && !edit.selection->first.is_all(), edit.selection && !edit.selection->second.is_all(), raw_present || creates_raw};
}
struct Generated {
    std::string path;
    std::optional<StructureSource> source;
    bool category_column = false;
    bool color = false;
};
std::vector<Generated> reindex_actions(const StructureEditImpl& edit, const Preflight& p) {
    std::vector<Generated> generated;
    const auto& selected = p.selected;
    auto add_array = [&](std::string path, BackedArray value) {
        if (!change_for(edit, path)) generated.push_back({std::move(path), StructureSource(std::move(value))});
    };
    auto add_element = [&](std::string path, BackedElement value) {
        if (!change_for(edit, path)) generated.push_back({std::move(path), StructureSource(std::move(value))});
    };
    auto add_table = [&](std::string path, BackedTable value) {
        if (!change_for(edit, path)) generated.push_back({std::move(path), StructureSource(std::move(value))});
    };
    if (!edit.selection) return generated;
    // Color plans depend on the original categorical codes. Capture the
    // transformed children now and stage every changed value before any
    // categorical source is replaced. Explicit edits to a color win.
    if (!change_for(edit, "/uns") && edit.dataset->session->backend->inspect("/uns").exists) {
        auto uns = selected.uns();
        const auto& uns_node = detail::BackedAccess::get(uns);
        for (const auto& key : backed_detail::color_changed_keys(uns_node)) {
            auto path = backed_detail::child("/uns", key);
            bool overridden = change_for(edit, path);
            if (!overridden) for (const auto& [changed, unused] : edit.changes) {
                (void)unused;
                if (starts_child(changed, path)) { overridden = true; break; }
            }
            if (overridden) continue;
            auto value = backed_detail::color_child(uns_node, key);
            if (value)
                generated.push_back({std::move(path), StructureSource(detail::BackedAccess::element(std::move(value))), false, true});
            else
                generated.push_back({std::move(path), std::nullopt, false, true});
        }
    }
    if (p.reindex_obs || p.reindex_var) {
        if (selected.has_x()) add_array("/X", selected.x());
        for (const auto& key : selected.layer_names()) add_array(backed_detail::child("/layers", key), selected.layer(key));
    }
    if (p.reindex_obs) {
        for (const auto& key : selected.obsm_names()) add_element(backed_detail::child("/obsm", key), selected.obsm(key));
        for (const auto& key : selected.obsp_names()) add_array(backed_detail::child("/obsp", key), selected.obsp(key));
        if (p.source.has_raw() && !is_erased(edit, "/raw") && selected.has_raw_x()) add_array("/raw/X", selected.raw_x());
    }
    if (p.reindex_var) {
        for (const auto& key : selected.varm_names()) add_element(backed_detail::child("/varm", key), selected.varm(key));
        for (const auto& key : selected.varp_names()) add_array(backed_detail::child("/varp", key), selected.varp(key));
    }
    // Axis tables are intentionally last. Slices and source handles captured
    // above keep their original axes until their individual payload is read.
    if (p.reindex_obs) add_table("/obs", selected.obs());
    if (p.reindex_var) add_table("/var", selected.var());
    // Explicit slicing also prunes unused categories on an identity axis.
    // Rewrite only those columns; unchanged numerical slots and table indexes
    // remain on disk. A caller-supplied table/column replacement wins.
    auto add_identity_categories = [&](std::string_view path, BackedTable table) {
        if (change_for(edit, path)) return;
        for (const auto& key : table.column_names()) {
            auto column_path = backed_detail::child(path, key);
            if (change_for(edit, column_path)) continue;
            auto column = table.column(key);
            if (column.kind() == ColumnKind::categorical)
                generated.push_back({std::move(column_path), StructureSource(std::move(column)), true});
        }
    };
    if (!p.reindex_obs) {
        add_identity_categories("/obs", selected.obs());
        for (const auto& key : selected.obsm_names()) {
            auto path = backed_detail::child("/obsm", key);
            if (change_for(edit, path)) continue;
            auto item = selected.obsm(key);
            if (item.kind() == ElementKind::table) add_identity_categories(path, item.table());
        }
    }
    if (!p.reindex_var) {
        add_identity_categories("/var", selected.var());
        for (const auto& key : selected.varm_names()) {
            auto path = backed_detail::child("/varm", key);
            if (change_for(edit, path)) continue;
            auto item = selected.varm(key);
            if (item.kind() == ElementKind::table) add_identity_categories(path, item.table());
        }
    }
    return generated;
}
Attributes coded(std::string type, std::string version) {
    return {{"encoding-type", std::move(type)}, {"encoding-version", std::move(version)}};
}
void ensure_parent(Session& session, std::string_view path) {
    auto& backend = *session.backend;
    const auto parent = parent_path(path);
    if (parent == "/") return;
    if (parent == "/raw" || parent == "/raw/varm") {
        const auto raw = backend.inspect("/raw");
        if (!raw.exists || backed_detail::encoding(raw, "/raw") == "null") {
            session.invalidate("/raw");
            if (raw.exists) backend.erase("/raw");
            backend.create_group("/raw", coded("raw", "0.1.0"));
        }
        if (parent == "/raw/varm" && !backend.inspect(parent).exists)
            backend.create_group(parent, coded("dict", "0.1.0"));
        return;
    }
    if (!backend.inspect(parent).exists) {
        if (parent == "/layers" || parent == "/obsm" || parent == "/varm" || parent == "/obsp" || parent == "/varp" || parent == "/uns") {
            session.invalidate(parent);
            backend.create_group(parent, coded("dict", "0.1.0"));
        } else fail(ErrorCode::out_of_bounds, parent, "parent disappeared before structural edit was applied");
    }
}
std::shared_ptr<detail::BackedNode> source_node(const StructureSource& source) {
    return std::visit([](const auto& value) -> std::shared_ptr<detail::BackedNode> {
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Element>) return {};
        else return detail::BackedAccess::get(value);
    }, source);
}
std::string unused_temporary(Backend& backend, std::size_t& counter) {
    for (;;) {
        auto path = "/.anndata_cppm_temporary_" + std::to_string(counter++);
        if (!backend.inspect(path).exists) return path;
    }
}
void stage(Session& session, const StructureSource& source, std::string& temporary,
           std::size_t& counter) {
    if (!temporary.empty()) return;
    auto node = source_node(source);
    temporary = unused_temporary(*session.backend, counter);
    if (node) backed_detail::copy_node(*session.backend, temporary, node, session.options.scratch_bytes);
    else backed_detail::write_element(*session.backend, temporary, std::get<Element>(source).view(), session.options.scratch_bytes);
}
void preserve_matching_attributes(Backend& backend, std::string_view old_path, std::string_view new_path) {
    auto old_info = backend.inspect(old_path);
    auto new_info = backend.inspect(new_path);
    if (!old_info.exists || !new_info.exists) return;
    backend.copy_attributes(old_path, new_path);
    if (!old_info.group || !new_info.group) return;
    for (const auto& name : backend.children(old_path)) {
        auto old_child = backed_detail::child(old_path, name);
        auto new_child = backed_detail::child(new_path, name);
        if (backend.inspect(new_child).exists) preserve_matching_attributes(backend, old_child, new_child);
    }
}
void replace(Session& session, std::string_view path, const StructureSource& source,
             std::string& temporary, std::size_t& counter) {
    if (source_node(source)) stage(session, source, temporary, counter);
    ensure_parent(session, path);
    auto& backend = *session.backend;
    const bool exists = backend.inspect(path).exists;
    // Preserve metadata that is not represented by the new semantic value.
    // Copy it while the old object still exists, so a failure on an unsupported
    // attribute cannot destroy the original target.
    if (exists) {
        stage(session, source, temporary, counter);
        preserve_matching_attributes(backend, path, temporary);
    }
    session.invalidate(path);
    if (exists) backend.erase(path);
    if (temporary.empty()) {
        const auto* value = std::get_if<Element>(&source);
        backed_detail::write_element(backend, path, value->view(), session.options.scratch_bytes);
    } else {
        backend.rename(temporary, path);
        temporary.clear();
    }
}
void erase_stored(Session& session, std::string_view path) {
    auto& backend = *session.backend;
    session.invalidate(path);
    if (!backend.inspect(path).exists) fail(ErrorCode::out_of_bounds, path, "element disappeared before it could be erased");
    backend.erase(path);
}
void apply_column(Session& session, std::string_view path, const std::optional<StructureSource>& value,
                  std::string& temporary, std::size_t& counter) {
    auto& backend = *session.backend;
    const auto table = parent_path(path);
    const auto key = last_component(path);
    const auto info = backend.inspect(table);
    if (!info.exists || backed_detail::encoding(info, table) != "dataframe")
        fail(ErrorCode::invalid_argument, table, "column parent is not a dataframe");
    const auto index_key = backed_detail::string_attr(info, "_index", table);
    auto columns = backed_detail::strings_attr(info, "column-order", table);
    const auto where = std::find(columns.begin(), columns.end(), key);
    // The named-index collision is a single on-disk array used by both the
    // index and the column; preflight has established that its values match.
    if (value) {
        if (key != index_key) replace(session, path, *value, temporary, counter);
        if (where == columns.end()) columns.push_back(key);
    } else {
        if (where == columns.end()) fail(ErrorCode::out_of_bounds, path, "column disappeared before it could be erased");
        columns.erase(where);
        if (key != index_key) erase_stored(session, path);
    }
    auto attrs = info.attrs;
    attrs.insert_or_assign("column-order", std::move(columns));
    session.invalidate(table);
    backend.attributes(table, attrs);
}
} // namespace

StructureEdit::StructureEdit() = default;
StructureEdit::StructureEdit(std::unique_ptr<detail::StructureEditImpl> impl) : impl_(std::move(impl)) {}
StructureEdit::~StructureEdit() = default;
StructureEdit::StructureEdit(StructureEdit&&) noexcept = default;
StructureEdit& StructureEdit::operator=(StructureEdit&&) noexcept = default;
void StructureEdit::set(std::string path, Element value) {
    usable(impl_.get());
    put(*impl_, std::move(path), std::optional<StructureSource>(std::in_place, std::move(value)));
}
void StructureEdit::set(std::string path, BackedArray value) {
    usable(impl_.get());
    put(*impl_, std::move(path), std::optional<StructureSource>(std::in_place, std::move(value)));
}
void StructureEdit::set(std::string path, BackedElement value) {
    usable(impl_.get());
    put(*impl_, std::move(path), std::optional<StructureSource>(std::in_place, std::move(value)));
}
void StructureEdit::set(std::string path, BackedTable value) {
    usable(impl_.get());
    put(*impl_, std::move(path), std::optional<StructureSource>(std::in_place, std::move(value)));
}
void StructureEdit::set(std::string path, BackedColumn value) {
    usable(impl_.get());
    put(*impl_, std::move(path), std::optional<StructureSource>(std::in_place, std::move(value)));
}
void StructureEdit::erase(std::string path) {
    usable(impl_.get());
    put(*impl_, std::move(path), std::nullopt);
}
void StructureEdit::set_obs(Table value) { set("/obs", Element(std::move(value))); }
void StructureEdit::set_var(Table value) { set("/var", Element(std::move(value))); }
void StructureEdit::reindex(Selection obs, Selection var) {
    usable(impl_.get());
    impl_->selection.emplace(std::move(obs), std::move(var));
}
void StructureEdit::apply() {
    usable(impl_.get());
    auto plan = preflight(*impl_);
    auto generated = reindex_actions(*impl_, plan);
    auto& session = *impl_->dataset->session;
    struct Action {
        std::string path;
        const StructureSource* source = nullptr;
        const std::optional<StructureSource>* column = nullptr;
        std::string temporary;
        bool category_column = false;
        bool color = false;
    };
    std::vector<Action> actions;
    for (const auto& action : generated)
        actions.push_back({action.path, action.source ? &*action.source : nullptr, nullptr, {}, action.category_column, action.color});
    for (const auto& [path, value] : impl_->changes)
        if (!is_dataframe_column(session, path)) actions.push_back({path, value ? &*value : nullptr, nullptr, {}});
    for (const auto& [path, value] : impl_->changes)
        if (is_dataframe_column(session, path)) actions.push_back({path, value ? &*value : nullptr, &value, {}});
    std::string current = "structure";
    std::size_t completed = 0;
    std::size_t temporary_counter = 0;
    bool attempted_mutation = false;
    impl_->consumed = true;
    auto cleanup = [&]() noexcept {
        bool failed = false;
        for (auto& action : actions) if (!action.temporary.empty()) {
            try {
                if (session.backend->inspect(action.temporary).exists) session.backend->erase(action.temporary);
                action.temporary.clear();
            } catch (...) { failed = true; }
        }
        return failed;
    };
    try {
        for (auto& action : actions) {
            if (!action.color || !action.source) continue;
            current = action.path;
            attempted_mutation = true;
            stage(session, *action.source, action.temporary, temporary_counter);
        }
        // A source can refer to another target in this batch (e.g. a swap).
        // Stage only such endangered sources before any target is changed.
        for (std::size_t i = 0; i < actions.size(); ++i) {
            auto& action = actions[i];
            if (!action.source) continue;
            auto node = source_node(*action.source);
            if (!node || !node->session || !node->session->backend) continue;
            if (node->session.get() != &session &&
                node->session->backend->identity() != session.backend->identity()) continue;
            bool endangered = false;
            for (std::size_t j = 0; j < actions.size(); ++j) {
                if (i == j) continue;
                const auto changed = actions[j].column ? parent_path(actions[j].path) : actions[j].path;
                if (node->path == changed || starts_child(node->path, changed)) { endangered = true; break; }
            }
            if (endangered) {
                current = action.path;
                attempted_mutation = true;
                stage(session, *action.source, action.temporary, temporary_counter);
            }
        }
        // Ordinary reindexing stages one node at a time; no full-store
        // snapshot or whole-store transaction is created.
        for (auto& action : actions) {
            current = action.path;
            attempted_mutation = true;
            if (action.column) apply_column(session, action.path, *action.column, action.temporary, temporary_counter);
            else if (action.source) replace(session, action.path, *action.source, action.temporary, temporary_counter);
            else erase_stored(session, action.path);
            ++completed;
        }
        for (const auto& action : actions)
            if (action.category_column) session.invalidate(parent_path(action.path));
        if (cleanup()) fail(ErrorCode::io_error, "structure", "could not remove an owned temporary node");
        impl_->dataset->obs = {plan.obs.size, 0, 1, {}};
        impl_->dataset->var = {plan.var.size, 0, 1, {}};
        if (attempted_mutation) impl_->dataset->observed_revision = ++session.dataset_revision;
    } catch (const Error& error) {
        const bool cleanup_failed = cleanup();
        if (attempted_mutation) {
            session.invalidate("/");
            impl_->dataset->observed_revision = ++session.dataset_revision;
        }
        const auto prefix = attempted_mutation
            ? "structural edit may be partially applied; reopen the store and inspect it; completed elements: "
            : "structural edit failed before any store mutation; completed elements: ";
        throw Error(error.code(), session.backend->location() + ":" + current,
                    std::string(prefix) + std::to_string(completed) + "; " + error.what() +
                    (cleanup_failed ? "; an owned temporary node could not be removed" : ""));
    } catch (const std::exception& error) {
        const bool cleanup_failed = cleanup();
        if (attempted_mutation) {
            session.invalidate("/");
            impl_->dataset->observed_revision = ++session.dataset_revision;
        }
        const auto prefix = attempted_mutation
            ? "structural edit may be partially applied; reopen the store and inspect it; completed elements: "
            : "structural edit failed before any store mutation; completed elements: ";
        throw Error(ErrorCode::io_error, session.backend->location() + ":" + current,
                    std::string(prefix) + std::to_string(completed) + "; " + error.what() +
                    (cleanup_failed ? "; an owned temporary node could not be removed" : ""));
    }
}
StructureEdit BackedAnnData::edit_structure() {
    if (!impl_ || !impl_->session) fail(ErrorCode::invalid_argument, "structure", "backed dataset is empty");
    impl_->session->check_writable();
    return StructureEdit(std::make_unique<detail::StructureEditImpl>(impl_));
}
void BackedAnnData::set(std::string path, Element value) {
    auto edit = edit_structure();
    edit.set(std::move(path), std::move(value));
    edit.apply();
}
void BackedAnnData::set(std::string path, BackedArray value) {
    auto edit = edit_structure(); edit.set(std::move(path), std::move(value)); edit.apply();
}
void BackedAnnData::set(std::string path, BackedElement value) {
    auto edit = edit_structure(); edit.set(std::move(path), std::move(value)); edit.apply();
}
void BackedAnnData::set(std::string path, BackedTable value) {
    auto edit = edit_structure(); edit.set(std::move(path), std::move(value)); edit.apply();
}
void BackedAnnData::set(std::string path, BackedColumn value) {
    auto edit = edit_structure(); edit.set(std::move(path), std::move(value)); edit.apply();
}
void BackedAnnData::erase(std::string path) {
    auto edit = edit_structure();
    edit.erase(std::move(path));
    edit.apply();
}
} // namespace anndata
