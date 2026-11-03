module;
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <variant>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

module anndata.backed;
import :storage;

namespace anndata::backed_detail {
namespace {
using Node = std::shared_ptr<detail::BackedNode>;
[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}
struct Resolution {
    Node value;
    bool changed = false;
};
std::optional<std::string> category_path(const Node& uns, std::string_view table, std::string_view name) {
    auto table_info = uns->session->backend->inspect(table);
    if (!table_info.exists || encoding(table_info, table) != "dataframe") return std::nullopt;
    const auto columns = strings_attr(table_info, "column-order", table);
    if (std::find(columns.begin(), columns.end(), name) == columns.end()) return std::nullopt;
    auto path = child(table, name);
    const auto info = uns->session->backend->inspect(path);
    if (encoding(info, path) != "categorical") return std::nullopt;
    return path;
}
Resolution resolve(const Node& uns, std::string_view key) {
    check_node(uns);
    if (uns->empty_mapping) return {};
    auto path = child(uns->path, key);
    auto info = uns->session->backend->inspect(path);
    if (!info.exists) return {};
    auto source = node(uns->session, path);
    source->writable = uns->writable;
    if (!uns->slice_history || uns->slice_history->empty() || !key.ends_with("_colors")) return {std::move(source), false};
    auto column_name = key.substr(0, key.size() - std::string_view("_colors").size());
    if (column_name.empty()) return {std::move(source), false};
    auto obs_path = category_path(uns, "/obs", column_name);
    auto var_path = category_path(uns, "/var", column_name);
    if (!obs_path && !var_path) return {std::move(source), false};

    auto code = encoding(info, path);
    auto shape = logical_shape(*uns->session, path);
    const bool scalar = code == "null" || code == "string" || code == "numeric-scalar" ||
                        ((code == "array" || code == "string-array") && shape.empty());
    bool valid = code == "array" || code == "csr_matrix" || code == "csc_matrix" ||
                 code == "string-array" || code == "categorical" ||
                 code == "nullable-integer" || code == "nullable-boolean" || code == "nullable-string-array";
    bool promoted = false, changed = false;
    std::size_t extent = shape.empty() ? 0 : shape.front();
    std::shared_ptr<CategoryPlan> color_positions;
    const auto depth = uns->slice_history->size();
    if (depth > (std::numeric_limits<std::size_t>::max() - 2) / 4)
        fail(ErrorCode::invalid_argument, path, "slice history exceeds supported size");
    // Both axes can share a palette name. Reserve payload for every before
    // and after plan retained across the full history; plan metadata is small.
    const auto plan_count = 4 * depth + 2;
    const auto scratch = uns->session->options.scratch_bytes;
    const auto plan_budget = std::max<std::size_t>(64, scratch / plan_count);
    const auto retained_per_plan = std::max<std::size_t>(sizeof(std::size_t), plan_budget / 8);
    if (plan_count > scratch / retained_per_plan)
        fail(ErrorCode::invalid_argument, path, "scratch budget is too small for categorical slice history");
    for (std::size_t step = 0; step < uns->slice_history->size(); ++step) {
        const auto& stage = (*uns->slice_history)[step];
        auto apply = [&](const std::optional<std::string>& categorical, const Axis& before, const Axis& after) {
            if (!categorical || !source) return;
            auto selected = node(uns->session, *categorical, {after});
            auto after_plan = make_category_plan(selected, plan_budget);
            std::shared_ptr<CategoryPlan> before_plan;
            if (step != 0) before_plan = make_category_plan(node(uns->session, *categorical, {before}), plan_budget);
            const auto previous = before_plan ? category_count(before_plan) : category_source_count(after_plan);
            const auto remaining = category_count(after_plan);
            if (scalar && !promoted) {
                promoted = true;
                valid = true;
                extent = 1;
                changed = true;
                ColorPromotion type;
                if (code == "null") type = ColorPromotion::nullable_string_column;
                else if (code == "string") type = ColorPromotion::string_column;
                else if (code == "string-array") type = ColorPromotion::string_array;
                else type = ColorPromotion::numeric_array;
                source->color_override = std::make_shared<ColorOverride>(ColorOverride{type, Axis{1}});
                return;
            }
            if (!valid || extent != previous) {
                source.reset(); changed = true; return;
            }
            if (remaining != previous) changed = true;
            extent = remaining;
            if (promoted) {
                // Once promoted, a scalar can only have logical extent zero or
                // one; its selected values all refer to the same stored cell.
                source->color_override->axis = Axis{remaining};
            } else {
                color_positions = compose_category_plans(color_positions, before_plan, after_plan);
            }
        };
        apply(obs_path, stage.obs_before, stage.obs_after);
        apply(var_path, stage.var_before, stage.var_after);
        if (!source) return {{}, true};
    }
    if (source && !promoted && color_positions) source->axes[0] = Axis{extent, 0, 1, {}, color_positions};
    return {std::move(source), changed};
}
std::string scalar_string(const Node& n) {
    auto value = n->session->backend->read_strings(n->path, {}, {});
    if (value.size() != 1) fail(ErrorCode::io_error, n->path, "string scalar transfer returned an incorrect length");
    if (n->session->options.validation == Validation::full) validate_utf8_string(value.front(), n->path);
    return std::move(value.front());
}
template<class T> Array scalar_array(const Node& n, std::size_t length) {
    auto result = Array::empty<T>(std::vector<std::size_t>{length});
    if (length) {
        T value{};
        if constexpr (std::is_same_v<T, bool>) {
            std::uint8_t stored = 0;
            n->session->backend->read_numeric(n->path, {}, {}, &stored);
            value = stored != 0;
        } else n->session->backend->read_numeric(n->path, {}, {}, &value);
        auto* data = result.edit().template dense<T>().data();
        std::fill_n(data, length, value);
    }
    auto info = n->session->backend->inspect(n->path);
    result.set_serialization_byte_order(info.byte_order);
    return result;
}
Array numeric_scalar_array(const Node& n, std::size_t length) {
    switch (n->session->backend->inspect(n->path).dtype) {
    case DType::boolean: return scalar_array<bool>(n, length);
    case DType::int8: return scalar_array<std::int8_t>(n, length);
    case DType::uint8: return scalar_array<std::uint8_t>(n, length);
    case DType::int16: return scalar_array<std::int16_t>(n, length);
    case DType::uint16: return scalar_array<std::uint16_t>(n, length);
    case DType::int32: return scalar_array<std::int32_t>(n, length);
    case DType::uint32: return scalar_array<std::uint32_t>(n, length);
    case DType::int64: return scalar_array<std::int64_t>(n, length);
    case DType::uint64: return scalar_array<std::uint64_t>(n, length);
    case DType::float32: return scalar_array<float>(n, length);
    case DType::float64: return scalar_array<double>(n, length);
    }
    fail(ErrorCode::unsupported_dtype, n->path, "unsupported scalar color dtype");
}
Attributes encoding_attrs(std::string type, std::string version) {
    return {{"encoding-type",std::move(type)},{"encoding-version",std::move(version)}};
}
} // namespace

std::vector<std::string> color_keys(const Node& uns) {
    check_node(uns);
    if (uns->empty_mapping) return {};
    auto names = uns->session->backend->children(uns->path);
    if (!uns->slice_history || uns->slice_history->empty()) return names;
    std::vector<std::string> result;
    result.reserve(names.size());
    for (auto& name : names) {
        if (!name.ends_with("_colors")) { result.push_back(std::move(name)); continue; }
        auto column_name = std::string_view(name).substr(0, name.size() - std::string_view("_colors").size());
        if (column_name.empty() || (!category_path(uns, "/obs", column_name) && !category_path(uns, "/var", column_name))) {
            result.push_back(std::move(name)); continue;
        }
        if (resolve(uns, name).value) result.push_back(std::move(name));
    }
    return result;
}
Node color_child(const Node& uns, std::string_view key) { return resolve(uns, key).value; }
std::vector<std::string> color_changed_keys(const Node& uns) {
    check_node(uns);
    std::vector<std::string> result;
    if (uns->empty_mapping) return result;
    if (!uns->slice_history || uns->slice_history->empty()) return result;
    for (const auto& name : uns->session->backend->children(uns->path)) {
        if (!name.ends_with("_colors")) continue;
        auto column_name = std::string_view(name).substr(0, name.size() - std::string_view("_colors").size());
        if (column_name.empty() || (!category_path(uns, "/obs", column_name) && !category_path(uns, "/var", column_name))) continue;
        if (resolve(uns, name).changed) result.push_back(name);
    }
    return result;
}
bool color_has_changes(const Node& uns) { return !color_changed_keys(uns).empty(); }
std::optional<Element> read_color_override(const Node& n) {
    check_node(n);
    if (!n->color_override) return std::nullopt;
    auto size = n->color_override->axis.size;
    switch (n->color_override->kind) {
    case ColorPromotion::numeric_array: return Element(numeric_scalar_array(n, size));
    case ColorPromotion::string_array: {
        auto value = size ? scalar_string(n) : std::string{};
        return Element(StringArray::from_strings({size}, std::vector<std::string>(size, value)));
    }
    case ColorPromotion::string_column: {
        auto value = size ? scalar_string(n) : std::string{};
        return Element(Column::strings(std::vector<std::string>(size, value)));
    }
    case ColorPromotion::nullable_string_column:
        return Element(Column::nullable_strings(std::vector<std::string>(size), std::vector<std::uint8_t>(size, 0)));
    }
    fail(ErrorCode::unsupported_encoding, n->path, "invalid scalar color promotion");
}
void copy_promoted_color(Backend& destination, std::string_view path, const Node& n, std::size_t scratch_bytes) {
    check_node(n);
    if (!n->color_override) fail(ErrorCode::invalid_argument, n->path, "color does not require scalar promotion");
    const auto size = n->color_override->axis.size;
    const auto budget = std::min(scratch_bytes, n->session->options.scratch_bytes);
    if (budget < 64) fail(ErrorCode::invalid_argument, path, "scratch budget must be at least 64 bytes");
    if (n->color_override->kind == ColorPromotion::numeric_array) {
        const auto info = n->session->backend->inspect(n->path);
        const auto width = dtype_size(info.dtype);
        destination.create_numeric(path, info.dtype, {size}, encoding_attrs("array", "0.2.0"), info.byte_order);
        if (!size) return;
        std::array<std::byte, 8> scalar{};
        n->session->backend->read_numeric(n->path, {}, {}, scalar.data());
        if (info.dtype == DType::boolean && n->session->options.validation == Validation::full)
            scalar[0] = scalar[0] == std::byte{0} ? std::byte{0} : std::byte{1};
        constexpr auto overhead = sizeof(std::array<std::byte, 8>) + sizeof(std::vector<std::byte>);
        if (budget < overhead + width) fail(ErrorCode::invalid_argument, path, "scratch budget is too small for promoted color transfer");
        const auto cap = (budget - overhead) / width;
        std::vector<std::byte> values;
        for (std::size_t first = 0; first < size;) {
            const auto count = std::min(cap, size - first);
            values.resize(count * width);
            for (std::size_t i = 0; i < count; ++i) std::memcpy(values.data() + i * width, scalar.data(), width);
            destination.write_numeric(path, {first}, {count}, values.data());
            first += count;
        }
        return;
    }
    const bool nullable = n->color_override->kind == ColorPromotion::nullable_string_column;
    auto values_path = std::string(path);
    std::string mask_path;
    if (nullable) {
        auto attributes = encoding_attrs("nullable-string-array", "0.1.0");
        attributes["na-value"] = std::string("NA");
        destination.create_group(path, attributes);
        values_path = child(path, "values"); mask_path = child(path, "mask");
    }
    destination.create_strings(values_path, {size}, encoding_attrs("string-array", "0.2.0"));
    if (nullable) destination.create_numeric(mask_path, DType::boolean, {size}, encoding_attrs("array", "0.2.0"), ByteOrder::native);
    if (!size) return;
    const auto value = nullable ? std::string{} : scalar_string(n);
    constexpr auto fixed = sizeof(std::string) + sizeof(std::vector<std::string>) + sizeof(std::vector<std::uint8_t>);
    if (value.capacity() >= budget || fixed > budget - value.capacity() - 1)
        fail(ErrorCode::invalid_argument, n->path, "string color exceeds staging budget");
    const auto available = budget - fixed - value.capacity() - 1;
    const auto stored_capacity = std::max(value.size(), std::string{}.capacity());
    if (stored_capacity > std::numeric_limits<std::size_t>::max() - sizeof(std::string) - 2)
        fail(ErrorCode::invalid_argument, n->path, "string color size overflows");
    const auto per_item = sizeof(std::string) + stored_capacity + 1 + (nullable ? 1 : 0);
    if (available < per_item) fail(ErrorCode::invalid_argument, n->path, "string color exceeds staging budget");
    const auto cap = available / per_item;
    for (std::size_t first = 0; first < size;) {
        const auto count = std::min(cap, size - first);
        std::vector<std::string> values(count, value);
        destination.write_strings(values_path, {first}, {count}, values);
        if (nullable) {
            std::vector<std::uint8_t> mask(count, 1);
            destination.write_numeric(mask_path, {first}, {count}, mask.data());
        }
        first += count;
    }
}
} // namespace anndata::backed_detail
