module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

module anndatapp.backed;
import :storage;

namespace anndatapp::backed_detail {

namespace {

template<class F> decltype(auto) visit_dtype(DType dtype, F&& function) {
    switch (dtype) {
        case DType::boolean: return function.template operator()<bool>();
        case DType::int8: return function.template operator()<std::int8_t>();
        case DType::uint8: return function.template operator()<std::uint8_t>();
        case DType::int16: return function.template operator()<std::int16_t>();
        case DType::uint16: return function.template operator()<std::uint16_t>();
        case DType::int32: return function.template operator()<std::int32_t>();
        case DType::uint32: return function.template operator()<std::uint32_t>();
        case DType::int64: return function.template operator()<std::int64_t>();
        case DType::uint64: return function.template operator()<std::uint64_t>();
        case DType::float32: return function.template operator()<float>();
        case DType::float64: return function.template operator()<double>();
    }
    throw Error(ErrorCode::unsupported_dtype, "backed element", "unknown numeric dtype");
}

[[noreturn]] void codec_fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}

NodeInfo inspect_node(const std::shared_ptr<detail::BackedNode>& value) {
    check_node(value);
    auto info = value->session->backend->inspect(value->path);
    if (!info.exists)
        codec_fail(ErrorCode::io_error, value->path, "storage element does not exist");
    return info;
}

void require_children(const std::shared_ptr<detail::BackedNode>& value,
                      std::vector<std::string> expected) {
    auto actual=value->session->backend->children(value->path);
    std::sort(actual.begin(),actual.end());
    std::sort(expected.begin(),expected.end());
    if(actual!=expected)codec_fail(ErrorCode::unsupported_encoding,value->path,
                                   "unexpected or missing stored children");
}

std::vector<std::size_t> base_shape(const std::shared_ptr<detail::BackedNode>& value) {
    const auto shape = logical_shape(*value->session, value->path);
    return shape;
}

std::vector<Axis> axes_for(const std::shared_ptr<detail::BackedNode>& value,
                           const std::vector<std::size_t>& shape) {
    if (value->axes.empty()) {
        std::vector<Axis> axes;
        axes.reserve(shape.size());
        for (const auto extent : shape) axes.push_back(Axis{extent, 0, 1, {}});
        return axes;
    }
    if (value->axes.size() != shape.size())
        codec_fail(ErrorCode::shape_mismatch, value->path,
                   "selected axes do not cover the storage rank");
    return value->axes;
}

std::vector<std::size_t> selected_shape(const std::shared_ptr<detail::BackedNode>& value) {
    const auto shape = base_shape(value);
    if (value->axes.empty()) return shape;
    if (value->axes.size() != shape.size())
        codec_fail(ErrorCode::shape_mismatch, value->path,
                   "selected axes do not cover the storage rank");
    std::vector<std::size_t> result;
    result.reserve(value->axes.size());
    for (const auto& axis : value->axes) result.push_back(axis.size);
    return result;
}

std::size_t element_count(const std::vector<std::size_t>& shape, std::string_view path) {
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) return 0;
    std::size_t count = 1;
    for (const auto extent : shape) {
        if (extent != 0 && count > std::numeric_limits<std::size_t>::max() / extent)
            codec_fail(ErrorCode::invalid_argument, path, "element count exceeds addressable range");
        count *= extent;
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        codec_fail(ErrorCode::invalid_argument, path, "element count exceeds addressable range");
    return count;
}

std::vector<std::size_t> coordinate(const std::vector<std::size_t>& shape,
                                    std::size_t flat) {
    std::vector<std::size_t> result(shape.size());
    for (std::size_t axis = shape.size(); axis-- > 0;) {
        result[axis] = shape[axis] == 0 ? 0 : flat % shape[axis];
        if (shape[axis] != 0) flat /= shape[axis];
    }
    return result;
}

std::vector<std::size_t> physical_coordinate(const std::vector<Axis>& axes,
                                             const std::vector<std::size_t>& logical,
                                             std::string_view path) {
    if (axes.size() != logical.size())
        codec_fail(ErrorCode::shape_mismatch, path, "selected coordinate rank differs");
    std::vector<std::size_t> result;
    result.reserve(axes.size());
    for (std::size_t i = 0; i < axes.size(); ++i) result.push_back(axes[i].at(logical[i]));
    return result;
}

bool valid_utf8(std::string_view value) noexcept {
    const auto continuation = [](unsigned char byte) { return byte >= 0x80 && byte <= 0xbf; };
    std::size_t i = 0;
    while (i < value.size()) {
        const auto first = static_cast<unsigned char>(value[i]);
        if (first <= 0x7f) { ++i; continue; }
        if (first >= 0xc2 && first <= 0xdf) {
            if (i + 1 >= value.size() ||
                !continuation(static_cast<unsigned char>(value[i + 1]))) return false;
            i += 2;
            continue;
        }
        if (first >= 0xe0 && first <= 0xef) {
            if (i + 2 >= value.size()) return false;
            const auto second = static_cast<unsigned char>(value[i + 1]);
            const auto third = static_cast<unsigned char>(value[i + 2]);
            if (!continuation(third)) return false;
            if (first == 0xe0) { if (second < 0xa0 || second > 0xbf) return false; }
            else if (first == 0xed) { if (second < 0x80 || second > 0x9f) return false; }
            else if (!continuation(second)) return false;
            i += 3;
            continue;
        }
        if (first >= 0xf0 && first <= 0xf4) {
            if (i + 3 >= value.size()) return false;
            const auto second = static_cast<unsigned char>(value[i + 1]);
            if (!continuation(static_cast<unsigned char>(value[i + 2])) ||
                !continuation(static_cast<unsigned char>(value[i + 3]))) return false;
            if (first == 0xf0) { if (second < 0x90 || second > 0xbf) return false; }
            else if (first == 0xf4) { if (second < 0x80 || second > 0x8f) return false; }
            else if (!continuation(second)) return false;
            i += 4;
            continue;
        }
        return false;
    }
    return true;
}

void validate_single_string(std::string_view value, std::string_view path) {
    if (!valid_utf8(value))
        codec_fail(ErrorCode::invalid_argument, path, "string is not valid UTF-8");
}

std::vector<std::string> read_strings_all(const std::shared_ptr<detail::BackedNode>& value,
                                          bool validate = true) {
    check_node(value);
    const auto shape = base_shape(value);
    const auto axes = axes_for(value, shape);
    const auto output_shape = selected_shape(value);
    const auto count = element_count(output_shape, value->path);
    std::vector<std::string> result;
    result.reserve(count);

    // Read no more than a scratch-sized run of adjacent last-axis values at a
    // time. A string that cannot fit is rejected by the backend admission check.
    const auto budget = scratch_remaining(value);
    const auto tile = std::max<std::size_t>(1, budget / (sizeof(std::string) + 1));
    std::size_t flat = 0;
    while (flat < count) {
        const auto logical = coordinate(output_shape, flat);
        auto start = physical_coordinate(axes, logical, value->path);
        std::vector<std::size_t> rectangle(logical.size(), 1);
        std::size_t run = 1;
        if (!logical.empty()) {
            const auto last = logical.size() - 1;
            const auto remaining = output_shape[last] - logical[last];
            const auto cap = std::min(remaining, tile);
            while (run < cap) {
                const auto next = axes[last].at(logical[last] + run);
                if (start[last] == std::numeric_limits<std::size_t>::max() ||
                    next != start[last] + run)
                    break;
                ++run;
            }
            rectangle[last] = run;
        }
        auto strings = value->session->backend->read_strings_bounded(
            value->path, start, rectangle, budget);
        if (strings.empty() || strings.size() > run)
            codec_fail(ErrorCode::io_error, value->path,
                       "backend returned an invalid bounded string prefix");
        const auto actual = strings.size();
        if (validate && value->session->options.validation == Validation::full) {
            for (const auto& string : strings) validate_single_string(string, value->path);
        }
        for (auto& string : strings) result.push_back(std::move(string));
        flat += actual;
    }
    return result;
}

struct PackedColumnStrings {
    std::vector<char> bytes;
    std::vector<std::size_t> offsets;
};
PackedColumnStrings read_packed_column_strings(const std::shared_ptr<detail::BackedNode>& value) {
    check_node(value);
    const auto original=base_shape(value);
    const auto selected=selected_shape(value);
    if(original.size()!=1 || selected.size()!=1)
        codec_fail(ErrorCode::shape_mismatch,value->path,"string column must be one-dimensional");
    const auto axes=axes_for(value,original);
    const auto budget=scratch_remaining(value);
    const auto cap=std::max<std::size_t>(1,budget/(sizeof(std::string)+1));
    PackedColumnStrings out;
    if(selected[0]==std::numeric_limits<std::size_t>::max())
        codec_fail(ErrorCode::invalid_argument,value->path,"string row count exceeds addressable range");
    out.offsets.reserve(selected[0]+1);
    out.offsets.push_back(0);
    auto walk=[&](const auto& visit) {
        for(std::size_t row=0;row<selected[0];) {
            auto first=axes[0].at(row);std::size_t run=1;
            while(run<cap && run<selected[0]-row && first<=std::numeric_limits<std::size_t>::max()-run && axes[0].at(row+run)==first+run)++run;
            auto batch=value->session->backend->read_strings_bounded(value->path,{first},{run},budget);
            if(batch.empty()||batch.size()>run)codec_fail(ErrorCode::io_error,value->path,"backend returned an invalid bounded string prefix");
            for(const auto& string:batch)visit(string);
            row+=batch.size();
        }
    };
    // A size pass allows a single final byte allocation. String payloads and
    // validity buffers never require a whole-column temporary allocation.
    walk([&](std::string_view string) {
        const auto current=out.offsets.back();
        const auto max=static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
        if(string.size()>max-current)codec_fail(ErrorCode::invalid_argument,value->path,"packed string bytes exceed addressable range");
        out.offsets.push_back(current+string.size());
    });
    out.bytes.resize(out.offsets.back());
    std::size_t row=0;
    walk([&](std::string_view string) {
        const auto begin=out.offsets[row],length=out.offsets[row+1]-begin;
        if(string.size()!=length)codec_fail(ErrorCode::io_error,value->path,"stored strings changed during materialization");
        if(length)std::memcpy(out.bytes.data()+begin,string.data(),length);
        ++row;
    });
    return out;
}

StringArray read_string_array(const std::shared_ptr<detail::BackedNode>& value) {
    check_node(value);
    const auto shape = base_shape(value);
    const auto axes = axes_for(value, shape);
    const auto output_shape = selected_shape(value);
    const auto count = element_count(output_shape, value->path);
    std::vector<char> bytes;
    std::vector<std::uint64_t> offsets;
    offsets.reserve(count + 1);
    offsets.push_back(0);
    const auto budget = scratch_remaining(value);
    const auto tile = std::max<std::size_t>(1, budget / (sizeof(std::string) + 1));
    std::size_t flat = 0;
    while (flat < count) {
        const auto logical = coordinate(output_shape, flat);
        auto start = physical_coordinate(axes, logical, value->path);
        std::vector<std::size_t> rectangle(logical.size(), 1);
        std::size_t run = 1;
        if (!logical.empty()) {
            const auto last = logical.size() - 1;
            const auto remaining = output_shape[last] - logical[last];
            const auto cap = std::min(remaining, tile);
            while (run < cap) {
                const auto next = axes[last].at(logical[last] + run);
                if (start[last] == std::numeric_limits<std::size_t>::max() ||
                    next != start[last] + run)
                    break;
                ++run;
            }
            rectangle[last] = run;
        }
        auto strings = value->session->backend->read_strings_bounded(
            value->path, start, rectangle, budget);
        if (strings.empty() || strings.size() > run)
            codec_fail(ErrorCode::io_error, value->path,
                       "backend returned an invalid bounded string prefix");
        const auto actual = strings.size();
        for (const auto& string : strings) {
            if (value->session->options.validation == Validation::full)
                validate_single_string(string, value->path);
            if (string.size() > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) - bytes.size())
                codec_fail(ErrorCode::invalid_argument, value->path,
                           "packed string bytes exceed addressable range");
            if (string.size() > std::numeric_limits<std::uint64_t>::max() - bytes.size())
                codec_fail(ErrorCode::invalid_argument, value->path,
                           "packed string offsets exceed uint64 range");
            bytes.insert(bytes.end(), string.begin(), string.end());
            offsets.push_back(static_cast<std::uint64_t>(bytes.size()));
        }
        flat += actual;
    }
    auto output = StringArray::from_packed(output_shape, std::move(bytes), std::move(offsets),
                                           Validation::none);
    return output;
}

std::string read_string_scalar(const std::shared_ptr<detail::BackedNode>& value) {
    check_node(value);
    if (!base_shape(value).empty())
        codec_fail(ErrorCode::shape_mismatch, value->path, "string scalar requires rank zero");
    const auto strings = value->session->backend->read_strings_bounded(value->path, {}, {}, scratch_remaining(value));
    if (strings.size() != 1)
        codec_fail(ErrorCode::io_error, value->path,
                   "backend returned an unexpected scalar string count");
    if (value->session->options.validation == Validation::full)
        validate_single_string(strings.front(), value->path);
    return strings.front();
}

bool required_bool(const NodeInfo& info, std::string_view key, std::string_view path) {
    const auto found = info.attrs.find(key);
    if (found == info.attrs.end() || !std::holds_alternative<bool>(found->second))
        codec_fail(ErrorCode::unsupported_encoding, path,
                   "required boolean attribute '" + std::string(key) + "' is missing or invalid");
    return std::get<bool>(found->second);
}

std::string optional_string(const NodeInfo& info, std::string_view key,
                            std::string_view path, std::string fallback) {
    const auto found = info.attrs.find(key);
    if (found == info.attrs.end()) return fallback;
    if (!std::holds_alternative<std::string>(found->second))
        codec_fail(ErrorCode::unsupported_encoding, path,
                   "attribute '" + std::string(key) + "' must be a string");
    return std::get<std::string>(found->second);
}

std::shared_ptr<detail::BackedNode> child_node(
    const std::shared_ptr<detail::BackedNode>& parent, std::string_view name,
    bool inherit_axes = true) {
    check_node(parent);
    const auto path = child(parent->path, name);
    std::vector<Axis> axes;
    if (inherit_axes && !parent->axes.empty()) {
        const auto info = parent->session->backend->inspect(path);
        if (!info.exists)
            codec_fail(ErrorCode::io_error, path, "storage element does not exist");
        if (info.group) {
            axes = parent->axes;
        } else if (info.shape.size() >= parent->axes.size()) {
            axes = parent->axes;
            for (std::size_t i = axes.size(); i < info.shape.size(); ++i)
                axes.push_back(Axis{info.shape[i], 0, 1, {}});
        } // A scalar child is not aligned with the parent's selected axes.
    }
    auto result = node(parent->session, path, std::move(axes));
    result->writable = parent->writable;
    result->prune_categories = parent->prune_categories;
    return result;
}

std::string element_encoding(const std::shared_ptr<detail::BackedNode>& value) {
    const auto info = inspect_node(value);
    return encoding(info, value->path);
}

std::string table_index_key(const std::shared_ptr<detail::BackedNode>& value,
                            const NodeInfo* known_info = nullptr) {
    const auto info = known_info ? *known_info : inspect_node(value);
    return string_attr(info, "_index", value->path);
}

std::shared_ptr<detail::BackedNode> table_index_node(
    const std::shared_ptr<detail::BackedNode>& value, bool selected = true) {
    const auto info = inspect_node(value);
    const auto key = table_index_key(value, &info);
    const auto index_path = child(value->path, key);
    if (!selected || value->axes.empty()) return node(value->session, index_path);
    if (value->axes.size() != 1)
        codec_fail(ErrorCode::shape_mismatch, value->path,
                   "dataframe selection must contain exactly one row axis");
    return node(value->session, index_path, {value->axes.front()});
}

Axis table_axis(const std::shared_ptr<detail::BackedNode>& value,
                const std::shared_ptr<detail::BackedNode>& index) {
    const auto index_shape = base_shape(index);
    if (index_shape.size() != 1)
        codec_fail(ErrorCode::shape_mismatch, index->path, "table index must be one-dimensional");
    if (value->axes.empty()) return Axis{index_shape.front(), 0, 1, {}};
    if (value->axes.size() != 1)
        codec_fail(ErrorCode::shape_mismatch, value->path,
                   "dataframe selection must contain exactly one row axis");
    return value->axes.front();
}

std::size_t column_length(const std::shared_ptr<detail::BackedNode>& value,
                          std::string_view code) {
    std::shared_ptr<detail::BackedNode> data = value;
    if (code == "categorical") data = child_node(value, "codes");
    else if (code == "nullable-integer" || code == "nullable-boolean" ||
             code == "nullable-string-array") data = child_node(value, "values");
    const auto shape = base_shape(data);
    if (shape.size() != 1)
        codec_fail(ErrorCode::shape_mismatch, data->path,
                   "table columns and categories must be one-dimensional");
    if (data->axes.empty()) return shape.front();
    if (data->axes.size() != 1)
        codec_fail(ErrorCode::shape_mismatch, data->path,
                   "selected column must contain exactly one row axis");
    return data->axes.front().size;
}

Column read_column(const std::shared_ptr<detail::BackedNode>& value);
Table read_table(const std::shared_ptr<detail::BackedNode>& value);
Element read_element(const std::shared_ptr<detail::BackedNode>& value);

Column read_column(const std::shared_ptr<detail::BackedNode>& value) {
    const auto info = inspect_node(value);
    const auto code = encoding(info, value->path);
    if (code == "array") {
        auto array = read_array(value);
        return Column::numeric(std::move(array));
    }
    if (code == "string-array") {
        auto strings=read_packed_column_strings(value);
        return Column::from_packed_strings(std::move(strings.bytes),std::move(strings.offsets),{},Validation::none);
    }
    if (code == "categorical") {
        if(!info.group)codec_fail(ErrorCode::unsupported_encoding,value->path,"categorical encoding requires a group");
        require_children(value,{"codes","categories"});
        auto codes_node = child_node(value, "codes");
        if (element_encoding(codes_node) != "array")
            codec_fail(ErrorCode::unsupported_encoding, codes_node->path,
                       "categorical codes must be a numerical array");
        auto categories_node = child_node(value, "categories", false);
        categories_node->prune_categories = false;
        if(value->prune_categories) {
            auto plan=make_category_plan(value,scratch_remaining(value));
            codes_node->code_recode=plan;
            categories_node->axes={Axis{category_count(plan),0,1,{},plan}};
        }
        auto codes = read_array(codes_node);
        auto categories = read_column(categories_node);
        if (categories.kind() == ColumnKind::categorical)
            codec_fail(ErrorCode::unsupported_encoding, categories_node->path,
                       "category dictionaries cannot themselves be categorical");
        const auto ordered = required_bool(info, "ordered", value->path);
        // The enclosing Column, Table, or Element read validates the fully
        // selected owner once after construction when full validation is set.
        return Column::categorical(std::move(codes), std::move(categories), ordered,
                                   Validation::none);
    }
    if (code == "nullable-integer" || code == "nullable-boolean" ||
        code == "nullable-string-array") {
        if(!info.group)codec_fail(ErrorCode::unsupported_encoding,value->path,"nullable encoding requires a group");
        require_children(value,{"values","mask"});
        auto values_node = child_node(value, "values");
        auto mask_node = child_node(value, "mask");
        if (element_encoding(mask_node) != "array")
            codec_fail(ErrorCode::unsupported_encoding, mask_node->path,
                       "nullable mask must be a numerical array");
        const auto mask_info=inspect_node(mask_node);
        const auto values_info=inspect_node(values_node);
        const auto selected_mask_shape=selected_shape(mask_node);
        if(mask_info.group || mask_info.strings || mask_info.dtype!=DType::boolean || selected_mask_shape.size()!=1)
            codec_fail(ErrorCode::unsupported_dtype,mask_node->path,
                       "nullable mask must be a one-dimensional boolean array");
        if(mask_info.shape!=values_info.shape)
            codec_fail(ErrorCode::shape_mismatch,value->path,"nullable values and mask lengths differ");
        // The validity vector is final Column storage. Read missing-mask bytes
        // directly into it and invert in place, avoiding an O(rows) temporary.
        std::vector<std::uint8_t> validity(selected_mask_shape[0]);
        auto destination=DenseBuffer{validity.data(),DType::boolean,selected_mask_shape,{}};
        detail::BackedAccess::array(mask_node).read_into(destination);
        for(auto& bit:validity)bit=bit?0:1;
        const auto values_code = element_encoding(values_node);
        if (code == "nullable-string-array") {
            if (values_code != "string-array")
                codec_fail(ErrorCode::unsupported_encoding, values_node->path,
                           "nullable strings require string-array values");
            auto strings=read_packed_column_strings(values_node);
            if(strings.offsets.size()!=validity.size()+1)
                codec_fail(ErrorCode::shape_mismatch,value->path,"nullable values and mask lengths differ");
            auto missing = optional_string(info, "na-value", value->path, "NA");
            StringMissingValue sentinel;
            if (missing == "NA") sentinel = StringMissingValue::NA;
            else if (missing == "NaN") sentinel = StringMissingValue::NaN;
            else codec_fail(ErrorCode::unsupported_encoding, value->path,
                             "unsupported nullable-string na-value '" + missing + "'");
            return Column::from_nullable_packed_strings(std::move(strings.bytes),std::move(strings.offsets),
                                                         std::move(validity),sentinel,Validation::none);
        }
        if (values_code != "array")
            codec_fail(ErrorCode::unsupported_encoding, values_node->path,
                       "nullable numerical values require array encoding");
        auto values = read_array(values_node);
        if (values.rank() != 1 || values.extent(0) != validity.size())
            codec_fail(ErrorCode::shape_mismatch, value->path,
                       "nullable values and mask lengths differ");
        if (code == "nullable-boolean" && values.dtype() != DType::boolean)
            codec_fail(ErrorCode::unsupported_dtype, values_node->path,
                       "nullable-boolean values must be boolean");
        if (code == "nullable-integer" && (values.dtype() == DType::boolean ||
            values.dtype() == DType::float32 || values.dtype() == DType::float64))
            codec_fail(ErrorCode::unsupported_dtype, values_node->path,
                       "nullable-integer values must be integer");
        return Column::nullable_numeric(std::move(values), std::move(validity));
    }
    codec_fail(ErrorCode::unsupported_encoding, value->path,
               "encoding '" + code + "' is not supported as a column");
}

Table read_table(const std::shared_ptr<detail::BackedNode>& value) {
    const auto info = inspect_node(value);
    if (encoding(info, value->path) != "dataframe")
        codec_fail(ErrorCode::unsupported_encoding, value->path,
                   "expected dataframe encoding");
    if (!info.group)
        codec_fail(ErrorCode::unsupported_encoding, value->path,
                   "dataframe encoding requires a group");
    const auto index_key = table_index_key(value, &info);
    auto index_node = table_index_node(value);
    if (element_encoding(index_node) != "string-array")
        codec_fail(ErrorCode::unsupported_dtype, index_node->path,
                   "AnnData table indexes currently require string-array encoding");
    const auto index_shape = base_shape(index_node);
    if (index_shape.size() != 1)
        codec_fail(ErrorCode::shape_mismatch, index_node->path,
                   "table index must be one-dimensional");
    auto index = read_strings_all(index_node,
        value->session->options.validation == Validation::full);
    auto table = Table::with_index(std::move(index), index_key == "_index" ? std::string{} : index_key);
    const auto names = strings_attr(info, "column-order", value->path);
    std::unordered_set<std::string> seen;
    for (const auto& name : names) {
        if (!seen.insert(name).second)
            codec_fail(ErrorCode::unsupported_encoding, value->path,
                       "duplicate dataframe column-order entry '" + name + "'");
        auto column = read_column(child_node(value, name));
        try { table.set(name, std::move(column)); }
        catch (const Error& error) {
            codec_fail(error.code(), child(value->path, name), error.what());
        }
    }
    for (const auto& name : value->session->backend->children(value->path)) {
        if (name != index_key && !seen.contains(name))
            codec_fail(ErrorCode::unsupported_encoding, child(value->path, name),
                       "dataframe member is absent from column-order");
    }
    return table;
}

Element read_element(const std::shared_ptr<detail::BackedNode>& value) {
    check_node(value);
    if(value->empty_mapping)return Element::mapping();
    if (value->color_override) {
        auto promoted=read_color_override(value);
        if(!promoted) codec_fail(ErrorCode::invalid_argument,value->path,"missing promoted color value");
        return std::move(*promoted);
    }
    const auto info = inspect_node(value);
    const auto code = encoding(info, value->path);
    if (code == "null") {
        if(info.group)require_children(value,{});
        return Element{};
    }
    if (code == "array" || code == "csr_matrix" || code == "csc_matrix")
        return Element(read_array(value));
    if (code == "rec-array") return Element(read_record(value, info));
    if (code == "string-array") return Element(read_string_array(value));
    if (code == "string") return Element(read_string_scalar(value));
    if (code == "numeric-scalar") {
        auto array = read_array(value);
        if (array.rank() != 0)
            codec_fail(ErrorCode::shape_mismatch, value->path,
                       "numeric scalar requires rank zero");
        auto scalar = visit_dtype(array.dtype(), [&]<class T> {
            return Element::scalar(array.view().dense<T>().data()[0]);
        });
        scalar.set_serialization_byte_order(array.serialization_byte_order());
        return scalar;
    }
    if (code == "dataframe") return Element(read_table(value));
    if (code == "categorical" || code == "nullable-integer" ||
        code == "nullable-boolean" || code == "nullable-string-array")
        return Element(read_column(value));
    if (code == "dict") {
        if (!info.group)
            codec_fail(ErrorCode::unsupported_encoding, value->path,
                       "dict encoding requires a group");
        auto mapping = Element::mapping();
        const auto names = value->session->backend->children(value->path);
        for (const auto& name : names) {
            auto item = value->slice_history ? color_child(value,name) : child_node(value, name);
            if(item)mapping.set(name, read_element(item));
        }
        return mapping;
    }
    codec_fail(ErrorCode::unsupported_encoding, value->path,
               "encoding '" + code + "' is not a nested metadata element");
}

} // namespace

} // namespace anndatapp::backed_detail

namespace anndatapp {

BackedStringArray::BackedStringArray() = default;
BackedStringArray::BackedStringArray(std::shared_ptr<detail::BackedNode> node)
    : node_(std::move(node)) {}

std::vector<std::size_t> BackedStringArray::shape() const {
    backed_detail::check_node(node_);
    if(node_->color_override) return {node_->color_override->axis.size};
    return backed_detail::selected_shape(node_);
}

std::size_t BackedStringArray::size() const {
    backed_detail::check_node(node_);
    if(node_->color_override) return node_->color_override->axis.size;
    return backed_detail::element_count(backed_detail::selected_shape(node_), node_->path);
}

std::string BackedStringArray::string(std::size_t flat_index) const {
    backed_detail::check_node(node_);
    if(node_->color_override) {
        if(flat_index>=node_->color_override->axis.size)throw Error(ErrorCode::out_of_bounds,node_->path,"string index is out of bounds");
        auto values=node_->session->backend->read_strings_bounded(node_->path,{},{},backed_detail::scratch_remaining(node_));
        if(values.size()!=1)throw Error(ErrorCode::io_error,node_->path,"invalid scalar string transfer");
        if(node_->session->options.validation==Validation::full)backed_detail::validate_utf8_string(values[0],node_->path);
        return std::move(values[0]);
    }
    const auto shape = backed_detail::selected_shape(node_);
    const auto count = backed_detail::element_count(shape, node_->path);
    if (flat_index >= count)
        throw Error(ErrorCode::out_of_bounds, node_->path, "string index is out of bounds");
    const auto physical_shape = backed_detail::base_shape(node_);
    const auto axes = backed_detail::axes_for(node_, physical_shape);
    const auto logical = backed_detail::coordinate(shape, flat_index);
    const auto start = backed_detail::physical_coordinate(axes, logical, node_->path);
    std::vector<std::size_t> count_one(shape.size(), 1);
    auto values = node_->session->backend->read_strings_bounded(node_->path, start, count_one, backed_detail::scratch_remaining(node_));
    if (values.size() != 1)
        throw Error(ErrorCode::io_error, node_->path,
                    "backend returned an unexpected single-string rectangle size");
    if (node_->session->options.validation == Validation::full)
        backed_detail::validate_single_string(values.front(), node_->path);
    return std::move(values.front());
}

BackedStringArray BackedStringArray::select(std::vector<Selection> selections) const {
    backed_detail::check_node(node_);
    if(node_->color_override) {
        if(selections.size()!=1)throw Error(ErrorCode::shape_mismatch,node_->path,"selection rank differs from promoted string array rank");
        auto result=std::make_shared<detail::BackedNode>(*node_);
        result->color_override=std::make_shared<backed_detail::ColorOverride>(*node_->color_override);
        result->color_override->axis=backed_detail::select_axis(node_->color_override->axis,selections[0]);
        return BackedStringArray(std::move(result));
    }
    const auto shape = backed_detail::base_shape(node_);
    const auto axes = backed_detail::axes_for(node_, shape);
    if (selections.size() != shape.size())
        throw Error(ErrorCode::shape_mismatch, node_->path,
                    "selection rank differs from string array rank");
    std::vector<backed_detail::Axis> selected;
    selected.reserve(axes.size());
    const auto labels = axes.size() == 1 ? node_ : std::shared_ptr<detail::BackedNode>{};
    for (std::size_t i = 0; i < axes.size(); ++i)
        selected.push_back(backed_detail::select_axis(axes[i], selections[i], labels));
    auto result = backed_detail::node(node_->session, node_->path, std::move(selected));
    result->writable = node_->writable;
    return BackedStringArray(std::move(result));
}

StringArray BackedStringArray::read() const {
    if(node_ && node_->color_override) {
        backed_detail::check_node(node_);
        auto promoted=backed_detail::read_color_override(node_);
        if(!promoted)throw Error(ErrorCode::invalid_argument,node_->path,"missing promoted color value");
        return promoted->view().string_array().copy();
    }
    auto result = backed_detail::read_string_array(node_);
    if (node_->session->options.validation == Validation::full) result.validate();
    return result;
}

BackedColumn::BackedColumn() = default;
BackedColumn::BackedColumn(std::shared_ptr<detail::BackedNode> node)
    : node_(std::move(node)) {}

std::size_t BackedColumn::size() const {
    backed_detail::check_node(node_);
    if(node_->color_override) return node_->color_override->axis.size;
    const auto code = backed_detail::element_encoding(node_);
    return backed_detail::column_length(node_, code);
}

ColumnKind BackedColumn::kind() const {
    backed_detail::check_node(node_);
    if(node_->color_override) return ColumnKind::string;
    const auto code = backed_detail::element_encoding(node_);
    if (code == "array" || code == "nullable-integer" || code == "nullable-boolean")
        return ColumnKind::numeric;
    if (code == "string-array" || code == "nullable-string-array")
        return ColumnKind::string;
    if (code == "categorical") return ColumnKind::categorical;
    throw Error(ErrorCode::unsupported_encoding, node_->path,
                "encoding '" + code + "' is not supported as a column");
}

bool BackedColumn::nullable() const {
    backed_detail::check_node(node_);
    if(node_->color_override) return node_->color_override->kind==backed_detail::ColorPromotion::nullable_string_column;
    const auto code = backed_detail::element_encoding(node_);
    return code == "categorical" || code == "nullable-integer" ||
           code == "nullable-boolean" || code == "nullable-string-array";
}

bool BackedColumn::ordered() const {
    backed_detail::check_node(node_);
    const auto info = backed_detail::inspect_node(node_);
    if (backed_detail::encoding(info, node_->path) != "categorical")
        throw Error(ErrorCode::invalid_argument, node_->path, "column is not categorical");
    return backed_detail::required_bool(info, "ordered", node_->path);
}

BackedArray BackedColumn::values() const {
    backed_detail::check_node(node_);
    const auto code = backed_detail::element_encoding(node_);
    if (code == "array") return BackedArray(node_);
    if (code == "nullable-integer" || code == "nullable-boolean")
        return BackedArray(backed_detail::child_node(node_, "values"));
    throw Error(ErrorCode::invalid_argument, node_->path,
                "column does not have numerical values");
}

BackedArray BackedColumn::codes() const {
    backed_detail::check_node(node_);
    if (backed_detail::element_encoding(node_) != "categorical")
        throw Error(ErrorCode::invalid_argument, node_->path, "column is not categorical");
    auto codes=backed_detail::child_node(node_, "codes");
    codes->writable=false;
    if(node_->prune_categories) codes->code_recode=backed_detail::make_category_plan(node_,backed_detail::scratch_remaining(node_));
    return BackedArray(std::move(codes));
}

BackedColumn BackedColumn::categories() const {
    backed_detail::check_node(node_);
    if (backed_detail::element_encoding(node_) != "categorical")
        throw Error(ErrorCode::invalid_argument, node_->path, "column is not categorical");
    auto categories=backed_detail::child_node(node_, "categories", false);
    categories->prune_categories=false;
    categories->writable=false;
    if(node_->prune_categories) {
        auto plan=backed_detail::make_category_plan(node_,backed_detail::scratch_remaining(node_));
        categories->axes={backed_detail::Axis{backed_detail::category_count(plan),0,1,{},plan}};
    }
    return BackedColumn(std::move(categories));
}

BackedArray BackedColumn::unsafe_codes() const {
    backed_detail::check_node(node_);
    if(backed_detail::element_encoding(node_)!="categorical")throw Error(ErrorCode::invalid_argument,node_->path,"column is not categorical");
    node_->session->check_writable();
    if(!node_->writable)throw Error(ErrorCode::invalid_argument,node_->path,"backed view is read-only");
    auto codes=backed_detail::child_node(node_,"codes");
    codes->prune_categories=false;
    codes->code_recode.reset();
    return BackedArray(std::move(codes));
}

BackedColumn BackedColumn::unsafe_categories() const {
    backed_detail::check_node(node_);
    if(backed_detail::element_encoding(node_)!="categorical")throw Error(ErrorCode::invalid_argument,node_->path,"column is not categorical");
    node_->session->check_writable();
    if(!node_->writable)throw Error(ErrorCode::invalid_argument,node_->path,"backed view is read-only");
    auto categories=backed_detail::child_node(node_,"categories",false);
    categories->prune_categories=false;
    return BackedColumn(std::move(categories));
}

BackedColumn BackedColumn::select(Selection rows) const {
    backed_detail::check_node(node_);
    if(node_->color_override) {
        auto result=std::make_shared<detail::BackedNode>(*node_);
        result->color_override=std::make_shared<backed_detail::ColorOverride>(*node_->color_override);
        result->color_override->axis=backed_detail::select_axis(node_->color_override->axis,rows);
        return BackedColumn(std::move(result));
    }
    const auto code = backed_detail::element_encoding(node_);
    const auto length = backed_detail::column_length(node_, code);
    const auto parent = node_->axes.empty() ? backed_detail::Axis{length, 0, 1, {}} : node_->axes.front();
    if (!node_->axes.empty() && node_->axes.size() != 1)
        throw Error(ErrorCode::shape_mismatch, node_->path,
                    "selected column must contain exactly one row axis");
    auto result = backed_detail::node(node_->session, node_->path,
        {backed_detail::select_axis(parent, rows)});
    result->writable = node_->writable;
    result->prune_categories = true;
    return BackedColumn(std::move(result));
}

Column BackedColumn::read() const {
    if(node_ && node_->color_override) {
        backed_detail::check_node(node_);
        auto promoted=backed_detail::read_color_override(node_);
        if(!promoted)throw Error(ErrorCode::invalid_argument,node_->path,"missing promoted color value");
        return promoted->view().column().copy();
    }
    auto result = backed_detail::read_column(node_);
    if (node_->session->options.validation == Validation::full) result.validate();
    return result;
}

BackedTable::BackedTable() = default;
BackedTable::BackedTable(std::shared_ptr<detail::BackedNode> node)
    : node_(std::move(node)) {}

std::size_t BackedTable::size() const {
    backed_detail::check_node(node_);
    auto index = backed_detail::table_index_node(node_);
    const auto code = backed_detail::element_encoding(index);
    if (code != "string-array")
        throw Error(ErrorCode::unsupported_dtype, index->path,
                    "AnnData table indexes currently require string-array encoding");
    return backed_detail::table_axis(node_, index).size;
}

std::string BackedTable::index_name() const {
    backed_detail::check_node(node_);
    const auto info = backed_detail::inspect_node(node_);
    const auto key = backed_detail::table_index_key(node_, &info);
    return key == "_index" ? std::string{} : key;
}

std::string BackedTable::name(std::size_t row) const {
    backed_detail::check_node(node_);
    const auto rows = size();
    if (row >= rows)
        throw Error(ErrorCode::out_of_bounds, node_->path, "table row is out of bounds");
    auto index = backed_detail::table_index_node(node_);
    const auto shape = backed_detail::base_shape(index);
    const auto axes = backed_detail::axes_for(index, shape);
    const std::vector<std::size_t> start{axes.front().at(row)};
    auto names = node_->session->backend->read_strings_bounded(index->path, start, {1}, backed_detail::scratch_remaining(index));
    if (names.size() != 1)
        throw Error(ErrorCode::io_error, index->path,
                    "backend returned an unexpected table index value count");
    if (node_->session->options.validation == Validation::full)
        backed_detail::validate_single_string(names.front(), index->path);
    return std::move(names.front());
}

std::vector<std::string> BackedTable::names() const {
    backed_detail::check_node(node_);
    auto index = backed_detail::table_index_node(node_);
    if (backed_detail::element_encoding(index) != "string-array")
        throw Error(ErrorCode::unsupported_dtype, index->path,
                    "AnnData table indexes currently require string-array encoding");
    if (backed_detail::base_shape(index).size() != 1)
        throw Error(ErrorCode::shape_mismatch, index->path, "table index must be one-dimensional");
    return backed_detail::read_strings_all(index,
        node_->session->options.validation == Validation::full);
}

std::vector<std::string> BackedTable::column_names() const {
    backed_detail::check_node(node_);
    const auto info = backed_detail::inspect_node(node_);
    if (backed_detail::encoding(info, node_->path) != "dataframe")
        throw Error(ErrorCode::unsupported_encoding, node_->path,
                    "expected dataframe encoding");
    return backed_detail::strings_attr(info, "column-order", node_->path);
}

bool BackedTable::contains(std::string_view column) const {
    const auto names = column_names();
    return std::find(names.begin(), names.end(), column) != names.end();
}

BackedColumn BackedTable::column(std::string_view column_name) const {
    backed_detail::check_node(node_);
    const auto names = column_names();
    if (std::find(names.begin(), names.end(), column_name) == names.end())
        throw Error(ErrorCode::out_of_bounds, node_->path,
                    "dataframe column does not exist");
    return BackedColumn(backed_detail::child_node(node_, column_name));
}

BackedTable BackedTable::select(Selection rows) const {
    backed_detail::check_node(node_);
    auto index = backed_detail::table_index_node(node_, false);
    const auto parent = backed_detail::table_axis(node_, index);
    auto result = backed_detail::node(node_->session, node_->path,
        {backed_detail::select_axis(parent, rows, index)});
    result->writable = node_->writable;
    result->prune_categories = true;
    return BackedTable(std::move(result));
}

Table BackedTable::read() const {
    auto result = backed_detail::read_table(node_);
    if (node_->session->options.validation == Validation::full) result.validate();
    return result;
}

BackedElement::BackedElement() = default;
BackedElement::BackedElement(std::shared_ptr<detail::BackedNode> node)
    : node_(std::move(node)) {}

ElementKind BackedElement::kind() const {
    backed_detail::check_node(node_);
    if(node_->empty_mapping)return ElementKind::mapping;
    if(node_->color_override) {
        switch(node_->color_override->kind) {
        case backed_detail::ColorPromotion::numeric_array: return ElementKind::array;
        case backed_detail::ColorPromotion::string_array: return ElementKind::string_array;
        default: return ElementKind::column;
        }
    }
    const auto code = backed_detail::element_encoding(node_);
    if (code == "null") return ElementKind::null;
    if (code == "dict") return ElementKind::mapping;
    if (code == "array" || code == "csr_matrix" || code == "csc_matrix")
        return ElementKind::array;
    if (code == "string-array") return ElementKind::string_array;
    if (code == "rec-array") return ElementKind::record_array;
    if (code == "string") return ElementKind::string;
    if (code == "dataframe") return ElementKind::table;
    if (code == "categorical" || code == "nullable-integer" ||
        code == "nullable-boolean" || code == "nullable-string-array")
        return ElementKind::column;
    if (code == "numeric-scalar") {
        const auto info = backed_detail::inspect_node(node_);
        if (info.dtype == DType::boolean) return ElementKind::boolean;
        if (info.dtype == DType::float32 || info.dtype == DType::float64)
            return ElementKind::floating;
        if (info.dtype == DType::int8 || info.dtype == DType::int16 ||
            info.dtype == DType::int32 || info.dtype == DType::int64)
            return ElementKind::signed_integer;
        return ElementKind::unsigned_integer;
    }
    throw Error(ErrorCode::unsupported_encoding, node_->path,
                "encoding '" + code + "' is not a metadata element");
}

std::vector<std::string> BackedElement::keys() const {
    backed_detail::check_node(node_);
    if(node_->empty_mapping)return {};
    if (backed_detail::element_encoding(node_) != "dict")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a mapping");
    return node_->slice_history ? backed_detail::color_keys(node_) : node_->session->backend->children(node_->path);
}

bool BackedElement::contains(std::string_view key) const {
    const auto names = keys();
    return std::find(names.begin(), names.end(), key) != names.end();
}

BackedElement BackedElement::at(std::string_view key) const {
    backed_detail::check_node(node_);
    if(node_->empty_mapping)throw Error(ErrorCode::out_of_bounds,node_->path,"mapping key does not exist");
    if (backed_detail::element_encoding(node_) != "dict")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a mapping");
    if(node_->slice_history) {
        auto item=backed_detail::color_child(node_,key);
        if(!item)throw Error(ErrorCode::out_of_bounds,node_->path,"mapping key does not exist");
        return BackedElement(std::move(item));
    }
    const auto names = node_->session->backend->children(node_->path);
    if (std::find(names.begin(), names.end(), key) == names.end())
        throw Error(ErrorCode::out_of_bounds, node_->path, "mapping key does not exist");
    return BackedElement(backed_detail::child_node(node_, key));
}

BackedArray BackedElement::array() const {
    backed_detail::check_node(node_);
    if(node_->color_override && node_->color_override->kind==backed_detail::ColorPromotion::numeric_array)return BackedArray(node_);
    const auto code = backed_detail::element_encoding(node_);
    if (code != "array" && code != "csr_matrix" && code != "csc_matrix" &&
        code != "numeric-scalar")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a numerical array");
    return BackedArray(node_);
}

BackedStringArray BackedElement::string_array() const {
    backed_detail::check_node(node_);
    if(node_->color_override && node_->color_override->kind==backed_detail::ColorPromotion::string_array)return BackedStringArray(node_);
    if (backed_detail::element_encoding(node_) != "string-array")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a string array");
    return BackedStringArray(node_);
}

BackedColumn BackedElement::column() const {
    backed_detail::check_node(node_);
    if(node_->color_override && (node_->color_override->kind==backed_detail::ColorPromotion::string_column || node_->color_override->kind==backed_detail::ColorPromotion::nullable_string_column))return BackedColumn(node_);
    const auto code = backed_detail::element_encoding(node_);
    if (code != "array" && code != "string-array" && code != "categorical" &&
        code != "nullable-integer" && code != "nullable-boolean" &&
        code != "nullable-string-array")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a column");
    return BackedColumn(node_);
}

BackedTable BackedElement::table() const {
    backed_detail::check_node(node_);
    if (backed_detail::element_encoding(node_) != "dataframe")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a table");
    return BackedTable(node_);
}

std::vector<std::string> BackedElement::field_names() const {
    backed_detail::check_node(node_);
    auto info = node_->session->backend->inspect(node_->path);
    if (backed_detail::encoding(info, node_->path) != "rec-array")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a record array");
    if (!info.record) throw Error(ErrorCode::unsupported_encoding, node_->path, "rec-array must be a compound dataset");
    std::vector<std::string> names;
    names.reserve(info.fields.size());
    for (const auto& field : info.fields) names.push_back(field.name);
    return names;
}

Element BackedElement::read_field(std::string_view name) const {
    backed_detail::check_node(node_);
    auto info = node_->session->backend->inspect(node_->path);
    if (backed_detail::encoding(info, node_->path) != "rec-array")
        throw Error(ErrorCode::invalid_argument, node_->path, "element is not a record array");
    if (!info.record) throw Error(ErrorCode::unsupported_encoding, node_->path, "rec-array must be a compound dataset");
    return backed_detail::read_record_field(node_, info, name);
}

Element BackedElement::read() const {
    auto result = backed_detail::read_element(node_);
    if (node_->session->options.validation == Validation::full) result.validate();
    return result;
}

} // namespace anndatapp
