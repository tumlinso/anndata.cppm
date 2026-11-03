module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

module anndata.backed;
import anndata.core;
import :storage;

namespace anndata::backed_detail {
namespace {
using Shape = std::vector<std::size_t>;
using Node = std::shared_ptr<detail::BackedNode>;

[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}

std::size_t count(const Shape& shape, std::string_view path) {
    std::size_t n = 1;
    for (auto x : shape) {
        if (!x) return 0;
        if (n > std::size_t(PTRDIFF_MAX)/x) fail(ErrorCode::shape_mismatch, path, "record shape exceeds addressable range");
        n *= x;
    }
    return n;
}

Shape shape_of(const Node& node) {
    Shape shape;
    for (const auto& axis : node->axes) shape.push_back(axis.size);
    return shape;
}

// One contiguous last-axis run. Other dimensions select one record each.
std::size_t run_at(const Shape& shape, std::size_t flat, std::size_t cap, Shape& start, Shape& size) {
    auto x = flat;
    for (std::size_t i = shape.size(); i-- > 0;) { start[i] = x%shape[i]; x /= shape[i]; size[i] = 1; }
    if (shape.empty()) return 1;
    auto n = std::min(cap, shape.back() - start.back());
    size.back() = n;
    return n;
}

std::size_t source_run(const Node& node, const Shape& logical, Shape& start, std::size_t n) {
    for (std::size_t i = 0; i < logical.size(); ++i) start[i] = node->axes[i].at(logical[i]);
    if (start.empty()) return 1;
    const auto& axis = node->axes.back();
    std::size_t run = 1;
    while (run < n && start.back() <= SIZE_MAX - run && axis.at(logical.back() + run) == start.back() + run) ++run;
    return run;
}

const RecordFieldInfo& field_info(const NodeInfo& info, std::string_view name, std::string_view path) {
    if (info.group || !info.record) fail(ErrorCode::unsupported_encoding, path, "rec-array must be a compound dataset");
    for (const auto& field : info.fields) if (field.name == name) return field;
    fail(ErrorCode::out_of_bounds, path, "record field '" + std::string(name) + "' does not exist");
}

Array make_array(DType t, const Shape& shape, std::string_view path) {
    switch (t) {
    case DType::boolean: return Array::empty<bool>(shape);
    case DType::int8: return Array::empty<std::int8_t>(shape);
    case DType::uint8: return Array::empty<std::uint8_t>(shape);
    case DType::int16: return Array::empty<std::int16_t>(shape);
    case DType::uint16: return Array::empty<std::uint16_t>(shape);
    case DType::int32: return Array::empty<std::int32_t>(shape);
    case DType::uint32: return Array::empty<std::uint32_t>(shape);
    case DType::int64: return Array::empty<std::int64_t>(shape);
    case DType::uint64: return Array::empty<std::uint64_t>(shape);
    case DType::float32: return Array::empty<float>(shape);
    case DType::float64: return Array::empty<double>(shape);
    }
    fail(ErrorCode::unsupported_dtype, path, "unknown record field dtype");
}

std::string field_path(std::string_view path, std::string_view name) {
    return std::string(path) + "[" + std::string(name) + "]";
}

} // namespace

std::variant<Array, StringArray> read_record_value(const Node& node, const NodeInfo& info, std::string_view name) {
    check_node(node);
    const auto& field = field_info(info, name, node->path);
    const auto path = field_path(node->path, name);
    const auto shape = shape_of(node);
    const auto n = count(shape, path), budget = scratch_remaining(node);
    auto& backend = *node->session->backend;
    Shape logical(shape.size()), start(shape.size()), size(shape.size(), 1);
    if (!field.strings) {
        auto out = make_array(field.dtype, shape, path);
        out.set_serialization_byte_order(field.byte_order);
        auto bytes = out.edit().bytes();
        const auto width = dtype_size(field.dtype);
        if (budget < width) fail(ErrorCode::invalid_argument, path, "scratch budget cannot hold one record value");
        for (std::size_t i = 0; i < n;) {
            auto run = run_at(shape, i, std::max<std::size_t>(1, budget/width), logical, size);
            run = source_run(node, logical, start, run);
            if (!size.empty()) size.back() = run;
            backend.read_record_numeric(node->path, name, start, size, bytes.data() + i*width);
            i += run;
        }
        if (field.dtype == DType::boolean && node->session->options.validation == Validation::full)
            for (std::size_t i = 0; i < n; ++i) bytes.data()[i] = bytes.data()[i] == std::byte{0} ? std::byte{0} : std::byte{1};
        return out;
    }
    std::vector<char> bytes;
    std::vector<std::uint64_t> offsets;
    if (n > std::size_t(PTRDIFF_MAX)/sizeof(std::uint64_t)-1)
        fail(ErrorCode::invalid_argument, path, "record string offsets exceed addressable range");
    offsets.reserve(n + 1); offsets.push_back(0);
    const auto cap = std::max<std::size_t>(1, budget/(sizeof(std::string) + 1));
    for (std::size_t i = 0; i < n;) {
        auto run = run_at(shape, i, cap, logical, size);
        run = source_run(node, logical, start, run);
        if (!size.empty()) size.back() = run;
        auto values = backend.read_record_strings_bounded(node->path, name, start, size, budget);
        if (values.empty() || values.size() > run) fail(ErrorCode::io_error, path, "backend returned an invalid string prefix");
        for (const auto& value : values) {
            if (node->session->options.validation == Validation::full) validate_utf8_string(value, path);
            if (value.size() > std::size_t(PTRDIFF_MAX) - bytes.size()) fail(ErrorCode::invalid_argument, path, "record string bytes exceed addressable range");
            bytes.insert(bytes.end(), value.begin(), value.end());
            offsets.push_back(std::uint64_t(bytes.size()));
        }
        i += values.size();
    }
    return StringArray::from_packed(shape, std::move(bytes), std::move(offsets), Validation::none);
}

Element read_record_field(const Node& node, const NodeInfo& info, std::string_view name) {
    auto value = read_record_value(node, info, name);
    if (auto* array = std::get_if<Array>(&value)) return Element(std::move(*array));
    return Element(std::move(std::get<StringArray>(value)));
}

RecordArray read_record(const Node& node, const NodeInfo& info) {
    if (info.group || !info.record) fail(ErrorCode::unsupported_encoding, node->path, "rec-array must be a compound dataset");
    auto result = RecordArray::with_shape(shape_of(node));
    for (const auto& field : info.fields) {
        auto value = read_record_value(node, info, field.name);
        // Payload validation was done during transfer; avoid scanning it again.
        if (field.strings) result.set(field.name, std::move(std::get<StringArray>(value)), Validation::none);
        else result.set(field.name, std::move(std::get<Array>(value)), Validation::none);
    }
    return result;
}

void write_record(Backend& backend, std::string_view path, RecordArrayView records, std::size_t budget) {
    auto shape = records.shape();
    const auto n = count(shape, path);
    const auto names = records.field_names();
    if (names.empty()) fail(ErrorCode::unsupported_encoding, path, "record arrays require at least one field");
    std::vector<RecordFieldInfo> fields;
    fields.reserve(names.size());
    for (const auto& name : names) {
        const auto field = records.field(name);
        fields.push_back({name, field.kind != RecordFieldKind::numeric, field.dtype, field.byte_order});
    }
    backend.create_record(path, shape, fields, {{"encoding-type", std::string("rec-array")}, {"encoding-version", std::string("0.2.0")}});
    Shape logical(shape.size()), size(shape.size(), 1);
    std::unique_ptr<std::byte[]> packed;
    std::size_t capacity = 0;
    std::vector<std::string> strings;
    for (const auto& field : fields) {
        if (!field.strings) {
            std::vector<std::string>().swap(strings);
            auto bytes = records.numeric_bytes(field.name);
            const auto width = dtype_size(field.dtype);
            if (budget < width) fail(ErrorCode::invalid_argument, path, "scratch budget cannot hold one record value");
            for (std::size_t i = 0; i < n;) {
                const auto run = run_at(shape, i, std::max<std::size_t>(1, budget/width), logical, size);
                if (run*width > capacity) {
                    packed.reset(); capacity = 0;
                    packed.reset(new std::byte[run*width]); capacity = run*width;
                }
                std::ptrdiff_t offset = 0;
                for (std::size_t a = 0; a < shape.size(); ++a) offset += std::ptrdiff_t(logical[a])*bytes.stride_bytes(a);
                const auto step = shape.empty() ? 0 : bytes.stride_bytes(shape.size()-1);
                if (step == std::ptrdiff_t(width)) std::memcpy(packed.get(), bytes.data() + offset, run*width);
                else for (std::size_t j = 0; j < run; ++j) std::memcpy(packed.get()+j*width, bytes.data()+offset+std::ptrdiff_t(j)*step, width);
                backend.write_record_numeric(path, field.name, logical, size, packed.get());
                i += run;
            }
            continue;
        }
        packed.reset(); capacity = 0;
        auto text = records.text(field.name);
        const auto metadata = sizeof(std::string) + sizeof(char*);
        const auto cap = budget > sizeof(strings) ? std::max<std::size_t>(1, (budget-sizeof(strings))/(2*metadata)) : 1;
        strings.reserve(std::min(cap, shape.empty() ? n : shape.back()));
        for (std::size_t i = 0; i < n;) {
            const auto run = run_at(shape, i, cap, logical, size);
            strings.clear();
            std::size_t used = sizeof(strings) + strings.capacity()*sizeof(std::string);
            for (std::size_t j = 0; j < run; ++j) {
                auto value = text.string(i+j);
                if (value.find('\0') != std::string_view::npos) fail(ErrorCode::invalid_argument, field_path(path, field.name), "serialized strings cannot contain embedded NUL bytes");
                const auto overhead = sizeof(char*) + 1;
                if (used > budget || overhead > budget-used || value.size() > budget-used-overhead) break;
                used += overhead + value.size();
                strings.emplace_back(value);
            }
            if (strings.empty()) fail(ErrorCode::invalid_argument, field_path(path, field.name), "one string exceeds the scratch budget");
            if (!size.empty()) size.back() = strings.size();
            backend.write_record_strings(path, field.name, logical, size, strings);
            i += strings.size();
        }
    }
}

void copy_record(Backend& out, std::string_view path, const Node& node,
                 const NodeInfo& info, std::size_t budget) {
    if (info.group || !info.record) fail(ErrorCode::unsupported_encoding, node->path, "rec-array must be a compound dataset");
    auto shape = shape_of(node);
    const auto n = count(shape, node->path);
    out.create_record(path, shape, info.fields, info.attrs);
    Shape logical(shape.size()), start(shape.size()), size(shape.size(), 1);
    std::unique_ptr<std::byte[]> packed;
    std::size_t capacity = 0;
    for (const auto& field : info.fields) {
        if (field.strings) { packed.reset(); capacity = 0; }
        auto width = field.strings ? sizeof(std::string)+sizeof(char*)+1 : dtype_size(field.dtype);
        if (budget < width) fail(ErrorCode::invalid_argument, node->path, "scratch budget cannot hold one record value");
        auto cap = std::max<std::size_t>(1, budget/width);
        for (std::size_t i = 0; i < n;) {
            auto run = run_at(shape, i, cap, logical, size);
            run = source_run(node, logical, start, run);
            if (!size.empty()) size.back() = run;
            if (field.strings) {
                auto values = node->session->backend->read_record_strings_bounded(node->path, field.name, start, size, budget/2);
                if (values.empty() || values.size() > run) fail(ErrorCode::io_error, node->path, "backend returned an invalid string prefix");
                if (node->session->options.validation == Validation::full)
                    for (const auto& value : values) validate_utf8_string(value, field_path(node->path, field.name));
                run = values.size();
                if (!size.empty()) size.back() = run;
                out.write_record_strings(path, field.name, logical, size, values);
            } else {
                if (run*width > capacity) {
                    packed.reset(); capacity = 0;
                    packed.reset(new std::byte[run*width]); capacity = run*width;
                }
                node->session->backend->read_record_numeric(node->path, field.name, start, size, packed.get());
                if (field.dtype == DType::boolean && node->session->options.validation == Validation::full)
                    for (std::size_t j = 0; j < run*width; ++j) packed[j] = packed[j] == std::byte{0} ? std::byte{0} : std::byte{1};
                out.write_record_numeric(path, field.name, logical, size, packed.get());
            }
            i += run;
        }
    }
}
} // namespace anndata::backed_detail
