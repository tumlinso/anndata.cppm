module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

module anndata.backed;
import anndata.core;
import :storage;

namespace anndata::backed_detail {
namespace {

using Shape = std::vector<std::size_t>;

[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}

template<class Function>
decltype(auto) with_validation_path(std::string_view path, Function&& function) {
    try {
        return std::forward<Function>(function)();
    } catch (const Error& error) {
        if (error.path().starts_with("/") || error.path().find(':') != std::string::npos) throw;
        throw Error(error.code(), std::string(path), error.what());
    }
}

std::string child_path(std::string_view parent, std::string_view key) {
    std::string result(parent);
    if (result.empty() || result.back() != '/') result.push_back('/');
    result.append(key);
    return result;
}

Attributes encoding(std::string type, std::string version) {
    return {{"encoding-type", std::move(type)}, {"encoding-version", std::move(version)}};
}

std::size_t element_count(const Shape& shape, std::string_view path) {
    std::size_t result = 1;
    for (const auto extent : shape) {
        if (extent == 0) return 0;
        if (result > std::numeric_limits<std::size_t>::max() / extent)
            fail(ErrorCode::invalid_argument, path, "element count overflows size_t");
        result *= extent;
    }
    return result;
}

Shape shape_of(ArrayView array) {
    Shape shape;
    shape.reserve(array.rank());
    for (std::size_t axis = 0; axis < array.rank(); ++axis) shape.push_back(array.extent(axis));
    return shape;
}

Shape shape_of(StringArrayView array) {
    Shape shape;
    shape.reserve(array.rank());
    for (std::size_t axis = 0; axis < array.rank(); ++axis) shape.push_back(array.extent(axis));
    return shape;
}

bool valid_utf8(std::string_view text) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t i = 0;
    while (i < text.size()) {
        const auto first = bytes[i++];
        if (first < 0x80) continue;
        unsigned remaining = 0;
        std::uint32_t value = 0, minimum = 0;
        if (first >= 0xc2 && first <= 0xdf) { remaining = 1; value = first & 0x1f; minimum = 0x80; }
        else if (first >= 0xe0 && first <= 0xef) { remaining = 2; value = first & 0x0f; minimum = 0x800; }
        else if (first >= 0xf0 && first <= 0xf4) { remaining = 3; value = first & 0x07; minimum = 0x10000; }
        else return false;
        if (remaining > text.size() - i) return false;
        while (remaining--) {
            const auto next = bytes[i++];
            if ((next & 0xc0) != 0x80) return false;
            value = (value << 6) | (next & 0x3f);
        }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
    }
    return true;
}

void validate_text(std::string_view text, std::string_view path, Validation validation = Validation::full) {
    if (text.find('\0') != std::string_view::npos)
        fail(ErrorCode::invalid_argument, path, "serialized strings cannot contain embedded NUL bytes");
    if (validation == Validation::full && !valid_utf8(text))
        fail(ErrorCode::unsupported_dtype, path, "serialized string is not valid UTF-8");
}

void validate_key(std::string_view key, std::string_view path, Validation validation = Validation::full) {
    // Keys and dataframe metadata are serialized into paths/JSON attributes, so
    // they must stay valid UTF-8 even when content scans are disabled.
    (void)validation;
    validate_text(key, path, Validation::full);
    if (key.empty() || key == "." || key == ".." || key.find('/') != std::string_view::npos)
        fail(ErrorCode::invalid_argument, path, "element keys must be nonempty path components other than '.' or '..'");
}

template<class T>
void validate_unique_values(ArrayView values, std::string_view path) {
    std::optional<Array> selected;
    // Copy named byte views into optionals throughout this file. Clang 18 can
    // leave shared_ptr move constructors unresolved when emplacing module views.
    std::optional<DenseBytesView<const std::byte>> bytes;
    try { auto view = values.bytes(); bytes.emplace(view); }
    catch (const Error& error) {
        if (error.code() != ErrorCode::non_contiguous) throw;
        selected.emplace(values.copy(Validation::none));
        auto view = selected->view().bytes();
        bytes.emplace(view);
    }
    std::unordered_set<T> seen;
    seen.reserve(values.extent(0));
    for (std::size_t i = 0; i < values.extent(0); ++i) {
        T value{};
        std::memcpy(&value, bytes->data() + static_cast<std::ptrdiff_t>(i) * bytes->stride_bytes(0), sizeof(T));
        if constexpr (std::is_floating_point_v<T>)
            if (std::isnan(value)) fail(ErrorCode::invalid_argument, path, "category dictionaries cannot contain NaN");
        if (!seen.insert(value).second)
            fail(ErrorCode::invalid_argument, path, "category dictionaries must contain unique values");
    }
}

void validate_unique_numeric(ArrayView values, std::string_view path) {
    switch (values.dtype()) {
        case DType::boolean: validate_unique_values<bool>(values, path); return;
        case DType::int8: validate_unique_values<std::int8_t>(values, path); return;
        case DType::uint8: validate_unique_values<std::uint8_t>(values, path); return;
        case DType::int16: validate_unique_values<std::int16_t>(values, path); return;
        case DType::uint16: validate_unique_values<std::uint16_t>(values, path); return;
        case DType::int32: validate_unique_values<std::int32_t>(values, path); return;
        case DType::uint32: validate_unique_values<std::uint32_t>(values, path); return;
        case DType::int64: validate_unique_values<std::int64_t>(values, path); return;
        case DType::uint64: validate_unique_values<std::uint64_t>(values, path); return;
        case DType::float32: validate_unique_values<float>(values, path); return;
        case DType::float64: validate_unique_values<double>(values, path); return;
    }
}

void validate_array(ArrayView array, std::string_view path, Validation validation = Validation::full) {
    with_validation_path(path, [&] {
        if (validation == Validation::full) array.validate();
        (void)element_count(shape_of(array), path);
        if (array.layout() != Layout::dense && array.rank() != 2)
            fail(ErrorCode::unsupported_layout, path, "sparse arrays must be rank two");
    });
}

void validate_column(ColumnView column, std::string_view path, Validation validation = Validation::full);
void validate_table(TableView table, std::string_view path, Validation validation = Validation::full);

void validate_column(ColumnView column, std::string_view path, Validation validation) {
    with_validation_path(path, [&] {
    if (column.kind() == ColumnKind::categorical) {
        const auto categories = column.categories();
        if (categories.kind() == ColumnKind::categorical)
            fail(ErrorCode::unsupported_encoding, child_path(path, "categories"), "category dictionaries cannot themselves be categorical");
        validate_column(categories, child_path(path, "categories"), validation);
        if (validation == Validation::full && categories.kind() == ColumnKind::string) {
            std::unordered_set<std::string_view> seen;
            seen.reserve(categories.size());
            for (std::size_t i = 0; i < categories.size(); ++i) {
                if (!categories.valid(i))
                    fail(ErrorCode::invalid_argument, child_path(path, "categories"), "category dictionaries cannot contain nulls");
                validate_text(categories.string(i), child_path(path, "categories"), validation);
                if (!seen.insert(categories.string(i)).second)
                    fail(ErrorCode::invalid_argument, child_path(path, "categories"), "category dictionaries must contain unique values");
            }
        } else if (validation == Validation::full && categories.kind() == ColumnKind::numeric) {
            for (std::size_t i = 0; i < categories.size(); ++i)
                if (!categories.valid(i))
                    fail(ErrorCode::invalid_argument, child_path(path, "categories"), "category dictionaries cannot contain nulls");
            validate_unique_numeric(categories.values(), child_path(path, "categories"));
        }
        validate_array(column.codes(), child_path(path, "codes"), validation);
        if (validation == Validation::full) for (std::size_t i = 0; i < column.size(); ++i) {
            const auto code = column.code(i);
            if (code < -1 || (code >= 0 && static_cast<std::uint64_t>(code) >= categories.size()))
                fail(ErrorCode::out_of_bounds, child_path(path, "codes"), "categorical code is outside dictionary bounds");
        }
        return;
    }
    if (column.kind() == ColumnKind::numeric) {
        auto values = column.values();
        validate_array(values, child_path(path, "values"), validation);
        if (column.nullable() && (values.dtype() == DType::float32 || values.dtype() == DType::float64))
            fail(ErrorCode::unsupported_dtype, path, "nullable floating-point columns have no supported AnnData codec");
        return;
    }
    for (std::size_t i = 0; i < column.size(); ++i)
        validate_text(column.string(i), path, validation);
    });
}

void validate_table(TableView table, std::string_view path, Validation validation) {
    with_validation_path(path, [&] {
    const auto index_name = table.index_name();
    if (index_name == "_index")
        fail(ErrorCode::invalid_argument, child_path(path, "_index"), "explicit index name '_index' is reserved as the unnamed dataframe index sentinel");
    const std::string index_key = index_name.empty() ? "_index" : std::string(index_name);
    validate_key(index_key, child_path(path, index_key), validation);
    for (std::size_t row = 0; row < table.size(); ++row) validate_text(table.name(row), child_path(path, index_key), validation);
    const auto names = table.column_names();
    for (const auto& name : names) {
        validate_key(name, child_path(path, name), validation);
        if (name == "_index")
            fail(ErrorCode::invalid_argument, child_path(path, name), "'_index' is reserved for dataframe indexes");
        auto column = table.column(name);
        validate_column(column, child_path(path, name), validation);
        if (name == index_key) {
            if (column.kind() != ColumnKind::string || column.nullable())
                fail(ErrorCode::invalid_argument, path, "index name collides with a column having a different representation");
            for (std::size_t row = 0; row < table.size(); ++row)
                if (column.string(row) != table.name(row))
                    fail(ErrorCode::invalid_argument, path, "index name collides with a column having different values");
        }
    }
    });
}

void validate_element(ElementView value, std::string_view path, Validation validation = Validation::full) {
    with_validation_path(path, [&] {
    switch (value.kind()) {
        case ElementKind::null: return;
        case ElementKind::boolean:
        case ElementKind::signed_integer:
        case ElementKind::unsigned_integer:
        case ElementKind::floating: (void)value.dtype(); return;
        case ElementKind::string: validate_text(value.string(), path, validation); return;
        case ElementKind::array: validate_array(value.array(), path, validation); return;
        case ElementKind::string_array: {
            auto array = value.string_array();
            if (validation == Validation::full) array.validate();
            const auto shape = shape_of(array);
            (void)element_count(shape, path);
            for (std::size_t i = 0; i < array.size(); ++i) validate_text(array.string(i), path, validation);
            return;
        }
        case ElementKind::column: validate_column(value.column(), path, validation); return;
        case ElementKind::table: validate_table(value.table(), path, validation); return;
        case ElementKind::mapping:
            for (const auto& key : value.keys()) {
                validate_key(key, child_path(path, key), validation);
                validate_element(value.at(key), child_path(path, key), validation);
            }
            return;
    }
    fail(ErrorCode::unsupported_encoding, path, "unsupported metadata element kind");
    });
}

std::vector<std::size_t> coordinates(Shape shape, std::size_t linear) {
    std::vector<std::size_t> result(shape.size(), 0);
    for (std::size_t axis = shape.size(); axis > 0; --axis) {
        const auto extent = shape[axis - 1];
        result[axis - 1] = linear % extent;
        linear /= extent;
    }
    return result;
}

void write_array(Backend& backend, std::string_view path, ArrayView array, std::size_t scratch_bytes,
                 Validation validation = Validation::full) {
    validate_array(array, path, validation);
    const auto type = array.dtype();
    if (array.layout() != Layout::dense) {
        std::optional<Array> copied;
        std::optional<SparseBytesView<const std::byte>> bytes;
        try { auto view = array.sparse_bytes(); bytes.emplace(view); }
        catch (const Error& error) {
            if (error.code() != ErrorCode::non_contiguous) throw;
            copied.emplace(array.copy(Validation::none));
            // As with byte views above, avoid Clang 18's unresolved module-view moves.
            const auto copied_view = copied->view();
            array = copied_view;
            auto view = array.sparse_bytes();
            bytes.emplace(view);
        }
        const auto major = array.extent(array.layout() == Layout::csr ? 0 : 1);
        if (major == std::numeric_limits<std::size_t>::max())
            fail(ErrorCode::invalid_argument, path, "sparse offset count overflows");
        const auto rows = array.extent(0), columns = array.extent(1);
        if (rows > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
            columns > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
            fail(ErrorCode::invalid_argument, path, "sparse shape exceeds the supported attribute integer range");
        const auto sparse_encoding = array.layout() == Layout::csr ? "csr_matrix" : "csc_matrix";
        auto attrs = encoding(sparse_encoding, "0.1.0");
        attrs.emplace("shape", std::vector<std::int64_t>{static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns)});
        backend.create_group(path, attrs);
        const auto nnz = bytes->nnz();
        backend.create_numeric(child_path(path, "data"), type, {nnz}, {} , array.serialization_byte_order());
        if (nnz) backend.write_numeric(child_path(path, "data"), {0}, {nnz}, bytes->values());
        backend.create_numeric(child_path(path, "indices"), array.index_dtype(), {nnz}, {}, array.index_serialization_byte_order());
        if (nnz) backend.write_numeric(child_path(path, "indices"), {0}, {nnz}, bytes->indices());
        const auto offset_count = major + 1;
        backend.create_numeric(child_path(path, "indptr"), array.offset_dtype(), {offset_count}, {}, array.offset_serialization_byte_order());
        backend.write_numeric(child_path(path, "indptr"), {0}, {offset_count}, bytes->offsets());
        return;
    }

    const auto shape = shape_of(array);
    const auto count = element_count(shape, path);
    const auto attrs = encoding("array", "0.2.0");
    backend.create_numeric(path, type, shape, attrs, array.serialization_byte_order());
    if (!count) return;

    std::optional<DenseBytesView<const std::byte>> bytes;
    try { auto view = array.bytes(); bytes.emplace(view); }
    catch (const Error& error) { if (error.code() != ErrorCode::non_contiguous) throw; }
    const Shape origin(shape.size(), 0);
    if (bytes && bytes->contiguous()) {
        backend.write_numeric(path, origin, shape, bytes->data());
        return;
    }
    if (shape.empty()) {
        auto scalar = array.copy(Validation::none);
        auto scalar_bytes = scalar.view().bytes();
        backend.write_numeric(path, {}, {}, scalar_bytes.data());
        return;
    }
    const auto width = dtype_size(type);
    if (scratch_bytes < width)
        fail(ErrorCode::invalid_argument, path, "scratch buffer must hold at least one numerical element for strided transfer");
    const auto tile_limit = std::max<std::size_t>(1, scratch_bytes / width);
    const auto columns = shape.back();
    const auto outer_count = count / columns;
    std::vector<std::byte> packed;
    for (std::size_t outer = 0; outer < outer_count; ++outer) {
        auto start = coordinates(Shape(shape.begin(), shape.end() - 1), outer);
        start.push_back(0);
        for (std::size_t first = 0; first < columns;) {
            const auto length = std::min(tile_limit, columns - first);
            start.back() = first;
            Shape tile_shape(shape.size(), 1);
            tile_shape.back() = length;
            auto transfer_start = start;
            if (bytes) {
                if (length > std::numeric_limits<std::size_t>::max() / width)
                    fail(ErrorCode::invalid_argument, path, "strided transfer buffer size overflows");
                packed.resize(length * width);
                std::ptrdiff_t offset = 0;
                auto coordinate = outer;
                for (std::size_t axis = shape.size() - 1; axis > 0; --axis) {
                    const auto previous_extent = shape[axis - 1];
                    const auto index = coordinate % previous_extent;
                    coordinate /= previous_extent;
                    offset += static_cast<std::ptrdiff_t>(index) * bytes->stride_bytes(axis - 1);
                }
                const auto column_stride = bytes->stride_bytes(shape.size() - 1);
                for (std::size_t i = 0; i < length; ++i)
                    std::memcpy(packed.data() + i * width,
                        bytes->data() + offset + static_cast<std::ptrdiff_t>(first + i) * column_stride, width);
                backend.write_numeric(path, transfer_start, tile_shape, packed.data());
            } else {
                auto coord = coordinates(Shape(shape.begin(), shape.end() - 1), outer);
                std::vector<std::optional<std::vector<std::size_t>>> selectors(shape.size());
                for (std::size_t axis = 0; axis + 1 < shape.size(); ++axis)
                    selectors[axis] = std::vector<std::size_t>{coord[axis]};
                auto& last = selectors.back().emplace();
                last.clear();
                last.reserve(length);
                for (std::size_t i = 0; i < length; ++i) last.push_back(first + i);
                auto tile = array.select(selectors).copy(Validation::none);
                auto tile_bytes = tile.view().bytes();
                backend.write_numeric(path, transfer_start, tile_shape, tile_bytes.data());
            }
            first += length;
        }
    }
}

template<class Getter>
void write_strings(Backend& backend, std::string_view path, Shape shape, Getter get,
                   std::size_t scratch_bytes, Attributes attrs = encoding("string-array", "0.2.0")) {
    const auto count = element_count(shape, path);
    const auto value_bytes = [&](std::size_t index) {
        const auto value = get(index);
        if (value.size() == std::numeric_limits<std::size_t>::max())
            fail(ErrorCode::invalid_argument, path, "string transfer size overflows size_t");
        return value.size() + 1; // Include the terminating byte in the scratch budget.
    };
    const auto storage_bytes = [&](std::size_t capacity) {
        if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(std::string))
            fail(ErrorCode::invalid_argument, path, "string vector capacity overflows size_t");
        return capacity * sizeof(std::string);
    };

    // Find the actual one-value metadata cost and reject an oversized value
    // before creating the target or copying any string payload.
    std::vector<std::string> single;
    if (count) {
        single.reserve(1);
        const auto metadata_bytes = storage_bytes(single.capacity());
        if (metadata_bytes > scratch_bytes)
            fail(ErrorCode::invalid_argument, path, "scratch buffer cannot hold one string value and its vector metadata");
        const auto payload_budget = scratch_bytes - metadata_bytes;
        for (std::size_t i = 0; i < count; ++i)
            if (value_bytes(i) > payload_budget)
                fail(ErrorCode::invalid_argument, path, "one string value exceeds the available scratch buffer");
    }
    std::vector<std::string>().swap(single);

    backend.create_strings(path, shape, attrs);
    if (!count) return;

    const auto write_batch = [&](std::size_t first, std::size_t maximum,
                                 const std::vector<std::size_t>& start,
                                 const std::vector<std::size_t>& prefix_count) {
        std::size_t requested = 0;
        std::size_t payload_bytes = 0;
        const auto requested_limit = std::min<std::size_t>(maximum, 4096);
        while (requested < requested_limit) {
            const auto payload = value_bytes(first + requested);
            if (requested > (scratch_bytes / sizeof(std::string))) break;
            const auto requested_metadata = (requested + 1) * sizeof(std::string);
            if (requested_metadata > scratch_bytes || payload_bytes > scratch_bytes - requested_metadata ||
                payload > scratch_bytes - requested_metadata - payload_bytes)
                break;
            payload_bytes += payload;
            ++requested;
        }
        if (!requested)
            fail(ErrorCode::invalid_argument, path, "scratch buffer cannot hold one string value and its vector metadata");

        for (;;) {
            std::vector<std::string> values;
            values.reserve(requested);
            const auto metadata_bytes = storage_bytes(values.capacity());
            payload_bytes = 0;
            bool fits = metadata_bytes <= scratch_bytes;
            for (std::size_t i = 0; i < requested; ++i) {
                const auto payload = value_bytes(first + i);
                if (metadata_bytes > scratch_bytes || payload_bytes > scratch_bytes - metadata_bytes ||
                    payload > scratch_bytes - metadata_bytes - payload_bytes) {
                    fits = false;
                    break;
                }
                payload_bytes += payload;
            }
            if (!fits) {
                if (requested == 1)
                    fail(ErrorCode::invalid_argument, path, "scratch buffer cannot hold one string value and its vector metadata");
                --requested;
                continue;
            }

            std::size_t copied_payload_bytes = 0;
            for (std::size_t i = 0; i < requested; ++i) {
                const auto value = get(first + i);
                if (value.size() == std::numeric_limits<std::size_t>::max())
                    fail(ErrorCode::invalid_argument, path, "string transfer size overflows size_t");
                const auto payload = value.size() + 1;
                if (copied_payload_bytes > scratch_bytes - metadata_bytes ||
                    payload > scratch_bytes - metadata_bytes - copied_payload_bytes)
                    fail(ErrorCode::invalid_argument, path, "string value changed after scratch preflight");
                values.emplace_back(value);
                copied_payload_bytes += payload;
            }
            auto tile_count = prefix_count;
            if (shape.empty()) tile_count.clear();
            else tile_count.back() = requested;
            backend.write_strings(path, start, tile_count, values);
            return requested;
        }
    };

    if (shape.empty()) {
        const auto written = write_batch(0, 1, {}, {});
        (void)written;
        return;
    }

    const auto columns = shape.back();
    const auto outer_count = count / columns;
    for (std::size_t outer = 0; outer < outer_count; ++outer) {
        auto start = coordinates(Shape(shape.begin(), shape.end() - 1), outer);
        start.push_back(0);
        Shape tile_shape(shape.size(), 1);
        for (std::size_t first = 0; first < columns;) {
            start.back() = first;
            const auto written = write_batch(outer * columns + first, columns - first, start, tile_shape);
            first += written;
        }
    }
}

void write_string_array(Backend& backend, std::string_view path, StringArrayView array,
                        std::size_t scratch_bytes, Validation validation = Validation::full) {
    if (validation == Validation::full) array.validate();
    const auto shape = shape_of(array);
    write_strings(backend, path, shape, [&](std::size_t i) { return array.string(i); }, scratch_bytes);
}

void write_column(Backend& backend, std::string_view path, ColumnView column, std::size_t scratch_bytes,
                  Validation validation = Validation::full) {
    if (column.kind() == ColumnKind::categorical) {
        auto attrs = encoding("categorical", "0.2.0");
        attrs.emplace("ordered", column.ordered());
        backend.create_group(path, attrs);
        if (validation == Validation::full) {
            const auto categories = column.categories();
            if (categories.kind() == ColumnKind::string) {
                std::unordered_set<std::string_view> seen;
                for (std::size_t i = 0; i < categories.size(); ++i)
                    if (!categories.valid(i) || !seen.insert(categories.string(i)).second)
                        fail(ErrorCode::invalid_argument, child_path(path, "categories"), "categorical categories must be non-null and unique");
            } else {
                for (std::size_t i = 0; i < categories.size(); ++i)
                    if (!categories.valid(i)) fail(ErrorCode::invalid_argument, child_path(path, "categories"), "category dictionaries cannot contain nulls");
                validate_unique_numeric(categories.values(), child_path(path, "categories"));
            }
            for (std::size_t i = 0; i < column.size(); ++i) {
                const auto code = column.code(i);
                if (code < -1 || (code >= 0 && static_cast<std::uint64_t>(code) >= categories.size()))
                    fail(ErrorCode::out_of_bounds, child_path(path, "codes"), "categorical code is outside dictionary bounds");
            }
        }
        write_array(backend, child_path(path, "codes"), column.codes(), scratch_bytes, validation);
        write_column(backend, child_path(path, "categories"), column.categories(), scratch_bytes, validation);
        return;
    }
    if (column.nullable()) {
        const bool strings = column.kind() == ColumnKind::string;
        const auto value_type = strings ? DType::boolean : column.values().dtype();
        auto attrs = encoding(strings ? "nullable-string-array" : value_type == DType::boolean ? "nullable-boolean" : "nullable-integer", "0.1.0");
        if (strings) attrs.emplace("na-value", std::string(column.string_missing_value() == StringMissingValue::NaN ? "NaN" : "NA"));
        backend.create_group(path, attrs);
        if (strings) {
            write_strings(backend, child_path(path, "values"), {column.size()},
                [&](std::size_t i) { return column.string(i); }, scratch_bytes);
        } else {
            write_array(backend, child_path(path, "values"), column.values(), scratch_bytes, validation);
        }
        auto mask_path = child_path(path, "mask");
        auto mask_attrs = encoding("array", "0.2.0");
        backend.create_numeric(mask_path, DType::boolean, {column.size()}, mask_attrs);
        const auto chunk_size = std::max<std::size_t>(1, scratch_bytes);
        std::vector<std::uint8_t> mask;
        for (std::size_t first = 0; first < column.size();) {
            const auto length = std::min(chunk_size, column.size() - first);
            mask.resize(length);
            for (std::size_t i = 0; i < length; ++i) mask[i] = static_cast<std::uint8_t>(!column.valid(first + i));
            backend.write_numeric(mask_path, {first}, {length}, mask.data());
            first += length;
        }
        return;
    }
    if (column.kind() == ColumnKind::numeric) {
        write_array(backend, path, column.values(), scratch_bytes, validation);
        return;
    }
    write_strings(backend, path, {column.size()}, [&](std::size_t i) { return column.string(i); }, scratch_bytes);
}

void write_table(Backend& backend, std::string_view path, TableView table, std::size_t scratch_bytes,
                 Validation validation = Validation::full) {
    validate_table(table, path, validation);
    const auto names = table.column_names();
    const std::string index_key = table.index_name().empty() ? "_index" : std::string(table.index_name());
    auto attrs = encoding("dataframe", "0.2.0");
    attrs.emplace("_index", index_key);
    attrs.emplace("column-order", names);
    backend.create_group(path, attrs);
    write_strings(backend, child_path(path, index_key), {table.size()},
        [&](std::size_t i) { return table.name(i); }, scratch_bytes);
    for (const auto& name : names)
        if (name != index_key) write_column(backend, child_path(path, name), table.column(name), scratch_bytes, validation);
}

void write_null(Backend& backend, std::string_view path) {
    backend.create_numeric(path, DType::float64, {}, encoding("null", "0.1.0"));
}

void write_element_impl(Backend& backend, std::string_view path, ElementView value, std::size_t scratch_bytes,
                        Validation validation = Validation::full) {
    switch (value.kind()) {
        case ElementKind::null: write_null(backend, path); return;
        case ElementKind::array: write_array(backend, path, value.array(), scratch_bytes, validation); return;
        case ElementKind::string_array: write_string_array(backend, path, value.string_array(), scratch_bytes, validation); return;
        case ElementKind::column: write_column(backend, path, value.column(), scratch_bytes, validation); return;
        case ElementKind::table: write_table(backend, path, value.table(), scratch_bytes, validation); return;
        case ElementKind::string:
            write_strings(backend, path, {}, [&](std::size_t) { return value.string(); }, scratch_bytes,
                          encoding("string", "0.2.0"));
            return;
        case ElementKind::mapping: {
            backend.create_group(path, encoding("dict", "0.1.0"));
            for (const auto& key : value.keys())
                write_element_impl(backend, child_path(path, key), value.at(key), scratch_bytes, validation);
            return;
        }
        case ElementKind::boolean:
        case ElementKind::signed_integer:
        case ElementKind::unsigned_integer:
        case ElementKind::floating: {
            const auto type = value.dtype();
            auto attrs = encoding("numeric-scalar", "0.2.0");
            backend.create_numeric(path, type, {}, attrs, value.serialization_byte_order());
            switch (type) {
                case DType::boolean: { const bool x = value.boolean(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::int8: { const auto x = value.scalar<std::int8_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::uint8: { const auto x = value.scalar<std::uint8_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::int16: { const auto x = value.scalar<std::int16_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::uint16: { const auto x = value.scalar<std::uint16_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::int32: { const auto x = value.scalar<std::int32_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::uint32: { const auto x = value.scalar<std::uint32_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::int64: { const auto x = value.scalar<std::int64_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::uint64: { const auto x = value.scalar<std::uint64_t>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::float32: { const auto x = value.scalar<float>(); backend.write_numeric(path, {}, {}, &x); return; }
                case DType::float64: { const auto x = value.scalar<double>(); backend.write_numeric(path, {}, {}, &x); return; }
            }
            return;
        }
    }
    fail(ErrorCode::unsupported_encoding, path, "unsupported metadata element kind");
}

template<class Getter>
void write_array_mapping(Backend& backend, std::string_view path, const std::vector<std::string>& keys,
                         Getter get, std::size_t scratch_bytes, Validation validation) {
    backend.create_group(path, encoding("dict", "0.1.0"));
    for (const auto& key : keys) write_array(backend, child_path(path, key), get(key), scratch_bytes, validation);
}

template<class Getter>
void write_element_mapping(Backend& backend, std::string_view path, const std::vector<std::string>& keys,
                           Getter get, std::size_t scratch_bytes, Validation validation) {
    backend.create_group(path, encoding("dict", "0.1.0"));
    for (const auto& key : keys)
        write_element_impl(backend, child_path(path, key), get(key), scratch_bytes, validation);
}

void validate_axis_element(ElementView value, std::size_t expected, TableView axis,
                           std::string_view path, Validation validation) {
    validate_element(value, path, validation);
    if (value.kind() == ElementKind::array) {
        const auto array = value.array();
        if (array.rank() == 0 || array.extent(0) != expected)
            fail(ErrorCode::shape_mismatch, path, "aligned array does not match its AnnData axis");
    } else if (value.kind() == ElementKind::string_array) {
        const auto array = value.string_array();
        if (array.rank() == 0 || array.extent(0) != expected)
            fail(ErrorCode::shape_mismatch, path, "aligned string array does not match its AnnData axis");
    } else if (value.kind() == ElementKind::table) {
        if (value.table().names() != axis.names())
            fail(ErrorCode::shape_mismatch, path, "aligned table index differs from the AnnData axis");
    } else {
        fail(ErrorCode::unsupported_encoding, path, "aligned AnnData slots must contain an array, string array, or table");
    }
}

void validate_dataset_impl(const AnnDataView& data, Validation validation) {
    const auto obs = data.obs();
    const auto var = data.var();
    if (obs.size() != data.n_obs() || var.size() != data.n_vars())
        fail(ErrorCode::shape_mismatch, "/", "obs or var table length differs from the AnnData axis");
    validate_table(obs, "/obs", validation);
    validate_table(var, "/var", validation);
    validate_element(data.uns(), "/uns", validation);
    if (data.has_x()) {
        const auto x = data.x();
        validate_array(x, "/X", validation);
        if (x.rank() != 2 || x.extent(0) != data.n_obs() || x.extent(1) != data.n_vars())
            fail(ErrorCode::shape_mismatch, "/X", "X shape differs from the observation and variable axes");
    }
    for (const auto& key : data.layer_names()) {
        validate_key(key, child_path("/layers", key), validation);
        const auto value = data.layer(key);
        validate_array(value, child_path("/layers", key), validation);
        if (value.rank() != 2 || value.extent(0) != data.n_obs() || value.extent(1) != data.n_vars())
            fail(ErrorCode::shape_mismatch, child_path("/layers", key), "layer shape differs from X axes");
    }
    for (const auto& key : data.obsp_names()) {
        validate_key(key, child_path("/obsp", key), validation);
        const auto value = data.obsp(key);
        validate_array(value, child_path("/obsp", key), validation);
        if (value.rank() < 2 || value.extent(0) != data.n_obs() || value.extent(1) != data.n_obs())
            fail(ErrorCode::shape_mismatch, child_path("/obsp", key), "first two obsp dimensions must be square over observations");
    }
    for (const auto& key : data.varp_names()) {
        validate_key(key, child_path("/varp", key), validation);
        const auto value = data.varp(key);
        validate_array(value, child_path("/varp", key), validation);
        if (value.rank() < 2 || value.extent(0) != data.n_vars() || value.extent(1) != data.n_vars())
            fail(ErrorCode::shape_mismatch, child_path("/varp", key), "first two varp dimensions must be square over variables");
    }
    for (const auto& key : data.obsm_names()) {
        validate_key(key, child_path("/obsm", key), validation);
        validate_axis_element(data.obsm(key), data.n_obs(), obs, child_path("/obsm", key), validation);
    }
    for (const auto& key : data.varm_names()) {
        validate_key(key, child_path("/varm", key), validation);
        validate_axis_element(data.varm(key), data.n_vars(), var, child_path("/varm", key), validation);
    }
    if (data.has_raw()) {
        const auto raw_var = data.raw_var();
        validate_table(raw_var, "/raw/var", validation);
        if (data.has_raw_x()) {
            const auto x = data.raw_x();
            validate_array(x, "/raw/X", validation);
            if (x.rank() != 2 || x.extent(0) != data.n_obs() || x.extent(1) != raw_var.size())
                fail(ErrorCode::shape_mismatch, "/raw/X", "raw X shape differs from its observation and variable axes");
        }
        for (const auto& key : data.raw_varm_names()) {
            validate_key(key, child_path("/raw/varm", key), validation);
            validate_axis_element(data.raw_varm(key), raw_var.size(), raw_var, child_path("/raw/varm", key), validation);
        }
    }
}

void validate_dataset(const AnnDataView& data) {
    validate_dataset_impl(data, Validation::full);
}

void validate_dataset_serializable(const AnnDataView& data, Validation validation) {
    validate_dataset_impl(data, validation);
}

} // namespace

void validate_serializable(ElementView value, std::string_view path) {
    validate_element(value, path);
}

void write_element(Backend& backend, std::string_view path, ElementView value, std::size_t scratch_bytes) {
    validate_serializable(value, path);
    write_element_impl(backend, path, value, scratch_bytes);
}

void initialize_store(Backend& backend, const AnnDataView& source, std::size_t scratch_bytes,
                      Validation validation) {
    if (validation == Validation::full) validate_dataset(source);
    else validate_dataset_serializable(source, validation);
    backend.attributes("/", encoding("anndata", "0.1.0"));
    write_table(backend, "/obs", source.obs(), scratch_bytes, validation);
    write_table(backend, "/var", source.var(), scratch_bytes, validation);
    if (source.has_x()) write_array(backend, "/X", source.x(), scratch_bytes, validation);
    write_array_mapping(backend, "/layers", source.layer_names(), [&](const auto& key) { return source.layer(key); }, scratch_bytes, validation);
    write_array_mapping(backend, "/obsp", source.obsp_names(), [&](const auto& key) { return source.obsp(key); }, scratch_bytes, validation);
    write_array_mapping(backend, "/varp", source.varp_names(), [&](const auto& key) { return source.varp(key); }, scratch_bytes, validation);
    write_element_mapping(backend, "/obsm", source.obsm_names(), [&](const auto& key) { return source.obsm(key); }, scratch_bytes, validation);
    write_element_mapping(backend, "/varm", source.varm_names(), [&](const auto& key) { return source.varm(key); }, scratch_bytes, validation);
    write_element_impl(backend, "/uns", source.uns(), scratch_bytes, validation);
    if (source.has_raw()) {
        backend.create_group("/raw", encoding("raw", "0.1.0"));
        if (source.has_raw_x()) write_array(backend, "/raw/X", source.raw_x(), scratch_bytes, validation);
        else write_null(backend, "/raw/X");
        write_table(backend, "/raw/var", source.raw_var(), scratch_bytes, validation);
        write_element_mapping(backend, "/raw/varm", source.raw_varm_names(), [&](const auto& key) { return source.raw_varm(key); }, scratch_bytes, validation);
    } else {
        write_null(backend, "/raw");
    }
    backend.flush();
}

} // namespace anndata::backed_detail
