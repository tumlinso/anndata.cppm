module;
#include <hdf5.h>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

module anndatapp.backed;
import :storage;

namespace anndatapp::backed_detail {
namespace {

class H5Handle {
public:
    H5Handle() = default;
    H5Handle(hid_t value, herr_t (*close)(hid_t)) : value_(value), close_(close) {}
    ~H5Handle() { reset(); }
    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;
    H5Handle(H5Handle&& other) noexcept : value_(std::exchange(other.value_, -1)), close_(other.close_) {}
    H5Handle& operator=(H5Handle&& other) noexcept {
        if (this != &other) { reset(); value_ = std::exchange(other.value_, -1); close_ = other.close_; }
        return *this;
    }
    void reset() noexcept { if (value_ >= 0 && close_) close_(std::exchange(value_, -1)); }
    operator hid_t() const noexcept { return value_; }
    bool valid() const noexcept { return value_ >= 0; }
private:
    hid_t value_ = -1;
    herr_t (*close_)(hid_t) = nullptr;
};

struct H5Context {
    std::string filename;
    Validation validation;
    [[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) const {
        throw Error(code, filename + ":" + std::string(path), std::move(message));
    }
    hid_t id(hid_t result, std::string_view path, std::string_view op) const {
        if (result < 0) fail(ErrorCode::io_error, path, std::string(op) + " failed");
        return result;
    }
    void check(herr_t result, std::string_view path, std::string_view op) const {
        if (result < 0) fail(ErrorCode::io_error, path, std::string(op) + " failed");
    }
};

std::size_t product(const H5Context& c, const std::vector<std::size_t>& shape, std::string_view path) {
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) return 0;
    std::size_t n = 1;
    for (auto d : shape) {
        if (d && n > static_cast<std::size_t>(PTRDIFF_MAX) / d)
            c.fail(ErrorCode::invalid_argument, path, "element count exceeds addressable range");
        n *= d;
    }
    return n;
}

std::vector<std::size_t> shape_of(const H5Context& c, hid_t space, std::string_view path) {
    const int rank = H5Sget_simple_extent_ndims(space);
    if (rank < 0) c.fail(ErrorCode::io_error, path, "inspect dataspace rank failed");
    std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
    c.check(H5Sget_simple_extent_dims(space, dims.data(), nullptr), path, "inspect dataspace dimensions");
    std::vector<std::size_t> result;
    for (auto d : dims) {
        if (d > std::numeric_limits<std::size_t>::max()) c.fail(ErrorCode::unsupported_layout, path, "dimension exceeds size_t");
        result.push_back(static_cast<std::size_t>(d));
    }
    return result;
}

H5Handle space_for(const H5Context& c, const std::vector<std::size_t>& shape, std::string_view path) {
    if (shape.empty()) return {c.id(H5Screate(H5S_SCALAR), path, "create scalar dataspace"), H5Sclose};
    if (shape.size() > H5S_MAX_RANK) c.fail(ErrorCode::unsupported_layout, path, "rank exceeds HDF5 maximum");
    std::vector<hsize_t> dims(shape.begin(), shape.end());
    return {c.id(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), path, "create dataspace"), H5Sclose};
}

H5Handle string_type(const H5Context& c, std::string_view path) {
    H5Handle type(c.id(H5Tcopy(H5T_C_S1), path, "create string datatype"), H5Tclose);
    c.check(H5Tset_size(type, H5T_VARIABLE), path, "set variable string width");
    c.check(H5Tset_cset(type, H5T_CSET_UTF8), path, "set UTF-8 strings");
    return type;
}

H5Handle boolean_type(const H5Context& c, std::string_view path) {
    static_assert(sizeof(bool) == 1);
    H5Handle type(c.id(H5Tenum_create(H5T_STD_U8LE), path, "create boolean datatype"), H5Tclose);
    const unsigned char no = 0, yes = 1;
    c.check(H5Tenum_insert(type, "FALSE", &no), path, "define FALSE");
    c.check(H5Tenum_insert(type, "TRUE", &yes), path, "define TRUE");
    return type;
}

DType dtype_of_h5(const H5Context& c, hid_t type, std::string_view path) {
    const auto kind = H5Tget_class(type);
    const auto width = H5Tget_size(type);
    if (kind == H5T_ENUM) {
        if (width == 1 && H5Tget_nmembers(type) == 2) {
            unsigned char no = 255, yes = 255;
            if (H5Tenum_valueof(type, "FALSE", &no) >= 0 && H5Tenum_valueof(type, "TRUE", &yes) >= 0 && no == 0 && yes == 1) return DType::boolean;
        }
        c.fail(ErrorCode::unsupported_dtype, path, "only the FALSE/TRUE boolean enum is supported");
    }
    if (kind == H5T_INTEGER && H5Tget_precision(type) == width * 8 && H5Tget_offset(type) == 0) {
        const auto sign = H5Tget_sign(type);
        if (sign != H5T_SGN_NONE && sign != H5T_SGN_2) c.fail(ErrorCode::unsupported_dtype, path, "unsupported integer sign representation");
        const bool s = sign == H5T_SGN_2;
        if (width == 1) return s ? DType::int8 : DType::uint8;
        if (width == 2) return s ? DType::int16 : DType::uint16;
        if (width == 4) return s ? DType::int32 : DType::uint32;
        if (width == 8) return s ? DType::int64 : DType::uint64;
    }
    if (kind == H5T_FLOAT) {
        if (H5Tequal(type, H5T_IEEE_F32LE) > 0 || H5Tequal(type, H5T_IEEE_F32BE) > 0) return DType::float32;
        if (H5Tequal(type, H5T_IEEE_F64LE) > 0 || H5Tequal(type, H5T_IEEE_F64BE) > 0) return DType::float64;
    }
    c.fail(ErrorCode::unsupported_dtype, path, "unsupported numerical datatype");
}

hid_t native_type(DType type) {
    switch (type) {
        case DType::boolean: return H5T_NATIVE_UCHAR;
        case DType::int8: return H5T_NATIVE_INT8;
        case DType::uint8: return H5T_NATIVE_UINT8;
        case DType::int16: return H5T_NATIVE_INT16;
        case DType::uint16: return H5T_NATIVE_UINT16;
        case DType::int32: return H5T_NATIVE_INT32;
        case DType::uint32: return H5T_NATIVE_UINT32;
        case DType::int64: return H5T_NATIVE_INT64;
        case DType::uint64: return H5T_NATIVE_UINT64;
        case DType::float32: return H5T_NATIVE_FLOAT;
        case DType::float64: return H5T_NATIVE_DOUBLE;
    }
    return -1;
}

hid_t file_type(DType type, ByteOrder order) {
    const bool little = order == ByteOrder::little || (order == ByteOrder::native && std::endian::native == std::endian::little);
    switch (type) {
        case DType::boolean: return -1;
        case DType::int8: return little ? H5T_STD_I8LE : H5T_STD_I8BE;
        case DType::uint8: return little ? H5T_STD_U8LE : H5T_STD_U8BE;
        case DType::int16: return little ? H5T_STD_I16LE : H5T_STD_I16BE;
        case DType::uint16: return little ? H5T_STD_U16LE : H5T_STD_U16BE;
        case DType::int32: return little ? H5T_STD_I32LE : H5T_STD_I32BE;
        case DType::uint32: return little ? H5T_STD_U32LE : H5T_STD_U32BE;
        case DType::int64: return little ? H5T_STD_I64LE : H5T_STD_I64BE;
        case DType::uint64: return little ? H5T_STD_U64LE : H5T_STD_U64BE;
        case DType::float32: return little ? H5T_IEEE_F32LE : H5T_IEEE_F32BE;
        case DType::float64: return little ? H5T_IEEE_F64LE : H5T_IEEE_F64BE;
    }
    return -1;
}

ByteOrder byte_order(hid_t type) {
    const auto order = H5Tget_order(type);
    return order == H5T_ORDER_LE ? ByteOrder::little : order == H5T_ORDER_BE ? ByteOrder::big : ByteOrder::native;
}

void check_filters(const H5Context& c, hid_t dataset, std::string_view path, bool writing = false) {
    H5Handle p(c.id(H5Dget_create_plist(dataset), path, "inspect dataset properties"), H5Pclose);
    const auto n = H5Pget_nfilters(p);
    if (n < 0) c.fail(ErrorCode::io_error, path, "inspect dataset filters failed");
    for (int i = 0; i < n; ++i) {
        unsigned flags = 0, config = 0;
        std::size_t count = 0;
        const auto filter = H5Pget_filter2(p, static_cast<unsigned>(i), &flags, &count, nullptr, 0, nullptr, nullptr);
        if (filter < 0) c.fail(ErrorCode::io_error, path, "inspect dataset filter failed");
        const unsigned required = writing ? H5Z_FILTER_CONFIG_ENCODE_ENABLED : H5Z_FILTER_CONFIG_DECODE_ENABLED;
        if (H5Zfilter_avail(filter) <= 0 || H5Zget_filter_info(filter, &config) < 0 || !(config & required))
            c.fail(ErrorCode::unavailable_filter, path, "HDF5 filter unavailable: " + std::to_string(filter));
    }
}

struct Reclaim {
    hid_t type, space;
    void* buffer;
    ~Reclaim() { H5Dvlen_reclaim(type, space, H5P_DEFAULT, buffer); }
};

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

template<class Read>
std::vector<std::string> read_text(const H5Context& c, hid_t type, hid_t space, Read read, std::string_view path) {
    const auto n = product(c, shape_of(c, space, path), path);
    std::vector<std::string> result;
    result.reserve(n);
    if (!n) return result;
    if (H5Tis_variable_str(type) > 0) {
        std::vector<char*> pointers(n, nullptr);
        Reclaim reclaim{type, space, pointers.data()};
        c.check(read(type, pointers.data()), path, "read variable strings");
        for (const auto* p : pointers) result.emplace_back(p ? p : "");
    } else {
        const auto width = H5Tget_size(type);
        if (width && n > static_cast<std::size_t>(PTRDIFF_MAX) / width) c.fail(ErrorCode::invalid_argument, path, "string byte count exceeds addressable range");
        std::vector<char> bytes(n * width);
        c.check(read(type, bytes.data()), path, "read fixed strings");
        const auto pad = H5Tget_strpad(type);
        for (std::size_t i = 0; i < n; ++i) {
            std::string_view s(bytes.data() + i * width, width);
            if (pad == H5T_STR_SPACEPAD) { while (!s.empty() && s.back() == ' ') s.remove_suffix(1); }
            else { const auto end = s.find('\0'); if (end != std::string_view::npos) s = s.substr(0, end); }
            result.emplace_back(s);
        }
    }
    if (c.validation == Validation::full)
        for (const auto& s : result) if (!valid_utf8(s)) c.fail(ErrorCode::unsupported_dtype, path, "string data is not valid UTF-8");
    return result;
}

Attributes read_attributes(const H5Context& c, hid_t object, std::string_view path, bool& opaque) {
    H5O_info_t info{};
    c.check(H5Oget_info(object, &info), path, "inspect attributes");
    std::string encoding;
    const auto has_encoding = H5Aexists(object, "encoding-type");
    if (has_encoding < 0) c.fail(ErrorCode::io_error, path, "inspect encoding attribute failed");
    if (has_encoding) {
        H5Handle attr(c.id(H5Aopen(object, "encoding-type", H5P_DEFAULT), path, "open encoding attribute"), H5Aclose);
        H5Handle type(c.id(H5Aget_type(attr), path, "inspect encoding datatype"), H5Tclose);
        H5Handle space(c.id(H5Aget_space(attr), path, "inspect encoding dataspace"), H5Sclose);
        if (H5Tget_class(type) == H5T_STRING && H5Sget_simple_extent_type(space) == H5S_SCALAR)
            encoding = read_text(c, type, space, [&](hid_t t, void* p) { return H5Aread(attr, t, p); }, path).front();
    }
    const auto semantic = [&](std::string_view key) {
        return key == "encoding-type" || key == "encoding-version" ||
            (encoding == "dataframe" && (key == "column-order" || key == "_index")) ||
            ((encoding == "csr_matrix" || encoding == "csc_matrix") && key == "shape") ||
            (encoding == "categorical" && key == "ordered") ||
            (encoding == "nullable-string-array" && key == "na-value");
    };
    Attributes result;
    for (hsize_t i = 0; i < info.num_attrs; ++i) {
        H5Handle attr(c.id(H5Aopen_by_idx(object, ".", H5_INDEX_NAME, H5_ITER_INC, i, H5P_DEFAULT, H5P_DEFAULT), path, "open attribute"), H5Aclose);
        const auto length = H5Aget_name(attr, 0, nullptr);
        if (length < 0) c.fail(ErrorCode::io_error, path, "inspect attribute name failed");
        std::string key(static_cast<std::size_t>(length) + 1, '\0');
        if (H5Aget_name(attr, key.size(), key.data()) < 0) c.fail(ErrorCode::io_error, path, "read attribute name failed");
        key.resize(static_cast<std::size_t>(length));
        // Semantic codecs consume only these attributes. Extension attributes
        // stay opaque so same-store replacement can preserve their exact HDF5
        // datatypes and dataspaces, rather than promoting them via Attribute.
        if (!semantic(key)) { opaque = true; continue; }
        H5Handle type(c.id(H5Aget_type(attr), path, "inspect attribute datatype"), H5Tclose);
        H5Handle space(c.id(H5Aget_space(attr), path, "inspect attribute dataspace"), H5Sclose);
        if (H5Sget_simple_extent_type(space) == H5S_NULL) continue;
        const auto shape = shape_of(c, space, path);
        const auto n = product(c, shape, path);
        const bool scalar = shape.empty();
        // Empty dataframe column orders are often stored as empty float arrays
        // by h5py. They carry no element datatype information.
        if (key == "column-order" && shape == std::vector<std::size_t>{0}) {
            result.emplace(std::move(key), std::vector<std::string>{});
        } else if (H5Tget_class(type) == H5T_STRING && (scalar || shape.size() == 1)) {
            auto values = read_text(c, type, space, [&](hid_t t, void* p) { return H5Aread(attr, t, p); }, path);
            if (scalar) result.emplace(std::move(key), std::move(values.front()));
            else result.emplace(std::move(key), std::move(values));
        } else if (H5Tget_class(type) == H5T_ENUM && scalar && key == "ordered" && dtype_of_h5(c, type, path) == DType::boolean) {
            unsigned char value = 0;
            c.check(H5Aread(attr, type, &value), path, "read boolean attribute");
            result.emplace(std::move(key), value != 0);
        } else if (H5Tget_class(type) == H5T_INTEGER && (scalar || shape.size() == 1)) {
            std::vector<std::int64_t> values(n);
            if (H5Tget_sign(type) == H5T_SGN_NONE) {
                std::vector<std::uint64_t> unsigned_values(n);
                if (n) c.check(H5Aread(attr, H5T_NATIVE_UINT64, unsigned_values.data()), path, "read unsigned attribute");
                bool representable = true;
                for (std::size_t j = 0; j < n; ++j) {
                    if (unsigned_values[j] > static_cast<std::uint64_t>(INT64_MAX)) { representable = false; break; }
                    values[j] = static_cast<std::int64_t>(unsigned_values[j]);
                }
                if (!representable) continue; // opaque extension metadata remains in place
            } else if (n) c.check(H5Aread(attr, H5T_NATIVE_INT64, values.data()), path, "read integer attribute");
            if (scalar) result.emplace(std::move(key), values.front());
            else result.emplace(std::move(key), std::move(values));
        } else if (H5Tget_class(type) == H5T_FLOAT && scalar) {
            double value = 0;
            c.check(H5Aread(attr, H5T_NATIVE_DOUBLE, &value), path, "read floating attribute");
            result.emplace(std::move(key), value);
        }
        // Unsupported extension attributes are not rewritten or removed. The
        // semantic codecs diagnose missing/invalid required metadata themselves.
    }
    return result;
}

void write_attributes(const H5Context& c, hid_t object, const Attributes& attrs, std::string_view path) {
    for (const auto& [key, value] : attrs) {
        if (key.empty() || key.find('\0') != std::string::npos) c.fail(ErrorCode::invalid_argument, path, "invalid attribute name");
        const auto exists = H5Aexists(object, key.c_str());
        if (exists < 0) c.fail(ErrorCode::io_error, path, "inspect attribute existence failed");
        if (exists) c.check(H5Adelete(object, key.c_str()), path, "replace attribute");
        std::visit([&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::vector<std::string>>) {
                auto type = string_type(c, path);
                const auto shape = [&] { if constexpr (std::is_same_v<T, std::string>) return std::vector<std::size_t>{}; else return std::vector<std::size_t>{v.size()}; }();
                auto space = space_for(c, shape, path);
                H5Handle attr(c.id(H5Acreate2(object, key.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), path, "create string attribute"), H5Aclose);
                std::vector<const char*> pointers;
                auto add = [&](const std::string& s) { if (s.find('\0') != std::string::npos) c.fail(ErrorCode::invalid_argument, path, "embedded NUL in string attribute"); pointers.push_back(s.c_str()); };
                if constexpr (std::is_same_v<T, std::string>) add(v); else for (const auto& s : v) add(s);
                if (!pointers.empty()) c.check(H5Awrite(attr, type, pointers.data()), path, "write string attribute");
            } else {
                const auto shape = [&] { if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) return std::vector<std::size_t>{v.size()}; else return std::vector<std::size_t>{}; }();
                auto space = space_for(c, shape, path);
                H5Handle boolean;
                const hid_t type = [&]() -> hid_t {
                    if constexpr (std::is_same_v<T, bool>) { boolean = boolean_type(c, path); return boolean; }
                    else if constexpr (std::is_same_v<T, double>) return H5T_IEEE_F64LE;
                    else return H5T_STD_I64LE;
                }();
                H5Handle attr(c.id(H5Acreate2(object, key.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT), path, "create numeric attribute"), H5Aclose);
                if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) { if (!v.empty()) c.check(H5Awrite(attr, H5T_NATIVE_INT64, v.data()), path, "write integer attribute"); }
                else if constexpr (std::is_same_v<T, bool>) { const unsigned char b = v ? 1 : 0; c.check(H5Awrite(attr, type, &b), path, "write boolean attribute"); }
                else c.check(H5Awrite(attr, std::is_same_v<T, double> ? H5T_NATIVE_DOUBLE : H5T_NATIVE_INT64, &v), path, "write numeric attribute");
            }
        }, value);
    }
}

bool has_vlen(const H5Context& c, hid_t type, std::string_view path) {
    const auto kind = H5Tget_class(type);
    if (kind == H5T_VLEN || H5Tis_variable_str(type) > 0) return true;
    if (kind == H5T_ARRAY) {
        H5Handle element(c.id(H5Tget_super(type), path, "inspect array attribute datatype"), H5Tclose);
        return has_vlen(c, element, path);
    }
    if (kind == H5T_COMPOUND) {
        const auto count = H5Tget_nmembers(type);
        if (count < 0) c.fail(ErrorCode::io_error, path, "inspect compound attribute datatype failed");
        for (int i = 0; i < count; ++i) {
            H5Handle member(c.id(H5Tget_member_type(type, static_cast<unsigned>(i)), path, "inspect compound attribute member"), H5Tclose);
            if (has_vlen(c, member, path)) return true;
        }
    }
    return false;
}

void account_bytes(const H5Context& c, std::size_t size, std::size_t& used,
                   std::size_t budget, std::string_view path) {
    if (size > budget - used) c.fail(ErrorCode::invalid_argument, path, "opaque attribute exceeds the metadata allocation budget");
    used += size;
}

// Attribute reads have no transfer-property argument in HDF5 1.10, so their
// variable-length allocator cannot be capped ahead of H5Aread. Account the
// resulting payload immediately and reclaim it before propagating any error.
void account_vlen(const H5Context& c, hid_t type, const std::byte* bytes,
                   std::size_t& used, std::size_t budget, std::string_view path) {
    if (H5Tis_variable_str(type) > 0) {
        const char* value = nullptr;
        std::memcpy(&value, bytes, sizeof(value));
        if (value) {
            const auto remaining = budget - used;
            const auto length = strnlen(value, remaining + 1);
            if (length >= remaining) c.fail(ErrorCode::invalid_argument, path, "opaque variable string exceeds the metadata allocation budget");
            account_bytes(c, length + 1, used, budget, path);
        }
        return;
    }
    const auto kind = H5Tget_class(type);
    if (kind == H5T_VLEN) {
        hvl_t value{};
        std::memcpy(&value, bytes, sizeof(value));
        H5Handle element(c.id(H5Tget_super(type), path, "inspect variable attribute datatype"), H5Tclose);
        const auto width = H5Tget_size(element);
        if (width && value.len > (budget - used) / width) c.fail(ErrorCode::invalid_argument, path, "opaque variable sequence exceeds the metadata allocation budget");
        account_bytes(c, value.len * width, used, budget, path);
        if (has_vlen(c, element, path))
            for (std::size_t i = 0; i < value.len; ++i) account_vlen(c, element, static_cast<const std::byte*>(value.p) + i * width, used, budget, path);
    } else if (kind == H5T_ARRAY) {
        const auto rank = H5Tget_array_ndims(type);
        if (rank < 0) c.fail(ErrorCode::io_error, path, "inspect array attribute rank failed");
        std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
        c.check(H5Tget_array_dims2(type, dims.data()), path, "inspect array attribute shape");
        std::vector<std::size_t> shape(dims.begin(), dims.end());
        const auto count = product(c, shape, path);
        H5Handle element(c.id(H5Tget_super(type), path, "inspect array attribute datatype"), H5Tclose);
        if (has_vlen(c, element, path))
            for (std::size_t i = 0; i < count; ++i) account_vlen(c, element, bytes + i * H5Tget_size(element), used, budget, path);
    } else if (kind == H5T_COMPOUND) {
        const auto count = H5Tget_nmembers(type);
        if (count < 0) c.fail(ErrorCode::io_error, path, "inspect compound attribute datatype failed");
        for (int i = 0; i < count; ++i) {
            H5Handle member(c.id(H5Tget_member_type(type, static_cast<unsigned>(i)), path, "inspect compound attribute member"), H5Tclose);
            if (has_vlen(c, member, path)) account_vlen(c, member, bytes + H5Tget_member_offset(type, static_cast<unsigned>(i)), used, budget, path);
        }
    }
}

std::string record_field_path(std::string_view path, std::string_view field) {
    return std::string(path) + "[" + std::string(field) + "]";
}

std::vector<RecordFieldInfo> record_fields(const H5Context& c, hid_t type, std::string_view path) {
    const auto count = H5Tget_nmembers(type);
    if (count < 0) c.fail(ErrorCode::io_error, path, "inspect record fields failed");
    if (!count) c.fail(ErrorCode::unsupported_layout, path, "record datatype has no fields");
    std::vector<RecordFieldInfo> fields;
    fields.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        std::unique_ptr<char, decltype(&H5free_memory)> name(
            H5Tget_member_name(type, static_cast<unsigned>(i)), H5free_memory);
        if (!name) c.fail(ErrorCode::io_error, path, "inspect record field name failed");
        const std::string field_name(name.get());
        const auto field_path = record_field_path(path, field_name);
        if (field_name.empty()) c.fail(ErrorCode::unsupported_layout, field_path, "record field name is empty");
        for (const auto& previous : fields)
            if (previous.name == field_name) c.fail(ErrorCode::unsupported_layout, field_path, "duplicate record field name");
        H5Handle member(c.id(H5Tget_member_type(type, static_cast<unsigned>(i)), field_path, "inspect record field datatype"), H5Tclose);
        const auto kind = H5Tget_class(member);
        RecordFieldInfo result;
        result.name = field_name;
        if (kind == H5T_STRING) {
            if (!H5Tget_size(member)) c.fail(ErrorCode::unsupported_dtype, field_path, "record string field has zero width");
            result.strings = true;
        } else {
            if (kind == H5T_COMPOUND || kind == H5T_ARRAY || kind == H5T_VLEN)
                c.fail(ErrorCode::unsupported_dtype, field_path, "nested, subarray, and variable-sequence record fields are unsupported");
            result.dtype = dtype_of_h5(c, member, field_path);
            result.byte_order = byte_order(member);
        }
        fields.push_back(std::move(result));
    }
    return fields;
}

H5Handle record_member(const H5Context& c, hid_t type, std::string_view field,
                       std::string_view path) {
    const auto field_path = record_field_path(path, field);
    if (H5Tget_class(type) != H5T_COMPOUND) c.fail(ErrorCode::unsupported_encoding, field_path, "record field access requires a compound dataset");
    if (field.empty() || field.find('\0') != std::string_view::npos)
        c.fail(ErrorCode::invalid_argument, field_path, "record field name must be nonempty and NUL-free");
    const std::string name(field);
    const auto index = H5Tget_member_index(type, name.c_str());
    if (index < 0) c.fail(ErrorCode::unsupported_encoding, field_path, "record field does not exist");
    H5Handle member(c.id(H5Tget_member_type(type, static_cast<unsigned>(index)), field_path, "inspect record field datatype"), H5Tclose);
    const auto kind = H5Tget_class(member);
    if (kind == H5T_COMPOUND || kind == H5T_ARRAY || kind == H5T_VLEN)
        c.fail(ErrorCode::unsupported_dtype, field_path, "nested, subarray, and variable-sequence record fields are unsupported");
    return member;
}

H5Handle compound_field_type(const H5Context& c, std::string_view field,
                             hid_t member, std::string_view path) {
    const auto width = H5Tget_size(member);
    if (!width) c.fail(ErrorCode::unsupported_dtype, record_field_path(path, field), "record field datatype has zero width");
    H5Handle type(c.id(H5Tcreate(H5T_COMPOUND, width), record_field_path(path, field), "create partial record datatype"), H5Tclose);
    const std::string name(field);
    c.check(H5Tinsert(type, name.c_str(), 0, member), record_field_path(path, field), "insert partial record field");
    return type;
}

H5Handle numeric_record_type(const H5Context& c, std::string_view field, hid_t member,
                             DType& dtype, std::string_view path) {
    const auto field_path = record_field_path(path, field);
    if (H5Tget_class(member) == H5T_STRING) c.fail(ErrorCode::unsupported_dtype, field_path, "numeric record access requires a numeric field");
    dtype = dtype_of_h5(c, member, field_path);
    H5Handle boolean;
    hid_t scalar = native_type(dtype);
    if (dtype == DType::boolean) {
        boolean = H5Handle(c.id(H5Tcopy(member), field_path, "copy boolean record datatype"), H5Tclose);
        scalar = boolean;
    }
    return compound_field_type(c, field, scalar, path);
}

// Open each hard-link component explicitly, rejecting links that could escape
// the file or introduce recursive metadata traversal.
H5Handle open_object(const H5Context& c, hid_t file, std::string_view path, bool allow_missing = false) {
    if (path.empty() || path.front() != '/' || path.find('\0') != std::string_view::npos)
        c.fail(ErrorCode::invalid_argument, path, "storage paths must be absolute and NUL-free");
    H5Handle current(c.id(H5Gopen2(file, "/", H5P_DEFAULT), path, "open root"), H5Oclose);
    if (path == "/") return current;
    std::vector<haddr_t> ancestors;
    H5O_info_t identity{};
    c.check(H5Oget_info(current, &identity), path, "inspect root identity");
    ancestors.push_back(identity.addr);
    std::size_t first = 1;
    while (first < path.size()) {
        const auto last = path.find('/', first);
        const auto component = path.substr(first, last == std::string_view::npos ? path.size() - first : last - first);
        if (component.empty() || component == "." || component == "..") c.fail(ErrorCode::invalid_argument, path, "invalid storage path component");
        const std::string key(component);
        const auto exists = H5Lexists(current, key.c_str(), H5P_DEFAULT);
        if (exists < 0) c.fail(ErrorCode::io_error, path, "inspect link existence failed");
        if (!exists) {
            if (allow_missing) return {};
            c.fail(ErrorCode::io_error, path, "storage element does not exist");
        }
        H5L_info_t link{};
        c.check(H5Lget_info(current, key.c_str(), &link, H5P_DEFAULT), path, "inspect link");
        if (link.type != H5L_TYPE_HARD) c.fail(ErrorCode::unsupported_encoding, path, "soft and external HDF5 links are unsupported");
        H5Handle next(c.id(H5Oopen(current, key.c_str(), H5P_DEFAULT), path, "open element"), H5Oclose);
        c.check(H5Oget_info(next, &identity), path, "inspect element identity");
        if (std::find(ancestors.begin(), ancestors.end(), identity.addr) != ancestors.end()) c.fail(ErrorCode::unsupported_encoding, path, "cyclic HDF5 links are unsupported");
        ancestors.push_back(identity.addr);
        current = std::move(next);
        if (last == std::string_view::npos) break;
        if (H5Iget_type(current) != H5I_GROUP) c.fail(ErrorCode::unsupported_encoding, path, "non-group storage path parent");
        first = last + 1;
    }
    return current;
}

struct Rectangle {
    H5Handle file_space;
    H5Handle memory_space;
    std::size_t elements = 0;
};

Rectangle rectangle(const H5Context& c, hid_t dataset, const std::vector<std::size_t>& start,
                    const std::vector<std::size_t>& count, std::string_view path) {
    Rectangle r{H5Handle(c.id(H5Dget_space(dataset), path, "inspect transfer dataspace"), H5Sclose), {}, product(c, count, path)};
    if (H5Sget_simple_extent_type(r.file_space) == H5S_NULL)
        c.fail(ErrorCode::unsupported_encoding, path, "null dataspaces have no transferable values");
    const auto shape = shape_of(c, r.file_space, path);
    if (shape.size() != start.size() || shape.size() != count.size()) c.fail(ErrorCode::shape_mismatch, path, "rectangle rank does not match dataset");
    for (std::size_t i = 0; i < shape.size(); ++i)
        if (start[i] > shape[i] || count[i] > shape[i] - start[i]) c.fail(ErrorCode::out_of_bounds, path, "rectangle exceeds dataset");
    r.memory_space = space_for(c, count, path);
    if (r.elements && !shape.empty()) {
        std::vector<hsize_t> begin(start.begin(), start.end()), size(count.begin(), count.end());
        c.check(H5Sselect_hyperslab(r.file_space, H5S_SELECT_SET, begin.data(), nullptr, size.data(), nullptr), path, "select rectangle");
    }
    return r;
}

// H5Aread has no transfer-property argument, but dataset VLEN allocations can
// use a capped allocator. Track live allocations so an unsuccessful H5Dread can
// release partially acquired strings without reclaiming uninitialized entries.
struct VlenAllocator {
    struct Header { std::size_t size; Header* previous; Header* next; };
    std::size_t capacity, maximum_count;
    std::size_t used = 0, count = 0;
    Header* first = nullptr;
    bool exceeded = false;
    ~VlenAllocator() {
        while (first) { auto* next = first->next; std::free(first); first = next; }
    }
    static void* allocate(std::size_t bytes, void* context) noexcept {
        auto& a = *static_cast<VlenAllocator*>(context);
        bytes = std::max<std::size_t>(bytes, 1);
        if (bytes > a.capacity - a.used || a.count == a.maximum_count || bytes > SIZE_MAX - sizeof(Header)) {
            a.exceeded = true;
            return nullptr;
        }
        auto* header = static_cast<Header*>(std::malloc(sizeof(Header) + bytes));
        if (!header) return nullptr;
        *header = Header{bytes, nullptr, a.first};
        if (a.first) a.first->previous = header;
        a.first = header;
        a.used += bytes;
        ++a.count;
        return header + 1;
    }
    static void release(void* pointer, void* context) noexcept {
        if (!pointer) return;
        auto& a = *static_cast<VlenAllocator*>(context);
        auto* header = static_cast<Header*>(pointer) - 1;
        if (header->previous) header->previous->next = header->next;
        else a.first = header->next;
        if (header->next) header->next->previous = header->previous;
        a.used -= header->size;
        --a.count;
        std::free(header);
    }
    static std::size_t allocation_size(const char* pointer) noexcept {
        return (reinterpret_cast<const Header*>(pointer) - 1)->size;
    }
};

struct StringPlan {
    bool variable;
    std::size_t raw_capacity;
    std::size_t fixed_width;
};

struct StringBudgetExceeded {};

std::optional<StringPlan> string_plan(hid_t type, const Rectangle& r, std::size_t budget) {
    const auto n = r.elements;
    if (H5Tis_variable_str(type) > 0) {
        constexpr auto base = sizeof(std::vector<std::string>) + sizeof(std::vector<char*>) + sizeof(VlenAllocator);
        constexpr auto per_string = sizeof(std::string) + sizeof(char*) + sizeof(VlenAllocator::Header);
        if (budget < base || n > (budget - base) / per_string) return std::nullopt;
        const auto metadata = base + n * per_string;
        // Both HDF5's raw strings and the final std::string copies coexist.
        const auto raw_capacity = (budget - metadata) / 2;
        // Admit each raw allocation through the capped DXPL allocator. This
        // avoids a second traversal by H5Dvlen_get_buf_size, whose HDF5 1.10.10
        // implementation is also unsafe for scalar dataspaces. A rejected batch
        // is fully reclaimed before the caller retries a shorter prefix.
        return StringPlan{true, raw_capacity, 0};
    }
    const auto width = H5Tget_size(type);
    constexpr auto base = sizeof(std::vector<std::string>) + sizeof(std::vector<char>);
    if (!width || budget < base || width > (SIZE_MAX - sizeof(std::string) - 1) / 2) return std::nullopt;
    const auto per_string = sizeof(std::string) + 2 * width + 1;
    if (n > (budget - base) / per_string) return std::nullopt;
    return StringPlan{false, n * width, width};
}

std::vector<std::string> read_planned_strings(const H5Context& c, hid_t dataset,
        hid_t memory_type, hid_t string_type, const Rectangle& r, const StringPlan& plan,
        std::string_view path, bool validate_utf8_values = true) {
    std::vector<std::string> result;
    result.reserve(r.elements);
    if (!r.elements) return result;
    if (plan.variable) {
        std::vector<char*> pointers(r.elements, nullptr);
        VlenAllocator allocator{plan.raw_capacity, r.elements};
        H5Handle transfer(c.id(H5Pcreate(H5P_DATASET_XFER), path, "create bounded string transfer"), H5Pclose);
        c.check(H5Pset_vlen_mem_manager(transfer, VlenAllocator::allocate, &allocator,
                                      VlenAllocator::release, &allocator), path, "set bounded string allocator");
        struct DatasetReclaim {
            hid_t type, space, transfer;
            void* buffer;
            bool active = false;
            ~DatasetReclaim() { if (active) H5Dvlen_reclaim(type, space, transfer, buffer); }
        } reclaim{memory_type, r.memory_space, transfer, pointers.data()};
        herr_t status;
        H5E_BEGIN_TRY { status = H5Dread(dataset, memory_type, r.memory_space, r.file_space, transfer, pointers.data()); } H5E_END_TRY;
        if (status < 0 && allocator.exceeded) throw StringBudgetExceeded{};
        c.check(status, path, "read bounded variable strings");
        reclaim.active = true;
        for (const auto* pointer : pointers) {
            if (!pointer) { result.emplace_back(); continue; }
            const auto allocation = VlenAllocator::allocation_size(pointer);
            const auto length = strnlen(pointer, allocation);
            if (length == allocation) c.fail(ErrorCode::unsupported_dtype, path, "variable string has no terminating NUL");
            result.emplace_back(pointer, length);
        }
    } else {
        std::vector<char> bytes(plan.raw_capacity);
        c.check(H5Dread(dataset, memory_type, r.memory_space, r.file_space, H5P_DEFAULT, bytes.data()), path, "read bounded fixed strings");
        const auto pad = H5Tget_strpad(string_type);
        for (std::size_t i = 0; i < r.elements; ++i) {
            std::string_view value(bytes.data() + i * plan.fixed_width, plan.fixed_width);
            if (pad == H5T_STR_SPACEPAD) { while (!value.empty() && value.back() == ' ') value.remove_suffix(1); }
            else if (const auto end = value.find('\0'); end != std::string_view::npos) value = value.substr(0, end);
            result.emplace_back(value);
        }
    }
    if (validate_utf8_values && c.validation == Validation::full)
        for (const auto& value : result) if (!valid_utf8(value)) c.fail(ErrorCode::unsupported_dtype, path, "string data is not valid UTF-8");
    return result;
}

class Hdf5Numeric final : public NumericTransfer {
public:
    Hdf5Numeric(H5Context c, std::string path, H5Handle dataset, bool writable)
        : c_(std::move(c)), path_(std::move(path)), dataset_(std::move(dataset)), writable_(writable) {
        type_ = H5Handle(c_.id(H5Dget_type(dataset_), path_, "inspect numerical datatype"), H5Tclose);
        dtype_ = dtype_of_h5(c_, type_, path_);
        file_space_ = H5Handle(c_.id(H5Dget_space(dataset_), path_, "inspect transfer dataspace"), H5Sclose);
        if (H5Sget_simple_extent_type(file_space_) == H5S_NULL)
            c_.fail(ErrorCode::unsupported_encoding, path_, "null dataspace has no transferable values");
        shape_ = shape_of(c_, file_space_, path_);
        begin_.resize(shape_.size()); size_.resize(shape_.size());
        check_filters(c_, dataset_, path_);
    }
    void read(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, void* out) override {
        const auto n = select(start, count);
        if (!n) return;
        if (!out) c_.fail(ErrorCode::invalid_argument, path_, "null numerical destination");
        const auto memory = dtype_ == DType::boolean ? static_cast<hid_t>(type_) : native_type(dtype_);
        c_.check(H5Dread(dataset_, memory, memory_space_, file_space_, H5P_DEFAULT, out), path_, "read numerical rectangle");
        ++stats_.read_calls; stats_.read_bytes += n * dtype_size(dtype_);
    }
    void write(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, const void* in) override {
        if (!writable_) c_.fail(ErrorCode::invalid_argument, path_, "H5AD session is read-only");
        const auto n = select(start, count);
        if (!n) return;
        if (!in) c_.fail(ErrorCode::invalid_argument, path_, "null numerical source");
        if (!write_checked_) { check_filters(c_, dataset_, path_, true); write_checked_ = true; }
        const auto memory = dtype_ == DType::boolean ? static_cast<hid_t>(type_) : native_type(dtype_);
        c_.check(H5Dwrite(dataset_, memory, memory_space_, file_space_, H5P_DEFAULT, in), path_, "write numerical rectangle");
        ++stats_.write_calls; stats_.write_bytes += n * dtype_size(dtype_);
    }
    NumericTransferStats stats() const override { return stats_; }
    std::size_t scratch_capacity() const override { return 0; }
private:
    std::size_t select(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count) {
        if (start.size() != shape_.size() || count.size() != shape_.size())
            c_.fail(ErrorCode::shape_mismatch, path_, "rectangle rank does not match dataset");
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            if (start[i] > shape_[i] || count[i] > shape_[i] - start[i])
                c_.fail(ErrorCode::out_of_bounds, path_, "rectangle exceeds dataset");
            begin_[i] = start[i]; size_[i] = count[i];
        }
        const auto n = product(c_, count, path_);
        if (!n) return 0;
        if (!memory_space_.valid() || memory_shape_ != count) {
            memory_space_ = space_for(c_, count, path_); memory_shape_ = count;
        }
        if (!shape_.empty())
            c_.check(H5Sselect_hyperslab(file_space_, H5S_SELECT_SET, begin_.data(), nullptr, size_.data(), nullptr), path_, "select rectangle");
        return n;
    }
    H5Context c_;
    std::string path_;
    H5Handle dataset_, type_, file_space_, memory_space_;
    std::vector<std::size_t> shape_, memory_shape_;
    std::vector<hsize_t> begin_, size_;
    DType dtype_ = DType::float64;
    bool writable_ = false, write_checked_ = false;
    NumericTransferStats stats_;
};

class Hdf5Backend final : public Backend {
public:
    Hdf5Backend(std::string_view path, BackedOptions options, bool create = false,
                bool exclusive = false, bool gzip = false, int gzip_level = 4,
                std::optional<ByteOrder> order_override = std::nullopt)
        : c_{std::string(path), options.validation}, writable_(create || options.mode == AccessMode::read_write),
          gzip_(gzip), gzip_level_(gzip_level), order_override_(order_override), metadata_budget_(options.scratch_bytes) {
        if (path.empty() || path.find('\0') != std::string_view::npos) c_.fail(ErrorCode::invalid_argument, "/", "filename must be nonempty and NUL-free");
        if (gzip_level < 0 || gzip_level > 9) c_.fail(ErrorCode::invalid_argument, "/", "gzip level must be between 0 and 9");
        H5Handle access(c_.id(H5Pcreate(H5P_FILE_ACCESS), "/", "create file properties"), H5Pclose);
        c_.check(H5Pset_fapl_sec2(access), "/", "select local POSIX file driver");
        const auto cache_bytes = std::min<std::size_t>(options.scratch_bytes, 1024u * 1024u);
        c_.check(H5Pset_cache(access, 0, 521, cache_bytes, 0.75), "/", "bound raw chunk cache");
        hid_t file;
        H5E_BEGIN_TRY {
            file = create ? H5Fcreate(c_.filename.c_str(), exclusive ? H5F_ACC_EXCL : H5F_ACC_TRUNC, H5P_DEFAULT, access)
                          : H5Fopen(c_.filename.c_str(), writable_ ? H5F_ACC_RDWR : H5F_ACC_RDONLY, access);
        } H5E_END_TRY;
        file_ = H5Handle(c_.id(file, "/", create ? "create H5AD file" : "open H5AD file"), H5Fclose);
        void* handle = nullptr;
        c_.check(H5Fget_vfd_handle(file_, H5P_DEFAULT, &handle), "/", "inspect store identity descriptor");
        if (!handle) c_.fail(ErrorCode::io_error, "/", "store identity descriptor is unavailable");
        struct stat info{};
        if (fstat(*static_cast<int*>(handle), &info) < 0) c_.fail(ErrorCode::io_error, "/", "inspect store identity failed");
        identity_ = "hdf5:" + std::to_string(static_cast<std::uintmax_t>(info.st_dev)) + ":" +
                    std::to_string(static_cast<std::uintmax_t>(info.st_ino));
    }

    std::string location() const override { return c_.filename; }
    std::string identity() const override { return identity_; }
    std::unique_ptr<Backend> fork(std::size_t scratch) override {
        ensure_open();
        // HDF5 cannot reopen an existing writable file with a read-only access
        // flag in the same process; preserve the original file access flags.
        BackedOptions options; options.mode = writable_ ? AccessMode::read_write : AccessMode::read_only;
        options.validation = c_.validation; options.scratch_bytes = scratch;
        return std::make_unique<Hdf5Backend>(c_.filename, options);
    }
    std::unique_ptr<NumericTransfer> prepare_numeric(std::string_view path) override {
        ensure_open();
        return std::make_unique<Hdf5Numeric>(c_, std::string(path), numeric_dataset(path), writable_);
    }
    bool same_file(std::string_view path) {
        ensure_open();
        if (path.empty() || path.find('\0') != std::string_view::npos) c_.fail(ErrorCode::invalid_argument, "/", "filename must be nonempty and NUL-free");
        struct stat destination{};
        if (stat(std::string(path).c_str(), &destination) < 0) {
            if (errno == ENOENT || errno == ENOTDIR) return false;
            c_.fail(ErrorCode::io_error, "/", "inspect output file identity failed");
        }
        void* handle = nullptr;
        c_.check(H5Fget_vfd_handle(file_, H5P_DEFAULT, &handle), "/", "inspect source file descriptor");
        if (!handle) c_.fail(ErrorCode::io_error, "/", "source POSIX file descriptor is unavailable");
        struct stat source{};
        if (fstat(*static_cast<int*>(handle), &source) < 0) c_.fail(ErrorCode::io_error, "/", "inspect source file identity failed");
        return source.st_dev == destination.st_dev && source.st_ino == destination.st_ino;
    }
    NodeInfo inspect(std::string_view path) override {
        ensure_open();
        auto object = open_object(c_, file_, path, true);
        if (!object.valid()) return {};
        NodeInfo result;
        result.exists = true;
        hbool_t threadsafe = 0;
        c_.check(H5is_library_threadsafe(&threadsafe), path, "inspect HDF5 threading capability");
        result.parallel_reads = threadsafe != 0;
        result.group = H5Iget_type(object) == H5I_GROUP;
        result.attrs = read_attributes(c_, object, path, result.has_opaque_attrs);
        const auto encoding = result.attrs.find("encoding-type");
        const bool null_encoding = encoding != result.attrs.end() && std::get_if<std::string>(&encoding->second) && std::get<std::string>(encoding->second) == "null";
        if (result.group && null_encoding) c_.fail(ErrorCode::unsupported_encoding, path, "null encoding requires an HDF5 dataset with a null dataspace");
        if (!result.group) {
            if (H5Iget_type(object) != H5I_DATASET) c_.fail(ErrorCode::unsupported_encoding, path, "only groups and datasets are supported");
            H5Handle space(c_.id(H5Dget_space(object), path, "inspect dataspace"), H5Sclose);
            result.shape = shape_of(c_, space, path);
            result.physical.shape = result.shape;
            H5Handle creation(c_.id(H5Dget_create_plist(object), path, "inspect dataset layout"), H5Pclose);
            const auto layout = H5Pget_layout(creation);
            result.physical.organization = layout == H5D_CHUNKED ? StorageOrganization::chunked :
                layout == H5D_CONTIGUOUS || layout == H5D_COMPACT ? StorageOrganization::contiguous : StorageOrganization::unknown;
            if (layout == H5D_CHUNKED) {
                std::vector<hsize_t> chunks(result.shape.size());
                if (H5Pget_chunk(creation, int(chunks.size()), chunks.data()) < 0)
                    c_.fail(ErrorCode::io_error, path, "inspect HDF5 chunks failed");
                result.physical.chunks.assign(chunks.begin(), chunks.end());
                result.physical.inner_chunks = result.physical.chunks;
                result.codec_tile_shape = result.physical.chunks;
            }
            result.physical.write_unit = result.physical.chunks.empty() ? result.shape : result.physical.chunks;
            if (null_encoding && H5Sget_simple_extent_type(space) != H5S_NULL) c_.fail(ErrorCode::unsupported_encoding, path, "null encoding requires an HDF5 null dataspace");
            if (!null_encoding && H5Sget_simple_extent_type(space) == H5S_NULL) c_.fail(ErrorCode::unsupported_encoding, path, "non-null encoding cannot use an HDF5 null dataspace");
            if (H5Sget_simple_extent_type(space) != H5S_NULL) {
                H5Handle type(c_.id(H5Dget_type(object), path, "inspect datatype"), H5Tclose);
                const auto kind = H5Tget_class(type);
                result.strings = kind == H5T_STRING;
                result.record = kind == H5T_COMPOUND;
                if (result.record) result.fields = record_fields(c_, type, path);
                else if (!result.strings) { result.dtype = dtype_of_h5(c_, type, path); result.byte_order = byte_order(type); }
            }
        }
        return result;
    }

    std::vector<std::string> children(std::string_view path) override {
        ensure_open();
        auto object = open_object(c_, file_, path);
        if (H5Iget_type(object) != H5I_GROUP) c_.fail(ErrorCode::unsupported_encoding, path, "children require a group");
        H5G_info_t info{};
        c_.check(H5Gget_info(object, &info), path, "inspect group members");
        std::vector<std::string> result;
        for (hsize_t i = 0; i < info.nlinks; ++i) {
            const auto n = H5Lget_name_by_idx(object, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
            if (n < 0) c_.fail(ErrorCode::io_error, path, "inspect member name failed");
            std::string name(static_cast<std::size_t>(n) + 1, '\0');
            if (H5Lget_name_by_idx(object, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), name.size(), H5P_DEFAULT) < 0) c_.fail(ErrorCode::io_error, path, "read member name failed");
            name.resize(static_cast<std::size_t>(n));
            result.push_back(std::move(name));
        }
        return result;
    }

    void read_numeric(std::string_view path, const std::vector<std::size_t>& start,
                      const std::vector<std::size_t>& count, void* destination) override {
        ensure_open();
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), path, "inspect numeric datatype"), H5Tclose);
        const auto dtype = dtype_of_h5(c_, type, path);
        auto r = rectangle(c_, dataset, start, count, path);
        if (!r.elements) return;
        if (!destination) c_.fail(ErrorCode::invalid_argument, path, "null numerical destination");
        check_filters(c_, dataset, path);
        const hid_t memory_type = dtype == DType::boolean ? static_cast<hid_t>(type) : native_type(dtype);
        c_.check(H5Dread(dataset, memory_type, r.memory_space, r.file_space, H5P_DEFAULT, destination), path, "read numerical rectangle");
        // Preserve enum bytes here. The shared transfer layer canonicalizes
        // full-validation boolean selections exactly once before typed access,
        // including operation-local validation of an otherwise trusted session.
    }

    void write_numeric(std::string_view path, const std::vector<std::size_t>& start,
                       const std::vector<std::size_t>& count, const void* source) override {
        ensure_writable();
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), path, "inspect numeric datatype"), H5Tclose);
        const auto dtype = dtype_of_h5(c_, type, path);
        auto r = rectangle(c_, dataset, start, count, path);
        if (!r.elements) return;
        if (!source) c_.fail(ErrorCode::invalid_argument, path, "null numerical source");
        check_filters(c_, dataset, path, true);
        const hid_t memory_type = dtype == DType::boolean ? static_cast<hid_t>(type) : native_type(dtype);
        c_.check(H5Dwrite(dataset, memory_type, r.memory_space, r.file_space, H5P_DEFAULT, source), path, "write numerical rectangle");
    }

    void read_record_numeric(std::string_view path, std::string_view field,
            const std::vector<std::size_t>& start, const std::vector<std::size_t>& count,
            void* destination) override {
        ensure_open();
        const auto field_path = record_field_path(path, field);
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), field_path, "inspect record datatype"), H5Tclose);
        auto member = record_member(c_, type, field, path);
        DType dtype;
        auto memory = numeric_record_type(c_, field, member, dtype, path);
        auto r = rectangle(c_, dataset, start, count, field_path);
        if (!r.elements) return;
        if (!destination) c_.fail(ErrorCode::invalid_argument, field_path, "null numeric record destination");
        check_filters(c_, dataset, field_path);
        c_.check(H5Dread(dataset, memory, r.memory_space, r.file_space, H5P_DEFAULT, destination), field_path, "read numeric record field");
    }

    void write_record_numeric(std::string_view path, std::string_view field,
            const std::vector<std::size_t>& start, const std::vector<std::size_t>& count,
            const void* source) override {
        ensure_writable();
        const auto field_path = record_field_path(path, field);
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), field_path, "inspect record datatype"), H5Tclose);
        auto member = record_member(c_, type, field, path);
        DType dtype;
        auto memory = numeric_record_type(c_, field, member, dtype, path);
        auto r = rectangle(c_, dataset, start, count, field_path);
        if (!r.elements) return;
        if (!source) c_.fail(ErrorCode::invalid_argument, field_path, "null numeric record source");
        check_filters(c_, dataset, field_path, true);
        H5Handle transfer(c_.id(H5Pcreate(H5P_DATASET_XFER), field_path, "create record transfer properties"), H5Pclose);
        c_.check(H5Pset_preserve(transfer, 1), field_path, "preserve other record fields");
        c_.check(H5Dwrite(dataset, memory, r.memory_space, r.file_space, transfer, source), field_path, "write numeric record field");
    }

    std::vector<std::string> read_strings(std::string_view path, const std::vector<std::size_t>& start,
                                          const std::vector<std::size_t>& count) override {
        ensure_open();
        auto dataset = string_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), path, "inspect string datatype"), H5Tclose);
        auto r = rectangle(c_, dataset, start, count, path);
        check_filters(c_, dataset, path);
        if (!r.elements) return {};
        const auto plan = string_plan(type, r, metadata_budget_);
        if (!plan) c_.fail(ErrorCode::invalid_argument, path, "string rectangle exceeds the decoded staging budget");
        try { return read_planned_strings(c_, dataset, type, type, r, *plan, path); }
        catch (const StringBudgetExceeded&) { c_.fail(ErrorCode::invalid_argument, path, "string rectangle exceeds the decoded staging budget"); }
    }

    std::vector<std::string> read_strings_bounded(std::string_view path,
            const std::vector<std::size_t>& start, const std::vector<std::size_t>& count,
            std::size_t byte_budget) override {
        ensure_open();
        auto dataset = string_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), path, "inspect string datatype"), H5Tclose);
        auto requested = rectangle(c_, dataset, start, count, path);
        if (!requested.elements) return {};
        for (std::size_t i = 0; i + 1 < count.size(); ++i)
            if (count[i] != 1) c_.fail(ErrorCode::shape_mismatch, path, "bounded string reads require a consecutive last-axis run");
        check_filters(c_, dataset, path);
        const auto budget = std::min(byte_budget, metadata_budget_);
        auto limited = count;
        while (true) {
            auto r = rectangle(c_, dataset, start, limited, path);
            if (const auto plan = string_plan(type, r, budget)) {
                try { return read_planned_strings(c_, dataset, type, type, r, *plan, path); }
                catch (const StringBudgetExceeded&) { /* retry after all staging was reclaimed */ }
            }
            if (limited.empty() || limited.back() == 1)
                c_.fail(ErrorCode::invalid_argument, path, "one string exceeds the decoded staging budget");
            limited.back() = (limited.back() + 1) / 2;
        }
    }

    std::vector<std::string> read_record_strings_bounded(std::string_view path,
            std::string_view field, const std::vector<std::size_t>& start,
            const std::vector<std::size_t>& count, std::size_t byte_budget) override {
        ensure_open();
        const auto field_path = record_field_path(path, field);
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), field_path, "inspect record datatype"), H5Tclose);
        auto member = record_member(c_, type, field, path);
        if (H5Tget_class(member) != H5T_STRING)
            c_.fail(ErrorCode::unsupported_dtype, field_path, "string record access requires a string field");
        const auto variable = H5Tis_variable_str(member);
        if (variable < 0) c_.fail(ErrorCode::io_error, field_path, "inspect record string width failed");
        auto memory = compound_field_type(c_, field, member, path);
        auto requested = rectangle(c_, dataset, start, count, field_path);
        if (!requested.elements) return {};
        for (std::size_t i = 0; i + 1 < count.size(); ++i)
            if (count[i] != 1) c_.fail(ErrorCode::shape_mismatch, field_path, "bounded string reads require a consecutive last-axis run");
        check_filters(c_, dataset, field_path);
        const auto budget = std::min(byte_budget, metadata_budget_);
        auto limited = count;
        while (true) {
            auto r = rectangle(c_, dataset, start, limited, field_path);
            if (const auto plan = string_plan(member, r, budget)) {
                try { return read_planned_strings(c_, dataset, memory, member, r, *plan, field_path, false); }
                catch (const StringBudgetExceeded&) { /* partial reads are reclaimed before retry */ }
            }
            if (limited.empty() || limited.back() == 1)
                c_.fail(ErrorCode::invalid_argument, field_path, "one string exceeds the decoded staging budget");
            limited.back() = (limited.back() + 1) / 2;
        }
    }

    void write_strings(std::string_view path, const std::vector<std::size_t>& start,
                       const std::vector<std::size_t>& count, const std::vector<std::string>& values) override {
        ensure_writable();
        auto dataset = string_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), path, "inspect string datatype"), H5Tclose);
        auto r = rectangle(c_, dataset, start, count, path);
        if (r.elements != values.size()) c_.fail(ErrorCode::shape_mismatch, path, "string value count differs from rectangle");
        if (!r.elements) return;
        check_filters(c_, dataset, path, true);
        for (const auto& s : values) if (s.find('\0') != std::string::npos) c_.fail(ErrorCode::invalid_argument, path, "embedded NUL in string data");
        if (H5Tis_variable_str(type) > 0) {
            std::vector<const char*> pointers;
            for (const auto& s : values) pointers.push_back(s.c_str());
            c_.check(H5Dwrite(dataset, type, r.memory_space, r.file_space, H5P_DEFAULT, pointers.data()), path, "write variable strings");
        } else {
            const auto width = H5Tget_size(type);
            if (width && values.size() > static_cast<std::size_t>(PTRDIFF_MAX) / width) c_.fail(ErrorCode::invalid_argument, path, "fixed string buffer exceeds addressable range");
            const bool space_pad = H5Tget_strpad(type) == H5T_STR_SPACEPAD;
            std::vector<char> bytes(values.size() * width, space_pad ? ' ' : '\0');
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (values[i].size() > width || (H5Tget_strpad(type) == H5T_STR_NULLTERM && values[i].size() == width)) c_.fail(ErrorCode::shape_mismatch, path, "string exceeds existing fixed-width datatype");
                std::memcpy(bytes.data() + i * width, values[i].data(), values[i].size());
            }
            c_.check(H5Dwrite(dataset, type, r.memory_space, r.file_space, H5P_DEFAULT, bytes.data()), path, "write fixed strings");
        }
    }

    void write_record_strings(std::string_view path, std::string_view field,
            const std::vector<std::size_t>& start, const std::vector<std::size_t>& count,
            const std::vector<std::string>& values) override {
        ensure_writable();
        const auto field_path = record_field_path(path, field);
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), field_path, "inspect record datatype"), H5Tclose);
        auto member = record_member(c_, type, field, path);
        if (H5Tget_class(member) != H5T_STRING)
            c_.fail(ErrorCode::unsupported_dtype, field_path, "string record access requires a string field");
        auto r = rectangle(c_, dataset, start, count, field_path);
        if (r.elements != values.size()) c_.fail(ErrorCode::shape_mismatch, field_path, "string value count differs from record rectangle");
        if (!r.elements) return;
        check_filters(c_, dataset, field_path, true);
        const auto is_variable = H5Tis_variable_str(member);
        if (is_variable < 0) c_.fail(ErrorCode::io_error, field_path, "inspect record string width failed");
        const bool variable = is_variable > 0;
        const auto width = H5Tget_size(member);
        const auto pad = H5Tget_strpad(member);
        std::vector<const char*> pointers;
        if (variable) pointers.reserve(values.size());
        for (const auto& value : values) {
            if (value.find('\0') != std::string::npos) c_.fail(ErrorCode::invalid_argument, field_path, "embedded NUL in string data");
            if (!variable && (value.size() > width || (pad == H5T_STR_NULLTERM && value.size() == width)))
                c_.fail(ErrorCode::shape_mismatch, field_path, "string exceeds existing fixed-width datatype");
            if (variable) pointers.push_back(value.c_str());
        }
        auto memory = compound_field_type(c_, field, member, path);
        H5Handle transfer(c_.id(H5Pcreate(H5P_DATASET_XFER), field_path, "create record transfer properties"), H5Pclose);
        c_.check(H5Pset_preserve(transfer, 1), field_path, "preserve other record fields");
        if (variable) c_.check(H5Dwrite(dataset, memory, r.memory_space, r.file_space, transfer, pointers.data()), field_path, "write variable string record field");
        else {
            if (width && values.size() > static_cast<std::size_t>(PTRDIFF_MAX) / width)
                c_.fail(ErrorCode::invalid_argument, field_path, "fixed string buffer exceeds addressable range");
            std::vector<char> bytes(values.size() * width, pad == H5T_STR_SPACEPAD ? ' ' : '\0');
            for (std::size_t i = 0; i < values.size(); ++i)
                std::memcpy(bytes.data() + i * width, values[i].data(), values[i].size());
            c_.check(H5Dwrite(dataset, memory, r.memory_space, r.file_space, transfer, bytes.data()), field_path, "write fixed string record field");
        }
    }

    void create_group(std::string_view path, const Attributes& attrs) override {
        ensure_writable();
        if (path == "/") { attributes(path, attrs); return; }
        auto [parent, name] = parent_for(path);
        auto links = link_properties(path);
        H5Handle group(c_.id(H5Gcreate2(parent, name.c_str(), links, H5P_DEFAULT, H5P_DEFAULT), path, "create group"), H5Gclose);
        write_attributes(c_, group, attrs, path);
    }

    void create_numeric(std::string_view path, DType dtype, const std::vector<std::size_t>& shape,
                        const Attributes& attrs, ByteOrder order) override {
        ensure_writable();
        const auto encoding = attrs.find("encoding-type");
        if (encoding != attrs.end() && std::get_if<std::string>(&encoding->second) && std::get<std::string>(encoding->second) == "null") {
            create_null(path, attrs);
            return;
        }
        auto [parent, name] = parent_for(path);
        auto space = space_for(c_, shape, path);
        auto properties = dataset_properties(shape, dtype_size(dtype), path);
        auto links = link_properties(path);
        H5Handle boolean;
        const hid_t type = dtype == DType::boolean ? static_cast<hid_t>(boolean = boolean_type(c_, path)) : file_type(dtype, order_override_.value_or(order));
        H5Handle dataset(c_.id(H5Dcreate2(parent, name.c_str(), type, space, links, properties, H5P_DEFAULT), path, "create numeric dataset"), H5Dclose);
        write_attributes(c_, dataset, attrs, path);
    }

    void create_strings(std::string_view path, const std::vector<std::size_t>& shape, const Attributes& attrs) override {
        ensure_writable();
        auto [parent, name] = parent_for(path);
        auto space = space_for(c_, shape, path);
        auto type = string_type(c_, path);
        auto properties = dataset_properties(shape, sizeof(char*), path);
        auto links = link_properties(path);
        H5Handle dataset(c_.id(H5Dcreate2(parent, name.c_str(), type, space, links, properties, H5P_DEFAULT), path, "create string dataset"), H5Dclose);
        write_attributes(c_, dataset, attrs, path);
    }

    void create_record(std::string_view path, const std::vector<std::size_t>& shape,
            const std::vector<RecordFieldInfo>& fields, const Attributes& attrs) override {
        ensure_writable();
        if (fields.empty()) c_.fail(ErrorCode::unsupported_layout, path, "record datatype requires at least one field");
        std::vector<H5Handle> member_types;
        std::vector<std::size_t> offsets;
        member_types.reserve(fields.size());
        offsets.reserve(fields.size());
        std::size_t width = 0;
        for (const auto& field : fields) {
            const auto field_path = record_field_path(path, field.name);
            if (field.name.empty() || field.name.find('\0') != std::string::npos)
                c_.fail(ErrorCode::invalid_argument, field_path, "record field name must be nonempty and NUL-free");
            for (const auto& previous : fields) {
                if (&previous == &field) break;
                if (previous.name == field.name) c_.fail(ErrorCode::invalid_argument, field_path, "duplicate record field name");
            }
            H5Handle member;
            if (field.strings) member = string_type(c_, field_path);
            else if (field.dtype == DType::boolean) member = boolean_type(c_, field_path);
            else {
                const auto file = file_type(field.dtype, order_override_.value_or(field.byte_order));
                if (file < 0) c_.fail(ErrorCode::unsupported_dtype, field_path, "unsupported numeric record datatype");
                member = H5Handle(c_.id(H5Tcopy(file), field_path, "copy numeric record datatype"), H5Tclose);
            }
            const auto member_width = H5Tget_size(member);
            if (!member_width || member_width > SIZE_MAX - width)
                c_.fail(ErrorCode::unsupported_layout, field_path, "record datatype exceeds addressable size");
            offsets.push_back(width);
            width += member_width;
            member_types.push_back(std::move(member));
        }
        H5Handle record(c_.id(H5Tcreate(H5T_COMPOUND, width), path, "create record datatype"), H5Tclose);
        for (std::size_t i = 0; i < fields.size(); ++i)
            c_.check(H5Tinsert(record, fields[i].name.c_str(), offsets[i], member_types[i]), record_field_path(path, fields[i].name), "insert record field");
        auto [parent, name] = parent_for(path);
        auto space = space_for(c_, shape, path);
        auto properties = dataset_properties(shape, width, path);
        auto links = link_properties(path);
        H5Handle dataset(c_.id(H5Dcreate2(parent, name.c_str(), record, space, links, properties, H5P_DEFAULT), path, "create record dataset"), H5Dclose);
        write_attributes(c_, dataset, attrs, path);
    }

    void create_null(std::string_view path, const Attributes& attrs) {
        ensure_writable();
        auto [parent, name] = parent_for(path);
        H5Handle space(c_.id(H5Screate(H5S_NULL), path, "create null dataspace"), H5Sclose);
        auto links = link_properties(path);
        H5Handle dataset(c_.id(H5Dcreate2(parent, name.c_str(), H5T_IEEE_F64LE, space, links, H5P_DEFAULT, H5P_DEFAULT), path, "create null dataset"), H5Dclose);
        write_attributes(c_, dataset, attrs, path);
    }

    void attributes(std::string_view path, const Attributes& attrs) override {
        ensure_writable();
        auto object = open_object(c_, file_, path);
        write_attributes(c_, object, attrs, path);
    }
    void copy_attributes(std::string_view source, std::string_view target) override {
        ensure_writable();
        auto original = open_object(c_, file_, source);
        auto replacement = open_object(c_, file_, target);
        H5O_info_t info{};
        c_.check(H5Oget_info(original, &info), source, "inspect source attributes");
        for (hsize_t i = 0; i < info.num_attrs; ++i) {
            H5Handle attr(c_.id(H5Aopen_by_idx(original, ".", H5_INDEX_NAME, H5_ITER_INC, i, H5P_DEFAULT, H5P_DEFAULT), source, "open preserved attribute"), H5Aclose);
            const auto length = H5Aget_name(attr, 0, nullptr);
            if (length < 0) c_.fail(ErrorCode::io_error, source, "inspect preserved attribute name failed");
            if (static_cast<std::size_t>(length) >= metadata_budget_) c_.fail(ErrorCode::invalid_argument, source, "attribute name exceeds metadata allocation budget");
            std::string key(static_cast<std::size_t>(length) + 1, '\0');
            if (H5Aget_name(attr, key.size(), key.data()) < 0) c_.fail(ErrorCode::io_error, source, "read preserved attribute name failed");
            key.resize(static_cast<std::size_t>(length));
            const auto exists = H5Aexists(replacement, key.c_str());
            if (exists < 0) c_.fail(ErrorCode::io_error, target, "inspect replacement attribute failed");
            if (exists) continue; // newly encoded structural metadata wins
            const auto attribute_path = std::string(source) + "@" + key;
            H5Handle type(c_.id(H5Aget_type(attr), attribute_path, "inspect preserved datatype"), H5Tclose);
            const auto reference = H5Tdetect_class(type, H5T_REFERENCE);
            if (reference < 0) c_.fail(ErrorCode::io_error, attribute_path, "inspect attribute references failed");
            if (reference) c_.fail(ErrorCode::unsupported_dtype, attribute_path, "reference attributes cannot be preserved across object replacement");
            H5Handle space(c_.id(H5Aget_space(attr), attribute_path, "inspect preserved dataspace"), H5Sclose);
            const auto points = H5Sget_simple_extent_npoints(space);
            if (points < 0) c_.fail(ErrorCode::io_error, attribute_path, "inspect preserved attribute size failed");
            const auto width = H5Tget_size(type);
            if (!width || static_cast<std::uintmax_t>(points) > metadata_budget_ / width) c_.fail(ErrorCode::invalid_argument, attribute_path, "opaque attribute exceeds metadata allocation budget");
            const auto bytes = static_cast<std::size_t>(points) * width;
            if (bytes > metadata_budget_ - key.size() - 1) c_.fail(ErrorCode::invalid_argument, attribute_path, "opaque attribute exceeds metadata allocation budget");
            // Zero initialization also makes partial VLEN read reclamation safe.
            std::vector<std::byte> buffer(bytes);
            const bool variable = has_vlen(c_, type, attribute_path);
            struct AttributeReclaim {
                hid_t type, space;
                void* bytes;
                bool active;
                ~AttributeReclaim() { if (active) H5Dvlen_reclaim(type, space, H5P_DEFAULT, bytes); }
            } reclaim{type, space, buffer.data(), variable && points > 0};
            if (points) c_.check(H5Aread(attr, type, buffer.data()), attribute_path, "read preserved attribute");
            std::size_t used = bytes + key.size() + 1;
            if (variable)
                for (hssize_t point = 0; point < points; ++point) account_vlen(c_, type, buffer.data() + static_cast<std::size_t>(point) * width, used, metadata_budget_, attribute_path);
            H5Handle properties(c_.id(H5Aget_create_plist(attr), attribute_path, "inspect preserved attribute properties"), H5Pclose);
            H5Handle copied(c_.id(H5Acreate2(replacement, key.c_str(), type, space, properties, H5P_DEFAULT), attribute_path, "create preserved attribute"), H5Aclose);
            if (points) c_.check(H5Awrite(copied, type, buffer.data()), attribute_path, "write preserved attribute");
        }
    }
    void erase(std::string_view path) override {
        ensure_writable();
        auto [parent, name] = parent_for(path);
        c_.check(H5Ldelete(parent, name.c_str(), H5P_DEFAULT), path, "erase storage link");
    }
    void rename(std::string_view source, std::string_view target) override {
        ensure_writable();
        auto [old_parent, old_name] = parent_for(source);
        auto [new_parent, new_name] = parent_for(target);
        auto links = link_properties(target);
        c_.check(H5Lmove(old_parent, old_name.c_str(), new_parent, new_name.c_str(), links, H5P_DEFAULT), target, "rename storage link");
    }
    void flush() override { ensure_open(); c_.check(H5Fflush(file_, H5F_SCOPE_LOCAL), "/", "flush H5AD file"); }
    void close() override { file_.reset(); }

    std::optional<MappedArray> map_numeric(std::string_view path) override {
        ensure_open();
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), path, "inspect mapped datatype"), H5Tclose);
        if (H5Tget_class(type) == H5T_STRING || H5Tis_variable_str(type) > 0) return std::nullopt;
        const auto dtype = dtype_of_h5(c_, type, path);
        if (dtype != DType::boolean && H5Tequal(type, native_type(dtype)) <= 0) return std::nullopt;
        H5Handle space(c_.id(H5Dget_space(dataset), path, "inspect mapped dataspace"), H5Sclose);
        if (H5Sget_simple_extent_type(space) == H5S_NULL) return std::nullopt;
        auto shape = shape_of(c_, space, path);
        const auto elements = product(c_, shape, path);
        const auto width = dtype_size(dtype);
        if (!elements || elements > static_cast<std::size_t>(PTRDIFF_MAX) / width) return std::nullopt;
        const auto bytes = elements * width;
        H5Handle properties(c_.id(H5Dget_create_plist(dataset), path, "inspect mapped layout"), H5Pclose);
        if (H5Pget_layout(properties) != H5D_CONTIGUOUS || H5Pget_nfilters(properties) != 0 || H5Pget_external_count(properties) != 0) return std::nullopt;
        H5D_space_status_t status;
        c_.check(H5Dget_space_status(dataset, &status), path, "inspect mapped allocation");
        if (status != H5D_SPACE_STATUS_ALLOCATED) return std::nullopt;
        const auto offset = H5Dget_offset(dataset);
        if (offset == HADDR_UNDEF || offset % width != 0 || H5Dget_storage_size(dataset) != bytes) return std::nullopt;
        H5Handle access(c_.id(H5Fget_access_plist(file_), path, "inspect mapping driver"), H5Pclose);
        if (H5Pget_driver(access) != H5FD_SEC2) return std::nullopt;
        if (writable_) flush();
        void* handle = nullptr;
        c_.check(H5Fget_vfd_handle(file_, H5P_DEFAULT, &handle), path, "get POSIX file descriptor");
        if (!handle) return std::nullopt;
        // The SEC2 handle is an int*, not an encoded descriptor value. Duplicate
        // the actual open file; reopening the pathname would race replacement.
        const int descriptor = fcntl(*static_cast<int*>(handle), F_DUPFD_CLOEXEC, 0);
        if (descriptor < 0) c_.fail(ErrorCode::io_error, path, "duplicate mapping descriptor failed");
        struct Mapping {
            int descriptor;
            void* address = MAP_FAILED;
            std::size_t length = 0;
            explicit Mapping(int fd) : descriptor(fd) {}
            ~Mapping() { if (address != MAP_FAILED) munmap(address, length); if (descriptor >= 0) ::close(descriptor); }
        };
        auto mapping = std::make_shared<Mapping>(descriptor);
        struct stat info{};
        if (fstat(descriptor, &info) < 0) c_.fail(ErrorCode::io_error, path, "inspect mapping file failed");
        if (!S_ISREG(info.st_mode) || info.st_size < 0 || offset > static_cast<std::uintmax_t>(info.st_size) || bytes > static_cast<std::uintmax_t>(info.st_size) - offset) return std::nullopt;
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) c_.fail(ErrorCode::io_error, path, "inspect page size failed");
        const auto base = offset - offset % static_cast<haddr_t>(page);
        const auto delta = static_cast<std::size_t>(offset - base);
        if (base > static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max()) || bytes > static_cast<std::size_t>(PTRDIFF_MAX) - delta) return std::nullopt;
        mapping->length = bytes + delta;
        mapping->address = mmap(nullptr, mapping->length, PROT_READ, MAP_SHARED, descriptor, static_cast<off_t>(base));
        if (mapping->address == MAP_FAILED) c_.fail(ErrorCode::io_error, path, "read-only file mapping failed");
        const auto* data = static_cast<const std::byte*>(mapping->address) + delta;
        // A mapping cannot repair malformed bool bytes. Reject that case under
        // full validation; ordinary reads canonicalize into caller storage.
        if (dtype == DType::boolean && c_.validation == Validation::full)
            for (std::size_t i = 0; i < elements; ++i) if (data[i] != std::byte{0} && data[i] != std::byte{1}) return std::nullopt;
        return MappedArray(std::move(mapping), data, bytes, dtype, std::move(shape));
    }

private:
    H5Context c_;
    H5Handle file_;
    std::string identity_;
    bool writable_, gzip_;
    int gzip_level_;
    std::optional<ByteOrder> order_override_;
    std::size_t metadata_budget_;
    void ensure_open() const { if (!file_.valid()) c_.fail(ErrorCode::io_error, "/", "H5AD session is closed"); }
    void ensure_writable() const { ensure_open(); if (!writable_) c_.fail(ErrorCode::invalid_argument, "/", "H5AD session is read-only"); }
    H5Handle numeric_dataset(std::string_view path) {
        auto dataset = open_object(c_, file_, path);
        if (H5Iget_type(dataset) != H5I_DATASET) c_.fail(ErrorCode::unsupported_encoding, path, "numerical access requires dataset");
        return dataset;
    }
    H5Handle string_dataset(std::string_view path) {
        auto dataset = numeric_dataset(path);
        H5Handle type(c_.id(H5Dget_type(dataset), path, "inspect string datatype"), H5Tclose);
        if (H5Tget_class(type) != H5T_STRING) c_.fail(ErrorCode::unsupported_dtype, path, "string access requires string datatype");
        return dataset;
    }
    std::pair<H5Handle, std::string> parent_for(std::string_view path) {
        const auto slash = path.rfind('/');
        if (slash == std::string_view::npos || path == "/" || slash + 1 == path.size()) c_.fail(ErrorCode::invalid_argument, path, "element path requires a final name");
        const auto name = path.substr(slash + 1);
        if (name == "." || name == ".." || name.find('\0') != std::string_view::npos) c_.fail(ErrorCode::invalid_argument, path, "invalid element name");
        auto parent = open_object(c_, file_, slash == 0 ? std::string_view("/") : path.substr(0, slash));
        if (H5Iget_type(parent) != H5I_GROUP) c_.fail(ErrorCode::unsupported_encoding, path, "storage parent must be a group");
        return {std::move(parent), std::string(name)};
    }
    H5Handle link_properties(std::string_view path) {
        H5Handle properties(c_.id(H5Pcreate(H5P_LINK_CREATE), path, "create link properties"), H5Pclose);
        c_.check(H5Pset_char_encoding(properties, H5T_CSET_UTF8), path, "set UTF-8 link names");
        return properties;
    }
    H5Handle dataset_properties(const std::vector<std::size_t>& shape, std::size_t width, std::string_view path) {
        H5Handle properties(c_.id(H5Pcreate(H5P_DATASET_CREATE), path, "create dataset properties"), H5Pclose);
        if (!gzip_ || shape.empty() || !product(c_, shape, path)) return properties;
        unsigned config = 0;
        if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0 || H5Zget_filter_info(H5Z_FILTER_DEFLATE, &config) < 0 || !(config & H5Z_FILTER_CONFIG_ENCODE_ENABLED)) c_.fail(ErrorCode::unavailable_filter, path, "HDF5 gzip encoder unavailable");
        std::vector<hsize_t> chunks(shape.begin(), shape.end());
        const auto target = std::max<std::size_t>(1, 1048576u / std::max<std::size_t>(1, width));
        auto count = [&] { std::size_t n = 1; for (auto d : chunks) { if (d > target || n > target / d) return target + 1; n *= d; } return n; };
        while (count() > target) { auto it = std::max_element(chunks.begin(), chunks.end()); *it = (*it + 1) / 2; }
        c_.check(H5Pset_chunk(properties, static_cast<int>(chunks.size()), chunks.data()), path, "set compressed chunk shape");
        c_.check(H5Pset_deflate(properties, static_cast<unsigned>(gzip_level_)), path, "set gzip compression");
        return properties;
    }
};

} // namespace
} // namespace anndatapp::backed_detail

namespace anndatapp::detail {
BackedAnnData open_h5ad_backed(std::string_view path, BackedOptions options) {
    return backed_detail::make_store(std::make_unique<backed_detail::Hdf5Backend>(path, options), options);
}

void write_h5ad_backed(std::string_view path, const BackedAnnDataView& source, bool exclusive,
                      bool gzip, int gzip_level, std::optional<ByteOrder> byte_order, Validation validation) {
    // Check session lifetime before opening a truncating output destination.
    (void)source.n_obs();
    const auto& retained = detail::BackedAccess::get(source);
    auto* backend = retained->session->backend.get();
    bool same = false;
    if (auto* hdf5 = dynamic_cast<backed_detail::Hdf5Backend*>(backend)) same = hdf5->same_file(path);
    else {
        std::error_code error;
        if (std::filesystem::exists(std::string(path), error) && !error)
            same = std::filesystem::equivalent(backend->location(), std::string(path), error) && !error;
    }
    if (same) throw Error(ErrorCode::invalid_argument, std::string(path), "cannot export a backed view over its open source file");
    BackedOptions options;
    options.mode = AccessMode::read_write;
    options.validation = validation;
    backed_detail::Hdf5Backend destination(path, options, true, exclusive, gzip, gzip_level, byte_order);
    backed_detail::export_store(destination, source, options.scratch_bytes, validation);
    destination.flush();
}
} // namespace anndatapp::detail
