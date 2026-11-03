module;
#include <hdf5.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

module anndata.h5ad;

namespace anndata {
namespace {

// No backend object is retained by an eager read. These handles, datatypes,
// conversion buffers and diagnostics stay out of all public module interfaces.
class Handle {
public:
    Handle() = default;
    Handle(hid_t value, herr_t (*close)(hid_t)) : value_(value), close_(close) {}
    ~Handle() { if (value_ >= 0 && close_) close_(value_); }
    Handle(Handle&& other) noexcept
        : value_(std::exchange(other.value_, -1)), close_(other.close_) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            if (value_ >= 0 && close_) close_(value_);
            value_ = std::exchange(other.value_, -1); close_ = other.close_;
        }
        return *this;
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    operator hid_t() const noexcept { return value_; }
    bool valid() const noexcept { return value_ >= 0; }
private:
    hid_t value_ = -1;
    herr_t (*close_)(hid_t) = nullptr;
};

struct Context {
    std::string filename;
    Validation validation = Validation::full;
    WriteOptions write;
    std::unordered_set<haddr_t> ancestors;

    [[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) const {
        throw Error(code, filename + ":" + std::string(path), std::move(message));
    }
    hid_t id(hid_t value, std::string_view path, std::string_view operation) const {
        if (value < 0) fail(ErrorCode::io_error, path, std::string(operation) + " failed");
        return value;
    }
    void check(herr_t status, std::string_view path, std::string_view operation) const {
        if (status < 0) fail(ErrorCode::io_error, path, std::string(operation) + " failed");
    }
};

[[noreturn]] void rethrow_at(const Context& c, const Error& error, std::string_view path) {
    if (error.path().starts_with(c.filename + ":")) throw;
    c.fail(error.code(), path, error.what());
}

std::string child(std::string_view parent, std::string_view name) {
    return parent == "/" ? "/" + std::string(name) : std::string(parent) + "/" + std::string(name);
}

std::size_t multiply(const Context& c, std::size_t left, std::size_t right, std::string_view path) {
    if (right && left > std::numeric_limits<std::size_t>::max() / right)
        c.fail(ErrorCode::invalid_argument, path, "element count or byte size overflows size_t");
    const auto result = left * right;
    if (result > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        c.fail(ErrorCode::invalid_argument, path, "buffer exceeds the addressable range");
    return result;
}

std::vector<std::size_t> shape_of(const Context& c, hid_t space, std::string_view path) {
    const auto rank = H5Sget_simple_extent_ndims(space);
    if (rank < 0) c.fail(ErrorCode::io_error, path, "cannot read dataspace rank");
    std::vector<hsize_t> dimensions(static_cast<std::size_t>(rank));
    c.check(H5Sget_simple_extent_dims(space, dimensions.data(), nullptr), path, "read dimensions");
    std::vector<std::size_t> result;
    for (auto dimension : dimensions) {
        if (dimension > std::numeric_limits<std::size_t>::max())
            c.fail(ErrorCode::invalid_argument, path, "dimension exceeds size_t");
        result.push_back(static_cast<std::size_t>(dimension));
    }
    return result;
}

std::size_t count_of(const Context& c, const std::vector<std::size_t>& shape, std::string_view path) {
    if (std::find(shape.begin(), shape.end(), std::size_t{0}) != shape.end()) return 0;
    std::size_t n = 1;
    for (auto size : shape) n = multiply(c, n, size, path);
    return n;
}

Handle space_for(const Context& c, const std::vector<std::size_t>& shape, std::string_view path) {
    if (shape.empty()) return {c.id(H5Screate(H5S_SCALAR), path, "create scalar dataspace"), H5Sclose};
    std::vector<hsize_t> dims(shape.begin(), shape.end());
    if (dims.size() > H5S_MAX_RANK) c.fail(ErrorCode::unsupported_layout, path, "rank exceeds HDF5 maximum");
    return {c.id(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), path, "create dataspace"), H5Sclose};
}

Handle string_type(const Context& c, std::string_view path) {
    Handle type(c.id(H5Tcopy(H5T_C_S1), path, "create string datatype"), H5Tclose);
    c.check(H5Tset_size(type, H5T_VARIABLE), path, "set variable-length strings");
    c.check(H5Tset_cset(type, H5T_CSET_UTF8), path, "set UTF-8 strings");
    return type;
}

Handle boolean_type(const Context& c, bool native, std::string_view path) {
    static_assert(sizeof(bool) == 1, "the initial HDF5 boolean adapter requires byte-sized bool");
    Handle type(c.id(H5Tenum_create(native ? H5T_NATIVE_UCHAR : H5T_STD_U8LE), path, "create boolean enum"), H5Tclose);
    const unsigned char no = 0, yes = 1;
    c.check(H5Tenum_insert(type, "FALSE", &no), path, "define FALSE");
    c.check(H5Tenum_insert(type, "TRUE", &yes), path, "define TRUE");
    return type;
}

bool exists(const Context& c, hid_t parent, std::string_view name, std::string_view path) {
    const auto value = H5Lexists(parent, std::string(name).c_str(), H5P_DEFAULT);
    if (value < 0) c.fail(ErrorCode::io_error, path, "cannot inspect element existence");
    return value > 0;
}

Handle open_object(const Context& c, hid_t parent, std::string_view name, std::string_view path) {
    H5L_info_t link{};
    c.check(H5Lget_info(parent, std::string(name).c_str(), &link, H5P_DEFAULT), path, "inspect link");
    if (link.type != H5L_TYPE_HARD)
        c.fail(ErrorCode::unsupported_encoding, path, "soft and external HDF5 links are not supported");
    return {c.id(H5Oopen(parent, std::string(name).c_str(), H5P_DEFAULT), path, "open element"), H5Oclose};
}

void require_kind(const Context& c, hid_t object, H5I_type_t kind, std::string_view path) {
    if (H5Iget_type(object) != kind)
        c.fail(ErrorCode::unsupported_encoding, path, kind == H5I_GROUP ? "encoding requires a group" : "encoding requires a dataset");
}

struct Ancestor {
    Context& context;
    haddr_t address;
    Ancestor(Context& c, hid_t object, std::string_view path) : context(c), address(0) {
        H5O_info_t info{};
        c.check(H5Oget_info(object, &info), path, "inspect object identity");
        address = info.addr;
        if (!c.ancestors.insert(address).second)
            c.fail(ErrorCode::unsupported_encoding, path, "cyclic HDF5 group links are not supported");
    }
    ~Ancestor() { context.ancestors.erase(address); }
};

std::vector<std::string> children(const Context& c, hid_t group, std::string_view path) {
    H5G_info_t info{};
    c.check(H5Gget_info(group, &info), path, "inspect group");
    std::vector<std::string> names;
    if (info.nlinks > std::numeric_limits<std::size_t>::max())
        c.fail(ErrorCode::invalid_argument, path, "too many group members");
    names.reserve(static_cast<std::size_t>(info.nlinks));
    for (hsize_t i = 0; i < info.nlinks; ++i) {
        const auto size = H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
        if (size < 0) c.fail(ErrorCode::io_error, path, "cannot inspect group member name");
        std::string name(static_cast<std::size_t>(size) + 1, '\0');
        if (H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), name.size(), H5P_DEFAULT) < 0)
            c.fail(ErrorCode::io_error, path, "cannot read group member name");
        name.resize(static_cast<std::size_t>(size));
        names.push_back(std::move(name));
    }
    return names;
}

void require_known_children(const Context& c, hid_t group,
                            std::initializer_list<std::string_view> known, std::string_view path) {
    for (const auto& name : children(c, group, path))
        if (std::find(known.begin(), known.end(), std::string_view(name)) == known.end())
            c.fail(ErrorCode::unsupported_encoding, child(path, name), "unrecognized child element in a structured encoding");
}

// HDF5 owns each pointer in a variable-length read. Reclaim even when packing
// into C++ owners throws, and do not retain backend allocations in public views.
struct Reclaim {
    hid_t type;
    hid_t space;
    void* buffer;
    bool active = true;
    ~Reclaim() { if (active) H5Dvlen_reclaim(type, space, H5P_DEFAULT, buffer); }
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
std::vector<std::string> read_strings(const Context& c, hid_t type, hid_t space, Read read, std::string_view path) {
    if (H5Tget_class(type) != H5T_STRING)
        c.fail(ErrorCode::unsupported_dtype, path, "expected an HDF5 string datatype");
    if (H5Sget_simple_extent_type(space) == H5S_NULL)
        c.fail(ErrorCode::unsupported_encoding, path, "string data cannot use a null dataspace");
    const auto shape = shape_of(c, space, path);
    const auto n = count_of(c, shape, path);
    std::vector<std::string> result;
    result.reserve(n);
    if (!n) return result;
    if (H5Tis_variable_str(type) > 0) {
        std::vector<char*> pointers(n, nullptr);
        Reclaim reclaim{type, space, pointers.data()};
        c.check(read(type, pointers.data()), path, "read UTF-8 strings");
        for (const auto* pointer : pointers) result.emplace_back(pointer ? pointer : "");
    } else {
        const auto width = H5Tget_size(type);
        std::vector<char> bytes(multiply(c, n, width, path));
        c.check(read(type, bytes.data()), path, "read fixed strings");
        const auto padding = H5Tget_strpad(type);
        for (std::size_t i = 0; i < n; ++i) {
            const auto* first = bytes.data() + i * width;
            std::size_t length = width;
            if (padding == H5T_STR_NULLTERM) length = static_cast<std::size_t>(std::find(first, first + width, '\0') - first);
            else if (padding == H5T_STR_NULLPAD) while (length && first[length - 1] == '\0') --length;
            else if (padding == H5T_STR_SPACEPAD) while (length && first[length - 1] == ' ') --length;
            result.emplace_back(first, length);
        }
    }
    if (c.validation == Validation::full)
        for (const auto& text : result)
            if (!valid_utf8(text)) c.fail(ErrorCode::unsupported_dtype, path, "string data is not valid UTF-8");
    return result;
}

std::vector<std::string> string_attribute(const Context& c, hid_t object, const char* name, std::string_view path) {
    if (H5Aexists(object, name) <= 0)
        c.fail(ErrorCode::unsupported_encoding, path, std::string("missing attribute '") + name + "'");
    Handle attr(c.id(H5Aopen(object, name, H5P_DEFAULT), path, "open attribute"), H5Aclose);
    Handle type(c.id(H5Aget_type(attr), path, "inspect attribute type"), H5Tclose);
    Handle space(c.id(H5Aget_space(attr), path, "inspect attribute dataspace"), H5Sclose);
    if (std::string_view(name) == "column-order") {
        const auto shape = shape_of(c, space, path);
        if (shape.size() != 1)
            c.fail(ErrorCode::unsupported_encoding, path, "column-order must be a one-dimensional attribute");
        // h5py/AnnData writes [] as float64 when a dataframe has no columns.
        // The empty sequence carries no string values or dtype information.
        if (shape[0] == 0) return {};
    }
    return read_strings(c, type, space, [&](hid_t t, void* data) { return H5Aread(attr, t, data); }, path);
}

std::string scalar_string_attribute(const Context& c, hid_t object, const char* name, std::string_view path) {
    auto strings = string_attribute(c, object, name, path);
    if (strings.size() != 1) c.fail(ErrorCode::unsupported_encoding, path, std::string(name) + " must be a scalar string");
    return std::move(strings.front());
}

std::string encoding(const Context& c, hid_t object, std::string_view path) {
    const auto type = scalar_string_attribute(c, object, "encoding-type", path);
    const auto version = scalar_string_attribute(c, object, "encoding-version", path);
    std::string_view expected;
    if (type == "anndata" || type == "raw" || type == "dict" || type == "null" ||
        type == "csr_matrix" || type == "csc_matrix" || type == "nullable-integer" ||
        type == "nullable-boolean" || type == "nullable-string-array") expected = "0.1.0";
    else if (type == "array" || type == "numeric-scalar" || type == "string" ||
             type == "string-array" || type == "dataframe" || type == "categorical") expected = "0.2.0";
    else c.fail(ErrorCode::unsupported_encoding, path, "unsupported encoding-type '" + type + "' (version '" + version + "')");
    if (version != expected)
        c.fail(ErrorCode::unsupported_version, path, "unsupported " + type + " encoding-version '" + version + "'; expected '" + std::string(expected) + "'");
    return type;
}

ByteOrder byte_order(hid_t type) {
    const auto order = H5Tget_order(type);
    return order == H5T_ORDER_LE ? ByteOrder::little : order == H5T_ORDER_BE ? ByteOrder::big : ByteOrder::native;
}

DType dtype(const Context& c, hid_t type, std::string_view path) {
    const auto kind = H5Tget_class(type);
    const auto width = H5Tget_size(type);
    if (kind == H5T_ENUM) {
        if (width == 1 && H5Tget_nmembers(type) == 2) {
            unsigned char no = 255, yes = 255;
            if (H5Tenum_valueof(type, "FALSE", &no) >= 0 && H5Tenum_valueof(type, "TRUE", &yes) >= 0 && no == 0 && yes == 1)
                return DType::boolean;
        }
        c.fail(ErrorCode::unsupported_dtype, path, "only the FALSE/TRUE boolean enum is supported");
    }
    if (kind == H5T_INTEGER && H5Tget_precision(type) == width * 8 && H5Tget_offset(type) == 0) {
        const bool sign = H5Tget_sign(type) == H5T_SGN_2;
        if (width == 1) return sign ? DType::int8 : DType::uint8;
        if (width == 2) return sign ? DType::int16 : DType::uint16;
        if (width == 4) return sign ? DType::int32 : DType::uint32;
        if (width == 8) return sign ? DType::int64 : DType::uint64;
    }
    if (kind == H5T_FLOAT) {
        if (H5Tequal(type, H5T_IEEE_F32LE) > 0 || H5Tequal(type, H5T_IEEE_F32BE) > 0) return DType::float32;
        if (H5Tequal(type, H5T_IEEE_F64LE) > 0 || H5Tequal(type, H5T_IEEE_F64BE) > 0) return DType::float64;
    }
    c.fail(ErrorCode::unsupported_dtype, path, "unsupported numerical datatype (class " + std::to_string(kind) + ", width " + std::to_string(width) + ")");
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

template<class F> decltype(auto) visit_dtype(DType type, F&& f) {
    switch (type) {
        case DType::boolean: return f.template operator()<bool>();
        case DType::int8: return f.template operator()<std::int8_t>();
        case DType::uint8: return f.template operator()<std::uint8_t>();
        case DType::int16: return f.template operator()<std::int16_t>();
        case DType::uint16: return f.template operator()<std::uint16_t>();
        case DType::int32: return f.template operator()<std::int32_t>();
        case DType::uint32: return f.template operator()<std::uint32_t>();
        case DType::int64: return f.template operator()<std::int64_t>();
        case DType::uint64: return f.template operator()<std::uint64_t>();
        case DType::float32: return f.template operator()<float>();
        case DType::float64: return f.template operator()<double>();
    }
    throw Error(ErrorCode::unsupported_dtype, "", "unknown numeric dtype");
}

void check_filters(const Context& c, hid_t dataset, std::string_view path) {
    Handle properties(c.id(H5Dget_create_plist(dataset), path, "inspect dataset filters"), H5Pclose);
    const int n = H5Pget_nfilters(properties);
    if (n < 0) c.fail(ErrorCode::io_error, path, "cannot inspect dataset filters");
    for (int i = 0; i < n; ++i) {
        unsigned flags = 0, config = 0;
        std::size_t count = 0;
        const auto filter = H5Pget_filter2(properties, static_cast<unsigned>(i), &flags, &count, nullptr, 0, nullptr, &config);
        if (filter < 0) c.fail(ErrorCode::io_error, path, "cannot inspect dataset filter");
        if (H5Zfilter_avail(filter) <= 0 || H5Zget_filter_info(filter, &config) < 0 || !(config & H5Z_FILTER_CONFIG_DECODE_ENABLED))
            c.fail(ErrorCode::unavailable_filter, path, "HDF5 decoder unavailable for filter " + std::to_string(filter));
    }
}

Array read_numeric(Context& c, hid_t object, std::string_view path) {
    require_kind(c, object, H5I_DATASET, path);
    Handle type(c.id(H5Dget_type(object), path, "inspect datatype"), H5Tclose);
    Handle space(c.id(H5Dget_space(object), path, "inspect dataspace"), H5Sclose);
    if (H5Sget_simple_extent_type(space) == H5S_NULL)
        c.fail(ErrorCode::unsupported_encoding, path, "numeric array cannot use a null dataspace");
    const auto shape = shape_of(c, space, path);
    const auto type_id = dtype(c, type, path);
    const auto count = count_of(c, shape, path);
    multiply(c, count, dtype_size(type_id), path);
    auto array = visit_dtype(type_id, [&]<class T> { return Array::empty<T>(shape); });
    array.set_serialization_byte_order(byte_order(type));
    if (count) {
        check_filters(c, object, path);
        // An identical enum transfer preserves its raw byte. Mapping malformed
        // enum values through another enum can instead produce 0xff, which is
        // not a C++ bool representation. Inspect bytes before any bool load.
        const hid_t memory_type = type_id == DType::boolean ? static_cast<hid_t>(type) : native_type(type_id);
        herr_t status;
        auto* data = array.edit().bytes().data();
        H5E_BEGIN_TRY { status = H5Dread(object, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data); } H5E_END_TRY;
        c.check(status, path, "read numerical dataset");
        if (type_id == DType::boolean && c.validation == Validation::full)
            for (std::size_t i = 0; i < count; ++i)
                data[i] = data[i] == std::byte{0} ? std::byte{0} : std::byte{1};
    }
    return array;
}

struct StringValues {
    std::vector<std::size_t> shape;
    std::vector<std::string> values;
};

StringValues read_string_values(Context& c, hid_t object, std::string_view path) {
    require_kind(c, object, H5I_DATASET, path);
    Handle type(c.id(H5Dget_type(object), path, "inspect string datatype"), H5Tclose);
    Handle space(c.id(H5Dget_space(object), path, "inspect string dataspace"), H5Sclose);
    if (H5Sget_simple_extent_type(space) == H5S_NULL)
        c.fail(ErrorCode::unsupported_encoding, path, "string array cannot use a null dataspace");
    check_filters(c, object, path);
    auto strings = read_strings(c, type, space, [&](hid_t t, void* data) { return H5Dread(object, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, data); }, path);
    return {shape_of(c, space, path), std::move(strings)};
}

StringArray read_string_array(Context& c, hid_t object, std::string_view path) try {
    auto values = read_string_values(c, object, path);
    std::size_t total = 0;
    for (const auto& string : values.values) {
        if (string.size() > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) - total)
            c.fail(ErrorCode::invalid_argument, path, "packed string bytes exceed addressable range");
        total += string.size();
    }
    std::vector<char> bytes;
    bytes.reserve(total);
    std::vector<std::uint64_t> offsets;
    offsets.reserve(values.values.size() + 1);
    offsets.push_back(0);
    for (const auto& string : values.values) {
        bytes.insert(bytes.end(), string.begin(), string.end());
        offsets.push_back(bytes.size());
    }
    // The decoder constructed monotonically increasing offsets itself; UTF-8
    // was checked once while decoding when full validation was requested.
    return StringArray::from_packed(std::move(values.shape), std::move(bytes), std::move(offsets), Validation::none);
} catch (const Error& error) {
    rethrow_at(c, error, path);
}

std::vector<std::size_t> sparse_shape(const Context& c, hid_t object, std::string_view path) {
    Handle attr(c.id(H5Aopen(object, "shape", H5P_DEFAULT), path, "open sparse shape"), H5Aclose);
    Handle space(c.id(H5Aget_space(attr), path, "inspect sparse shape"), H5Sclose);
    const auto shape = shape_of(c, space, path);
    if (shape != std::vector<std::size_t>{2})
        c.fail(ErrorCode::shape_mismatch, path, "sparse shape must contain two dimensions");
    Handle type(c.id(H5Aget_type(attr), path, "inspect sparse shape datatype"), H5Tclose);
    const auto kind = dtype(c, type, path);
    if (kind == DType::boolean || kind == DType::float32 || kind == DType::float64)
        c.fail(ErrorCode::unsupported_dtype, path, "sparse shape must be an integer array");
    std::uint64_t dims[2]{};
    if (H5Tget_sign(type) == H5T_SGN_2) {
        std::int64_t signed_dims[2]{};
        c.check(H5Aread(attr, H5T_NATIVE_INT64, signed_dims), path, "read sparse shape");
        if (signed_dims[0] < 0 || signed_dims[1] < 0) c.fail(ErrorCode::shape_mismatch, path, "negative sparse shape");
        dims[0] = static_cast<std::uint64_t>(signed_dims[0]); dims[1] = static_cast<std::uint64_t>(signed_dims[1]);
    } else c.check(H5Aread(attr, H5T_NATIVE_UINT64, dims), path, "read sparse shape");
    for (auto dim : dims) if (dim > std::numeric_limits<std::size_t>::max()) c.fail(ErrorCode::shape_mismatch, path, "sparse dimension exceeds size_t");
    return {static_cast<std::size_t>(dims[0]), static_cast<std::size_t>(dims[1])};
}

Array read_array(Context& c, hid_t object, std::string_view path, std::string_view code) try {
    if (code == "array") return read_numeric(c, object, path);
    if (code != "csr_matrix" && code != "csc_matrix")
        c.fail(ErrorCode::unsupported_encoding, path, "expected numerical array, found '" + std::string(code) + "'");
    require_kind(c, object, H5I_GROUP, path);
    Ancestor ancestor(c, object, path);
    require_known_children(c, object, {"data", "indices", "indptr"}, path);
    const auto shape = sparse_shape(c, object, path);
    auto values_object = open_object(c, object, "data", child(path, "data"));
    auto indices_object = open_object(c, object, "indices", child(path, "indices"));
    auto offsets_object = open_object(c, object, "indptr", child(path, "indptr"));
    auto values = read_numeric(c, values_object, child(path, "data"));
    auto indices = read_numeric(c, indices_object, child(path, "indices"));
    auto offsets = read_numeric(c, offsets_object, child(path, "indptr"));
    return Array::from_sparse_buffers(code == "csr_matrix" ? Layout::csr : Layout::csc,
        std::move(values), std::move(indices), std::move(offsets), shape[0], shape[1], c.validation);
} catch (const Error& error) { rethrow_at(c, error, path); }

Column read_column(Context&, hid_t, std::string_view, std::string_view);
Table read_table(Context&, hid_t, std::string_view);
Element read_element(Context&, hid_t, std::string_view, std::string_view);

Column read_column(Context& c, hid_t object, std::string_view path, std::string_view code) try {
    if (code == "array") return Column::numeric(read_numeric(c, object, path));
    if (code == "string-array") {
        auto strings = read_string_values(c, object, path);
        if (strings.shape.size() != 1) c.fail(ErrorCode::shape_mismatch, path, "table columns and categories must be one-dimensional");
        return Column::strings(std::move(strings.values));
    }
    require_kind(c, object, H5I_GROUP, path);
    Ancestor ancestor(c, object, path);
    if (code == "categorical") {
        require_known_children(c, object, {"codes", "categories"}, path);
        auto codes_object = open_object(c, object, "codes", child(path, "codes"));
        if (encoding(c, codes_object, child(path, "codes")) != "array") c.fail(ErrorCode::unsupported_encoding, path, "categorical codes must be a numerical array");
        auto codes = read_numeric(c, codes_object, child(path, "codes"));
        auto categories_object = open_object(c, object, "categories", child(path, "categories"));
        auto categories = read_column(c, categories_object, child(path, "categories"), encoding(c, categories_object, child(path, "categories")));
        Handle ordered(c.id(H5Aopen(object, "ordered", H5P_DEFAULT), path, "open categorical ordering"), H5Aclose);
        Handle ordered_space(c.id(H5Aget_space(ordered), path, "inspect categorical ordering dataspace"), H5Sclose);
        Handle ordered_type(c.id(H5Aget_type(ordered), path, "inspect categorical ordering datatype"), H5Tclose);
        if (H5Sget_simple_extent_type(ordered_space) != H5S_SCALAR || dtype(c, ordered_type, path) != DType::boolean)
            c.fail(ErrorCode::unsupported_encoding, path, "categorical ordered attribute must be a scalar boolean");
        auto bool_type = boolean_type(c, true, path);
        unsigned char is_ordered = 0;
        c.check(H5Aread(ordered, bool_type, &is_ordered), path, "read categorical ordering");
        return Column::categorical(std::move(codes), std::move(categories), is_ordered != 0, c.validation);
    }
    if (code == "nullable-integer" || code == "nullable-boolean" || code == "nullable-string-array") {
        require_known_children(c, object, {"values", "mask"}, path);
        auto values_object = open_object(c, object, "values", child(path, "values"));
        auto mask_object = open_object(c, object, "mask", child(path, "mask"));
        if (encoding(c, mask_object, child(path, "mask")) != "array") c.fail(ErrorCode::unsupported_encoding, path, "nullable mask must be a numerical array");
        auto mask = read_numeric(c, mask_object, child(path, "mask"));
        if (mask.rank() != 1 || mask.dtype() != DType::boolean)
            c.fail(ErrorCode::unsupported_dtype, child(path, "mask"), "nullable mask must be a one-dimensional boolean array");
        auto mask_view = mask.view().dense<bool>();
        std::vector<std::uint8_t> validity(mask.extent(0));
        for (std::size_t i = 0; i < validity.size(); ++i) validity[i] = !mask_view.data()[i];
        const auto values_code = encoding(c, values_object, child(path, "values"));
        if (code == "nullable-string-array") {
            if (values_code != "string-array") c.fail(ErrorCode::unsupported_encoding, path, "nullable strings require string-array values");
            auto strings = read_string_values(c, values_object, child(path, "values"));
            if (strings.shape.size() != 1 || strings.values.size() != validity.size()) c.fail(ErrorCode::shape_mismatch, path, "nullable values and mask lengths differ");
            auto sentinel = StringMissingValue::NA;
            if (H5Aexists(object, "na-value") > 0) {
                const auto value = scalar_string_attribute(c, object, "na-value", path);
                if (value == "NaN") sentinel = StringMissingValue::NaN;
                else if (value != "NA") c.fail(ErrorCode::unsupported_encoding, path, "unsupported nullable-string na-value '" + value + "'");
            }
            return Column::nullable_strings(std::move(strings.values), std::move(validity), sentinel);
        }
        if (values_code != "array") c.fail(ErrorCode::unsupported_encoding, path, "nullable numerical values require array encoding");
        auto values = read_numeric(c, values_object, child(path, "values"));
        if (code == "nullable-boolean" && values.dtype() != DType::boolean)
            c.fail(ErrorCode::unsupported_dtype, path, "nullable-boolean values must be boolean");
        if (code == "nullable-integer" && (values.dtype() == DType::boolean || values.dtype() == DType::float32 || values.dtype() == DType::float64))
            c.fail(ErrorCode::unsupported_dtype, path, "nullable-integer values must be integer");
        return Column::nullable_numeric(std::move(values), std::move(validity));
    }
    c.fail(ErrorCode::unsupported_encoding, path, "encoding '" + std::string(code) + "' is not supported as a column");
} catch (const Error& error) { rethrow_at(c, error, path); }

Table read_table(Context& c, hid_t object, std::string_view path) {
    require_kind(c, object, H5I_GROUP, path);
    Ancestor ancestor(c, object, path);
    const auto index_key = scalar_string_attribute(c, object, "_index", path);
    auto index_object = open_object(c, object, index_key, child(path, index_key));
    const auto index_code = encoding(c, index_object, child(path, index_key));
    if (index_code != "string-array") c.fail(ErrorCode::unsupported_dtype, child(path, index_key), "AnnData table indexes currently require string-array encoding");
    auto index = read_string_values(c, index_object, child(path, index_key));
    if (index.shape.size() != 1) c.fail(ErrorCode::shape_mismatch, path, "table index must be one-dimensional");
    auto table = Table::with_index(std::move(index.values), index_key == "_index" ? std::string{} : index_key);
    auto names = string_attribute(c, object, "column-order", path);
    std::unordered_set<std::string> seen;
    for (const auto& name : names) {
        if (!seen.insert(name).second) c.fail(ErrorCode::unsupported_encoding, path, "duplicate dataframe column-order entry '" + name + "'");
        auto column_object = open_object(c, object, name, child(path, name));
        auto column = read_column(c, column_object, child(path, name), encoding(c, column_object, child(path, name)));
        try { table.set(name, std::move(column)); }
        catch (const Error& error) { rethrow_at(c, error, child(path, name)); }
    }
    for (const auto& name : children(c, object, path)) {
        if (name != index_key && !seen.contains(name)) c.fail(ErrorCode::unsupported_encoding, child(path, name), "dataframe member is absent from column-order");
    }
    return table;
}

Element read_element(Context& c, hid_t object, std::string_view path, std::string_view code) {
    if (code == "null") {
        require_kind(c, object, H5I_DATASET, path);
        Handle space(c.id(H5Dget_space(object), path, "inspect null dataspace"), H5Sclose);
        if (H5Sget_simple_extent_type(space) != H5S_NULL) c.fail(ErrorCode::unsupported_encoding, path, "null encoding requires an HDF5 null dataspace");
        return Element{};
    }
    if (code == "array" || code == "csr_matrix" || code == "csc_matrix") return Element(read_array(c, object, path, code));
    if (code == "string-array") return Element(read_string_array(c, object, path));
    if (code == "string") {
        auto values = read_string_values(c, object, path);
        if (!values.shape.empty()) c.fail(ErrorCode::shape_mismatch, path, "string scalar requires rank zero");
        return Element(std::move(values.values.front()));
    }
    if (code == "numeric-scalar") {
        auto values = read_numeric(c, object, path);
        if (values.rank() != 0) c.fail(ErrorCode::shape_mismatch, path, "numeric scalar requires rank zero");
        auto scalar = visit_dtype(values.dtype(), [&]<class T> { return Element::scalar(values.view().dense<T>().data()[0]); });
        scalar.set_serialization_byte_order(values.serialization_byte_order());
        return scalar;
    }
    if (code == "dataframe") return Element(read_table(c, object, path));
    if (code == "categorical" || code == "nullable-integer" || code == "nullable-boolean" || code == "nullable-string-array")
        return Element(read_column(c, object, path, code));
    if (code == "dict") {
        require_kind(c, object, H5I_GROUP, path);
        Ancestor ancestor(c, object, path);
        auto mapping = Element::mapping();
        for (const auto& name : children(c, object, path)) {
            const auto item_path = child(path, name);
            auto item = open_object(c, object, name, item_path);
            mapping.set(name, read_element(c, item, item_path, encoding(c, item, item_path)));
        }
        return mapping;
    }
    c.fail(ErrorCode::unsupported_encoding, path, "encoding '" + std::string(code) + "' is not a nested metadata element");
}

template<class F> void read_mapping(Context& c, hid_t root, std::string_view name, F&& read) {
    const auto path = child("/", name);
    if (!exists(c, root, name, path)) return;
    auto group = open_object(c, root, name, path);
    if (encoding(c, group, path) != "dict") c.fail(ErrorCode::unsupported_encoding, path, "aligned mapping requires dict encoding");
    require_kind(c, group, H5I_GROUP, path);
    Ancestor ancestor(c, group, path);
    for (const auto& key : children(c, group, path)) {
        const auto item_path = child(path, key);
        auto object = open_object(c, group, key, item_path);
        read(key, object, item_path, encoding(c, object, item_path));
    }
}

template<class FArray, class FTable, class FStrings>
void read_aligned(Context& c, hid_t object, std::string_view path, std::string_view code, FArray array, FTable table, FStrings strings) {
    if (code == "dataframe") table(read_table(c, object, path));
    else if (code == "string-array") strings(read_string_array(c, object, path));
    else array(read_array(c, object, path, code));
}

AnnData read_dataset(Context& c, hid_t root) {
    if (encoding(c, root, "/") != "anndata") c.fail(ErrorCode::unsupported_encoding, "/", "root must use anndata encoding");
    Ancestor ancestor(c, root, "/");
    const std::unordered_set<std::string> allowed{"X", "obs", "var", "layers", "obsm", "varm", "obsp", "varp", "uns", "raw"};
    for (const auto& name : children(c, root, "/"))
        if (!allowed.contains(name)) c.fail(ErrorCode::unsupported_encoding, child("/", name), "unrecognized AnnData root slot");
    auto obs_object = open_object(c, root, "obs", "/obs");
    auto var_object = open_object(c, root, "var", "/var");
    if (encoding(c, obs_object, "/obs") != "dataframe" || encoding(c, var_object, "/var") != "dataframe")
        c.fail(ErrorCode::unsupported_encoding, "/", "obs and var require dataframe encoding");
    auto obs = read_table(c, obs_object, "/obs");
    auto var = read_table(c, var_object, "/var");
    auto data = AnnData::empty(obs.size(), var.size());
    data.set_obs(std::move(obs)); data.set_var(std::move(var));
    if (exists(c, root, "X", "/X")) {
        auto x = open_object(c, root, "X", "/X");
        auto code = encoding(c, x, "/X");
        if (code == "null") (void)read_element(c, x, "/X", code);
        else data.set_x(read_array(c, x, "/X", code));
    }
    read_mapping(c, root, "layers", [&](const auto& key, hid_t object, const auto& path, const auto& code) { data.set_layer(key, read_array(c, object, path, code)); });
    read_mapping(c, root, "obsp", [&](const auto& key, hid_t object, const auto& path, const auto& code) { data.set_obsp(key, read_array(c, object, path, code)); });
    read_mapping(c, root, "varp", [&](const auto& key, hid_t object, const auto& path, const auto& code) { data.set_varp(key, read_array(c, object, path, code)); });
    read_mapping(c, root, "obsm", [&](const auto& key, hid_t object, const auto& path, const auto& code) {
        read_aligned(c, object, path, code, [&](Array a) { data.set_obsm(key, std::move(a)); },
            [&](Table t) { data.set_obsm(key, std::move(t)); }, [&](StringArray s) { data.set_obsm(key, std::move(s)); });
    });
    read_mapping(c, root, "varm", [&](const auto& key, hid_t object, const auto& path, const auto& code) {
        read_aligned(c, object, path, code, [&](Array a) { data.set_varm(key, std::move(a)); },
            [&](Table t) { data.set_varm(key, std::move(t)); }, [&](StringArray s) { data.set_varm(key, std::move(s)); });
    });
    if (exists(c, root, "uns", "/uns")) {
        auto object = open_object(c, root, "uns", "/uns");
        auto code = encoding(c, object, "/uns");
        if (code != "dict") c.fail(ErrorCode::unsupported_encoding, "/uns", "uns requires dict encoding");
        data.set_uns(read_element(c, object, "/uns", code));
    }
    if (exists(c, root, "raw", "/raw")) {
        auto raw = open_object(c, root, "raw", "/raw");
        const auto code = encoding(c, raw, "/raw");
        if (code == "null") (void)read_element(c, raw, "/raw", code);
        else {
            if (code != "raw") c.fail(ErrorCode::unsupported_encoding, "/raw", "raw slot requires raw encoding");
            require_kind(c, raw, H5I_GROUP, "/raw");
            Ancestor raw_ancestor(c, raw, "/raw");
            auto raw_var_object = open_object(c, raw, "var", "/raw/var");
            if (encoding(c, raw_var_object, "/raw/var") != "dataframe") c.fail(ErrorCode::unsupported_encoding, "/raw/var", "raw var requires dataframe encoding");
            auto raw_var = read_table(c, raw_var_object, "/raw/var");
            if (exists(c, raw, "X", "/raw/X")) {
                auto x = open_object(c, raw, "X", "/raw/X");
                auto x_code = encoding(c, x, "/raw/X");
                if (x_code == "null") { (void)read_element(c, x, "/raw/X", x_code); data.set_raw(std::move(raw_var)); }
                else data.set_raw(read_array(c, x, "/raw/X", x_code), std::move(raw_var));
            } else data.set_raw(std::move(raw_var));
            if (exists(c, raw, "varm", "/raw/varm")) {
                auto varm = open_object(c, raw, "varm", "/raw/varm");
                if (encoding(c, varm, "/raw/varm") != "dict") c.fail(ErrorCode::unsupported_encoding, "/raw/varm", "raw varm requires dict encoding");
                require_kind(c, varm, H5I_GROUP, "/raw/varm");
                Ancestor varm_ancestor(c, varm, "/raw/varm");
                for (const auto& key : children(c, varm, "/raw/varm")) {
                    const auto path = child("/raw/varm", key);
                    auto item = open_object(c, varm, key, path);
                    const auto item_code = encoding(c, item, path);
                    read_aligned(c, item, path, item_code, [&](Array a) { data.set_raw_varm(key, std::move(a)); },
                        [&](Table t) { data.set_raw_varm(key, std::move(t)); }, [&](StringArray s) { data.set_raw_varm(key, std::move(s)); });
                }
            }
            for (const auto& name : children(c, raw, "/raw"))
                if (name != "X" && name != "var" && name != "varm") c.fail(ErrorCode::unsupported_encoding, child("/raw", name), "unrecognized raw slot");
        }
    }
    // Sparse structures were validated once at assembly, not again here.
    return data;
}

void check_text(const Context& c, std::string_view text, std::string_view path) {
    if (text.find('\0') != std::string_view::npos)
        c.fail(ErrorCode::invalid_argument, path, "HDF5 variable-length strings cannot contain embedded NUL bytes");
    if (c.validation == Validation::full && !valid_utf8(text))
        c.fail(ErrorCode::unsupported_dtype, path, "string data is not valid UTF-8");
}

void check_key(const Context& c, std::string_view key, std::string_view path) {
    check_text(c, key, path);
    if (key.empty() || key == "." || key.find('/') != std::string_view::npos)
        c.fail(ErrorCode::invalid_argument, path, "HDF5 element keys must be nonempty single path components other than '.'");
}

Handle link_properties(const Context& c, std::string_view path) {
    Handle properties(c.id(H5Pcreate(H5P_LINK_CREATE), path, "create UTF-8 link properties"), H5Pclose);
    c.check(H5Pset_char_encoding(properties, H5T_CSET_UTF8), path, "set UTF-8 link name");
    return properties;
}

Handle create_group(const Context& c, hid_t parent, std::string_view key, std::string_view path) {
    check_key(c, key, path);
    auto properties = link_properties(c, path);
    return {c.id(H5Gcreate2(parent, std::string(key).c_str(), properties, H5P_DEFAULT, H5P_DEFAULT), path, "create group"), H5Gclose};
}

void write_string_attribute(const Context& c, hid_t object, const char* key,
                            const std::vector<std::string>& strings, bool scalar, std::string_view path) {
    auto type = string_type(c, path);
    auto space = space_for(c, scalar ? std::vector<std::size_t>{} : std::vector<std::size_t>{strings.size()}, path);
    Handle attr(c.id(H5Acreate2(object, key, type, space, H5P_DEFAULT, H5P_DEFAULT), path, "create string attribute"), H5Aclose);
    std::vector<const char*> pointers;
    pointers.reserve(strings.size());
    for (const auto& string : strings) { check_text(c, string, path); pointers.push_back(string.c_str()); }
    if (!pointers.empty()) c.check(H5Awrite(attr, type, pointers.data()), path, "write string attribute");
}

void write_encoding(const Context& c, hid_t object, std::string_view code, std::string_view version, std::string_view path) {
    write_string_attribute(c, object, "encoding-type", {std::string(code)}, true, path);
    write_string_attribute(c, object, "encoding-version", {std::string(version)}, true, path);
}

ByteOrder output_order(const Context& c, ByteOrder original) {
    if (c.write.byte_order == FileByteOrder::little) return ByteOrder::little;
    if (c.write.byte_order == FileByteOrder::big) return ByteOrder::big;
    if (c.write.byte_order == FileByteOrder::native) return ByteOrder::native;
    return original;
}

hid_t file_numeric_type(DType type, ByteOrder order) {
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

Handle dataset_properties(const Context& c, const std::vector<std::size_t>& shape, std::size_t width, std::string_view path) {
    Handle properties(c.id(H5Pcreate(H5P_DATASET_CREATE), path, "create dataset properties"), H5Pclose);
    if (c.write.compression == Compression::none || shape.empty() || count_of(c, shape, path) == 0) return properties;
    unsigned config = 0;
    if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0 || H5Zget_filter_info(H5Z_FILTER_DEFLATE, &config) < 0 || !(config & H5Z_FILTER_CONFIG_ENCODE_ENABLED))
        c.fail(ErrorCode::unavailable_filter, path, "HDF5 gzip encoder unavailable");
    std::vector<hsize_t> chunks(shape.begin(), shape.end());
    const auto target = std::max<std::size_t>(1, 1048576 / std::max<std::size_t>(1, width));
    auto product = [&] {
        std::size_t n = 1;
        for (auto chunk : chunks) {
            if (chunk > target || n > target / static_cast<std::size_t>(chunk)) return target + 1;
            n *= static_cast<std::size_t>(chunk);
        }
        return n;
    };
    while (product() > target) {
        auto largest = std::max_element(chunks.begin(), chunks.end());
        *largest = (*largest + 1) / 2;
    }
    c.check(H5Pset_chunk(properties, static_cast<int>(chunks.size()), chunks.data()), path, "set chunk shape");
    c.check(H5Pset_deflate(properties, static_cast<unsigned>(c.write.gzip_level)), path, "set gzip compression");
    return properties;
}

Handle create_dataset(const Context& c, hid_t parent, std::string_view key, hid_t type,
                      const std::vector<std::size_t>& shape, std::size_t width, std::string_view path) {
    check_key(c, key, path);
    auto space = space_for(c, shape, path);
    auto properties = dataset_properties(c, shape, width, path);
    auto links = link_properties(c, path);
    return {c.id(H5Dcreate2(parent, std::string(key).c_str(), type, space, links, properties, H5P_DEFAULT), path, "create dataset"), H5Dclose};
}

std::vector<std::size_t> array_shape(ArrayView array) {
    std::vector<std::size_t> shape;
    for (std::size_t axis = 0; axis < array.rank(); ++axis) shape.push_back(array.extent(axis));
    return shape;
}

// Select one row fragment in file coordinates. The matching memory buffer is
// one-dimensional even for a higher-dimensional destination.
void write_tile(const Context& c, hid_t dataset, hid_t memory_type,
                 const std::vector<std::size_t>& shape, std::size_t outer, std::size_t first,
                 std::size_t length, const void* data, std::string_view path) {
    std::vector<hsize_t> start(shape.size(), 0), count(shape.size(), 1);
    start.back() = first; count.back() = length;
    for (std::size_t axis = shape.size() - 1; axis > 0; --axis) {
        start[axis - 1] = outer % shape[axis - 1]; outer /= shape[axis - 1];
    }
    Handle file_space(c.id(H5Dget_space(dataset), path, "inspect output dataspace"), H5Sclose);
    c.check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr), path, "select output tile");
    auto memory_space = space_for(c, {length}, path);
    c.check(H5Dwrite(dataset, memory_type, memory_space, file_space, H5P_DEFAULT, data), path, "write dataset tile");
}

void write_dense_data(Context& c, hid_t dataset, ArrayView array, hid_t memory_type, std::string_view path) {
    const auto shape = array_shape(array);
    const auto count = count_of(c, shape, path);
    if (!count) return;
    std::optional<DenseBytesView<const std::byte>> bytes;
    try { bytes.emplace(array.bytes()); }
    catch (const Error& error) { if (error.code() != ErrorCode::non_contiguous) throw; }
    if (bytes && bytes->contiguous()) {
        c.check(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, bytes->data()), path, "write contiguous numerical dataset");
        return;
    }
    const auto width = dtype_size(array.dtype());
    const auto tile_elements = std::max<std::size_t>(1, 1048576 / width);
    const auto columns = shape.back();
    const auto outer_count = count / columns;
    std::vector<std::byte> packed;
    if (bytes) packed.resize(multiply(c, std::min(columns, tile_elements), width, path));
    for (std::size_t outer = 0; outer < outer_count; ++outer) {
        std::ptrdiff_t row_offset = 0;
        auto coordinate = outer;
        if (bytes) for (std::size_t axis = shape.size() - 1; axis > 0; --axis) {
            row_offset += static_cast<std::ptrdiff_t>(coordinate % shape[axis - 1]) * bytes->stride_bytes(axis - 1);
            coordinate /= shape[axis - 1];
        }
        for (std::size_t first = 0; first < columns;) {
            const auto length = std::min(tile_elements, columns - first);
            if (bytes) {
                const auto stride = bytes->stride_bytes(shape.size() - 1);
                for (std::size_t j = 0; j < length; ++j)
                    std::memcpy(packed.data() + j * width, bytes->data() + row_offset + static_cast<std::ptrdiff_t>(first + j) * stride, width);
                write_tile(c, dataset, memory_type, shape, outer, first, length, packed.data(), path);
            } else {
                // A gather cannot expose affine strides. Materialize only this
                // bounded row fragment, not the array or its owning dataset.
                std::vector<std::optional<std::vector<std::size_t>>> axes(shape.size());
                coordinate = outer;
                for (std::size_t axis = shape.size() - 1; axis > 0; --axis) {
                    axes[axis - 1].emplace(1, coordinate % shape[axis - 1]); coordinate /= shape[axis - 1];
                }
                axes.back().emplace(); axes.back()->reserve(length);
                for (std::size_t j = 0; j < length; ++j) axes.back()->push_back(first + j);
                auto tile = array.select(std::move(axes)).copy();
                write_tile(c, dataset, memory_type, shape, outer, first, length, tile.view().bytes().data(), path);
            }
            first += length;
        }
    }
}

void write_numeric_buffer(Context& c, hid_t parent, std::string_view key, DType type,
                           ByteOrder order, const void* data, const std::vector<std::size_t>& shape,
                           std::string_view path, bool element_encoding = true) {
    auto file_bool = type == DType::boolean ? boolean_type(c, false, path) : Handle{};
    auto memory_bool = type == DType::boolean ? boolean_type(c, true, path) : Handle{};
    const hid_t file_type = type == DType::boolean ? static_cast<hid_t>(file_bool) : file_numeric_type(type, output_order(c, order));
    const hid_t memory_type = type == DType::boolean ? static_cast<hid_t>(memory_bool) : native_type(type);
    auto dataset = create_dataset(c, parent, key, file_type, shape, dtype_size(type), path);
    if (element_encoding) write_encoding(c, dataset, "array", "0.2.0", path);
    if (count_of(c, shape, path)) c.check(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), path, "write numerical buffer");
}

void write_array(Context& c, hid_t parent, std::string_view key, ArrayView array, std::string_view path) try {
    if (c.validation == Validation::full) array.validate();
    if (array.layout() == Layout::dense) {
        const auto type = array.dtype();
        auto file_bool = type == DType::boolean ? boolean_type(c, false, path) : Handle{};
        auto memory_bool = type == DType::boolean ? boolean_type(c, true, path) : Handle{};
        const hid_t file_type = type == DType::boolean ? static_cast<hid_t>(file_bool) : file_numeric_type(type, output_order(c, array.serialization_byte_order()));
        const hid_t memory_type = type == DType::boolean ? static_cast<hid_t>(memory_bool) : native_type(type);
        auto dataset = create_dataset(c, parent, key, file_type, array_shape(array), dtype_size(type), path);
        write_encoding(c, dataset, "array", "0.2.0", path);
        write_dense_data(c, dataset, array, memory_type, path);
        return;
    }
    std::optional<Array> selected;
    std::optional<SparseBytesView<const std::byte>> bytes;
    try { bytes.emplace(array.sparse_bytes()); }
    catch (const Error& error) {
        if (error.code() != ErrorCode::non_contiguous) throw;
        selected.emplace(array.copy(Validation::none)); array = selected->view(); bytes.emplace(array.sparse_bytes());
    }
    auto group = create_group(c, parent, key, path);
    write_encoding(c, group, array.layout() == Layout::csr ? "csr_matrix" : "csc_matrix", "0.1.0", path);
    const std::uint64_t shape[2]{array.extent(0), array.extent(1)};
    auto space = space_for(c, {2}, path);
    Handle attr(c.id(H5Acreate2(group, "shape", H5T_STD_U64LE, space, H5P_DEFAULT, H5P_DEFAULT), path, "create sparse shape"), H5Aclose);
    c.check(H5Awrite(attr, H5T_NATIVE_UINT64, shape), path, "write sparse shape");
    write_numeric_buffer(c, group, "data", array.dtype(), array.serialization_byte_order(), bytes->values(), {bytes->nnz()}, child(path, "data"), false);
    write_numeric_buffer(c, group, "indices", array.index_dtype(), array.index_serialization_byte_order(), bytes->indices(), {bytes->nnz()}, child(path, "indices"), false);
    const auto major = array.extent(array.layout() == Layout::csr ? 0 : 1);
    if (major == std::numeric_limits<std::size_t>::max()) c.fail(ErrorCode::invalid_argument, path, "sparse offset count overflows");
    write_numeric_buffer(c, group, "indptr", array.offset_dtype(), array.offset_serialization_byte_order(), bytes->offsets(), {major + 1}, child(path, "indptr"), false);
} catch (const Error& error) { rethrow_at(c, error, path); }

template<class Getter>
void write_strings(Context& c, hid_t parent, std::string_view key, const std::vector<std::size_t>& shape,
                    Getter get, std::string_view path, bool scalar = false) {
    auto type = string_type(c, path);
    auto dataset = create_dataset(c, parent, key, type, shape, sizeof(char*), path);
    write_encoding(c, dataset, scalar ? "string" : "string-array", "0.2.0", path);
    const auto n = count_of(c, shape, path);
    if (!n) return;
    if (shape.empty()) {
        std::string value(get(0)); check_text(c, value, path);
        const char* pointer = value.c_str();
        c.check(H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &pointer), path, "write string scalar");
        return;
    }
    const auto columns = shape.back();
    const auto rows = n / columns;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t first = 0; first < columns;) {
            std::vector<std::string> strings;
            std::vector<const char*> pointers;
            std::size_t byte_count = 0;
            while (first + strings.size() < columns && strings.size() < 4096) {
                const auto value = get(row * columns + first + strings.size());
                if (!strings.empty() && value.size() + 1 > 1048576 - std::min(byte_count, std::size_t{1048576})) break;
                check_text(c, value, path);
                strings.emplace_back(value); byte_count += value.size() + 1;
            }
            pointers.reserve(strings.size());
            for (const auto& string : strings) pointers.push_back(string.c_str());
            write_tile(c, dataset, type, shape, row, first, strings.size(), pointers.data(), path);
            first += strings.size();
        }
    }
}

void write_string_array(Context& c, hid_t parent, std::string_view key, StringArrayView array, std::string_view path) {
    std::vector<std::size_t> shape;
    for (std::size_t i = 0; i < array.rank(); ++i) shape.push_back(array.extent(i));
    write_strings(c, parent, key, shape, [&](std::size_t i) { return array.string(i); }, path);
}

void write_column(Context&, hid_t, std::string_view, ColumnView, std::string_view);
void write_table(Context&, hid_t, std::string_view, TableView, std::string_view);
void write_element(Context&, hid_t, std::string_view, ElementView, std::string_view);

void validate_dictionary(Context& c, ColumnView categories, std::string_view path) {
    for (std::size_t i = 0; i < categories.size(); ++i)
        if (!categories.valid(i)) c.fail(ErrorCode::invalid_argument, path, "category dictionaries cannot contain nulls");
    if (categories.kind() == ColumnKind::string) {
        std::unordered_set<std::string_view> seen;
        for (std::size_t i = 0; i < categories.size(); ++i)
            if (!seen.insert(categories.string(i)).second) c.fail(ErrorCode::invalid_argument, path, "category dictionaries must contain unique values");
    } else {
        auto array = categories.values();
        std::optional<Array> selected;
        std::optional<DenseBytesView<const std::byte>> bytes;
        try { bytes.emplace(array.bytes()); }
        catch (const Error& error) {
            if (error.code() != ErrorCode::non_contiguous) throw;
            selected.emplace(array.copy(Validation::none)); bytes.emplace(selected->view().bytes());
        }
        visit_dtype(array.dtype(), [&]<class T> {
            std::unordered_set<T> seen;
            for (std::size_t i = 0; i < categories.size(); ++i) {
                T value;
                std::memcpy(&value, bytes->data() + static_cast<std::ptrdiff_t>(i) * bytes->stride_bytes(0), sizeof(T));
                if constexpr (std::is_floating_point_v<T>)
                    if (std::isnan(value)) c.fail(ErrorCode::invalid_argument, path, "category dictionaries cannot contain NaN");
                if (!seen.insert(value).second) c.fail(ErrorCode::invalid_argument, path, "category dictionaries must contain unique values");
            }
        });
    }
}

void write_column(Context& c, hid_t parent, std::string_view key, ColumnView column, std::string_view path) try {
    if (column.kind() == ColumnKind::categorical) {
        auto group = create_group(c, parent, key, path);
        write_encoding(c, group, "categorical", "0.2.0", path);
        auto type = boolean_type(c, false, path), memory_type = boolean_type(c, true, path);
        auto space = space_for(c, {}, path);
        Handle ordered(c.id(H5Acreate2(group, "ordered", type, space, H5P_DEFAULT, H5P_DEFAULT), path, "create categorical ordering"), H5Aclose);
        const unsigned char value = column.ordered();
        c.check(H5Awrite(ordered, memory_type, &value), path, "write categorical ordering");
        const auto categories = column.categories();
        if (c.validation == Validation::full) {
            validate_dictionary(c, categories, child(path, "categories"));
            for (std::size_t i = 0; i < column.size(); ++i) {
                const auto code = column.code(i);
                if (code < -1 || (code >= 0 && static_cast<std::uint64_t>(code) >= categories.size()))
                    c.fail(ErrorCode::out_of_bounds, child(path, "codes"), "categorical code is outside dictionary bounds");
            }
        }
        write_array(c, group, "codes", column.codes(), child(path, "codes"));
        write_column(c, group, "categories", categories, child(path, "categories"));
        return;
    }
    if (column.nullable()) {
        const bool strings = column.kind() == ColumnKind::string;
        auto type = strings ? DType::boolean : column.values().dtype();
        if (!strings && (type == DType::float32 || type == DType::float64))
            c.fail(ErrorCode::unsupported_dtype, path, "nullable floating-point columns have no supported AnnData codec; nulls cannot be replaced by NaN without losing their distinction");
        auto group = create_group(c, parent, key, path);
        write_encoding(c, group, strings ? "nullable-string-array" : type == DType::boolean ? "nullable-boolean" : "nullable-integer", "0.1.0", path);
        if (strings) {
            write_string_attribute(c, group, "na-value", {column.string_missing_value() == StringMissingValue::NaN ? "NaN" : "NA"}, true, path);
            write_strings(c, group, "values", {column.size()}, [&](std::size_t i) { return column.string(i); }, child(path, "values"));
        } else write_array(c, group, "values", column.values(), child(path, "values"));
        auto mask = Array::empty<bool>({column.size()});
        auto out = mask.edit().dense<bool>();
        for (std::size_t i = 0; i < column.size(); ++i) out.data()[i] = !column.valid(i);
        write_array(c, group, "mask", mask.view(), child(path, "mask"));
        return;
    }
    if (column.kind() == ColumnKind::numeric) write_array(c, parent, key, column.values(), path);
    else write_strings(c, parent, key, {column.size()}, [&](std::size_t i) { return column.string(i); }, path);
} catch (const Error& error) { rethrow_at(c, error, path); }

void write_table(Context& c, hid_t parent, std::string_view key, TableView table, std::string_view path) {
    const auto names = table.column_names();
    if (table.index_name() == "_index")
        c.fail(ErrorCode::invalid_argument, child(path, "_index"), "explicit index name '_index' is reserved as the unnamed dataframe index sentinel");
    const std::string index_key = table.index_name().empty() ? "_index" : std::string(table.index_name());
    check_key(c, index_key, child(path, index_key));
    for (const auto& name : names) {
        check_key(c, name, child(path, name));
        if (name == "_index") c.fail(ErrorCode::invalid_argument, child(path, name), "'_index' is reserved for dataframe indexes");
        if (name == index_key) {
            const auto column = table.column(name);
            if (column.kind() != ColumnKind::string || column.nullable())
                c.fail(ErrorCode::invalid_argument, path, "index name collides with a column having a different representation");
            for (std::size_t i = 0; i < table.size(); ++i)
                if (column.string(i) != table.name(i)) c.fail(ErrorCode::invalid_argument, path, "index name collides with a column having different values");
        }
    }
    auto group = create_group(c, parent, key, path);
    write_encoding(c, group, "dataframe", "0.2.0", path);
    write_string_attribute(c, group, "_index", {index_key}, true, path);
    write_string_attribute(c, group, "column-order", names, false, path);
    write_strings(c, group, index_key, {table.size()}, [&](std::size_t i) { return table.name(i); }, child(path, index_key));
    for (const auto& name : names)
        if (name != index_key) write_column(c, group, name, table.column(name), child(path, name));
}

void write_null(Context& c, hid_t parent, std::string_view key, std::string_view path) {
    check_key(c, key, path);
    Handle space(c.id(H5Screate(H5S_NULL), path, "create null dataspace"), H5Sclose);
    auto links = link_properties(c, path);
    Handle dataset(c.id(H5Dcreate2(parent, std::string(key).c_str(), H5T_IEEE_F32LE, space, links, H5P_DEFAULT, H5P_DEFAULT), path, "create null element"), H5Dclose);
    write_encoding(c, dataset, "null", "0.1.0", path);
}

void write_element(Context& c, hid_t parent, std::string_view key, ElementView element, std::string_view path) {
    switch (element.kind()) {
        case ElementKind::null: write_null(c, parent, key, path); return;
        case ElementKind::array: write_array(c, parent, key, element.array(), path); return;
        case ElementKind::string_array: write_string_array(c, parent, key, element.string_array(), path); return;
        case ElementKind::column: write_column(c, parent, key, element.column(), path); return;
        case ElementKind::table: write_table(c, parent, key, element.table(), path); return;
        case ElementKind::string:
            write_strings(c, parent, key, {}, [&](std::size_t) { return element.string(); }, path, true); return;
        case ElementKind::mapping: {
            auto group = create_group(c, parent, key, path);
            write_encoding(c, group, "dict", "0.1.0", path);
            for (const auto& name : element.keys()) write_element(c, group, name, element.at(name), child(path, name));
            return;
        }
        case ElementKind::boolean: case ElementKind::signed_integer:
        case ElementKind::unsigned_integer: case ElementKind::floating: {
            const auto type = element.dtype();
            auto file_bool = type == DType::boolean ? boolean_type(c, false, path) : Handle{};
            auto memory_bool = type == DType::boolean ? boolean_type(c, true, path) : Handle{};
            const hid_t file_type = type == DType::boolean ? static_cast<hid_t>(file_bool) : file_numeric_type(type, output_order(c, element.serialization_byte_order()));
            const hid_t memory_type = type == DType::boolean ? static_cast<hid_t>(memory_bool) : native_type(type);
            auto dataset = create_dataset(c, parent, key, file_type, {}, dtype_size(type), path);
            write_encoding(c, dataset, "numeric-scalar", "0.2.0", path);
            visit_dtype(type, [&]<class T> {
                const T value = element.scalar<T>();
                c.check(H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value), path, "write numerical scalar");
            });
            return;
        }
    }
    c.fail(ErrorCode::unsupported_encoding, path, "unsupported metadata element kind");
}

template<class Get>
void write_array_mapping(Context& c, hid_t root, std::string_view name, const std::vector<std::string>& keys, Get get) {
    const auto path = child("/", name);
    auto group = create_group(c, root, name, path);
    write_encoding(c, group, "dict", "0.1.0", path);
    for (const auto& key : keys) write_array(c, group, key, get(key), child(path, key));
}

template<class Get>
void write_element_mapping(Context& c, hid_t root, std::string_view name, const std::vector<std::string>& keys, Get get, std::string_view parent_path = "/") {
    const auto path = child(parent_path, name);
    auto group = create_group(c, root, name, path);
    write_encoding(c, group, "dict", "0.1.0", path);
    for (const auto& key : keys) write_element(c, group, key, get(key), child(path, key));
}

void write_dataset(Context& c, hid_t root, const AnnDataView& data) {
    write_encoding(c, root, "anndata", "0.1.0", "/");
    write_table(c, root, "obs", data.obs(), "/obs");
    write_table(c, root, "var", data.var(), "/var");
    if (data.has_x()) write_array(c, root, "X", data.x(), "/X");
    write_array_mapping(c, root, "layers", data.layer_names(), [&](const auto& key) { return data.layer(key); });
    write_array_mapping(c, root, "obsp", data.obsp_names(), [&](const auto& key) { return data.obsp(key); });
    write_array_mapping(c, root, "varp", data.varp_names(), [&](const auto& key) { return data.varp(key); });
    write_element_mapping(c, root, "obsm", data.obsm_names(), [&](const auto& key) { return data.obsm(key); });
    write_element_mapping(c, root, "varm", data.varm_names(), [&](const auto& key) { return data.varm(key); });
    write_element(c, root, "uns", data.uns(), "/uns");
    if (data.has_raw()) {
        auto raw = create_group(c, root, "raw", "/raw");
        write_encoding(c, raw, "raw", "0.1.0", "/raw");
        if (data.has_raw_x()) write_array(c, raw, "X", data.raw_x(), "/raw/X");
        else write_null(c, raw, "X", "/raw/X");
        write_table(c, raw, "var", data.raw_var(), "/raw/var");
        write_element_mapping(c, raw, "varm", data.raw_varm_names(), [&](const auto& key) { return data.raw_varm(key); }, "/raw");
    } else write_null(c, root, "raw", "/raw");
}

std::string path_string(std::string_view path) {
    if (path.empty() || path.find('\0') != std::string_view::npos)
        throw Error(ErrorCode::invalid_argument, "file", "file path must be nonempty and contain no NUL bytes");
    return std::string(path);
}

[[noreturn]] void contextualize(const Context& c, const Error& error) {
    if (error.path().starts_with(c.filename + ":")) throw;
    std::string path = error.path();
    if (path.empty() || path.front() != '/') path = "/" + path;
    c.fail(error.code(), path, error.what());
}

} // namespace

AnnData read_h5ad(std::string_view path, ReadOptions options) {
    Context c{path_string(path), options.validation, {}, {}};
    hid_t value;
    H5E_BEGIN_TRY { value = H5Fopen(c.filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT); } H5E_END_TRY;
    Handle file(c.id(value, "/", "open H5AD file"), H5Fclose);
    try { return read_dataset(c, file); }
    catch (const Error& error) { contextualize(c, error); }
}

BackedAnnData open_h5ad(std::string_view path, BackedOptions options) {
    return detail::open_h5ad_backed(path, options);
}

bool hdf5_library_threadsafe() {
    hbool_t result = false;
    if (H5is_library_threadsafe(&result) < 0) throw Error(ErrorCode::io_error, "HDF5", "cannot inspect HDF5 thread-safety capability");
    return result != 0;
}

void write_h5ad(std::string_view path, const BackedAnnData& data, WriteOptions options) {
    write_h5ad(path, data.view(), options);
}

void write_h5ad(std::string_view path, const BackedAnnDataView& data, WriteOptions options) {
    std::optional<ByteOrder> order;
    if (options.byte_order == FileByteOrder::native) order = ByteOrder::native;
    else if (options.byte_order == FileByteOrder::little) order = ByteOrder::little;
    else if (options.byte_order == FileByteOrder::big) order = ByteOrder::big;
    detail::write_h5ad_backed(path, data, options.mode == WriteMode::exclusive,
                             options.compression == Compression::gzip, options.gzip_level, order, options.validation);
}

void write_h5ad(std::string_view path, const AnnData& data, WriteOptions options) {
    write_h5ad(path, data.view(), options);
}

void write_h5ad(std::string_view path, const AnnDataView& data, WriteOptions options) {
    Context c{path_string(path), options.validation, options, {}};
    if (options.gzip_level < 0 || options.gzip_level > 9)
        c.fail(ErrorCode::invalid_argument, "/", "gzip_level must be between 0 and 9");
    const unsigned flags = options.mode == WriteMode::exclusive ? H5F_ACC_EXCL : H5F_ACC_TRUNC;
    hid_t value;
    H5E_BEGIN_TRY { value = H5Fcreate(c.filename.c_str(), flags, H5P_DEFAULT, H5P_DEFAULT); } H5E_END_TRY;
    Handle file(c.id(value, "/", "create H5AD file"), H5Fclose);
    try {
        write_dataset(c, file, data);
        c.check(H5Fflush(file, H5F_SCOPE_LOCAL), "/", "flush H5AD file");
    } catch (const Error& error) { contextualize(c, error); }
}

} // namespace anndata
