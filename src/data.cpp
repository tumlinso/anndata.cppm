module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

module anndatapp.core;

namespace anndatapp {
namespace detail {

struct ColumnImpl {
    ColumnKind kind{ColumnKind::numeric};
    std::optional<Array> values;
    std::vector<std::size_t> offsets;
    std::vector<char> bytes;
    std::vector<std::uint8_t> validity;
    bool nullable{false};
    StringMissingValue string_missing_value{StringMissingValue::NA};
    std::shared_ptr<ColumnImpl> categories;
    bool ordered{false};
    std::size_t size() const {
        return kind == ColumnKind::string ? offsets.size() - 1 : values->extent(0);
    }
};

struct ColumnViewImpl {
    std::shared_ptr<ColumnImpl> source;
    Axis rows;
    std::vector<std::size_t> category_rows;
    std::vector<std::int64_t> recode;
    bool pruned{false};
    std::size_t size() const { return rows.extent(source->size()); }
    std::size_t row(std::size_t i) const {
        if (i >= size()) throw Error(ErrorCode::out_of_bounds, "column", "row is out of bounds");
        return rows.index(i);
    }
};

struct TableImpl {
    std::shared_ptr<std::vector<std::string>> names;
    std::string index_name;
    std::vector<std::pair<std::string, Column>> columns;
};

struct TableViewImpl {
    std::shared_ptr<const std::vector<std::string>> names;
    std::string index_name;
    Axis rows;
    std::vector<std::pair<std::string, ColumnView>> columns;
};

struct ElementImpl {
    ElementKind kind{ElementKind::null};
    DType scalar_dtype{DType::boolean};
    ByteOrder scalar_byte_order{ByteOrder::native};
    bool boolean{};
    std::int64_t signed_integer{};
    std::uint64_t unsigned_integer{};
    double floating{};
    std::string string;
    std::optional<Array> array;
    std::optional<ArrayView> view_array;
    std::optional<StringArray> string_array;
    std::optional<StringArrayView> view_string_array;
    std::optional<RecordArray> record_array;
    std::optional<Column> column;
    std::optional<ColumnView> view_column;
    std::optional<Table> table;
    std::optional<TableView> view_table;
    std::vector<std::pair<std::string, Element>> mapping;
};

struct RawData {
    std::optional<Array> x;
    Table var;
    std::vector<std::pair<std::string, Element>> varm;
};

struct AnnDataImpl {
    std::size_t n_obs{}, n_vars{};
    std::optional<Array> x;
    Table obs, var;
    std::vector<std::pair<std::string, Array>> layers, obsp, varp;
    std::vector<std::pair<std::string, Element>> obsm, varm;
    Element uns{Element::mapping()};
    std::optional<RawData> raw;
};

struct AnnDataViewImpl {
    // All aligned slots share these absolute maps into the original snapshot.
    std::shared_ptr<AnnDataViewImpl> source;
    Axis obs_axis, var_axis;
    std::size_t n_obs{}, n_vars{};
    std::optional<ArrayView> x;
    TableView obs, var;
    std::vector<std::pair<std::string, ArrayView>> layers, obsp, varp;
    std::vector<std::pair<std::string, ElementView>> obsm, varm;
    ElementView uns;
    bool raw_present{false};
    std::optional<ArrayView> raw_x;
    TableView raw_var;
    std::vector<std::pair<std::string, ElementView>> raw_varm;
};

} // namespace detail

namespace {

[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}

template<class T>
void validate_at(const T& value, std::string_view path) {
    try { value.validate(); }
    catch (const Error& e) {
        const auto& inner = e.path();
        const bool generic = inner == "array" || inner == "column" || inner == "table" || inner == "element";
        throw Error(e.code(), generic || inner.empty() ? std::string(path) : std::string(path) + "/" + inner, e.what());
    }
}

template<class V>
auto find_key(V& values, std::string_view key) {
    return std::find_if(values.begin(), values.end(), [&](const auto& p) { return p.first == key; });
}

template<class V>
auto& require_key(V& values, std::string_view key, std::string_view path) {
    auto it = find_key(values, key);
    if (it == values.end()) fail(ErrorCode::out_of_bounds, std::string(path) + "/" + std::string(key), "key does not exist");
    return it->second;
}

template<class V, class T>
void put(V& values, std::string key, T value) {
    auto it = find_key(values, key);
    if (it == values.end()) values.emplace_back(std::move(key), std::move(value));
    else it->second = std::move(value);
}

template<class V>
std::vector<std::string> keys(const V& values) {
    std::vector<std::string> out;
    out.reserve(values.size());
    for (const auto& p : values) out.push_back(p.first);
    return out;
}

bool same_index(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

template<class F>
decltype(auto) visit_numeric(DType dtype, F&& f) {
    switch (dtype) {
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
    fail(ErrorCode::dtype_mismatch, "column", "unsupported dtype");
}

std::int64_t read_code(const ArrayView& a, std::size_t i) {
    const auto v = a.bytes();
    const auto* p = v.data() + static_cast<std::ptrdiff_t>(i) * v.stride_bytes(0);
    switch (a.dtype()) {
        case DType::int8: { std::int8_t value; std::memcpy(&value, p, sizeof value); return value; }
        case DType::uint8: { std::uint8_t value; std::memcpy(&value, p, sizeof value); return value; }
        case DType::int16: { std::int16_t value; std::memcpy(&value, p, sizeof value); return value; }
        case DType::uint16: { std::uint16_t value; std::memcpy(&value, p, sizeof value); return value; }
        case DType::int32: { std::int32_t value; std::memcpy(&value, p, sizeof value); return value; }
        case DType::uint32: { std::uint32_t value; std::memcpy(&value, p, sizeof value); return value; }
        case DType::int64: { std::int64_t value; std::memcpy(&value, p, sizeof value); return value; }
        case DType::uint64: {
            std::uint64_t value;
            std::memcpy(&value, p, sizeof value);
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                fail(ErrorCode::out_of_bounds, "categorical/codes", "unsigned category code exceeds the supported dictionary index range");
            return static_cast<std::int64_t>(value);
        }
        default: fail(ErrorCode::dtype_mismatch, "categorical/codes", "categorical codes must be signed or unsigned integers");
    }
}

Array make_codes(const ArrayView& selected, const std::vector<std::int64_t>& values) {
    auto out = selected.copy();
    auto bytes = out.edit().bytes();
    visit_numeric(selected.dtype(), [&]<class T>() {
        if constexpr (!std::is_integral_v<T> || std::is_same_v<T, bool>) {
            fail(ErrorCode::dtype_mismatch, "categorical/codes", "categorical codes must be signed or unsigned integers");
        } else {
            for (std::size_t i = 0; i < values.size(); ++i) {
                if constexpr (std::is_unsigned_v<T>) {
                    if (values[i] < 0 || static_cast<std::uint64_t>(values[i]) > std::numeric_limits<T>::max())
                        fail(ErrorCode::out_of_bounds, "categorical/codes", "recoded category value does not fit the original unsigned dtype");
                } else {
                    if (values[i] < -1 || values[i] > std::numeric_limits<T>::max())
                        fail(ErrorCode::out_of_bounds, "categorical/codes", "recoded category value does not fit the original signed dtype");
                }
                const auto value = static_cast<T>(values[i]);
                std::memcpy(bytes.data() + static_cast<std::ptrdiff_t>(i) * bytes.stride_bytes(0), &value, sizeof value);
            }
        }
    });
    return out;
}

Axis compose_rows(const Axis& parent, const Axis& child, std::size_t n, std::string_view path) {
    try { return detail::compose_axis(parent, child, n); }
    catch (const Error& e) { fail(e.code(), path, "selected row is out of bounds"); }
}

bool valid_utf8_string(std::string_view value) noexcept {
    const auto* bytes=reinterpret_cast<const unsigned char*>(value.data());
    std::size_t i=0;
    while(i<value.size()) {
        const auto first=bytes[i++];
        if(first<0x80)continue;
        unsigned remaining=0;std::uint32_t code=0,minimum=0;
        if(first>=0xc2 && first<=0xdf){remaining=1;code=first&0x1f;minimum=0x80;}
        else if(first>=0xe0 && first<=0xef){remaining=2;code=first&0x0f;minimum=0x800;}
        else if(first>=0xf0 && first<=0xf4){remaining=3;code=first&0x07;minimum=0x10000;}
        else return false;
        if(remaining>value.size()-i)return false;
        while(remaining--){const auto next=bytes[i++];if((next&0xc0)!=0x80)return false;code=(code<<6)|(next&0x3f);}
        if(code<minimum||code>0x10ffff||(code>=0xd800&&code<=0xdfff))return false;
    }
    return true;
}
void validate_packed_strings(const std::vector<char>& bytes,const std::vector<std::size_t>& offsets) {
    if(offsets.empty()||offsets.front()!=0)
        fail(ErrorCode::invalid_argument,"column/strings","packed offsets must start at zero");
    if(bytes.size()>static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        fail(ErrorCode::invalid_argument,"column/strings","packed bytes exceed addressable range");
    for(std::size_t i=1;i<offsets.size();++i) {
        const auto first=offsets[i-1],last=offsets[i];
        if(last<first||last>bytes.size())fail(ErrorCode::invalid_argument,"column/strings","packed offsets are nonmonotonic or exceed byte size");
        if(last!=first && !valid_utf8_string(std::string_view(bytes.data()+first,last-first)))
            fail(ErrorCode::invalid_argument,"column/strings","string is not valid UTF-8");
    }
    if(offsets.back()!=bytes.size())fail(ErrorCode::invalid_argument,"column/strings","final packed offset differs from byte size");
}

void check_validity(std::size_t n, const std::vector<std::uint8_t>& validity) {
    if (!validity.empty() && validity.size() != n)
        fail(ErrorCode::shape_mismatch, "column/validity", "validity length differs from column length");
    for (auto v : validity) if (v > 1)
        fail(ErrorCode::invalid_argument, "column/validity", "validity entries must be zero or one");
}

void check_category_dictionary(const ColumnView& cats) {
    for (std::size_t i = 0; i < cats.size(); ++i)
        if (!cats.valid(i)) fail(ErrorCode::invalid_argument, "categorical/categories", "category dictionaries cannot contain nulls");
    if (cats.kind() == ColumnKind::string) {
        std::unordered_set<std::string_view> seen;
        for (std::size_t i = 0; i < cats.size(); ++i)
            if (!seen.insert(cats.string(i)).second)
                fail(ErrorCode::invalid_argument, "categorical/categories", "categories must be unique");
    } else {
        auto a = cats.values();
        visit_numeric(a.dtype(), [&]<class T>() {
            auto v = a.bytes();
            std::unordered_set<T> seen;
            for (std::size_t i = 0; i < cats.size(); ++i) {
                T value;
                std::memcpy(&value, v.data() + static_cast<std::ptrdiff_t>(i) * v.stride_bytes(0), sizeof value);
                if constexpr (std::is_floating_point_v<T>)
                    if (std::isnan(value)) fail(ErrorCode::invalid_argument, "categorical/categories", "categories cannot contain NaN");
                if (!seen.insert(value).second)
                    fail(ErrorCode::invalid_argument, "categorical/categories", "categories must be unique");
            }
        });
    }
}

void check_kind(ElementKind actual, ElementKind wanted) {
    if (actual != wanted) fail(ErrorCode::invalid_argument, "element", "element has a different kind");
}

void check_matrix(const ArrayView& a, std::size_t rows, std::size_t cols, std::string_view path) {
    if (a.rank() != 2 || a.extent(0) != rows || a.extent(1) != cols)
        fail(ErrorCode::shape_mismatch, path, "matrix shape does not match aligned axes");
}

void check_first(const ArrayView& a, std::size_t rows, std::string_view path) {
    if (a.rank() < 1 || a.extent(0) != rows)
        fail(ErrorCode::shape_mismatch, path, "first dimension does not match aligned axis");
}

void check_first(const StringArrayView& a, std::size_t rows, std::string_view path) {
    if (a.rank() < 1 || a.extent(0) != rows)
        fail(ErrorCode::shape_mismatch, path, "first dimension does not match aligned axis");
}

void check_pairwise(const ArrayView& a, std::size_t n, std::string_view path) {
    if (a.rank() < 2 || a.extent(0) != n || a.extent(1) != n)
        fail(ErrorCode::shape_mismatch, path, "first two dimensions must match the aligned axis");
}

ArrayView select_first(const ArrayView& a, const Axis& rows) {
    if (rows.is_all()) return a;
    std::vector<Axis> axes(a.rank());
    axes[0] = rows;
    return a.select(axes.data(), axes.size());
}

StringArrayView select_first(const StringArrayView& a, const Axis& rows) {
    if (rows.is_all()) return a;
    if (a.rank() == 0) fail(ErrorCode::shape_mismatch, "string-array", "aligned string array must have at least one dimension");
    std::vector<Axis> axes(a.rank());
    axes[0] = rows;
    return a.select(axes.data(), axes.size());
}

ArrayView select_both(const ArrayView& a, const Axis& rows, const Axis& cols) {
    if (rows.is_all() && cols.is_all()) return a;
    std::vector<Axis> axes(a.rank());
    axes[0] = rows; axes[1] = cols;
    return a.select(axes.data(), axes.size());
}

} // namespace

Selection::Selection(all_t) {}
Selection::Selection(indices v) : kind_(Kind::indices), indices_(std::move(v.values)) {}
Selection::Selection(range v) : kind_(Kind::range), range_(v) {}
Selection::Selection(mask v) : kind_(Kind::mask), mask_(std::move(v.values)) {}
Selection::Selection(labels v) : kind_(Kind::labels), labels_(std::move(v.values)) {}
Selection::Selection(std::ptrdiff_t v) : kind_(Kind::indices), indices_{v} {}
Selection::Selection(std::string_view v) : kind_(Kind::label), labels_{std::string(v)} {}
Selection::Selection(const char* v) : Selection(std::string_view(v)) {}
bool Selection::is_all() const noexcept { return kind_ == Kind::all; }

Axis Selection::resolve(const std::vector<std::string>& names,
                                           std::string_view path) const {
    return resolve_impl(names.size(), path, &names, nullptr);
}
Axis Selection::resolve(const TableView& index, std::string_view path) const {
    return resolve_impl(index.size(), path, nullptr, &index);
}
Axis Selection::resolve_impl(std::size_t n, std::string_view path,
    const std::vector<std::string>* names, const TableView* table) const {
    auto name_at = [&](std::size_t i) -> std::string_view { return table ? table->name(i) : (*names)[i]; };
    if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        fail(ErrorCode::out_of_bounds, path, "axis exceeds supported signed index range");
    const auto length = static_cast<std::ptrdiff_t>(n);
    std::vector<std::size_t> out;
    switch (kind_) {
    case Kind::all: return Axis::all();
    case Kind::indices:
        out.reserve(indices_.size());
        for (auto i : indices_) {
            if (i < 0) i += length;
            if (i < 0 || i >= length) fail(ErrorCode::out_of_bounds, path, "position is out of bounds");
            out.push_back(static_cast<std::size_t>(i));
        }
        return detail::gather_axis(std::move(out));
    case Kind::mask:
        if (mask_.size() != n) fail(ErrorCode::shape_mismatch, path, "boolean selector length differs from axis length");
        for (std::size_t i = 0; i < n; ++i) {
            if (mask_[i] > 1) fail(ErrorCode::invalid_argument, path, "boolean selector entries must be zero or one");
            if (mask_[i]) out.push_back(i);
        }
        return detail::gather_axis(std::move(out));
    case Kind::range: {
        const auto step = range_.step;
        if (step == 0) fail(ErrorCode::invalid_argument, path, "slice step cannot be zero");
        if (step == std::numeric_limits<std::ptrdiff_t>::min())
            fail(ErrorCode::invalid_argument, path, "slice step magnitude is too large");
        auto norm = [&](std::optional<std::ptrdiff_t> v, bool start) {
            if (!v) return step > 0 ? (start ? std::ptrdiff_t{0} : length) : (start ? length - 1 : std::ptrdiff_t{-1});
            auto x = *v;
            if (x < 0) x += length;
            return step > 0 ? std::clamp(x, std::ptrdiff_t{0}, length)
                            : std::clamp(x, std::ptrdiff_t{-1}, length - 1);
        };
        const auto first = norm(range_.start, true), stop = norm(range_.stop, false);
        const auto span = step > 0 ? stop - first : first - stop;
        const auto magnitude = step > 0 ? step : -step;
        const auto count = span <= 0 ? std::size_t{0} :
            static_cast<std::size_t>((span - 1) / magnitude + 1);
        return Axis::stride(count ? static_cast<std::size_t>(first) : 0, count,
                            static_cast<std::size_t>(magnitude), step < 0);
    }
    case Kind::label:
        for (std::size_t i = 0; i < n; ++i) if (name_at(i) == labels_[0]) out.push_back(i);
        if (out.empty()) fail(ErrorCode::out_of_bounds, path, "label does not exist");
        return detail::gather_axis(std::move(out));
    case Kind::labels: {
        std::unordered_map<std::string_view, std::size_t> index;
        index.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            if (!index.emplace(name_at(i), i).second)
                fail(ErrorCode::invalid_argument, path, "label-list indexing requires unique axis names");
        out.reserve(labels_.size());
        for (const auto& label : labels_) {
            auto it = index.find(label);
            if (it == index.end()) fail(ErrorCode::out_of_bounds, path, "label does not exist: " + label);
            out.push_back(it->second);
        }
        return detail::gather_axis(std::move(out));
    }
    }
    return detail::gather_axis(std::move(out));
}

Column::Column() : impl_(std::make_shared<detail::ColumnImpl>()) {
    impl_->values.emplace(Array::empty<double>({0}));
}
Column::~Column() = default;
Column::Column(Column&&) noexcept = default;
Column& Column::operator=(Column&&) noexcept = default;
Column::Column(std::shared_ptr<detail::ColumnImpl> p) : impl_(std::move(p)) {}

Column Column::numeric(Array values, std::vector<std::uint8_t> validity) {
    if (values.rank() != 1 || values.layout() != Layout::dense)
        fail(ErrorCode::shape_mismatch, "column/values", "numeric columns require a one-dimensional dense array");
    check_validity(values.extent(0), validity);
    auto p = std::make_shared<detail::ColumnImpl>();
    p->values.emplace(std::move(values));
    p->validity = std::move(validity);
    p->nullable = !p->validity.empty();
    return Column(std::move(p));
}
Column Column::nullable_numeric(Array values, std::vector<std::uint8_t> validity) {
    if (values.rank() != 1 || validity.size() != values.extent(0))
        fail(ErrorCode::shape_mismatch, "column/validity", "nullable validity length differs from column length");
    auto c = numeric(std::move(values), std::move(validity));
    c.impl_->nullable = true;
    return c;
}

Column Column::strings(std::vector<std::string> values, std::vector<std::uint8_t> validity) {
    check_validity(values.size(), validity);
    auto p = std::make_shared<detail::ColumnImpl>();
    p->kind = ColumnKind::string;
    p->offsets.push_back(0);
    for (const auto& value : values) {
        p->bytes.insert(p->bytes.end(), value.begin(), value.end());
        p->offsets.push_back(p->bytes.size());
    }
    p->validity = std::move(validity);
    p->nullable = !p->validity.empty();
    return Column(std::move(p));
}

Column Column::strings(std::initializer_list<std::string_view> values, std::vector<std::uint8_t> validity) {
    std::vector<std::string> copied;
    copied.reserve(values.size());
    for (auto value : values) copied.emplace_back(value);
    return strings(std::move(copied), std::move(validity));
}
Column Column::nullable_strings(std::vector<std::string> values, std::vector<std::uint8_t> validity,
                                StringMissingValue missing) {
    if (validity.size() != values.size())
        fail(ErrorCode::shape_mismatch, "column/validity", "nullable validity length differs from column length");
    auto c = strings(std::move(values), std::move(validity));
    c.impl_->nullable = true;
    c.impl_->string_missing_value = missing;
    return c;
}
Column Column::nullable_strings(std::initializer_list<std::string_view> values,
                                std::vector<std::uint8_t> validity,
                                StringMissingValue missing) {
    std::vector<std::string> copied;
    copied.reserve(values.size());
    for (auto value : values) copied.emplace_back(value);
    return nullable_strings(std::move(copied), std::move(validity), missing);
}

Column Column::from_packed_strings(std::vector<char> bytes, std::vector<std::size_t> offsets,
                                   std::vector<std::uint8_t> validity, Validation validation) {
    if(offsets.empty())fail(ErrorCode::invalid_argument,"column/strings","packed offsets must include an initial zero");
    if(validation==Validation::full) {
        validate_packed_strings(bytes,offsets);
        check_validity(offsets.size()-1,validity);
    }
    auto p=std::make_shared<detail::ColumnImpl>();
    p->kind=ColumnKind::string;
    p->bytes=std::move(bytes);
    p->offsets=std::move(offsets);
    p->validity=std::move(validity);
    p->nullable=!p->validity.empty();
    return Column(std::move(p));
}
Column Column::from_nullable_packed_strings(std::vector<char> bytes, std::vector<std::size_t> offsets,
                                            std::vector<std::uint8_t> validity,
                                            StringMissingValue missing, Validation validation) {
    if(offsets.empty() || validity.size()!=offsets.size()-1)
        fail(ErrorCode::shape_mismatch,"column/validity","nullable validity length differs from packed row count");
    auto result=from_packed_strings(std::move(bytes),std::move(offsets),std::move(validity),validation);
    result.impl_->nullable=true;
    result.impl_->string_missing_value=missing;
    return result;
}

Column Column::categorical(Array codes, Column categories, bool ordered, Validation validation) {
    if (codes.rank() != 1 || codes.layout() != Layout::dense)
        fail(ErrorCode::shape_mismatch, "categorical/codes", "categorical codes require a one-dimensional dense array");
    if (categories.kind() == ColumnKind::categorical)
        fail(ErrorCode::invalid_argument, "categorical/categories", "category dictionaries cannot themselves be categorical");
    auto cats = categories.view();
    if (validation == Validation::full) check_category_dictionary(cats);
    auto cv = codes.view();
    // Signed codes use -1 for missing values. Unsigned codes have no missing
    // sentinel; retain their actual width instead of coercing or wrapping.
    if (cv.dtype() != DType::int8 && cv.dtype() != DType::uint8 &&
        cv.dtype() != DType::int16 && cv.dtype() != DType::uint16 &&
        cv.dtype() != DType::int32 && cv.dtype() != DType::uint32 &&
        cv.dtype() != DType::int64 && cv.dtype() != DType::uint64)
        fail(ErrorCode::dtype_mismatch, "categorical/codes", "categorical codes must be signed or unsigned integers");
    if (validation == Validation::full) {
        for (std::size_t i = 0; i < cv.extent(0); ++i) {
            auto code = read_code(cv, i);
            if (code < -1 || (code >= 0 && static_cast<std::uint64_t>(code) >= cats.size()))
                fail(ErrorCode::out_of_bounds, "categorical/codes", "category code is outside dictionary bounds");
        }
    }
    auto p = std::make_shared<detail::ColumnImpl>();
    p->kind = ColumnKind::categorical;
    p->values.emplace(std::move(codes));
    p->categories = std::move(categories.impl_);
    p->ordered = ordered;
    return Column(std::move(p));
}

ColumnView::ColumnView() = default;
ColumnView::ColumnView(std::shared_ptr<detail::ColumnViewImpl> p) : impl_(std::move(p)) {}
ColumnView Column::view() const {
    auto p = std::make_shared<detail::ColumnViewImpl>();
    p->source = impl_;
    return ColumnView(std::move(p));
}
std::size_t Column::size() const { return impl_->size(); }
ColumnKind Column::kind() const { return impl_->kind; }
Column Column::copy() const { return view().copy(); }
std::size_t ColumnView::size() const { return impl_ ? impl_->size() : 0; }
ColumnKind ColumnView::kind() const { return impl_ ? impl_->source->kind : ColumnKind::numeric; }
bool ColumnView::nullable() const {
    return impl_ && (kind() == ColumnKind::categorical || impl_->source->nullable);
}
bool ColumnView::valid(std::size_t row) const {
    if (!impl_) fail(ErrorCode::out_of_bounds, "column", "row is out of bounds");
    auto r = impl_->row(row);
    if (kind() == ColumnKind::categorical) return code(row) != -1;
    return impl_->source->validity.empty() || impl_->source->validity[r] != 0;
}
ArrayView ColumnView::values() const {
    if (!impl_) fail(ErrorCode::invalid_argument, "column", "column has no storage");
    if (kind() != ColumnKind::numeric) fail(ErrorCode::invalid_argument, "column", "column is not numeric");
    auto v = impl_->source->values->view();
    return impl_->rows.is_all() ? v : v.select({impl_->rows});
}
std::string_view ColumnView::string(std::size_t row) const {
    if (kind() != ColumnKind::string) fail(ErrorCode::invalid_argument, "column", "column is not a string column");
    auto r = impl_->row(row);
    const auto& s = *impl_->source;
    const auto n = s.offsets[r + 1] - s.offsets[r];
    return n == 0 ? std::string_view{} : std::string_view(s.bytes.data() + s.offsets[r], n);
}
StringMissingValue ColumnView::string_missing_value() const {
    if (kind() != ColumnKind::string) fail(ErrorCode::invalid_argument, "column", "column is not a string column");
    return impl_->source->string_missing_value;
}
bool ColumnView::ordered() const {
    if (kind() != ColumnKind::categorical) fail(ErrorCode::invalid_argument, "column", "column is not categorical");
    return impl_->source->ordered;
}
ColumnView ColumnView::categories() const {
    if (kind() != ColumnKind::categorical) fail(ErrorCode::invalid_argument, "column", "column is not categorical");
    auto p = std::make_shared<detail::ColumnViewImpl>();
    p->source = impl_->source->categories;
    if (impl_->pruned) p->rows = Axis::gather(impl_->category_rows.data(), impl_->category_rows.size());
    return ColumnView(std::move(p));
}
std::int64_t ColumnView::code(std::size_t row) const {
    if (kind() != ColumnKind::categorical) fail(ErrorCode::invalid_argument, "column", "column is not categorical");
    auto value = read_code(impl_->source->values->view(), impl_->row(row));
    if (value < -1 || (value >= 0 && static_cast<std::uint64_t>(value) >= impl_->source->categories->size()))
        fail(ErrorCode::out_of_bounds, "categorical/codes", "category code is outside dictionary bounds");
    if (value >= 0 && impl_->pruned) {
        const auto mapped = impl_->recode[static_cast<std::size_t>(value)];
        if (mapped < 0)
            fail(ErrorCode::invalid_argument, "categorical/codes", "source code changed outside the selected dictionary; create a fresh view");
        return mapped;
    }
    return value;
}
ArrayView ColumnView::codes() const {
    if (kind() != ColumnKind::categorical) fail(ErrorCode::invalid_argument, "column", "column is not categorical");
    bool identity_recode = true;
    for (std::size_t i = 0; i < impl_->category_rows.size(); ++i)
        if (impl_->category_rows[i] != i) { identity_recode = false; break; }
    if (!impl_->pruned || identity_recode) {
        if (impl_->pruned && impl_->category_rows.size() != impl_->source->categories->size())
            for (std::size_t i = 0; i < size(); ++i) (void)code(i);
        auto a = impl_->source->values->view();
        return impl_->rows.is_all() ? a : a.select({impl_->rows});
    }
    std::vector<std::int64_t> values(size());
    for (std::size_t i = 0; i < size(); ++i) values[i] = code(i);
    auto selected = impl_->source->values->view();
    if (!impl_->rows.is_all()) selected = selected.select({impl_->rows});
    return make_codes(selected, values).view();
}
ColumnView ColumnView::select(const Axis& rows, bool prune_categories) const {
    auto x = compose_rows(impl_ ? impl_->rows : Axis::all(), rows,
                          impl_ ? impl_->source->size() : 0, "column");
    if (!impl_) return ColumnView{};
    if (rows.is_all()) return select_all(prune_categories);
    auto p = std::make_shared<detail::ColumnViewImpl>(*impl_);
    p->rows = std::move(x);
    return ColumnView(std::move(p)).select_all(prune_categories);
}
ColumnView ColumnView::select_all(bool prune_categories) const {
    if (!impl_ || kind() != ColumnKind::categorical || !prune_categories) return *this;
    auto p = std::make_shared<detail::ColumnViewImpl>(*impl_);
    const auto ncats = p->source->categories->size();
    std::vector<bool> used(ncats, false);
    auto codes = p->source->values->view();
    for (std::size_t i = 0; i < size(); ++i) {
        (void)code(i); // Detect invalid or stale source codes before remapping.
        const auto original = read_code(codes, impl_->row(i));
        if (original >= 0) used[static_cast<std::size_t>(original)] = true;
    }
    p->pruned = true;
    p->recode.assign(ncats, -1);
    p->category_rows.clear();
    for (std::size_t i = 0; i < ncats; ++i) if (used[i]) {
        p->recode[i] = static_cast<std::int64_t>(p->category_rows.size());
        p->category_rows.push_back(i);
    }
    return ColumnView(std::move(p));
}
Column ColumnView::copy() const {
    if (!impl_) return Column{};
    if (kind() == ColumnKind::categorical)
        return Column::categorical(codes().copy(), categories().copy(), ordered());
    std::vector<std::uint8_t> validity;
    if (impl_->source->nullable) {
        validity.reserve(size());
        for (std::size_t i = 0; i < size(); ++i) validity.push_back(valid(i));
    }
    if (kind() == ColumnKind::numeric)
        return nullable() ? Column::nullable_numeric(values().copy(), std::move(validity))
                          : Column::numeric(values().copy());
    std::vector<std::string> strings;
    strings.reserve(size());
    for (std::size_t i = 0; i < size(); ++i) strings.emplace_back(string(i));
    return nullable() ? Column::nullable_strings(std::move(strings), std::move(validity), string_missing_value())
                      : Column::strings(std::move(strings));
}
ArrayEdit Column::edit_values() {
    if (kind() != ColumnKind::numeric) fail(ErrorCode::invalid_argument, "column", "column is not numeric");
    return impl_->values->edit();
}
ArrayEdit Column::unsafe_edit_codes() {
    if (kind() != ColumnKind::categorical) fail(ErrorCode::invalid_argument, "column", "column is not categorical");
    return impl_->values->edit();
}
void Column::set_valid(std::size_t row, bool valid) {
    if (row >= size()) fail(ErrorCode::out_of_bounds, "column/validity", "row is out of bounds");
    if (kind() == ColumnKind::categorical)
        fail(ErrorCode::invalid_argument, "column/validity", "categorical missing values use code -1");
    if (impl_->validity.empty()) impl_->validity.assign(size(), 1);
    impl_->nullable = true;
    impl_->validity[row] = valid;
}
void Column::validate() const {
    if (kind()==ColumnKind::string) validate_packed_strings(impl_->bytes,impl_->offsets);
    if (impl_->nullable && impl_->validity.size() != size())
        fail(ErrorCode::shape_mismatch, "column/validity", "nullable validity length differs from column length");
    check_validity(size(), impl_->validity);
    if (impl_->values) impl_->values->validate();
    if (kind() == ColumnKind::categorical) {
        Column categories(impl_->categories);
        validate_at(categories, "categorical/categories");
        check_category_dictionary(categories.view());
        const auto n = impl_->categories->size();
        for (std::size_t i = 0; i < size(); ++i) {
            auto c = read_code(impl_->values->view(), i);
            if (c < -1 || (c >= 0 && static_cast<std::uint64_t>(c) >= n))
                fail(ErrorCode::out_of_bounds, "categorical/codes", "category code is outside dictionary bounds");
        }
    }
}

Table::Table() : impl_(std::make_shared<detail::TableImpl>()) {
    impl_->names = std::make_shared<std::vector<std::string>>();
}
Table::~Table() = default;
Table::Table(Table&&) noexcept = default;
Table& Table::operator=(Table&&) noexcept = default;
Table Table::with_index(std::vector<std::string> names, std::string index_name) {
    Table t;
    t.impl_->names = std::make_shared<std::vector<std::string>>(std::move(names));
    t.impl_->index_name = std::move(index_name);
    return t;
}
Table Table::with_index(std::initializer_list<std::string_view> names, std::string index_name) {
    std::vector<std::string> n;
    n.reserve(names.size());
    for (auto s : names) n.emplace_back(s);
    return with_index(std::move(n), std::move(index_name));
}
Table Table::numbered(std::size_t rows) {
    std::vector<std::string> names;
    names.reserve(rows);
    for (std::size_t i = 0; i < rows; ++i) names.push_back(std::to_string(i));
    return with_index(std::move(names));
}
std::size_t Table::size() const { return impl_->names->size(); }
TableView::TableView() = default;
TableView::TableView(std::shared_ptr<detail::TableViewImpl> p) : impl_(std::move(p)) {}
TableView Table::view() const {
    auto p = std::make_shared<detail::TableViewImpl>();
    p->names = impl_->names;
    p->index_name = impl_->index_name;
    for (const auto& [name, col] : impl_->columns) p->columns.emplace_back(name, col.view());
    return TableView(std::move(p));
}
Table Table::copy() const { return view().copy(); }
void Table::set(std::string name, Column column) {
    if (column.size() != size()) fail(ErrorCode::shape_mismatch, "table/" + name, "column length differs from table length");
    put(impl_->columns, std::move(name), std::move(column));
}
void Table::erase(std::string_view name) {
    auto it = find_key(impl_->columns, name);
    if (it != impl_->columns.end()) impl_->columns.erase(it);
}
ColumnView Table::column(std::string_view name) const { return require_key(impl_->columns, name, "table").view(); }
ArrayEdit Table::edit_values(std::string_view name) { return require_key(impl_->columns, name, "table").edit_values(); }
void Table::set_names(std::vector<std::string> names, std::string index_name) {
    if (names.size() != size()) fail(ErrorCode::shape_mismatch, "table/index", "index length differs from table length");
    impl_->names = std::make_shared<std::vector<std::string>>(std::move(names));
    impl_->index_name = std::move(index_name);
}
void Table::validate() const {
    for (const auto& [name, c] : impl_->columns) {
        if (c.size() != size()) fail(ErrorCode::shape_mismatch, "table/" + name, "column length differs from table length");
        validate_at(c, name);
    }
}
std::size_t TableView::size() const { return impl_ ? impl_->rows.extent(impl_->names->size()) : 0; }
std::size_t TableView::n_columns() const { return impl_ ? impl_->columns.size() : 0; }
std::string_view TableView::index_name() const { return impl_ ? impl_->index_name : std::string_view{}; }
std::string_view TableView::name(std::size_t row) const {
    if (row >= size()) fail(ErrorCode::out_of_bounds, "table/index", "row is out of bounds");
    return (*impl_->names)[impl_->rows.index(row)];
}
std::vector<std::string> TableView::names() const {
    std::vector<std::string> out;
    out.reserve(size());
    for (std::size_t i = 0; i < size(); ++i) out.emplace_back(name(i));
    return out;
}
std::vector<std::string> TableView::column_names() const { return impl_ ? keys(impl_->columns) : std::vector<std::string>{}; }
bool TableView::contains(std::string_view column) const { return impl_ && find_key(impl_->columns, column) != impl_->columns.end(); }
ColumnView TableView::column(std::string_view column) const {
    if (!impl_) fail(ErrorCode::out_of_bounds, "table", "column does not exist");
    return require_key(impl_->columns, column, "table");
}
TableView TableView::select(const Axis& rows, bool prune_categories) const {
    auto x = compose_rows(impl_ ? impl_->rows : Axis::all(), rows,
                          impl_ ? impl_->names->size() : 0, "table");
    if (!impl_) return TableView{};
    if (rows.is_all()) return select_all(prune_categories);
    return with_rows(x, prune_categories);
}
TableView TableView::with_rows(const Axis& rows, bool prune_categories) const {
    if (!impl_) return TableView{};
    auto p = std::make_shared<detail::TableViewImpl>();
    p->names = impl_->names; p->index_name = impl_->index_name;
    p->rows = rows;
    for (const auto& [name, col] : impl_->columns) {
        auto c = std::make_shared<detail::ColumnViewImpl>(*col.impl_);
        c->rows = p->rows;
        p->columns.emplace_back(name, ColumnView(std::move(c)).select_all(prune_categories));
    }
    return TableView(std::move(p));
}
TableView TableView::select_all(bool prune_categories) const {
    if (!impl_) return *this;
    auto p = std::make_shared<detail::TableViewImpl>();
    p->names = impl_->names; p->index_name = impl_->index_name; p->rows = impl_->rows;
    for (const auto& [name, col] : impl_->columns)
        p->columns.emplace_back(name, col.select_all(prune_categories));
    return TableView(std::move(p));
}
Table TableView::copy() const {
    auto t = Table::with_index(names(), std::string(index_name()));
    if (impl_) for (const auto& [name, col] : impl_->columns) t.set(name, col.copy());
    return t;
}

Element::Element() : impl_(std::make_shared<detail::ElementImpl>()) {}
Element::Element(std::nullptr_t) : Element() {}
Element::~Element() = default;
Element::Element(Element&&) noexcept = default;
Element& Element::operator=(Element&&) noexcept = default;
Element::Element(std::shared_ptr<detail::ElementImpl> p) : impl_(std::move(p)) {}
Element::Element(bool v) : Element() { impl_->kind = ElementKind::boolean; impl_->boolean = v; impl_->scalar_dtype = DType::boolean; }
Element::Element(std::int64_t v) : Element() { impl_->kind = ElementKind::signed_integer; impl_->signed_integer = v; impl_->scalar_dtype = DType::int64; }
Element::Element(std::uint64_t v) : Element() { impl_->kind = ElementKind::unsigned_integer; impl_->unsigned_integer = v; impl_->scalar_dtype = DType::uint64; }
Element::Element(double v) : Element() { impl_->kind = ElementKind::floating; impl_->floating = v; impl_->scalar_dtype = DType::float64; }
Element::Element(std::string v) : Element() { impl_->kind = ElementKind::string; impl_->string = std::move(v); }
Element::Element(const char* v) : Element() {
    if (!v) fail(ErrorCode::invalid_argument, "element", "string pointer is null");
    impl_->kind = ElementKind::string; impl_->string = v;
}
Element::Element(Array v) : Element() { impl_->kind = ElementKind::array; impl_->array.emplace(std::move(v)); }
Element::Element(StringArray v) : Element() { impl_->kind = ElementKind::string_array; impl_->string_array.emplace(std::move(v)); }
Element::Element(RecordArray v) : Element() { impl_->kind = ElementKind::record_array; impl_->record_array.emplace(std::move(v)); }
Element::Element(Column v) : Element() { impl_->kind = ElementKind::column; impl_->column.emplace(std::move(v)); }
Element::Element(Table v) : Element() { impl_->kind = ElementKind::table; impl_->table.emplace(std::move(v)); }
Element Element::mapping() { Element e; e.impl_->kind = ElementKind::mapping; return e; }
Element Element::scalar_from_bytes(DType dtype, const void* bytes) {
    return visit_numeric(dtype, [&]<class T>() {
        T value;
        std::memcpy(&value, bytes, sizeof value);
        Element e;
        e.impl_->scalar_dtype = dtype;
        if constexpr (std::is_same_v<T, bool>) {
            e.impl_->kind = ElementKind::boolean; e.impl_->boolean = value;
        } else if constexpr (std::is_floating_point_v<T>) {
            e.impl_->kind = ElementKind::floating; e.impl_->floating = value;
        } else if constexpr (std::is_signed_v<T>) {
            e.impl_->kind = ElementKind::signed_integer; e.impl_->signed_integer = value;
        } else {
            e.impl_->kind = ElementKind::unsigned_integer; e.impl_->unsigned_integer = value;
        }
        return e;
    });
}
ElementView::ElementView() = default;
ElementView::ElementView(std::shared_ptr<detail::ElementImpl> p) : impl_(std::move(p)) {}
ElementView Element::view() const { return ElementView(impl_); }
Element Element::copy() const { return view().copy(); }
void Element::set(std::string key, Element value) {
    check_kind(impl_->kind, ElementKind::mapping);
    put(impl_->mapping, std::move(key), std::move(value));
}
void Element::set_serialization_byte_order(ByteOrder byte_order) {
    if (impl_->kind == ElementKind::array) {
        impl_->array->set_serialization_byte_order(byte_order);
        return;
    }
    if (impl_->kind != ElementKind::boolean && impl_->kind != ElementKind::signed_integer &&
        impl_->kind != ElementKind::unsigned_integer && impl_->kind != ElementKind::floating)
        fail(ErrorCode::invalid_argument, "element", "storage byte order applies to numeric scalars and arrays");
    impl_->scalar_byte_order = byte_order;
}
void Element::erase(std::string_view key) {
    check_kind(impl_->kind, ElementKind::mapping);
    auto it = find_key(impl_->mapping, key);
    if (it != impl_->mapping.end()) impl_->mapping.erase(it);
}
ElementView Element::at(std::string_view key) const { return view().at(key); }
void Element::validate() const {
    if (impl_->array) impl_->array->validate();
    if (impl_->string_array) impl_->string_array->validate();
    if (impl_->record_array) impl_->record_array->validate();
    if (impl_->column) impl_->column->validate();
    if (impl_->table) impl_->table->validate();
    for (const auto& [name, value] : impl_->mapping) validate_at(value, name);
}
ElementKind ElementView::kind() const { return impl_ ? impl_->kind : ElementKind::null; }
DType ElementView::dtype() const {
    if (kind() != ElementKind::boolean && kind() != ElementKind::signed_integer &&
        kind() != ElementKind::unsigned_integer && kind() != ElementKind::floating)
        fail(ErrorCode::dtype_mismatch, "element", "element is not a numerical scalar");
    return impl_->scalar_dtype;
}
bool ElementView::boolean() const { check_kind(kind(), ElementKind::boolean); return impl_->boolean; }
std::int64_t ElementView::signed_integer() const { check_kind(kind(), ElementKind::signed_integer); return impl_->signed_integer; }
std::uint64_t ElementView::unsigned_integer() const { check_kind(kind(), ElementKind::unsigned_integer); return impl_->unsigned_integer; }
double ElementView::floating() const { check_kind(kind(), ElementKind::floating); return impl_->floating; }
std::string_view ElementView::string() const { check_kind(kind(), ElementKind::string); return impl_->string; }
ByteOrder ElementView::serialization_byte_order() const {
    if (kind() == ElementKind::array) return array().serialization_byte_order();
    if (kind() != ElementKind::boolean && kind() != ElementKind::signed_integer &&
        kind() != ElementKind::unsigned_integer && kind() != ElementKind::floating)
        return ByteOrder::native;
    return impl_->scalar_byte_order;
}
ArrayView ElementView::array() const {
    check_kind(kind(), ElementKind::array);
    return impl_->view_array ? *impl_->view_array : impl_->array->view();
}
StringArrayView ElementView::string_array() const {
    check_kind(kind(), ElementKind::string_array);
    return impl_->view_string_array ? *impl_->view_string_array : impl_->string_array->view();
}
RecordArrayView ElementView::record_array() const {
    check_kind(kind(), ElementKind::record_array);
    return impl_->record_array->view();
}
ColumnView ElementView::column() const {
    check_kind(kind(), ElementKind::column);
    return impl_->view_column ? *impl_->view_column : impl_->column->view();
}
TableView ElementView::table() const {
    check_kind(kind(), ElementKind::table);
    return impl_->view_table ? *impl_->view_table : impl_->table->view();
}
std::vector<std::string> ElementView::keys() const {
    check_kind(kind(), ElementKind::mapping);
    return anndatapp::keys(impl_->mapping);
}
bool ElementView::contains(std::string_view key) const {
    check_kind(kind(), ElementKind::mapping);
    return find_key(impl_->mapping, key) != impl_->mapping.end();
}
ElementView ElementView::at(std::string_view key) const {
    check_kind(kind(), ElementKind::mapping);
    return require_key(impl_->mapping, key, "element").view();
}
Element ElementView::copy() const {
    switch (kind()) {
    case ElementKind::null: return Element{};
    case ElementKind::boolean:
    case ElementKind::signed_integer:
    case ElementKind::unsigned_integer:
    case ElementKind::floating: {
        Element e;
        e.impl_->kind = kind(); e.impl_->scalar_dtype = impl_->scalar_dtype;
        e.impl_->scalar_byte_order = impl_->scalar_byte_order;
        e.impl_->boolean = impl_->boolean; e.impl_->signed_integer = impl_->signed_integer;
        e.impl_->unsigned_integer = impl_->unsigned_integer; e.impl_->floating = impl_->floating;
        return e;
    }
    case ElementKind::string: return Element(std::string(string()));
    case ElementKind::array: return Element(array().copy());
    case ElementKind::string_array: return Element(string_array().copy());
    case ElementKind::record_array: return Element(record_array().copy());
    case ElementKind::column: return Element(column().copy());
    case ElementKind::table: return Element(table().copy());
    case ElementKind::mapping: {
        auto e = Element::mapping();
        for (const auto& [name, value] : impl_->mapping) e.set(name, value.copy());
        return e;
    }
    }
    return Element{};
}

AnnData::AnnData() : impl_(std::make_unique<detail::AnnDataImpl>()) {}
AnnData::~AnnData() = default;
AnnData::AnnData(AnnData&&) noexcept = default;
AnnData& AnnData::operator=(AnnData&&) noexcept = default;
AnnData AnnData::empty(std::size_t n_obs, std::size_t n_vars) {
    AnnData a;
    a.impl_->n_obs = n_obs; a.impl_->n_vars = n_vars;
    a.impl_->obs = Table::numbered(n_obs); a.impl_->var = Table::numbered(n_vars);
    return a;
}
std::size_t AnnData::n_obs() const { return impl_->n_obs; }
std::size_t AnnData::n_vars() const { return impl_->n_vars; }
bool AnnData::has_x() const { return impl_->x.has_value(); }
ArrayView AnnData::x() const {
    if (!has_x()) fail(ErrorCode::invalid_argument, "X", "X is absent");
    return impl_->x->view();
}
ArrayEdit AnnData::edit_x() {
    if (!has_x()) fail(ErrorCode::invalid_argument, "X", "X is absent");
    return impl_->x->edit();
}
void AnnData::set_x(Array x) { check_matrix(x.view(), n_obs(), n_vars(), "X"); impl_->x.emplace(std::move(x)); }
void AnnData::clear_x() { impl_->x.reset(); }
TableView AnnData::obs() const { return impl_->obs.view(); }
TableView AnnData::var() const { return impl_->var.view(); }
ColumnView AnnData::obs_column(std::string_view name) const { return impl_->obs.column(name); }
ColumnView AnnData::var_column(std::string_view name) const { return impl_->var.column(name); }
void AnnData::set_obs(Table t) {
    if (t.size() != n_obs()) fail(ErrorCode::shape_mismatch, "obs", "table length differs from observation axis");
    for (auto& [name, e] : impl_->obsm) if (e.impl_->kind == ElementKind::table)
        { e.impl_->table->impl_->names = t.impl_->names; e.impl_->table->impl_->index_name = t.impl_->index_name; }
    impl_->obs = std::move(t);
}
void AnnData::set_var(Table t) {
    if (t.size() != n_vars()) fail(ErrorCode::shape_mismatch, "var", "table length differs from variable axis");
    for (auto& [name, e] : impl_->varm) if (e.impl_->kind == ElementKind::table)
        { e.impl_->table->impl_->names = t.impl_->names; e.impl_->table->impl_->index_name = t.impl_->index_name; }
    impl_->var = std::move(t);
}
void AnnData::set_obs_column(std::string name, Column c) {
    if (c.size() != n_obs()) fail(ErrorCode::shape_mismatch, "obs/" + name, "column length differs from observation axis");
    impl_->obs.set(std::move(name), std::move(c));
}
void AnnData::set_var_column(std::string name, Column c) {
    if (c.size() != n_vars()) fail(ErrorCode::shape_mismatch, "var/" + name, "column length differs from variable axis");
    impl_->var.set(std::move(name), std::move(c));
}
ArrayEdit AnnData::edit_obs_values(std::string_view name) { return impl_->obs.edit_values(name); }
ArrayEdit AnnData::edit_var_values(std::string_view name) { return impl_->var.edit_values(name); }
void AnnData::set_layer(std::string name, Array a) {
    check_matrix(a.view(), n_obs(), n_vars(), "layers/" + name); put(impl_->layers, std::move(name), std::move(a));
}
void AnnData::set_obsm(std::string name, Array a) {
    check_first(a.view(), n_obs(), "obsm/" + name); put(impl_->obsm, std::move(name), Element(std::move(a)));
}
void AnnData::set_obsm(std::string name, StringArray a) {
    check_first(a.view(), n_obs(), "obsm/" + name); put(impl_->obsm, std::move(name), Element(std::move(a)));
}
void AnnData::set_obsm(std::string name, Table t) {
    if (!same_index(*t.impl_->names, *impl_->obs.impl_->names))
        fail(ErrorCode::shape_mismatch, "obsm/" + name, "table index differs from observation index");
    put(impl_->obsm, std::move(name), Element(std::move(t)));
}
void AnnData::set_varm(std::string name, Array a) {
    check_first(a.view(), n_vars(), "varm/" + name); put(impl_->varm, std::move(name), Element(std::move(a)));
}
void AnnData::set_varm(std::string name, StringArray a) {
    check_first(a.view(), n_vars(), "varm/" + name); put(impl_->varm, std::move(name), Element(std::move(a)));
}
void AnnData::set_varm(std::string name, Table t) {
    if (!same_index(*t.impl_->names, *impl_->var.impl_->names))
        fail(ErrorCode::shape_mismatch, "varm/" + name, "table index differs from variable index");
    put(impl_->varm, std::move(name), Element(std::move(t)));
}
void AnnData::set_obsp(std::string name, Array a) {
    check_pairwise(a.view(), n_obs(), "obsp/" + name); put(impl_->obsp, std::move(name), std::move(a));
}
void AnnData::set_varp(std::string name, Array a) {
    check_pairwise(a.view(), n_vars(), "varp/" + name); put(impl_->varp, std::move(name), std::move(a));
}
ArrayView AnnData::layer(std::string_view name) const { return require_key(impl_->layers, name, "layers").view(); }
ArrayEdit AnnData::edit_layer(std::string_view name) { return require_key(impl_->layers, name, "layers").edit(); }
ElementView AnnData::obsm(std::string_view name) const { return require_key(impl_->obsm, name, "obsm").view(); }
ElementView AnnData::varm(std::string_view name) const { return require_key(impl_->varm, name, "varm").view(); }
ArrayEdit AnnData::edit_obsm_array(std::string_view name) {
    auto& e = require_key(impl_->obsm, name, "obsm"); check_kind(e.impl_->kind, ElementKind::array); return e.impl_->array->edit();
}
ArrayEdit AnnData::edit_varm_array(std::string_view name) {
    auto& e = require_key(impl_->varm, name, "varm"); check_kind(e.impl_->kind, ElementKind::array); return e.impl_->array->edit();
}
StringArrayEdit AnnData::edit_obsm_string_array(std::string_view name) {
    auto& e = require_key(impl_->obsm, name, "obsm");
    check_kind(e.impl_->kind, ElementKind::string_array);
    return e.impl_->string_array->edit();
}
StringArrayEdit AnnData::edit_varm_string_array(std::string_view name) {
    auto& e = require_key(impl_->varm, name, "varm");
    check_kind(e.impl_->kind, ElementKind::string_array);
    return e.impl_->string_array->edit();
}
ArrayView AnnData::obsp(std::string_view name) const { return require_key(impl_->obsp, name, "obsp").view(); }
ArrayView AnnData::varp(std::string_view name) const { return require_key(impl_->varp, name, "varp").view(); }
ArrayEdit AnnData::edit_obsp(std::string_view name) { return require_key(impl_->obsp, name, "obsp").edit(); }
ArrayEdit AnnData::edit_varp(std::string_view name) { return require_key(impl_->varp, name, "varp").edit(); }
std::vector<std::string> AnnData::layer_names() const { return keys(impl_->layers); }
std::vector<std::string> AnnData::obsm_names() const { return keys(impl_->obsm); }
std::vector<std::string> AnnData::varm_names() const { return keys(impl_->varm); }
std::vector<std::string> AnnData::obsp_names() const { return keys(impl_->obsp); }
std::vector<std::string> AnnData::varp_names() const { return keys(impl_->varp); }
ElementView AnnData::uns() const { return impl_->uns.view(); }
void AnnData::set_uns(Element e) { check_kind(e.view().kind(), ElementKind::mapping); impl_->uns = std::move(e); }
void AnnData::set_uns(std::string name, Element e) { impl_->uns.set(std::move(name), std::move(e)); }
void AnnData::set_raw(Array x, Table var) {
    check_matrix(x.view(), n_obs(), var.size(), "raw/X");
    impl_->raw.emplace(detail::RawData{std::move(x), std::move(var), {}});
}
void AnnData::set_raw(Table var) {
    impl_->raw.emplace(detail::RawData{std::nullopt, std::move(var), {}});
}
void AnnData::set_raw_varm(std::string name, Array a) {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    check_first(a.view(), impl_->raw->var.size(), "raw/varm/" + name);
    put(impl_->raw->varm, std::move(name), Element(std::move(a)));
}
void AnnData::set_raw_varm(std::string name, StringArray a) {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    check_first(a.view(), impl_->raw->var.size(), "raw/varm/" + name);
    put(impl_->raw->varm, std::move(name), Element(std::move(a)));
}
void AnnData::set_raw_varm(std::string name, Table t) {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    if (!same_index(*t.impl_->names, *impl_->raw->var.impl_->names))
        fail(ErrorCode::shape_mismatch, "raw/varm/" + name, "table index differs from raw variable index");
    put(impl_->raw->varm, std::move(name), Element(std::move(t)));
}
bool AnnData::has_raw() const { return impl_->raw.has_value(); }
bool AnnData::has_raw_x() const { return has_raw() && impl_->raw->x.has_value(); }
ArrayView AnnData::raw_x() const {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    if (!has_raw_x()) fail(ErrorCode::invalid_argument, "raw/X", "raw X is absent");
    return impl_->raw->x->view();
}
TableView AnnData::raw_var() const {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    return impl_->raw->var.view();
}
ElementView AnnData::raw_varm(std::string_view name) const {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    return require_key(impl_->raw->varm, name, "raw/varm").view();
}
std::vector<std::string> AnnData::raw_varm_names() const {
    return has_raw() ? keys(impl_->raw->varm) : std::vector<std::string>{};
}
void AnnData::clear_raw() { impl_->raw.reset(); }

AnnDataView::AnnDataView() = default;
AnnDataView::AnnDataView(std::shared_ptr<detail::AnnDataViewImpl> p) : impl_(std::move(p)) {}
AnnDataView AnnData::view() const {
    auto p = std::make_shared<detail::AnnDataViewImpl>();
    p->n_obs = n_obs(); p->n_vars = n_vars();
    if (has_x()) p->x = x();
    p->obs = obs(); p->var = var();
    auto uns_snapshot = std::make_shared<detail::ElementImpl>();
    uns_snapshot->kind = ElementKind::mapping;
    for (const auto& [name, value] : impl_->uns.impl_->mapping)
        uns_snapshot->mapping.emplace_back(name, Element(value.impl_));
    p->uns = ElementView(std::move(uns_snapshot));
    for (const auto& [name, a] : impl_->layers) p->layers.emplace_back(name, a.view());
    for (const auto& [name, a] : impl_->obsp) p->obsp.emplace_back(name, a.view());
    for (const auto& [name, a] : impl_->varp) p->varp.emplace_back(name, a.view());
    auto snapshot_aligned = [&](const auto& source, auto& dest) {
        for (const auto& [name, e] : source) {
            auto snapshot = std::make_shared<detail::ElementImpl>();
            snapshot->kind = e.view().kind();
            if (snapshot->kind == ElementKind::array) snapshot->view_array = e.view().array();
            else if (snapshot->kind == ElementKind::string_array) snapshot->view_string_array = e.view().string_array();
            else snapshot->view_table = e.view().table();
            dest.emplace_back(name, ElementView(std::move(snapshot)));
        }
    };
    snapshot_aligned(impl_->obsm, p->obsm);
    snapshot_aligned(impl_->varm, p->varm);
    if (has_raw()) {
        p->raw_present = true;
        if (has_raw_x()) p->raw_x = raw_x();
        p->raw_var = raw_var();
        snapshot_aligned(impl_->raw->varm, p->raw_varm);
    }
    return AnnDataView(std::move(p));
}
AnnDataView AnnData::slice(Selection o, Selection v) const { return view().slice(std::move(o), std::move(v)); }
AnnData AnnData::copy() const { return view().copy(); }
std::size_t AnnDataView::n_obs() const { return impl_ ? impl_->n_obs : 0; }
std::size_t AnnDataView::n_vars() const { return impl_ ? impl_->n_vars : 0; }
bool AnnDataView::has_x() const { return impl_ && impl_->x.has_value(); }
ArrayView AnnDataView::x() const {
    if (!has_x()) fail(ErrorCode::invalid_argument, "X", "X is absent");
    return *impl_->x;
}
TableView AnnDataView::obs() const { return impl_ ? impl_->obs : TableView{}; }
TableView AnnDataView::var() const { return impl_ ? impl_->var : TableView{}; }
ColumnView AnnDataView::obs_column(std::string_view name) const { return obs().column(name); }
ColumnView AnnDataView::var_column(std::string_view name) const { return var().column(name); }
ArrayView AnnDataView::layer(std::string_view name) const {
    if (!impl_) fail(ErrorCode::out_of_bounds, "layers/" + std::string(name), "key does not exist");
    return require_key(impl_->layers, name, "layers");
}
ElementView AnnDataView::obsm(std::string_view name) const {
    if (!impl_) fail(ErrorCode::out_of_bounds, "obsm/" + std::string(name), "key does not exist");
    return require_key(impl_->obsm, name, "obsm");
}
ElementView AnnDataView::varm(std::string_view name) const {
    if (!impl_) fail(ErrorCode::out_of_bounds, "varm/" + std::string(name), "key does not exist");
    return require_key(impl_->varm, name, "varm");
}
ArrayView AnnDataView::obsp(std::string_view name) const {
    if (!impl_) fail(ErrorCode::out_of_bounds, "obsp/" + std::string(name), "key does not exist");
    return require_key(impl_->obsp, name, "obsp");
}
ArrayView AnnDataView::varp(std::string_view name) const {
    if (!impl_) fail(ErrorCode::out_of_bounds, "varp/" + std::string(name), "key does not exist");
    return require_key(impl_->varp, name, "varp");
}
std::vector<std::string> AnnDataView::layer_names() const { return impl_ ? keys(impl_->layers) : std::vector<std::string>{}; }
std::vector<std::string> AnnDataView::obsm_names() const { return impl_ ? keys(impl_->obsm) : std::vector<std::string>{}; }
std::vector<std::string> AnnDataView::varm_names() const { return impl_ ? keys(impl_->varm) : std::vector<std::string>{}; }
std::vector<std::string> AnnDataView::obsp_names() const { return impl_ ? keys(impl_->obsp) : std::vector<std::string>{}; }
std::vector<std::string> AnnDataView::varp_names() const { return impl_ ? keys(impl_->varp) : std::vector<std::string>{}; }
ElementView AnnDataView::uns() const { return impl_ ? impl_->uns : ElementView{}; }
bool AnnDataView::has_raw() const { return impl_ && impl_->raw_present; }
bool AnnDataView::has_raw_x() const { return has_raw() && impl_->raw_x.has_value(); }
ArrayView AnnDataView::raw_x() const {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    if (!has_raw_x()) fail(ErrorCode::invalid_argument, "raw/X", "raw X is absent");
    return *impl_->raw_x;
}
TableView AnnDataView::raw_var() const {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    return impl_->raw_var;
}
ElementView AnnDataView::raw_varm(std::string_view name) const {
    if (!has_raw()) fail(ErrorCode::invalid_argument, "raw", "raw is absent");
    return require_key(impl_->raw_varm, name, "raw/varm");
}
std::vector<std::string> AnnDataView::raw_varm_names() const {
    return has_raw() ? keys(impl_->raw_varm) : std::vector<std::string>{};
}

AnnDataView AnnDataView::slice(Selection obs_selection, Selection var_selection) const {
    const auto rows = obs_selection.resolve(obs(), "obs/index");
    const auto cols = var_selection.resolve(var(), "var/index");
    const auto base = impl_ && impl_->source ? impl_->source : impl_;
    const auto r = detail::compose_axis(impl_ ? impl_->obs_axis : Axis::all(), rows, base ? base->n_obs : 0);
    const auto c = detail::compose_axis(impl_ ? impl_->var_axis : Axis::all(), cols, base ? base->n_vars : 0);
    auto p = std::make_shared<detail::AnnDataViewImpl>();
    p->source = base; p->obs_axis = r; p->var_axis = c;
    p->n_obs = rows.extent(n_obs()); p->n_vars = cols.extent(n_vars());
    p->obs = obs().with_rows(r, true);
    p->var = var().with_rows(c, true);
    if (has_x()) p->x = select_both(*base->x, r, c);
    if (impl_) {
        for (const auto& [name, a] : base->layers) p->layers.emplace_back(name, select_both(a, r, c));
        for (const auto& [name, a] : base->obsp) p->obsp.emplace_back(name, select_both(a, r, r));
        for (const auto& [name, a] : base->varp) p->varp.emplace_back(name, select_both(a, c, c));
        auto aligned = [&](const auto& source, const auto& current, auto& dest, const Axis& selected) {
            for (std::size_t i = 0; i < source.size(); ++i) {
                const auto& [name, e] = source[i];
                auto ep = std::make_shared<detail::ElementImpl>();
                ep->kind = e.kind();
                if (ep->kind == ElementKind::array) ep->view_array = select_first(e.array(), selected);
                else if (ep->kind == ElementKind::string_array) ep->view_string_array = select_first(e.string_array(), selected);
                else ep->view_table = current[i].second.table().with_rows(selected, true);
                dest.emplace_back(name, ElementView(std::move(ep)));
            }
        };
        aligned(base->obsm, impl_->obsm, p->obsm, r);
        aligned(base->varm, impl_->varm, p->varm, c);
        // Preserve unknown uns content by sharing its bindings. Only the mapping
        // and categorical color arrays need adjusted bindings for this slice.
        auto uns_impl = std::make_shared<detail::ElementImpl>();
        uns_impl->kind = ElementKind::mapping;
        for (const auto& key : impl_->uns.keys()) {
            auto e = impl_->uns.at(key);
            uns_impl->mapping.emplace_back(key, Element(e.impl_));
        }
        auto fix_colors = [&](const TableView& before, const Axis& selected) {
            for (const auto& name : before.column_names()) {
                auto column = before.column(name);
                if (column.kind() != ColumnKind::categorical) continue;
                const auto color_key = name + "_colors";
                auto it = find_key(uns_impl->mapping, color_key);
                if (it == uns_impl->mapping.end()) continue;
                auto color = it->second.view();
                // Upstream's scalar branch promotes to one dimension and skips
                // the length/pruning branches, even for an empty selection.
                bool scalar_color = false;
                if (color.kind() == ElementKind::null) {
                    color = Element(Column::nullable_strings({""}, {0})).view();
                    scalar_color = true;
                } else if (color.kind() == ElementKind::string) {
                    color = Element(Column::strings({color.string()})).view();
                    scalar_color = true;
                } else if (color.kind() == ElementKind::boolean || color.kind() == ElementKind::signed_integer ||
                         color.kind() == ElementKind::unsigned_integer || color.kind() == ElementKind::floating) {
                    color = visit_numeric(color.dtype(), [&]<class T>() {
                        auto a = Array::empty<T>({1});
                        a.edit().template dense<T>().data()[0] = color.template scalar<T>();
                        return Element(std::move(a)).view();
                    });
                    scalar_color = true;
                } else if (color.kind() == ElementKind::array && color.array().rank() == 0) {
                    color = visit_numeric(color.array().dtype(), [&]<class T>() {
                        T value;
                        std::memcpy(&value, color.array().bytes().data(), sizeof value);
                        auto a = Array::empty<T>({1});
                        a.edit().template dense<T>().data()[0] = value;
                        return Element(std::move(a)).view();
                    });
                    scalar_color = true;
                } else if (color.kind() == ElementKind::string_array && color.string_array().rank() == 0) {
                    color = Element(StringArray::from_strings({1}, {color.string_array().string(0)})).view();
                    scalar_color = true;
                }
                if (scalar_color) { it->second = Element(color.impl_); continue; }
                const bool array_colors = color.kind() == ElementKind::array && color.array().rank() >= 1 &&
                    color.array().extent(0) == column.categories().size();
                const bool column_colors = color.kind() == ElementKind::column && color.column().size() == column.categories().size();
                const bool string_array_colors = color.kind() == ElementKind::string_array && color.string_array().rank() >= 1 &&
                    color.string_array().extent(0) == column.categories().size();
                if (!array_colors && !column_colors && !string_array_colors) {
                    uns_impl->mapping.erase(it);
                    continue;
                }
                // Derive positions in the current category dictionary using codes.
                std::vector<bool> used(column.categories().size(), false);
                for (std::size_t i = 0, n = selected.extent(before.size()); i < n; ++i) {
                    auto c = column.code(selected.index(i));
                    if (c >= 0) used[static_cast<std::size_t>(c)] = true;
                }
                std::vector<std::size_t> category_positions;
                for (std::size_t i = 0; i < used.size(); ++i) if (used[i]) category_positions.push_back(i);
                const auto categories = detail::gather_axis(std::move(category_positions));
                auto ep = std::make_shared<detail::ElementImpl>();
                if (array_colors) {
                    ep->kind = ElementKind::array;
                    ep->view_array = select_first(color.array(), categories);
                } else if (string_array_colors) {
                    ep->kind = ElementKind::string_array;
                    ep->view_string_array = select_first(color.string_array(), categories);
                } else {
                    ep->kind = ElementKind::column;
                    ep->view_column = color.column().select(categories, false);
                }
                it->second = Element(std::move(ep));
            }
        };
        fix_colors(obs(), rows);
        fix_colors(var(), cols);
        p->uns = ElementView(std::move(uns_impl));
        if (has_raw()) {
            p->raw_present = true;
            if (has_raw_x()) p->raw_x = select_first(*base->raw_x, r);
            p->raw_var = raw_var(); p->raw_varm = impl_->raw_varm;
        }
    }
    return AnnDataView(std::move(p));
}

AnnData AnnDataView::copy() const {
    auto a = AnnData::empty(n_obs(), n_vars());
    a.set_obs(obs().copy()); a.set_var(var().copy());
    if (has_x()) a.set_x(x().copy());
    if (impl_) {
        for (const auto& [name, v] : impl_->layers) a.set_layer(name, v.copy());
        for (const auto& [name, v] : impl_->obsp) a.set_obsp(name, v.copy());
        for (const auto& [name, v] : impl_->varp) a.set_varp(name, v.copy());
        for (const auto& [name, v] : impl_->obsm) {
            if (v.kind() == ElementKind::array) a.set_obsm(name, v.array().copy());
            else if (v.kind() == ElementKind::string_array) a.set_obsm(name, v.string_array().copy());
            else a.set_obsm(name, v.table().copy());
        }
        for (const auto& [name, v] : impl_->varm) {
            if (v.kind() == ElementKind::array) a.set_varm(name, v.array().copy());
            else if (v.kind() == ElementKind::string_array) a.set_varm(name, v.string_array().copy());
            else a.set_varm(name, v.table().copy());
        }
        a.set_uns(uns().copy());
        if (has_raw()) {
            if (has_raw_x()) a.set_raw(raw_x().copy(), raw_var().copy());
            else a.set_raw(raw_var().copy());
            for (const auto& [name, v] : impl_->raw_varm) {
                if (v.kind() == ElementKind::array) a.set_raw_varm(name, v.array().copy());
                else if (v.kind() == ElementKind::string_array) a.set_raw_varm(name, v.string_array().copy());
                else a.set_raw_varm(name, v.table().copy());
            }
        }
    }
    return a;
}

void AnnData::validate() const {
    validate_at(impl_->obs, "obs"); validate_at(impl_->var, "var"); validate_at(impl_->uns, "uns");
    if (impl_->obs.size() != n_obs()) fail(ErrorCode::shape_mismatch, "obs", "table length differs from observation axis");
    if (impl_->var.size() != n_vars()) fail(ErrorCode::shape_mismatch, "var", "table length differs from variable axis");
    if (has_x()) { check_matrix(x(), n_obs(), n_vars(), "X"); validate_at(*impl_->x, "X"); }
    for (const auto& [name, a] : impl_->layers) { check_matrix(a.view(), n_obs(), n_vars(), "layers/" + name); validate_at(a, "layers/" + name); }
    for (const auto& [name, a] : impl_->obsp) { check_pairwise(a.view(), n_obs(), "obsp/" + name); validate_at(a, "obsp/" + name); }
    for (const auto& [name, a] : impl_->varp) { check_pairwise(a.view(), n_vars(), "varp/" + name); validate_at(a, "varp/" + name); }
    for (const auto& [name, e] : impl_->obsm) {
        validate_at(e, "obsm/" + name);
        if (e.view().kind() == ElementKind::array) check_first(e.view().array(), n_obs(), "obsm/" + name);
        else if (e.view().kind() == ElementKind::string_array) check_first(e.view().string_array(), n_obs(), "obsm/" + name);
        else if (!same_index(*e.impl_->table->impl_->names, *impl_->obs.impl_->names)) fail(ErrorCode::shape_mismatch, "obsm/" + name, "table index differs from observation index");
    }
    for (const auto& [name, e] : impl_->varm) {
        validate_at(e, "varm/" + name);
        if (e.view().kind() == ElementKind::array) check_first(e.view().array(), n_vars(), "varm/" + name);
        else if (e.view().kind() == ElementKind::string_array) check_first(e.view().string_array(), n_vars(), "varm/" + name);
        else if (!same_index(*e.impl_->table->impl_->names, *impl_->var.impl_->names)) fail(ErrorCode::shape_mismatch, "varm/" + name, "table index differs from variable index");
    }
    if (has_raw()) {
        validate_at(impl_->raw->var, "raw/var");
        if (has_raw_x()) {
            check_matrix(raw_x(), n_obs(), raw_var().size(), "raw/X");
            validate_at(*impl_->raw->x, "raw/X");
        }
        for (const auto& [name, e] : impl_->raw->varm) {
            validate_at(e, "raw/varm/" + name);
            if (e.view().kind() == ElementKind::array) check_first(e.view().array(), raw_var().size(), "raw/varm/" + name);
            else if (e.view().kind() == ElementKind::string_array) check_first(e.view().string_array(), raw_var().size(), "raw/varm/" + name);
            else if (!same_index(*e.impl_->table->impl_->names, *impl_->raw->var.impl_->names))
                fail(ErrorCode::shape_mismatch, "raw/varm/" + name, "table index differs from raw variable index");
        }
    }
}

} // namespace anndatapp
