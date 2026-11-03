module;
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

export module anndata.core:data;
export import :arrays;
export import :strings;
export import :records;

export namespace anndata {

namespace detail {
struct ColumnImpl;
struct ColumnViewImpl;
struct TableImpl;
struct TableViewImpl;
struct ElementImpl;
struct AnnDataImpl;
struct AnnDataViewImpl;
}

struct all_t {};
inline constexpr all_t all{};

// Positions use Python/AnnData indexing: negative positions count from the end.
struct indices {
    std::vector<std::ptrdiff_t> values;
    indices(std::initializer_list<std::ptrdiff_t> v) : values(v) {}
    explicit indices(std::vector<std::ptrdiff_t> v) : values(std::move(v)) {}
};

struct range {
    std::optional<std::ptrdiff_t> start;
    std::optional<std::ptrdiff_t> stop;
    std::ptrdiff_t step{1};
};

struct mask {
    std::vector<std::uint8_t> values;
    mask(std::initializer_list<std::uint8_t> v) : values(v) {}
    explicit mask(std::vector<std::uint8_t> v) : values(std::move(v)) {}
};

struct labels {
    std::vector<std::string> values;
    labels(std::initializer_list<std::string_view> v) {
        for (auto s : v) values.emplace_back(s);
    }
    explicit labels(std::vector<std::string> v) : values(std::move(v)) {}
};

class TableView;

class Selection {
public:
    Selection(all_t = all);
    Selection(indices);
    Selection(range);
    Selection(mask);
    Selection(labels);
    Selection(std::ptrdiff_t);
    template<class I> requires (std::is_integral_v<I> && !std::is_same_v<I, bool>
                               && !std::is_same_v<I, std::ptrdiff_t>)
    Selection(I value) : Selection() {
        if constexpr (std::is_unsigned_v<I>) {
            if (value > static_cast<std::make_unsigned_t<std::ptrdiff_t>>(std::numeric_limits<std::ptrdiff_t>::max()))
                throw Error(ErrorCode::out_of_bounds, "index", "position exceeds the signed index range");
        } else if constexpr (sizeof(I) > sizeof(std::ptrdiff_t)) {
            if (value < std::numeric_limits<std::ptrdiff_t>::min() || value > std::numeric_limits<std::ptrdiff_t>::max())
                throw Error(ErrorCode::out_of_bounds, "index", "position exceeds the signed index range");
        }
        kind_ = Kind::indices;
        indices_.push_back(static_cast<std::ptrdiff_t>(value));
    }
    Selection(std::string_view);
    Selection(const char*);
    bool is_all() const noexcept;
    enum class Kind { all, indices, range, mask, labels, label };
    Kind kind() const noexcept { return kind_; }
    const std::vector<std::ptrdiff_t>& index_values() const noexcept { return indices_; }
    const range& range_value() const noexcept { return range_; }
    const std::vector<std::uint8_t>& mask_values() const noexcept { return mask_; }
    const std::vector<std::string>& label_values() const noexcept { return labels_; }
    Axis resolve(const std::vector<std::string>& names,
                                     std::string_view path) const;
    Axis resolve(const TableView& index,
                                     std::string_view path) const;
private:
    Axis resolve_impl(std::size_t length, std::string_view path,
        const std::vector<std::string>* names, const TableView* table) const;
    Kind kind_{Kind::all};
    std::vector<std::ptrdiff_t> indices_;
    range range_;
    std::vector<std::uint8_t> mask_;
    std::vector<std::string> labels_;
};

enum class ColumnKind { numeric, string, categorical };
enum class StringMissingValue { NA, NaN };

class Column;
class Table;
class TableView;

class ColumnView {
public:
    ColumnView();
    std::size_t size() const;
    ColumnKind kind() const;
    bool nullable() const;
    bool valid(std::size_t row) const;
    ArrayView values() const;
    std::string_view string(std::size_t row) const;
    [[nodiscard]] StringMissingValue string_missing_value() const;
    bool ordered() const;
    ColumnView categories() const;
    std::int64_t code(std::size_t row) const;
    // Codes may be recoded for the selected category dictionary.
    ArrayView codes() const;
    Column copy() const;
    ColumnView select(const Axis& rows,
                      bool prune_categories = true) const;
    ColumnView select_all(bool prune_categories = true) const;
private:
    std::shared_ptr<detail::ColumnViewImpl> impl_;
    explicit ColumnView(std::shared_ptr<detail::ColumnViewImpl>);
    friend class Column;
    friend class TableView;
};

class Column {
public:
    Column();
    ~Column();
    Column(Column&&) noexcept;
    Column& operator=(Column&&) noexcept;
    Column(const Column&) = delete;
    Column& operator=(const Column&) = delete;
    static Column numeric(Array values, std::vector<std::uint8_t> validity = {});
    static Column nullable_numeric(Array values, std::vector<std::uint8_t> validity);
    static Column strings(std::vector<std::string> values,
                          std::vector<std::uint8_t> validity = {});
    static Column strings(std::initializer_list<std::string_view> values,
                          std::vector<std::uint8_t> validity = {});
    static Column nullable_strings(std::vector<std::string> values,
                                   std::vector<std::uint8_t> validity,
                                   StringMissingValue missing = StringMissingValue::NA);
    static Column nullable_strings(std::initializer_list<std::string_view> values,
                                   std::vector<std::uint8_t> validity,
                                   StringMissingValue missing = StringMissingValue::NA);
    // Adopt packed UTF-8 bytes and byte offsets (one more offset than rows).
    // Validation::none trusts buffer contents; validate() checks them later.
    static Column from_packed_strings(std::vector<char> bytes,
                                      std::vector<std::size_t> offsets,
                                      std::vector<std::uint8_t> validity = {},
                                      Validation validation = Validation::full);
    static Column from_nullable_packed_strings(std::vector<char> bytes,
                                               std::vector<std::size_t> offsets,
                                               std::vector<std::uint8_t> validity,
                                               StringMissingValue missing = StringMissingValue::NA,
                                               Validation validation = Validation::full);
    static Column categorical(Array codes, Column categories, bool ordered = false,
                              Validation validation = Validation::full);
    template<class T>
    static Column from_values(std::initializer_list<T> values,
                              std::vector<std::uint8_t> validity = {}) {
        auto a = Array::empty<T>({values.size()});
        auto v = a.edit().template dense<T>();
        std::size_t i = 0;
        for (auto value : values) v.data()[i++] = value;
        return numeric(std::move(a), std::move(validity));
    }
    ColumnView view() const;
    Column copy() const;
    std::size_t size() const;
    ColumnKind kind() const;
    ArrayEdit edit_values();
    ArrayEdit unsafe_edit_codes();
    void set_valid(std::size_t row, bool valid);
    void validate() const;
private:
    std::shared_ptr<detail::ColumnImpl> impl_;
    explicit Column(std::shared_ptr<detail::ColumnImpl>);
    friend class ColumnView;
};

class TableView {
public:
    TableView();
    std::size_t size() const;
    std::size_t n_columns() const;
    std::string_view index_name() const;
    std::string_view name(std::size_t row) const;
    std::vector<std::string> names() const;
    std::vector<std::string> column_names() const;
    bool contains(std::string_view column) const;
    ColumnView column(std::string_view column) const;
    TableView select(const Axis& rows,
                     bool prune_categories = true) const;
    TableView select_all(bool prune_categories = true) const;
    Table copy() const;
private:
    std::shared_ptr<detail::TableViewImpl> impl_;
    explicit TableView(std::shared_ptr<detail::TableViewImpl>);
    TableView with_rows(const Axis&, bool prune_categories) const;
    friend class Table;
    friend class AnnDataView;
};

class Table {
public:
    Table();
    ~Table();
    Table(Table&&) noexcept;
    Table& operator=(Table&&) noexcept;
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;
    static Table with_index(std::vector<std::string> names,
                            std::string index_name = {});
    static Table with_index(std::initializer_list<std::string_view> names,
                            std::string index_name = {});
    static Table numbered(std::size_t rows);
    std::size_t size() const;
    TableView view() const;
    Table copy() const;
    void set(std::string name, Column column);
    void erase(std::string_view name);
    ColumnView column(std::string_view name) const;
    ArrayEdit edit_values(std::string_view name);
    void set_names(std::vector<std::string> names, std::string index_name = {});
    void validate() const;
private:
    std::shared_ptr<detail::TableImpl> impl_;
    friend class AnnData;
};

enum class ElementKind { null, boolean, signed_integer, unsigned_integer,
                         floating, string, array, string_array, column, table, mapping, record_array };
class ElementView;

class Element {
public:
    Element();
    explicit Element(std::nullptr_t);
    ~Element();
    Element(Element&&) noexcept;
    Element& operator=(Element&&) noexcept;
    Element(const Element&) = delete;
    Element& operator=(const Element&) = delete;
    explicit Element(bool);
    explicit Element(std::int64_t);
    explicit Element(std::uint64_t);
    explicit Element(double);
    template<class T> requires (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>
                               && !std::is_same_v<T, std::int64_t>
                               && !std::is_same_v<T, std::uint64_t>
                               && !std::is_same_v<T, double>)
    explicit Element(T value) : Element(scalar(value)) {}
    explicit Element(std::string);
    explicit Element(const char*);
    explicit Element(Array);
    explicit Element(StringArray);
    explicit Element(RecordArray);
    explicit Element(Column);
    explicit Element(Table);
    static Element mapping();
    template<class T> static Element scalar(T value) {
        return scalar_from_bytes(dtype_of<T>(), &value);
    }
    ElementView view() const;
    Element copy() const;
    void set(std::string key, Element value);
    void set_serialization_byte_order(ByteOrder);
    void erase(std::string_view key);
    ElementView at(std::string_view key) const;
    void validate() const;
private:
    std::shared_ptr<detail::ElementImpl> impl_;
    explicit Element(std::shared_ptr<detail::ElementImpl>);
    static Element scalar_from_bytes(DType, const void*);
    friend class ElementView;
    friend class AnnData;
    friend class AnnDataView;
};

class ElementView {
public:
    ElementView();
    ElementKind kind() const;
    DType dtype() const;
    template<class T> T scalar() const {
        if (dtype() != dtype_of<T>())
            throw Error(ErrorCode::dtype_mismatch, "element", "scalar dtype differs from requested type");
        if constexpr (std::is_same_v<T, bool>) return boolean();
        else if constexpr (std::is_floating_point_v<T>) return static_cast<T>(floating());
        else if constexpr (std::is_signed_v<T>) return static_cast<T>(signed_integer());
        else return static_cast<T>(unsigned_integer());
    }
    bool boolean() const;
    std::int64_t signed_integer() const;
    std::uint64_t unsigned_integer() const;
    double floating() const;
    std::string_view string() const;
    [[nodiscard]] ByteOrder serialization_byte_order() const;
    ArrayView array() const;
    StringArrayView string_array() const;
    RecordArrayView record_array() const;
    ColumnView column() const;
    TableView table() const;
    std::vector<std::string> keys() const;
    bool contains(std::string_view key) const;
    ElementView at(std::string_view key) const;
    Element copy() const;
private:
    std::shared_ptr<detail::ElementImpl> impl_;
    explicit ElementView(std::shared_ptr<detail::ElementImpl>);
    friend class Element;
    friend class AnnData;
    friend class AnnDataView;
};

class AnnDataView;

class AnnData {
public:
    AnnData();
    ~AnnData();
    AnnData(AnnData&&) noexcept;
    AnnData& operator=(AnnData&&) noexcept;
    AnnData(const AnnData&) = delete;
    AnnData& operator=(const AnnData&) = delete;
    static AnnData empty(std::size_t n_obs, std::size_t n_vars);
    std::size_t n_obs() const;
    std::size_t n_vars() const;
    bool has_x() const;
    ArrayView x() const;
    ArrayEdit edit_x();
    void set_x(Array);
    void clear_x();
    TableView obs() const;
    TableView var() const;
    ColumnView obs_column(std::string_view name) const;
    ColumnView var_column(std::string_view name) const;
    void set_obs(Table);
    void set_var(Table);
    void set_obs_column(std::string, Column);
    void set_var_column(std::string, Column);
    ArrayEdit edit_obs_values(std::string_view);
    ArrayEdit edit_var_values(std::string_view);
    void set_layer(std::string, Array);
    void set_obsm(std::string, Array);
    void set_obsm(std::string, StringArray);
    void set_obsm(std::string, Table);
    void set_varm(std::string, Array);
    void set_varm(std::string, StringArray);
    void set_varm(std::string, Table);
    void set_obsp(std::string, Array);
    void set_varp(std::string, Array);
    ArrayView layer(std::string_view) const;
    ArrayEdit edit_layer(std::string_view);
    ElementView obsm(std::string_view) const;
    ElementView varm(std::string_view) const;
    ArrayEdit edit_obsm_array(std::string_view);
    ArrayEdit edit_varm_array(std::string_view);
    StringArrayEdit edit_obsm_string_array(std::string_view);
    StringArrayEdit edit_varm_string_array(std::string_view);
    ArrayView obsp(std::string_view) const;
    ArrayView varp(std::string_view) const;
    ArrayEdit edit_obsp(std::string_view);
    ArrayEdit edit_varp(std::string_view);
    std::vector<std::string> layer_names() const;
    std::vector<std::string> obsm_names() const;
    std::vector<std::string> varm_names() const;
    std::vector<std::string> obsp_names() const;
    std::vector<std::string> varp_names() const;
    ElementView uns() const;
    void set_uns(Element);
    void set_uns(std::string, Element);
    // raw carries X, var and varm; observation alignment is shared with this object.
    void set_raw(Array, Table);
    void set_raw(Table);
    void set_raw_varm(std::string, Array);
    void set_raw_varm(std::string, StringArray);
    void set_raw_varm(std::string, Table);
    bool has_raw() const;
    bool has_raw_x() const;
    ArrayView raw_x() const;
    TableView raw_var() const;
    ElementView raw_varm(std::string_view) const;
    std::vector<std::string> raw_varm_names() const;
    void clear_raw();
    AnnDataView view() const;
    AnnDataView slice(Selection obs = all, Selection var = all) const;
    AnnData copy() const;
    void validate() const;
private:
    std::unique_ptr<detail::AnnDataImpl> impl_;
};

class AnnDataView {
public:
    AnnDataView();
    std::size_t n_obs() const;
    std::size_t n_vars() const;
    bool has_x() const;
    ArrayView x() const;
    TableView obs() const;
    TableView var() const;
    ColumnView obs_column(std::string_view name) const;
    ColumnView var_column(std::string_view name) const;
    ArrayView layer(std::string_view) const;
    ElementView obsm(std::string_view) const;
    ElementView varm(std::string_view) const;
    ArrayView obsp(std::string_view) const;
    ArrayView varp(std::string_view) const;
    std::vector<std::string> layer_names() const;
    std::vector<std::string> obsm_names() const;
    std::vector<std::string> varm_names() const;
    std::vector<std::string> obsp_names() const;
    std::vector<std::string> varp_names() const;
    ElementView uns() const;
    bool has_raw() const;
    bool has_raw_x() const;
    ArrayView raw_x() const;
    TableView raw_var() const;
    ElementView raw_varm(std::string_view) const;
    std::vector<std::string> raw_varm_names() const;
    AnnDataView slice(Selection obs = all, Selection var = all) const;
    AnnData copy() const;
private:
    std::shared_ptr<detail::AnnDataViewImpl> impl_;
    explicit AnnDataView(std::shared_ptr<detail::AnnDataViewImpl>);
    friend class AnnData;
};

} // namespace anndata
