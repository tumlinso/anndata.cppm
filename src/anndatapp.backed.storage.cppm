module;
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <utility>
#include <vector>

module anndatapp.backed:storage;
import anndatapp.backed;

namespace anndatapp {
namespace backed_detail {
using Attribute = std::variant<std::string, std::vector<std::string>, std::int64_t,
                               std::vector<std::int64_t>, bool, double>;
using Attributes = std::map<std::string, Attribute, std::less<>>;
struct RecordFieldInfo {
    std::string name;
    bool strings = false;
    DType dtype = DType::float64;
    ByteOrder byte_order = ByteOrder::native;
};
struct NodeInfo {
    PhysicalArrayGeometry physical;
    bool parallel_reads = false, parallel_writes = false;
    bool exists = false;
    bool group = false;
    bool strings = false;
    bool record = false;
    std::vector<RecordFieldInfo> fields;
    bool has_opaque_attrs = false;
    DType dtype = DType::float64;
    ByteOrder byte_order = ByteOrder::native;
    std::vector<std::size_t> shape;
    Attributes attrs;
    // Nominal numeric codec unit in logical coordinates; empty means unknown
    // or rectangle-friendly storage. A shard can contain several such units.
    std::vector<std::size_t> codec_tile_shape;
    bool sharded_payload = false;
};
struct NumericTransferStats {
    std::size_t read_calls = 0, write_calls = 0, read_bytes = 0, write_bytes = 0;
    std::size_t scratch_allocations = 0, scratch_bytes = 0;
};
struct NumericWriteAxis {
    std::size_t count = 0;
    std::ptrdiff_t start = 0, step = 1;
    const std::size_t* positions = nullptr;
};
struct NumericWrite {
    std::vector<NumericWriteAxis> axes;
    const void* data = nullptr;
    std::vector<std::ptrdiff_t> strides;
    void* context = nullptr;
    void (*pack)(void*, const std::vector<std::vector<std::size_t>>&, void*) = nullptr;
    std::size_t scratch_bytes = 0, packing_metadata_bytes = 0;
};
// One worker owns each prepared transfer. Its backend must outlive it.
class NumericTransfer {
public:
    virtual ~NumericTransfer() = default;
    virtual void read(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, void*) = 0;
    virtual void write(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, const void*) = 0;
    virtual bool write_selection(const NumericWrite&) { return false; }
    virtual NumericTransferStats stats() const = 0;
    virtual std::size_t scratch_capacity() const = 0;
};
// Native-endian packed rectangles. start/count have the array rank; scalar
// transfers use empty vectors. The caller supplies sufficient final storage.
// Backends do no application caching and introduce no threads or locks.
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string location() const = 0;
    // Stable per-open-store identity for control-plane alias checks.
    virtual std::string identity() const { return location(); }
    // Worker-private backend and pre-inspected array; owner serializes context
    // preparation with structural mutation, and destroys transfers first.
    virtual std::unique_ptr<Backend> fork(std::size_t) {
        throw Error(ErrorCode::unsupported_layout, location(), "backend cannot create transfer context");
    }
    virtual std::unique_ptr<NumericTransfer> prepare_numeric(std::string_view path) {
        throw Error(ErrorCode::unsupported_layout, std::string(path), "backend cannot prepare numerical transfer");
    }
    virtual std::size_t retained_scratch_bytes() const { return 0; }
    virtual std::size_t scratch_allocations() const { return 0; }
    virtual NodeInfo inspect(std::string_view path) = 0;
    // Conservative payload workspace retained during one numeric operation.
    // Read and write calls are serial; a copying caller reserves their maximum.
    // Third-party internal allocations are outside this private-owned hint.
    virtual std::size_t numeric_workspace_bytes(const NodeInfo&, bool) { return 0; }
    virtual std::vector<std::string> children(std::string_view path) = 0;
    virtual void read_numeric(std::string_view path, const std::vector<std::size_t>& start,
                              const std::vector<std::size_t>& count, void* destination) = 0;
    virtual void write_numeric(std::string_view path, const std::vector<std::size_t>& start,
                               const std::vector<std::size_t>& count, const void* source) = 0;
    virtual std::vector<std::string> read_strings(std::string_view path,
        const std::vector<std::size_t>& start, const std::vector<std::size_t>& count) = 0;
    // A consecutive run along the last dimension (all earlier counts are one).
    // Return a nonempty prefix fitting decoded staging in byte_budget; reject
    // an indivisible string exceeding it. Default is conservative for adapters.
    virtual std::vector<std::string> read_strings_bounded(std::string_view path,
        const std::vector<std::size_t>& start, const std::vector<std::size_t>& count,
        std::size_t byte_budget) {
        auto limited = count;
        if (!limited.empty() && limited.back() > 1) limited.back() = 1;
        auto result = read_strings(path, start, limited);
        std::size_t used = sizeof(result) + result.capacity() * sizeof(std::string);
        for (const auto& value : result) {
            if (used > byte_budget || value.capacity() >= byte_budget - used)
                throw Error(ErrorCode::invalid_argument, std::string(path), "decoded string exceeds scratch budget");
            used += value.capacity() + 1;
        }
        return result;
    }
    virtual void write_strings(std::string_view path, const std::vector<std::size_t>& start,
        const std::vector<std::size_t>& count, const std::vector<std::string>& values) = 0;
    virtual void create_group(std::string_view path, const Attributes& attrs) = 0;
    virtual void create_numeric(std::string_view path, DType dtype,
        const std::vector<std::size_t>& shape, const Attributes& attrs,
        ByteOrder byte_order = ByteOrder::native) = 0;
    virtual void create_strings(std::string_view path,
        const std::vector<std::size_t>& shape, const Attributes& attrs) = 0;
    virtual void create_record(std::string_view path, const std::vector<std::size_t>&,
        const std::vector<RecordFieldInfo>&, const Attributes&) {
        throw Error(ErrorCode::unsupported_encoding, std::string(path), "record arrays are unsupported by this backend");
    }
    virtual void read_record_numeric(std::string_view path, std::string_view,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&, void*) {
        throw Error(ErrorCode::unsupported_encoding, std::string(path), "record arrays are unsupported by this backend");
    }
    virtual void write_record_numeric(std::string_view path, std::string_view,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&, const void*) {
        throw Error(ErrorCode::unsupported_encoding, std::string(path), "record arrays are unsupported by this backend");
    }
    virtual std::vector<std::string> read_record_strings_bounded(std::string_view path, std::string_view,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&, std::size_t) {
        throw Error(ErrorCode::unsupported_encoding, std::string(path), "record arrays are unsupported by this backend");
    }
    virtual void write_record_strings(std::string_view path, std::string_view,
        const std::vector<std::size_t>&, const std::vector<std::size_t>&, const std::vector<std::string>&) {
        throw Error(ErrorCode::unsupported_encoding, std::string(path), "record arrays are unsupported by this backend");
    }
    virtual void attributes(std::string_view path, const Attributes& attrs) = 0;
    virtual void erase(std::string_view path) = 0;
    virtual void rename(std::string_view source, std::string_view target) = 0;
    // Preserve backend-native extension attributes during a same-store rewrite.
    // Known attributes already present on target take precedence.
    virtual void copy_attributes(std::string_view, std::string_view) {}
    virtual void flush() = 0;
    virtual void close() = 0;
    // Read-only native-endian contiguous numerical mapping. Its returned owner
    // retains an independent mapping lifetime, including after session close.
    virtual std::optional<MappedArray> map_numeric(std::string_view) { return std::nullopt; }
};
BackedAnnData make_store(std::unique_ptr<Backend> backend, BackedOptions options);
void write_element(Backend&, std::string_view path, ElementView, std::size_t scratch_bytes);
void validate_serializable(ElementView, std::string_view path);
void initialize_store(Backend& backend, const AnnDataView& source, std::size_t scratch_bytes,
                      Validation validation = Validation::full);
void export_store(Backend&, const BackedAnnDataView&, std::size_t scratch_bytes, Validation validation = Validation::full);
struct CategoryPlan;
std::size_t category_count(const std::shared_ptr<CategoryPlan>&);
std::size_t category_source_count(const std::shared_ptr<CategoryPlan>&);
std::size_t category_position(const std::shared_ptr<CategoryPlan>&, std::size_t);
std::int64_t category_recode(const std::shared_ptr<CategoryPlan>&, std::int64_t);
std::shared_ptr<CategoryPlan> compose_category_plans(const std::shared_ptr<CategoryPlan>& outer,
    const std::shared_ptr<CategoryPlan>& before, const std::shared_ptr<CategoryPlan>& after);
void recode_codes_buffer(const std::shared_ptr<CategoryPlan>&, DType, void*, std::size_t count, std::ptrdiff_t byte_stride, std::string_view path);
void copy_recoded_codes(Backend&, std::string_view destination_path,
    const std::shared_ptr<detail::BackedNode>&, std::size_t scratch_bytes,
    const std::shared_ptr<CategoryPlan>&);
std::shared_ptr<CategoryPlan> make_category_plan(const std::shared_ptr<detail::BackedNode>&, std::size_t scratch_bytes);
std::size_t scratch_remaining(const std::shared_ptr<detail::BackedNode>&);
void validate_category_dictionary(const std::shared_ptr<detail::BackedNode>&, std::size_t scratch_bytes);
void copy_pruned_category(Backend&, std::string_view destination_path,
    const std::shared_ptr<detail::BackedNode>&, std::size_t scratch_bytes,
    std::shared_ptr<CategoryPlan> plan = {});
struct Axis {
    std::size_t size = 0;
    std::ptrdiff_t start = 0;
    std::ptrdiff_t step = 1;
    std::shared_ptr<const std::vector<std::size_t>> positions;
    std::shared_ptr<CategoryPlan> category_positions;
    Axis(std::size_t length = 0, std::ptrdiff_t first = 0, std::ptrdiff_t stride = 1,
         std::shared_ptr<const std::vector<std::size_t>> mapping = {},
         std::shared_ptr<CategoryPlan> category_mapping = {})
        : size(length), start(first), step(stride), positions(std::move(mapping)),
          category_positions(std::move(category_mapping)) {}
    std::size_t at(std::size_t i) const {
        if (positions) return (*positions)[i];
        auto index = static_cast<std::size_t>(start + static_cast<std::ptrdiff_t>(i) * step);
        return category_positions ? category_position(category_positions, index) : index;
    }
    bool has_positions() const noexcept { return bool(positions) || bool(category_positions); }
    bool all(std::size_t length) const { return !has_positions() && start == 0 && step == 1 && size == length; }
};
struct SliceStage {
    Axis obs_before, var_before, obs_after, var_after;
};
enum class ColorPromotion { numeric_array, string_array, string_column, nullable_string_column };
struct ColorOverride {
    ColorPromotion kind;
    Axis axis;
};

struct Revision { std::size_t value = 0; std::optional<bool> unique_index; };
struct Session {
    std::unique_ptr<Backend> backend;
    BackedOptions options;
    bool open = true;
    std::size_t dataset_revision = 0;
    std::map<std::string, std::shared_ptr<Revision>, std::less<>> revisions;
    void check_open() const;
    void check_writable() const;
    void invalidate(std::string_view path);
};
std::string child(std::string_view parent, std::string_view name);
std::string encoding(const NodeInfo&, std::string_view path);
std::string string_attr(const NodeInfo&, std::string_view key, std::string_view path);
std::vector<std::string> strings_attr(const NodeInfo&, std::string_view key, std::string_view path);
std::vector<std::size_t> logical_shape(Session&, std::string_view path);
Array read_array(const std::shared_ptr<detail::BackedNode>&);
Element read_record_field(const std::shared_ptr<detail::BackedNode>&, const NodeInfo&, std::string_view);
RecordArray read_record(const std::shared_ptr<detail::BackedNode>&, const NodeInfo&);
void write_record(Backend&, std::string_view, RecordArrayView, std::size_t);
void copy_record(Backend&, std::string_view, const std::shared_ptr<detail::BackedNode>&, const NodeInfo&, std::size_t);
void copy_array(Backend&, std::string_view destination_path, const std::shared_ptr<detail::BackedNode>&, std::size_t scratch_bytes);
void copy_node(Backend&, std::string_view destination_path, const std::shared_ptr<detail::BackedNode>&,
               std::size_t scratch_bytes, std::optional<std::size_t> string_scratch_bytes = {});
void require_unique_index(const std::shared_ptr<detail::BackedNode>&, const Axis&, std::size_t scratch_bytes = 0);
void require_unique_categories(const std::shared_ptr<detail::BackedNode>&, std::size_t scratch_bytes = 0);
void validate_utf8_string(std::string_view, std::string_view path);
void compare_table_indexes(const BackedTable&, const BackedTable&);
void compare_table_indexes(const BackedTable&, const TableView&);
std::vector<std::string> color_keys(const std::shared_ptr<detail::BackedNode>&);
std::shared_ptr<detail::BackedNode> color_child(const std::shared_ptr<detail::BackedNode>&, std::string_view key);
bool color_has_changes(const std::shared_ptr<detail::BackedNode>&);
std::vector<std::string> color_changed_keys(const std::shared_ptr<detail::BackedNode>&);
std::optional<Element> read_color_override(const std::shared_ptr<detail::BackedNode>&);
void copy_promoted_color(Backend&, std::string_view path, const std::shared_ptr<detail::BackedNode>&, std::size_t scratch_bytes);

void check_node(const std::shared_ptr<detail::BackedNode>&);
std::shared_ptr<detail::BackedNode> node(std::shared_ptr<Session>, std::string path, std::vector<Axis> axes = {});
Axis select_axis(const Axis& parent, const Selection& selection, const std::shared_ptr<detail::BackedNode>& labels = {});
BackedAnnDataView root_view(const std::shared_ptr<detail::BackedDataset>&);
} // namespace backed_detail
namespace detail {
struct BackedNode {
    std::shared_ptr<backed_detail::Session> session;
    std::string path;
    std::vector<backed_detail::Axis> axes;
    std::shared_ptr<backed_detail::Revision> revision;
    std::size_t observed_revision = 0;
    bool writable = false;
    bool empty_mapping = false;
    std::size_t observed_dataset_revision = 0;
    bool prune_categories = false;
    std::shared_ptr<backed_detail::CategoryPlan> code_recode;
    std::shared_ptr<const std::vector<backed_detail::SliceStage>> slice_history;
    std::shared_ptr<backed_detail::ColorOverride> color_override;
};
struct BackedAccess {
    static BackedElement element(std::shared_ptr<BackedNode> value) { return BackedElement(std::move(value)); }
    static BackedArray array(std::shared_ptr<BackedNode> value) { return BackedArray(std::move(value)); }
    static const std::shared_ptr<BackedNode>& get(const BackedArray& v) { return v.node_; }
    static const std::shared_ptr<BackedNode>& get(const BackedStringArray& v) { return v.node_; }
    static const std::shared_ptr<BackedNode>& get(const BackedColumn& v) { return v.node_; }
    static const std::shared_ptr<BackedNode>& get(const BackedTable& v) { return v.node_; }
    static const std::shared_ptr<BackedNode>& get(const BackedElement& v) { return v.node_; }
    static const std::shared_ptr<BackedDataset>& get(const BackedAnnDataView& v) { return v.impl_; }
};
struct BackedDataset {
    std::shared_ptr<backed_detail::Session> session;
    backed_detail::Axis obs, var;
    std::size_t observed_revision = 0;
    bool prune_categories = false;
    std::shared_ptr<const std::vector<backed_detail::SliceStage>> slice_history;
};
} // namespace detail
} // namespace anndatapp
