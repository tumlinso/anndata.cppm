module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <variant>

module anndata.backed;
import :storage;

namespace anndata {
namespace backed_detail {
namespace {
[[noreturn]] void fail(ErrorCode code, std::string_view path, std::string message) {
    throw Error(code, std::string(path), std::move(message));
}
std::string index_path(Session& s, std::string_view table) {
    auto info = s.backend->inspect(table);
    if (encoding(info, table) != "dataframe") fail(ErrorCode::unsupported_encoding, table, "expected dataframe");
    return child(table, string_attr(info, "_index", table));
}
void require_dataset(const std::shared_ptr<detail::BackedDataset>& data) {
    if (!data || !data->session) fail(ErrorCode::invalid_argument, "store", "empty backed dataset");
    data->session->check_open();
    if (data->observed_revision != data->session->dataset_revision)
        fail(ErrorCode::invalid_argument, "store", "backed dataset view is stale after structural mutation");
}
bool present(Session& s, std::string_view path) {
    auto info = s.backend->inspect(path);
    return info.exists && encoding(info, path) != "null";
}
std::vector<Axis> aligned_axes(Session& s, std::string_view path, const Axis& first,
                               const std::optional<Axis>& second = {}) {
    auto shape = logical_shape(s, path);
    if (shape.empty() || (second && shape.size() < 2))
        fail(ErrorCode::shape_mismatch, path, "aligned element has insufficient rank");
    std::vector<Axis> axes;
    for (auto n : shape) axes.push_back(Axis{n});
    axes[0] = first;
    if (second) axes[1] = *second;
    return axes;
}
std::vector<std::string> mapping_names(const std::shared_ptr<detail::BackedDataset>& d, std::string_view path) {
    require_dataset(d);
    const auto info = d->session->backend->inspect(path);
    if (!info.exists) return {};
    if (encoding(info, path) != "dict") fail(ErrorCode::unsupported_encoding, path, "expected mapping");
    return d->session->backend->children(path);
}
}

std::string child(std::string_view parent, std::string_view name) {
    if (name.empty() || name.find('/') != std::string_view::npos || name.find('\0') != std::string_view::npos)
        fail(ErrorCode::invalid_argument, parent, "node key must be nonempty and contain no slash or NUL");
    return parent == "/" ? "/" + std::string(name) : std::string(parent) + "/" + std::string(name);
}
std::string string_attr(const NodeInfo& info, std::string_view key, std::string_view path) {
    auto it = info.attrs.find(key);
    if (it == info.attrs.end()) fail(ErrorCode::unsupported_encoding, path, "missing attribute '" + std::string(key) + "'");
    if (auto p = std::get_if<std::string>(&it->second)) return *p;
    fail(ErrorCode::unsupported_encoding, path, "attribute '" + std::string(key) + "' must be scalar text");
}
std::vector<std::string> strings_attr(const NodeInfo& info, std::string_view key, std::string_view path) {
    auto it = info.attrs.find(key);
    if (it == info.attrs.end()) fail(ErrorCode::unsupported_encoding, path, "missing attribute '" + std::string(key) + "'");
    if (auto p = std::get_if<std::vector<std::string>>(&it->second)) return *p;
    fail(ErrorCode::unsupported_encoding, path, "attribute '" + std::string(key) + "' must be a text array");
}
std::string encoding(const NodeInfo& info, std::string_view path) {
    if (!info.exists) fail(ErrorCode::out_of_bounds, path, "stored element does not exist");
    auto code = string_attr(info, "encoding-type", path);
    auto version = string_attr(info, "encoding-version", path);
    std::string_view expected;
    if (code == "array" || code == "numeric-scalar" || code == "string" || code == "string-array" || code == "dataframe" || code == "categorical") expected = "0.2.0";
    else if (code == "anndata" || code == "raw" || code == "dict" || code == "null" || code == "csr_matrix" || code == "csc_matrix" || code == "nullable-integer" || code == "nullable-boolean" || code == "nullable-string-array") expected = "0.1.0";
    else fail(ErrorCode::unsupported_encoding, path, "unsupported encoding '" + code + "'");
    if (version != expected) fail(ErrorCode::unsupported_version, path, "unsupported " + code + " version '" + version + "'");
    return code;
}
void Session::check_open() const {
    if (!open || !backend) fail(ErrorCode::invalid_argument, "store", "backed store is closed");
}
void Session::check_writable() const {
    check_open();
    if (options.mode != AccessMode::read_write) fail(ErrorCode::invalid_argument, backend->location(), "backed store is read-only");
}
void Session::invalidate(std::string_view path) {
    for (auto& [key, revision] : revisions)
        if (key == path || (key.size() > path.size() && key.starts_with(path) && (path == "/" || key[path.size()] == '/')))
            { ++revision->value; revision->unique_index.reset(); }
}
void check_node(const std::shared_ptr<detail::BackedNode>& n) {
    if (!n || !n->session) fail(ErrorCode::invalid_argument, "store", "empty backed handle");
    n->session->check_open();
    if(n->slice_history && n->observed_dataset_revision != n->session->dataset_revision)
        fail(ErrorCode::invalid_argument,n->path,"backed color context is stale after structural mutation");
    if (!n->revision || n->revision->value != n->observed_revision)
        fail(ErrorCode::invalid_argument, n->path, "backed handle is stale after replacement or deletion");
}
std::vector<std::size_t> logical_shape(Session& s, std::string_view path) {
    s.check_open();
    const auto info = s.backend->inspect(path);
    if (!info.exists) fail(ErrorCode::out_of_bounds, path, "stored element does not exist");
    // Sparse components are physical arrays without element-encoding attrs.
    if (!info.attrs.contains("encoding-type")) {
        if (info.group) fail(ErrorCode::unsupported_encoding, path, "group has no element encoding");
        return info.shape;
    }
    auto code = encoding(info, path);
    if (code == "csr_matrix" || code == "csc_matrix") {
        auto it = info.attrs.find("shape");
        if (it == info.attrs.end()) fail(ErrorCode::unsupported_encoding, path, "sparse matrix lacks shape");
        const auto* shape = std::get_if<std::vector<std::int64_t>>(&it->second);
        if (!shape || shape->size() != 2) fail(ErrorCode::shape_mismatch, path, "sparse shape must have two signed dimensions");
        std::vector<std::size_t> out;
        for (auto n : *shape) {
            if (n < 0 || static_cast<std::uint64_t>(n) > std::numeric_limits<std::size_t>::max()) fail(ErrorCode::shape_mismatch, path, "invalid sparse dimension");
            out.push_back(static_cast<std::size_t>(n));
        }
        return out;
    }
    if (code == "dataframe") {
        auto shape = logical_shape(s, child(path, string_attr(info, "_index", path)));
        if (shape.size() != 1) fail(ErrorCode::shape_mismatch, path, "dataframe index must be one-dimensional");
        return shape;
    }
    if (code == "categorical") return logical_shape(s, child(path, "codes"));
    if (code == "nullable-integer" || code == "nullable-boolean" || code == "nullable-string-array") return logical_shape(s, child(path, "values"));
    if (code == "array" || code == "string-array" || code == "numeric-scalar" || code == "string") return info.shape;
    return {};
}
std::shared_ptr<detail::BackedNode> node(std::shared_ptr<Session> s, std::string path, std::vector<Axis> axes) {
    s->check_open();
    auto shape = logical_shape(*s, path);
    // Alignment is a metadata invariant of the stored object, including when
    // only a subset is requested. A subset must not hide oversized slots.
    auto component_count = static_cast<std::size_t>(std::count(path.begin(), path.end(), '/'));
    auto aligned = [&](std::size_t first, std::optional<std::size_t> second, bool matrix) {
        if (shape.empty() || shape[0] != first || (second && (shape.size() < 2 || shape[1] != *second)) || (matrix && shape.size() != 2))
            fail(ErrorCode::shape_mismatch, path, "stored shape does not match its AnnData axes");
    };
    if (path == "/X" || (component_count == 2 && path.starts_with("/layers/"))) {
        aligned(logical_shape(*s,"/obs")[0],logical_shape(*s,"/var")[0],true);
    } else if (component_count == 2 && (path.starts_with("/obsm/") || path.starts_with("/obsp/"))) {
        auto length=logical_shape(*s,"/obs")[0];aligned(length,path.starts_with("/obsp/")?std::optional<std::size_t>{length}:std::nullopt,false);
    } else if (component_count == 2 && (path.starts_with("/varm/") || path.starts_with("/varp/"))) {
        auto length=logical_shape(*s,"/var")[0];aligned(length,path.starts_with("/varp/")?std::optional<std::size_t>{length}:std::nullopt,false);
    } else if(path == "/raw/X") {
        aligned(logical_shape(*s,"/obs")[0],logical_shape(*s,"/raw/var")[0],true);
    } else if(component_count == 3 && path.starts_with("/raw/varm/")) {
        aligned(logical_shape(*s,"/raw/var")[0],std::nullopt,false);
    }
    if (axes.empty()) for (auto n : shape) axes.push_back(Axis{n});
    if (axes.size() != shape.size()) fail(ErrorCode::shape_mismatch, path, "selection rank differs from stored rank");
    auto& revision = s->revisions[path];
    if (!revision) revision = std::make_shared<Revision>();
    auto result = std::make_shared<detail::BackedNode>();
    result->session = std::move(s); result->path = std::move(path); result->axes = std::move(axes);
    result->revision = revision; result->observed_revision = revision->value;
    result->observed_dataset_revision = result->session->dataset_revision;
    return result;
}
Axis select_axis(const Axis& parent, const Selection& selection, const std::shared_ptr<detail::BackedNode>& index) {
    const auto path = index ? index->path : std::string("selection");
    if (parent.size > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) fail(ErrorCode::out_of_bounds, path, "axis exceeds signed index range");
    if (selection.kind() == Selection::Kind::all) return parent;
    const auto length = static_cast<std::ptrdiff_t>(parent.size);
    if (selection.kind() == Selection::Kind::range) {
        const auto& r = selection.range_value();
        if (!r.step || r.step == std::numeric_limits<std::ptrdiff_t>::min()) fail(ErrorCode::invalid_argument, path, "invalid range step");
        auto norm = [&](std::optional<std::ptrdiff_t> v, bool first) {
            if (!v) return r.step > 0 ? (first ? std::ptrdiff_t{0} : length) : (first ? length - 1 : std::ptrdiff_t{-1});
            auto x = *v;
            if (x < 0) x += length;
            return r.step > 0 ? std::clamp(x, std::ptrdiff_t{0}, length) : std::clamp(x, std::ptrdiff_t{-1}, length - 1);
        };
        auto first = norm(r.start, true), stop = norm(r.stop, false);
        std::size_t size = 0;
        if (r.step > 0 && first < stop) size = static_cast<std::size_t>((stop - first - 1) / r.step + 1);
        if (r.step < 0 && first > stop) size = static_cast<std::size_t>((first - stop - 1) / (-r.step) + 1);
        if (parent.category_positions) {
            // Compose in the compact category-map coordinate system.
            std::ptrdiff_t mapped_start = 0, mapped_step = 1, offset = 0;
            if (size && (__builtin_mul_overflow(first, parent.step, &offset) ||
                         __builtin_add_overflow(parent.start, offset, &mapped_start)))
                fail(ErrorCode::out_of_bounds, path, "category selection overflows");
            if (size > 1 && __builtin_mul_overflow(parent.step, r.step, &mapped_step))
                fail(ErrorCode::out_of_bounds, path, "category selection step overflows");
            return Axis{size, mapped_start, mapped_step, {}, parent.category_positions};
        }
        if (!parent.positions) {
            // The selected endpoints are in the original axis range, so start
            // and the useful progression are representable. Avoid an overflow
            // in a step that is irrelevant for zero/one-element selections.
            auto new_start = size ? static_cast<std::ptrdiff_t>(parent.at(static_cast<std::size_t>(first))) : std::ptrdiff_t{0};
            std::ptrdiff_t new_step = 1;
            if (size > 1) new_step = static_cast<std::ptrdiff_t>(parent.at(static_cast<std::size_t>(first + r.step))) - new_start;
            return Axis{size, new_start, new_step};
        }
        auto positions = std::make_shared<std::vector<std::size_t>>(); positions->reserve(size);
        auto i = first;
        for (std::size_t k = 0; k < size; ++k) { positions->push_back(parent.at(static_cast<std::size_t>(i))); if (k + 1 < size) i += r.step; }
        return Axis{size, 0, 1, positions};
    }
    auto positions = std::make_shared<std::vector<std::size_t>>();
    if (selection.kind() == Selection::Kind::indices) {
        positions->reserve(selection.index_values().size());
        for (auto i : selection.index_values()) {
            if (i < 0) i += length;
            if (i < 0 || i >= length) fail(ErrorCode::out_of_bounds, path, "position is out of bounds");
            positions->push_back(parent.at(static_cast<std::size_t>(i)));
        }
    } else if (selection.kind() == Selection::Kind::mask) {
        const auto& mask = selection.mask_values();
        if (mask.size() != parent.size) fail(ErrorCode::shape_mismatch, path, "mask length differs from axis");
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] > 1) fail(ErrorCode::invalid_argument, path, "mask entries must be zero or one");
            if (mask[i]) positions->push_back(parent.at(i));
        }
    } else {
        if (!index) fail(ErrorCode::invalid_argument, path, "label selectors require a backed table index");
        check_node(index);
        if (selection.kind() == Selection::Kind::labels) require_unique_index(index, parent);
        std::unordered_map<std::string, std::vector<std::size_t>> found;
        for (const auto& query : selection.label_values()) found.try_emplace(query);
        const auto max_rows = std::max<std::size_t>(1, std::min<std::size_t>(8192,index->session->options.scratch_bytes/(2*sizeof(std::string))));
        // Read consecutive selected positions as a block. Arbitrary parent
        // selectors retain output order while requesting only their own labels.
        for (std::size_t i = 0; i < parent.size;) {
            std::size_t run = 1;
            while (run < max_rows && i + run < parent.size && parent.at(i + run) == parent.at(i) + run) ++run;
            auto values = index->session->backend->read_strings_bounded(index->path, {parent.at(i)}, {run}, index->session->options.scratch_bytes);
            if (values.empty() || values.size() > run) fail(ErrorCode::io_error, path, "index transfer returned an incorrect length");
            run = values.size();
            for (std::size_t j = 0; j < run; ++j) {
                if(index->session->options.validation==Validation::full)validate_utf8_string(values[j],index->path);
                auto it = found.find(values[j]);
                if (it != found.end()) it->second.push_back(parent.at(i + j));
            }
            i += run;
        }
        for (const auto& query : selection.label_values()) {
            const auto& matches = found.at(query);
            if (matches.empty()) fail(ErrorCode::out_of_bounds, path, "label does not exist: " + query);
            if (selection.kind() == Selection::Kind::labels && matches.size() != 1)
                fail(ErrorCode::invalid_argument, path, "label-list indexing requires unique matching axis names");
            positions->insert(positions->end(), matches.begin(), matches.end());
        }
    }
    return Axis{positions->size(), 0, 1, positions};
}
BackedAnnData make_store(std::unique_ptr<Backend> backend, BackedOptions options) {
    if (!backend) fail(ErrorCode::invalid_argument, "store", "backend is empty");
    if (options.scratch_bytes < 64) fail(ErrorCode::invalid_argument, "store", "scratch budget must be at least 64 bytes");
    if (options.decoded_cache_bytes) fail(ErrorCode::invalid_argument, "store", "decoded application caching is not implemented; use zero");
    auto session = std::make_shared<Session>(); session->backend = std::move(backend); session->options = options;
    if (encoding(session->backend->inspect("/"), "/") != "anndata") fail(ErrorCode::unsupported_encoding, "/", "root is not AnnData");
    const auto obs = logical_shape(*session, "/obs"); const auto var = logical_shape(*session, "/var");
    auto data = std::make_shared<detail::BackedDataset>(); data->session = std::move(session); data->obs = Axis{obs[0]}; data->var = Axis{var[0]};
    return BackedAnnData(std::move(data));
}
void check_dataset_children(const BackedAnnDataView& source) {
    const auto& data=detail::BackedAccess::get(source);require_dataset(data);
    const std::set<std::string,std::less<>> allowed{"obs","var","X","layers","obsm","varm","obsp","varp","uns","raw"};
    for(const auto& key:data->session->backend->children("/"))
        if(!allowed.contains(key))fail(ErrorCode::unsupported_encoding,child("/",key),"unknown AnnData root element cannot be materialized");
    const auto raw=data->session->backend->inspect("/raw");
    if(raw.exists){
        const auto code=encoding(raw,"/raw");
        if(code!="null" && (code!="raw" || !raw.group))fail(ErrorCode::unsupported_encoding,"/raw","expected raw encoding");
        if(raw.group)for(const auto& key:data->session->backend->children("/raw"))
            if(code=="null" || (key!="X"&&key!="var"&&key!="varm"))fail(ErrorCode::unsupported_encoding,child("/raw",key),"unknown raw element cannot be materialized");
    }
}
BackedAnnDataView root_view(const std::shared_ptr<detail::BackedDataset>& data) { return BackedAnnData(data).view(); }
} // namespace backed_detail

using backed_detail::Axis;
using backed_detail::node;
using backed_detail::child;
using backed_detail::encoding;
using backed_detail::logical_shape;

BackedAnnDataView::BackedAnnDataView() = default;
BackedAnnDataView::BackedAnnDataView(std::shared_ptr<detail::BackedDataset> d) : impl_(std::move(d)) {}
std::size_t BackedAnnDataView::n_obs() const { backed_detail::require_dataset(impl_); return impl_->obs.size; }
std::size_t BackedAnnDataView::n_vars() const { backed_detail::require_dataset(impl_); return impl_->var.size; }
bool BackedAnnDataView::has_x() const { backed_detail::require_dataset(impl_); return backed_detail::present(*impl_->session, "/X"); }
BackedArray BackedAnnDataView::x() const {
    backed_detail::require_dataset(impl_);
    if (!has_x()) throw Error(ErrorCode::out_of_bounds, "/X", "dataset has no X");
    return BackedArray(node(impl_->session, "/X", {impl_->obs, impl_->var}));
}
BackedTable BackedAnnDataView::obs() const { backed_detail::require_dataset(impl_); auto n=node(impl_->session, "/obs", {impl_->obs}); n->prune_categories=impl_->prune_categories; return BackedTable(std::move(n)); }
BackedTable BackedAnnDataView::var() const { backed_detail::require_dataset(impl_); auto n=node(impl_->session, "/var", {impl_->var}); n->prune_categories=impl_->prune_categories; return BackedTable(std::move(n)); }
BackedArray BackedAnnDataView::layer(std::string_view key) const { backed_detail::require_dataset(impl_); return BackedArray(node(impl_->session, child("/layers",key), {impl_->obs, impl_->var})); }
BackedElement BackedAnnDataView::obsm(std::string_view key) const { backed_detail::require_dataset(impl_); auto path=child("/obsm",key); auto n=node(impl_->session,path,backed_detail::aligned_axes(*impl_->session,path,impl_->obs)); n->prune_categories=impl_->prune_categories; return BackedElement(std::move(n)); }
BackedElement BackedAnnDataView::varm(std::string_view key) const { backed_detail::require_dataset(impl_); auto path=child("/varm",key); auto n=node(impl_->session,path,backed_detail::aligned_axes(*impl_->session,path,impl_->var)); n->prune_categories=impl_->prune_categories; return BackedElement(std::move(n)); }
BackedArray BackedAnnDataView::obsp(std::string_view key) const { backed_detail::require_dataset(impl_); auto path=child("/obsp",key); return BackedArray(node(impl_->session,path,backed_detail::aligned_axes(*impl_->session,path,impl_->obs,impl_->obs))); }
BackedArray BackedAnnDataView::varp(std::string_view key) const { backed_detail::require_dataset(impl_); auto path=child("/varp",key); return BackedArray(node(impl_->session,path,backed_detail::aligned_axes(*impl_->session,path,impl_->var,impl_->var))); }
std::vector<std::string> BackedAnnDataView::layer_names() const { return backed_detail::mapping_names(impl_,"/layers"); }
std::vector<std::string> BackedAnnDataView::obsm_names() const { return backed_detail::mapping_names(impl_,"/obsm"); }
std::vector<std::string> BackedAnnDataView::varm_names() const { return backed_detail::mapping_names(impl_,"/varm"); }
std::vector<std::string> BackedAnnDataView::obsp_names() const { return backed_detail::mapping_names(impl_,"/obsp"); }
std::vector<std::string> BackedAnnDataView::varp_names() const { return backed_detail::mapping_names(impl_,"/varp"); }
BackedElement BackedAnnDataView::uns() const {
    backed_detail::require_dataset(impl_);
    std::shared_ptr<detail::BackedNode> n;
    if(impl_->session->backend->inspect("/uns").exists)n=node(impl_->session,"/uns");
    else {
        n=std::make_shared<detail::BackedNode>();n->session=impl_->session;n->path="/uns";n->empty_mapping=true;
        auto& revision=impl_->session->revisions[n->path];if(!revision)revision=std::make_shared<backed_detail::Revision>();
        n->revision=revision;n->observed_revision=revision->value;n->observed_dataset_revision=impl_->session->dataset_revision;
    }
    n->slice_history=impl_->slice_history;return BackedElement(std::move(n));
}
bool BackedAnnDataView::has_raw() const { backed_detail::require_dataset(impl_); return backed_detail::present(*impl_->session,"/raw"); }
bool BackedAnnDataView::has_raw_x() const { return has_raw() && backed_detail::present(*impl_->session,"/raw/X"); }
BackedArray BackedAnnDataView::raw_x() const { if (!has_raw_x()) throw Error(ErrorCode::out_of_bounds,"/raw/X","dataset has no raw X"); auto shape=logical_shape(*impl_->session,"/raw/X"); if(shape.size()!=2) throw Error(ErrorCode::shape_mismatch,"/raw/X","raw X must have rank two"); return BackedArray(node(impl_->session,"/raw/X",{impl_->obs,Axis{shape[1]}})); }
BackedTable BackedAnnDataView::raw_var() const { if (!has_raw()) throw Error(ErrorCode::out_of_bounds,"/raw","dataset has no raw"); return BackedTable(node(impl_->session,"/raw/var")); }
BackedElement BackedAnnDataView::raw_varm(std::string_view key) const { if (!has_raw()) throw Error(ErrorCode::out_of_bounds,"/raw","dataset has no raw"); return BackedElement(node(impl_->session,child("/raw/varm",key))); }
std::vector<std::string> BackedAnnDataView::raw_varm_names() const { if(!has_raw()) return {}; return backed_detail::mapping_names(impl_,"/raw/varm"); }
BackedAnnDataView BackedAnnDataView::slice(Selection obs, Selection var) const {
    backed_detail::require_dataset(impl_);
    auto result=std::make_shared<detail::BackedDataset>(*impl_);
    auto obs_index=node(impl_->session,backed_detail::index_path(*impl_->session,"/obs"));
    auto var_index=node(impl_->session,backed_detail::index_path(*impl_->session,"/var"));
    result->obs=backed_detail::select_axis(impl_->obs,obs,obs_index);
    result->var=backed_detail::select_axis(impl_->var,var,var_index);
    auto history=std::make_shared<std::vector<backed_detail::SliceStage>>();
    if(impl_->slice_history)*history=*impl_->slice_history;
    history->push_back({impl_->obs,impl_->var,result->obs,result->var});
    result->slice_history=std::move(history);
    result->prune_categories=true;
    return BackedAnnDataView(std::move(result));
}
AnnData BackedAnnDataView::read() const {
    backed_detail::check_dataset_children(*this);
    auto result=AnnData::empty(n_obs(),n_vars());
    result.set_obs(obs().read()); result.set_var(var().read());
    if(has_x()) result.set_x(x().read());
    for(const auto& key:layer_names()) result.set_layer(key,layer(key).read());
    for(const auto& key:obsp_names()) result.set_obsp(key,obsp(key).read());
    for(const auto& key:varp_names()) result.set_varp(key,varp(key).read());
    auto aligned=[&](BackedElement value, const std::string& key, bool observations, bool raw){
        switch(value.kind()) {
            case ElementKind::array: if(raw) result.set_raw_varm(key,value.array().read()); else if(observations) result.set_obsm(key,value.array().read()); else result.set_varm(key,value.array().read()); break;
            case ElementKind::string_array: if(raw) result.set_raw_varm(key,value.string_array().read()); else if(observations) result.set_obsm(key,value.string_array().read()); else result.set_varm(key,value.string_array().read()); break;
            case ElementKind::table: if(raw) result.set_raw_varm(key,value.table().read()); else if(observations) result.set_obsm(key,value.table().read()); else result.set_varm(key,value.table().read()); break;
            default: throw Error(ErrorCode::unsupported_encoding,"aligned","aligned element must be numerical/string array or dataframe");
        }
    };
    for(const auto& key:obsm_names()) aligned(obsm(key),key,true,false);
    for(const auto& key:varm_names()) aligned(varm(key),key,false,false);
    if(impl_->session->backend->inspect("/uns").exists) result.set_uns(uns().read());
    if(has_raw()) {
        if(has_raw_x()) result.set_raw(raw_x().read(),raw_var().read()); else result.set_raw(raw_var().read());
        for(const auto& key:raw_varm_names()) aligned(raw_varm(key),key,false,true);
    }
    return result;
}
BackedAnnData::BackedAnnData()=default;
BackedAnnData::~BackedAnnData()=default;
BackedAnnData::BackedAnnData(BackedAnnData&&) noexcept=default;
BackedAnnData& BackedAnnData::operator=(BackedAnnData&&) noexcept=default;
BackedAnnData::BackedAnnData(std::shared_ptr<detail::BackedDataset> d):impl_(std::move(d)){}
bool BackedAnnData::is_open() const noexcept { return impl_ && impl_->session && impl_->session->open; }
bool BackedAnnData::writable() const { backed_detail::require_dataset(impl_); return impl_->session->options.mode==AccessMode::read_write; }
std::string BackedAnnData::location() const { backed_detail::require_dataset(impl_); return impl_->session->backend->location(); }
BackedAnnDataView BackedAnnData::view() const { backed_detail::require_dataset(impl_); return BackedAnnDataView(std::make_shared<detail::BackedDataset>(*impl_)); }
BackedAnnDataView BackedAnnData::slice(Selection o,Selection v) const {return view().slice(std::move(o),std::move(v));}
std::size_t BackedAnnData::n_obs() const{return view().n_obs();}
std::size_t BackedAnnData::n_vars() const{return view().n_vars();}
bool BackedAnnData::has_x() const{return view().has_x();}
BackedArray BackedAnnData::x() const{auto a=view().x();a.node_->writable=true;return a;}
BackedTable BackedAnnData::obs() const{auto a=view().obs();a.node_->writable=true;return a;}
BackedTable BackedAnnData::var() const{auto a=view().var();a.node_->writable=true;return a;}
BackedElement BackedAnnData::uns() const{auto a=view().uns();a.node_->writable=true;return a;}
BackedArray BackedAnnData::layer(std::string_view k) const{auto a=view().layer(k);a.node_->writable=true;return a;}
BackedElement BackedAnnData::obsm(std::string_view k) const{auto a=view().obsm(k);a.node_->writable=true;return a;}
BackedElement BackedAnnData::varm(std::string_view k) const{auto a=view().varm(k);a.node_->writable=true;return a;}
BackedArray BackedAnnData::obsp(std::string_view k) const{auto a=view().obsp(k);a.node_->writable=true;return a;}
BackedArray BackedAnnData::varp(std::string_view k) const{auto a=view().varp(k);a.node_->writable=true;return a;}
std::vector<std::string> BackedAnnData::layer_names() const{return view().layer_names();}
std::vector<std::string> BackedAnnData::obsm_names() const{return view().obsm_names();}
std::vector<std::string> BackedAnnData::varm_names() const{return view().varm_names();}
std::vector<std::string> BackedAnnData::obsp_names() const{return view().obsp_names();}
std::vector<std::string> BackedAnnData::varp_names() const{return view().varp_names();}
std::vector<std::string> BackedAnnData::raw_varm_names() const{return view().raw_varm_names();}
bool BackedAnnData::has_raw() const{return view().has_raw();}
bool BackedAnnData::has_raw_x() const{return view().has_raw_x();}
BackedArray BackedAnnData::raw_x() const{auto a=view().raw_x();a.node_->writable=true;return a;}
BackedTable BackedAnnData::raw_var() const{auto a=view().raw_var();a.node_->writable=true;return a;}
BackedElement BackedAnnData::raw_varm(std::string_view k) const{auto a=view().raw_varm(k);a.node_->writable=true;return a;}
AnnData BackedAnnData::read() const{return view().read();}
void BackedAnnData::flush(){backed_detail::require_dataset(impl_);impl_->session->backend->flush();}
void BackedAnnData::close(){if(is_open()){impl_->session->backend->close();impl_->session->open=false;}}
} // namespace anndata
