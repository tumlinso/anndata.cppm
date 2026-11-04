module;
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

module anndata.backed;
import :storage;

namespace anndata {
StorageGeometry BackedArray::storage_geometry() const {
    backed_detail::check_node(node_);
    auto& s = *node_->session;
    auto info = s.backend->inspect(node_->path);
    auto encoding = backed_detail::encoding(info, node_->path);
    StorageGeometry g;
    if (encoding == "csr_matrix" || encoding == "csc_matrix") {
        g.layout = encoding == "csr_matrix" ? Layout::csr : Layout::csc;
        g.source_shape = backed_detail::logical_shape(s, node_->path);
        auto data = s.backend->inspect(backed_detail::child(node_->path, "data"));
        g.data = data.physical; g.dtype = data.dtype;
        g.indices = s.backend->inspect(backed_detail::child(node_->path, "indices")).physical;
        g.offsets = s.backend->inspect(backed_detail::child(node_->path, "indptr")).physical;
        g.parallel_reads = data.parallel_reads;
    } else if ((encoding == "array" || encoding == "numeric-scalar") && !info.group && !info.strings && !info.record) {
        g.source_shape = info.shape; g.dtype = info.dtype; g.data = info.physical;
        g.parallel_reads = info.parallel_reads; g.parallel_writes = info.parallel_writes;
    } else throw Error(ErrorCode::unsupported_layout, node_->path, "storage geometry requires a numerical array");
    if (node_->color_override) {
        g.source_shape = {1}; g.axes = {{false, 0, 1}};
        g.shape = {node_->color_override->axis.size};
        return g;
    }
    g.shape = g.source_shape; g.axes.resize(g.shape.size());
    for (std::size_t i = 0; i < node_->axes.size(); ++i) {
        const auto& a = node_->axes[i];
        g.shape[i] = a.size; g.axes[i] = {!a.has_positions(), a.start, a.step};
    }
    g.writable = node_->writable && s.options.mode == AccessMode::read_write && !node_->code_recode;
    return g;
}
}
