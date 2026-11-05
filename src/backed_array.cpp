module;
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <variant>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

module anndatapp.backed;
import :storage;

namespace anndatapp {
namespace {
using backed_detail::Axis;
using backed_detail::Backend;
using backed_detail::NodeInfo;
using Node = detail::BackedNode;
using NodePtr = std::shared_ptr<Node>;

[[noreturn]] void fail(ErrorCode code, const NodePtr& n, std::string message) {
    throw Error(code, n ? n->path : "backed array", std::move(message));
}
std::size_t multiply(std::size_t a, std::size_t b, const NodePtr& n) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        fail(ErrorCode::invalid_argument, n, "array size overflows size_t");
    return a * b;
}
std::size_t elements(const std::vector<std::size_t>& shape, const NodePtr& n) {
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
    std::size_t result = 1;
    for (auto d : shape) result = multiply(result, d, n);
    return result;
}
bool integer_type(DType t) {
    return t == DType::int8 || t == DType::uint8 || t == DType::int16 || t == DType::uint16 ||
           t == DType::int32 || t == DType::uint32 || t == DType::int64 || t == DType::uint64;
}
Array allocate(DType t, std::vector<std::size_t> shape) {
    switch (t) {
    case DType::boolean: return Array::empty<bool>(std::move(shape));
    case DType::int8: return Array::empty<std::int8_t>(std::move(shape));
    case DType::uint8: return Array::empty<std::uint8_t>(std::move(shape));
    case DType::int16: return Array::empty<std::int16_t>(std::move(shape));
    case DType::uint16: return Array::empty<std::uint16_t>(std::move(shape));
    case DType::int32: return Array::empty<std::int32_t>(std::move(shape));
    case DType::uint32: return Array::empty<std::uint32_t>(std::move(shape));
    case DType::int64: return Array::empty<std::int64_t>(std::move(shape));
    case DType::uint64: return Array::empty<std::uint64_t>(std::move(shape));
    case DType::float32: return Array::empty<float>(std::move(shape));
    case DType::float64: return Array::empty<double>(std::move(shape));
    }
    throw Error(ErrorCode::unsupported_dtype, "backed array", "unknown numerical dtype");
}
struct Descriptor {
    Layout layout = Layout::dense;
    NodeInfo data, indices, offsets;
    std::vector<std::size_t> source_shape;
    std::vector<Axis> axes;
    std::vector<std::size_t> shape;
};
Descriptor describe(const NodePtr& n) {
    backed_detail::check_node(n);
    auto& s = *n->session;
    auto info = s.backend->inspect(n->path);
    if (!info.exists) fail(ErrorCode::out_of_bounds, n, "backed array does not exist");
    auto code = backed_detail::encoding(info, n->path);
    Descriptor d;
    if (n->color_override) {
        if (n->color_override->kind != backed_detail::ColorPromotion::numeric_array ||
            (code != "numeric-scalar" && code != "array") || info.group || info.strings || !info.shape.empty())
            fail(ErrorCode::unsupported_encoding, n, "invalid promoted scalar color");
        d.data = std::move(info); d.source_shape = {1}; d.axes = {n->color_override->axis};
        d.shape = {n->color_override->axis.size};
        if (d.axes[0].size && (d.axes[0].at(0) != 0 || d.axes[0].at(d.axes[0].size - 1) != 0))
            fail(ErrorCode::out_of_bounds, n, "promoted scalar color selection exceeds its single value");
        return d;
    }
    if (code == "array" || code == "numeric-scalar") {
        if (info.group || info.strings) fail(ErrorCode::unsupported_encoding, n, "numeric encoding is not a numeric dataset");
        d.data = std::move(info);
        d.source_shape = d.data.shape;
    } else if (code == "csr_matrix" || code == "csc_matrix") {
        if (!info.group) fail(ErrorCode::unsupported_encoding, n, "sparse encoding is not a group");
        auto children = s.backend->children(n->path);
        std::sort(children.begin(), children.end());
        if (children != std::vector<std::string>{"data", "indices", "indptr"})
            fail(ErrorCode::unsupported_encoding, n, "sparse group has unexpected or missing children");
        d.layout = code == "csr_matrix" ? Layout::csr : Layout::csc;
        d.source_shape = backed_detail::logical_shape(s, n->path);
        if (d.source_shape.size() != 2) fail(ErrorCode::shape_mismatch, n, "sparse shape must have rank two");
        d.data = s.backend->inspect(backed_detail::child(n->path, "data"));
        d.indices = s.backend->inspect(backed_detail::child(n->path, "indices"));
        d.offsets = s.backend->inspect(backed_detail::child(n->path, "indptr"));
        if (!d.data.exists || !d.indices.exists || !d.offsets.exists || d.data.group || d.indices.group || d.offsets.group ||
            d.data.strings || d.indices.strings || d.offsets.strings || d.data.shape.size() != 1 ||
            d.indices.shape.size() != 1 || d.offsets.shape.size() != 1)
            fail(ErrorCode::unsupported_encoding, n, "sparse buffers must be one-dimensional numeric datasets");
        if (!integer_type(d.indices.dtype) || !integer_type(d.offsets.dtype))
            fail(ErrorCode::unsupported_dtype, n, "sparse indices and offsets must have integer dtypes");
        const auto major = d.source_shape[d.layout == Layout::csr ? 0 : 1];
        if (major == std::numeric_limits<std::size_t>::max() || d.offsets.shape[0] != major + 1 || d.data.shape[0] != d.indices.shape[0])
            fail(ErrorCode::invalid_sparse_structure, n, "inconsistent sparse buffer lengths");
    } else {
        fail(ErrorCode::unsupported_encoding, n, "element is not a numerical array: " + code);
    }
    d.axes = n->axes;
    if (d.axes.empty()) {
        for (auto len : d.source_shape) d.axes.push_back(Axis{len, 0, 1, {}});
    }
    if (d.axes.size() != d.source_shape.size()) fail(ErrorCode::shape_mismatch, n, "selection rank differs from array rank");
    for (std::size_t i = 0; i < d.axes.size(); ++i) {
        if (d.axes[i].size && (d.axes[i].at(0) >= d.source_shape[i] || d.axes[i].at(d.axes[i].size - 1) >= d.source_shape[i]))
            fail(ErrorCode::out_of_bounds, n, "selection exceeds source extent");
        d.shape.push_back(d.axes[i].size);
    }
    return d;
}
std::ptrdiff_t add_stride(std::ptrdiff_t a, std::ptrdiff_t b, const NodePtr& n) {
    std::ptrdiff_t out;
    if (__builtin_add_overflow(a, b, &out)) fail(ErrorCode::invalid_argument, n, "buffer byte offset overflows ptrdiff_t");
    return out;
}
std::ptrdiff_t mul_stride(std::ptrdiff_t a, std::size_t b, const NodePtr& n) {
    if (b > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        fail(ErrorCode::invalid_argument, n, "buffer extent exceeds ptrdiff_t");
    std::ptrdiff_t out;
    if (__builtin_mul_overflow(a, static_cast<std::ptrdiff_t>(b), &out))
        fail(ErrorCode::invalid_argument, n, "buffer byte offset overflows ptrdiff_t");
    return out;
}
std::vector<std::ptrdiff_t> contiguous_strides(const std::vector<std::size_t>& shape, std::size_t width, const NodePtr& n) {
    std::vector<std::ptrdiff_t> result(shape.size());
    auto step = static_cast<std::ptrdiff_t>(width);
    for (std::size_t i = shape.size(); i-- > 0;) {
        result[i] = step;
        step = mul_stride(step, shape[i], n);
    }
    return result;
}
void validate_buffer(const Descriptor& d, const DenseBuffer& b, const NodePtr& n) {
    if (d.layout != Layout::dense) fail(ErrorCode::unsupported_layout, n, "dense destination supplied for sparse array");
    if (b.dtype != d.data.dtype) fail(ErrorCode::dtype_mismatch, n, "destination dtype does not match stored dtype");
    if (b.shape != d.shape) fail(ErrorCode::shape_mismatch, n, "destination shape does not match selected shape");
    if (!b.strides.empty() && b.strides.size() != b.shape.size()) fail(ErrorCode::shape_mismatch, n, "destination strides differ from rank");
    if (!b.data && elements(b.shape, n)) fail(ErrorCode::invalid_argument, n, "destination data is null");
    auto strides = b.strides.empty() ? contiguous_strides(b.shape, dtype_size(b.dtype), n) : b.strides;
    std::ptrdiff_t lo = 0, hi = 0;
    for (std::size_t i = 0; i < b.shape.size(); ++i) {
        if (b.shape[i] == 0) continue;
        auto span = mul_stride(strides[i], b.shape[i] - 1, n);
        if (span < 0) lo = add_stride(lo, span, n); else hi = add_stride(hi, span, n);
    }
    (void)lo; (void)hi;
}
bool forward_rectangle(const Descriptor& d) {
    for (const auto& axis : d.axes)
        if (axis.has_positions() || axis.step != 1) return false;
    return true;
}
std::size_t staging_budget(std::size_t available, std::size_t workspace, const NodePtr& n) {
    if (workspace > available)
        fail(ErrorCode::invalid_argument, n, "backend numerical workspace exceeds remaining scratch budget");
    return available - workspace;
}
std::vector<std::size_t> dense_block_shape(const Descriptor& d, std::size_t maximum_cells,
                                         const std::vector<std::size_t>& codec_tile, const NodePtr& n) {
    const auto rank = d.shape.size();
    bool advised = codec_tile.size() == rank;
    for (auto extent : codec_tile) advised = advised && extent != 0;
    std::vector<std::size_t> block(rank, 1);
    std::size_t cells = 1;
    if (advised) {
        for (std::size_t axis = 0; axis < rank; ++axis) {
            block[axis] = std::min(d.shape[axis], codec_tile[axis]);
            cells = multiply(cells, block[axis], n);
        }
        // When a complete unit cannot fit, keep a bounded rectangular portion
        // rather than discarding its outer-axis geometry.
        while (cells > maximum_cells) {
            const auto axis = static_cast<std::size_t>(std::max_element(block.begin(), block.end()) - block.begin());
            const auto extent = block[axis] / 2 + block[axis] % 2;
            cells = multiply(cells / block[axis], extent, n);
            block[axis] = extent;
        }
    }
    for (std::size_t axis = rank; axis-- > 0;) {
        const auto other_cells = cells / block[axis];
        auto extent = std::min(d.shape[axis], maximum_cells / other_cells);
        if (advised && extent < d.shape[axis] && extent >= codec_tile[axis])
            extent -= extent % codec_tile[axis];
        block[axis] = extent;
        cells = multiply(other_cells, extent, n);
    }
    return block;
}

struct ReadPoint { std::size_t physical, logical; };
struct ReadAxis {
    const Axis* axis = nullptr;
    std::size_t start = 0, count = 0;
    ReadPoint* points = nullptr;
    std::size_t logical(std::size_t i) const {
        return points ? points[i].logical : axis->step < 0 ? count-1-i : i;
    }
    std::size_t physical(std::size_t i) const {
        return points ? points[i].physical : axis->at(start+logical(i));
    }
    std::size_t lower(std::size_t value) const {
        std::size_t a=0,b=count;
        while(a<b) {auto k=a+(b-a)/2;if(physical(k)<value)a=k+1;else b=k;}
        return a;
    }
};

// Reordering scatter is valid only if distinct destination elements cannot
// alias. Conservative rejection preserves the original logical write order.
bool disjoint_dense(const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& strides,std::size_t width) {
    std::vector<std::size_t> order(shape.size());
    for(std::size_t i=0;i<order.size();++i)order[i]=i;
    auto magnitude=[&](std::size_t i) {auto s=strides[i];return s<0?std::size_t(-(s+1))+1:std::size_t(s);};
    std::sort(order.begin(),order.end(),[&](auto a,auto b){return magnitude(a)<magnitude(b);});
    std::size_t span=width;
    for(auto i:order) if(shape[i]>1) {
        const auto step=magnitude(i);
        if(step<span || (step && shape[i]-1>(std::numeric_limits<std::size_t>::max()-span)/step))return false;
        span+=(shape[i]-1)*step;
    }
    return true;
}

// Group only the independently selected axes; never construct their Cartesian
// product or scan unselected chunk coordinates. Each touched codec tile is
// read once, then scattered to its distinct logical destinations.
bool grouped_shard_read(const NodePtr& node,const Descriptor& d,const std::vector<std::size_t>& start,
                        const std::vector<std::size_t>& count,const std::vector<std::ptrdiff_t>& strides,
                        std::byte* output,backed_detail::NumericTransfer& source,
                        std::vector<std::byte>& tile,std::size_t available,std::size_t& tile_allocations) {
    const auto rank=count.size(),width=dtype_size(d.data.dtype);
    const auto& unit=d.data.codec_tile_shape;
    if(!rank || !d.data.sharded_payload || unit.size()!=rank || available<4096 || rank>(available-4096)/256)return false;
    if(!disjoint_dense(count,strides,width))return false;
    std::size_t mapped=0;
    for(std::size_t i=0;i<rank;++i) {
        if(!unit[i])return false;
        if(d.axes[i].has_positions()) {
            if(count[i]>(std::numeric_limits<std::size_t>::max()-mapped))return false;
            mapped+=count[i];
        }
    }
    if(mapped>available/sizeof(ReadPoint))return false;
    const auto map_bytes=mapped*sizeof(ReadPoint);
    if(rank>(std::numeric_limits<std::size_t>::max()-4096)/256)return false;
    const auto control=4096+rank*256;
    if(map_bytes>available || control>available-map_bytes)return false;
    const auto tile_bytes=multiply(elements(unit,node),width,node),limit=available-map_bytes-control;
    if(tile_bytes>limit)return false;
    if(tile.capacity()>limit || tile.capacity()<tile_bytes) {
        std::vector<std::byte>().swap(tile);tile.resize(tile_bytes);tile_allocations+=tile_bytes!=0;
    }
    else tile.resize(tile_bytes);

    std::vector<ReadAxis> axes(rank);
    std::unique_ptr<ReadPoint[]> points(mapped?new ReadPoint[mapped]:nullptr);
    std::size_t offset=0;
    for(std::size_t i=0;i<rank;++i) {
        auto& a=axes[i];a.axis=&d.axes[i];a.start=start[i];a.count=count[i];
        if(!a.axis->has_positions())continue;
        a.points=points.get()+offset;offset+=a.count;
        for(std::size_t j=0;j<a.count;++j) {
            const auto physical=a.axis->at(a.start+j);
            if(physical>=d.source_shape[i])fail(ErrorCode::out_of_bounds,node,"gathered read exceeds source extent");
            a.points[j]={physical,j};
        }
        std::sort(a.points,a.points+a.count,[](auto a,auto b){return a.physical<b.physical || (a.physical==b.physical && a.logical<b.logical);});
    }
    std::vector<std::size_t> lo(rank),hi(rank),base(rank),extent(rank),index(rank);
    auto end_group=[&](std::size_t axis,std::size_t first) {
        const auto b=(axes[axis].physical(first)/unit[axis])*unit[axis];
        return axes[axis].lower(b+std::min(unit[axis],d.source_shape[axis]-b));
    };
    for(std::size_t i=0;i<rank;++i)hi[i]=end_group(i,0);
    const bool canonical=d.data.dtype==DType::boolean && node->session->options.validation==Validation::full;
    while(true) {
        for(std::size_t i=0;i<rank;++i) {
            base[i]=(axes[i].physical(lo[i])/unit[i])*unit[i];
            extent[i]=std::min(unit[i],d.source_shape[i]-base[i]);
            index[i]=lo[i];
        }
        source.read(base,extent,tile.data());
        if(canonical)for(std::size_t i=0,n=elements(extent,node);i<n;++i)tile[i]=tile[i]==std::byte{0}?std::byte{0}:std::byte{1};
        while(true) {
            std::size_t from=0;std::ptrdiff_t to=0;
            for(std::size_t i=0;i<rank;++i) {
                from=from*extent[i]+axes[i].physical(index[i])-base[i];
                to+=std::ptrdiff_t(axes[i].logical(index[i]))*strides[i];
            }
            std::memcpy(output+to,tile.data()+from*width,width);
            std::size_t i=rank;
            while(i) {--i;if(++index[i]<hi[i])break;index[i]=lo[i];}
            if(!i && index[0]==lo[0])break;
        }
        std::size_t i=rank;
        while(i) {
            --i;lo[i]=hi[i];
            if(lo[i]<count[i]) {hi[i]=end_group(i,lo[i]);break;}
            lo[i]=0;hi[i]=end_group(i,0);
        }
        if(!i && lo[0]==0)break;
    }
    return true;
}
// Each transfer is a rectangle in storage. Contiguous selected runs go directly
// to/from the caller's buffer; non-contiguous memory uses a bounded packing tile.
void dense_transfer(const NodePtr& n, const Descriptor& d, const DenseBuffer& b, bool writing) {
    validate_buffer(d, b, n);
    const auto available_scratch = backed_detail::scratch_remaining(n);
    const auto count = elements(d.shape, n);
    if (!count) return;
    auto& backend = *n->session->backend;
    const auto stage_bytes = staging_budget(available_scratch,
        backend.numeric_workspace_bytes(d.data, writing), n);
    const auto width = dtype_size(d.data.dtype);
    auto strides = b.strides.empty() ? contiguous_strides(b.shape, width, n) : b.strides;
    auto transfer = [&](const std::vector<std::size_t>& start, const std::vector<std::size_t>& ext, std::byte* p) {
        if (writing) backend.write_numeric(n->path, start, ext, p);
        else {
            backend.read_numeric(n->path, start, ext, p);
            if (n->code_recode) {
                if (d.shape.size() != 1 || ext.size() != 1)
                    fail(ErrorCode::shape_mismatch, n, "categorical codes require a one-dimensional selection");
                backed_detail::recode_codes_buffer(n->code_recode, d.data.dtype, p, ext[0], static_cast<std::ptrdiff_t>(width), n->path);
            }
        }
    };
    auto* dest = static_cast<std::byte*>(b.data);
    if (n->color_override) {
        if (writing) fail(ErrorCode::invalid_argument, n, "promoted scalar colors are read-only");
        if (stage_bytes < 8) fail(ErrorCode::invalid_argument, n, "remaining scratch budget is smaller than one promoted scalar");
        std::array<std::byte, 8> scalar{};
        backend.read_numeric(n->path, {}, {}, scalar.data());
        if (d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
            scalar[0] = scalar[0] == std::byte{0} ? std::byte{0} : std::byte{1};
        for (std::size_t i = 0; i < d.shape[0]; ++i) {
            if (d.axes[0].at(i) != 0) fail(ErrorCode::out_of_bounds, n, "promoted scalar color selection exceeds its single value");
            std::memcpy(dest + mul_stride(strides[0], i, n), scalar.data(), width);
        }
        return;
    }
    const auto rank = d.shape.size();
    if (!rank) {
        transfer({}, {}, dest);
        if (!writing && d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
            *dest = *dest == std::byte{0} ? std::byte{0} : std::byte{1};
        return;
    }
    const bool packed = strides == contiguous_strides(d.shape, width, n);
    if (!writing && d.data.sharded_payload && !forward_rectangle(d) && !n->code_recode) {
        const auto workspace=std::max(backend.numeric_workspace_bytes(d.data,false),
                                      n->writable && n->session->options.mode==AccessMode::read_write?backend.numeric_workspace_bytes(d.data,true):std::size_t{0});
        const auto available=staging_budget(available_scratch,workspace,n);
        auto source=backend.prepare_numeric(n->path);
        std::vector<std::byte> tile;
        std::size_t tile_allocations=0;
        if(grouped_shard_read(n,d,std::vector<std::size_t>(rank,0),d.shape,strides,dest,*source,tile,available,tile_allocations))return;
    }
    if (packed && forward_rectangle(d)) {
        std::vector<std::size_t> start(rank);
        for (std::size_t axis = 0; axis < rank; ++axis) start[axis] = d.axes[axis].at(0);
        transfer(start, d.shape, dest);
        if (!writing && d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
            for (std::size_t i = 0; i < count; ++i) dest[i] = dest[i] == std::byte{0} ? std::byte{0} : std::byte{1};
        return;
    }
    std::size_t outer = 1;
    for (std::size_t i = 0; i + 1 < rank; ++i) outer = multiply(outer, d.shape[i], n);
    const auto last = rank - 1;
    std::vector<std::size_t> start(rank), ext(rank, 1), coord(rank, 0);
    const bool packing = strides[last] != static_cast<std::ptrdiff_t>(width);
    const auto run_budget = packing ? stage_bytes : available_scratch;
    if (run_budget < width) fail(ErrorCode::invalid_argument, n, "remaining scratch budget is smaller than one numeric element");
    auto max_run = run_budget / width;
    std::vector<std::byte> tile;
    for (std::size_t row = 0; row < outer; ++row) {
        auto rem = row;
        std::ptrdiff_t row_off = 0;
        for (std::size_t i = last; i-- > 0;) {
            coord[i] = rem % d.shape[i]; rem /= d.shape[i];
            start[i] = d.axes[i].at(coord[i]);
            row_off = add_stride(row_off, mul_stride(strides[i], coord[i], n), n);
        }
        std::size_t col = 0;
        while (col < d.shape[last]) {
            start[last] = d.axes[last].at(col);
            std::size_t run = 1;
            const auto limit = std::min(max_run, d.shape[last] - col);
            while (run < limit && d.axes[last].at(col + run) == start[last] + run) ++run;
            ext[last] = run;
            auto off = add_stride(row_off, mul_stride(strides[last], col, n), n);
            auto* out = dest + off;
            if (strides[last] == static_cast<std::ptrdiff_t>(width)) {
                transfer(start, ext, out);
                if (!writing && d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
                    for (std::size_t j = 0; j < run; ++j) out[j] = out[j] == std::byte{0} ? std::byte{0} : std::byte{1};
            } else {
                tile.resize(multiply(run, width, n));
                if (writing) for (std::size_t j = 0; j < run; ++j)
                    std::memcpy(tile.data() + j * width, out + mul_stride(strides[last], j, n), width);
                transfer(start, ext, tile.data());
                if (!writing) {
                    if (d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
                        for (std::size_t j = 0; j < run; ++j) tile[j] = tile[j] == std::byte{0} ? std::byte{0} : std::byte{1};
                    for (std::size_t j = 0; j < run; ++j)
                        std::memcpy(out + mul_stride(strides[last], j, n), tile.data() + j * width, width);
                }
            }
            col += run;
        }
    }
}
template<class T> std::size_t unpack_index(const std::byte* p, const NodePtr& n) {
    T value;
    std::memcpy(&value, p, sizeof(T));
    if constexpr (std::numeric_limits<T>::is_signed)
        if (value < 0) fail(ErrorCode::invalid_sparse_structure, n, "negative sparse index or offset");
    const auto wide = static_cast<std::uint64_t>(value);
    if (wide > std::numeric_limits<std::size_t>::max())
        fail(ErrorCode::invalid_sparse_structure, n, "sparse index or offset exceeds size_t");
    return static_cast<std::size_t>(wide);
}
template<class F> void visit_integer(DType t, const NodePtr& n, F&& f) {
    switch (t) {
    case DType::int8: return f.template operator()<std::int8_t>();
    case DType::uint8: return f.template operator()<std::uint8_t>();
    case DType::int16: return f.template operator()<std::int16_t>();
    case DType::uint16: return f.template operator()<std::uint16_t>();
    case DType::int32: return f.template operator()<std::int32_t>();
    case DType::uint32: return f.template operator()<std::uint32_t>();
    case DType::int64: return f.template operator()<std::int64_t>();
    case DType::uint64: return f.template operator()<std::uint64_t>();
    default: fail(ErrorCode::unsupported_dtype, n, "unsupported sparse integer dtype");
    }
}
std::size_t read_index(const std::byte* p, DType t, const NodePtr& n) {
    switch (t) {
    case DType::int8: return unpack_index<std::int8_t>(p, n);
    case DType::uint8: return unpack_index<std::uint8_t>(p, n);
    case DType::int16: return unpack_index<std::int16_t>(p, n);
    case DType::uint16: return unpack_index<std::uint16_t>(p, n);
    case DType::int32: return unpack_index<std::int32_t>(p, n);
    case DType::uint32: return unpack_index<std::uint32_t>(p, n);
    case DType::int64: return unpack_index<std::int64_t>(p, n);
    case DType::uint64: return unpack_index<std::uint64_t>(p, n);
    default: fail(ErrorCode::unsupported_dtype, n, "unsupported sparse integer dtype");
    }
}
template<class T> void pack_index(std::byte* p, std::size_t value, const NodePtr& n) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
        fail(ErrorCode::invalid_sparse_structure, n, "selected sparse index does not fit stored integer width");
    const auto v = static_cast<T>(value);
    std::memcpy(p, &v, sizeof v);
}
void write_index(std::byte* p, DType t, std::size_t value, const NodePtr& n) {
    switch (t) {
    case DType::int8: return pack_index<std::int8_t>(p, value, n);
    case DType::uint8: return pack_index<std::uint8_t>(p, value, n);
    case DType::int16: return pack_index<std::int16_t>(p, value, n);
    case DType::uint16: return pack_index<std::uint16_t>(p, value, n);
    case DType::int32: return pack_index<std::int32_t>(p, value, n);
    case DType::uint32: return pack_index<std::uint32_t>(p, value, n);
    case DType::int64: return pack_index<std::int64_t>(p, value, n);
    case DType::uint64: return pack_index<std::uint64_t>(p, value, n);
    default: fail(ErrorCode::unsupported_dtype, n, "unsupported sparse integer dtype");
    }
}
struct MinorLookup {
    const Axis& axis;
    std::unordered_map<std::size_t, std::vector<std::size_t>> explicit_indices;
    bool indexed = false;
    explicit MinorLookup(const Axis& a, std::size_t budget) : axis(a) {
        // The explicit selector already owns its positions. A bounded optional
        // lookup avoids allocating a hash table proportional to a huge axis;
        // the fallback scans the supplied selector without additional storage.
        indexed = axis.positions && axis.size <= budget / 128;
        if (indexed) for (std::size_t i = 0; i < axis.size; ++i) explicit_indices[axis.at(i)].push_back(i);
    }
    template<class F> void each(std::size_t physical, F&& f) const {
        if (axis.category_positions && !axis.positions) {
            if (physical > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return;
            const auto mapped = backed_detail::category_recode(axis.category_positions, static_cast<std::int64_t>(physical));
            if (mapped < 0) return;
            const auto candidate = static_cast<std::size_t>(mapped);
            if (!axis.size || axis.start < 0 || axis.step == 0) return;
            const auto start = static_cast<std::size_t>(axis.start);
            std::size_t distance, step;
            if (axis.step > 0) {
                if (candidate < start) return;
                distance = candidate - start; step = static_cast<std::size_t>(axis.step);
            } else {
                if (candidate > start) return;
                distance = start - candidate; step = static_cast<std::size_t>(-(axis.step + 1)) + 1;
            }
            if (distance % step == 0 && distance / step < axis.size) f(distance / step);
            return;
        }
        if (axis.positions) {
            if (indexed) {
                auto it = explicit_indices.find(physical);
                if (it != explicit_indices.end()) for (auto i : it->second) f(i);
            } else {
                for (std::size_t i = 0; i < axis.size; ++i) if (axis.at(i) == physical) f(i);
            }
            return;
        }
        if (!axis.size || axis.start < 0 || axis.step == 0) return;
        auto start = static_cast<std::size_t>(axis.start);
        std::size_t distance;
        std::size_t step;
        if (axis.step > 0) {
            if (physical < start) return;
            distance = physical - start;
            step = static_cast<std::size_t>(axis.step);
        } else {
            if (physical > start) return;
            distance = start - physical;
            step = static_cast<std::size_t>(-(axis.step + 1)) + 1;
        }
        if (distance % step == 0 && distance / step < axis.size) f(distance / step);
    }
};
// Scans one selected sparse major segment at a time. No buffer scales with the
// source nnz; duplicate minor selectors intentionally emit duplicate entries.
std::size_t scan_sparse(const NodePtr& n, const Descriptor& d, const SparseBuffer* out,
                        bool require_original_widths = false,
                        std::size_t scratch_limit = std::numeric_limits<std::size_t>::max()) {
    auto& backend = *n->session->backend;
    const auto major_axis = d.layout == Layout::csr ? std::size_t{0} : std::size_t{1};
    const auto minor_axis = 1 - major_axis;
    const auto& major = d.axes[major_axis];
    const auto& minor = d.axes[minor_axis];
    const auto nnz = d.data.shape[0];
    const auto isz = dtype_size(d.indices.dtype), osz = dtype_size(d.offsets.dtype), vsz = dtype_size(d.data.dtype);
    const auto indices_path = backed_detail::child(n->path, "indices");
    const auto offsets_path = backed_detail::child(n->path, "indptr");
    const auto data_path = backed_detail::child(n->path, "data");
    const auto scratch = std::min(backed_detail::scratch_remaining(n), scratch_limit);
    MinorLookup lookup(minor, scratch / 4);
    std::array<std::byte, 16> pair{};
    auto per = isz + (out ? vsz : 0);
    if (scratch / 2 < per) fail(ErrorCode::invalid_argument, n, "remaining scratch budget is smaller than one sparse transfer cell");
    auto chunk = (scratch / 2) / per;
    std::vector<std::byte> index_chunk, value_chunk;
    std::size_t emitted = 0;
    if (major.size == 0 && d.source_shape[major_axis] == 0 && n->session->options.validation == Validation::full) {
        backend.read_numeric(offsets_path, {0}, {1}, pair.data());
        if (read_index(pair.data(), d.offsets.dtype, n) != 0 || nnz != 0)
            fail(ErrorCode::invalid_sparse_structure, n, "empty sparse source has invalid boundary offsets");
    }
    const auto out_isz = out ? dtype_size(out->index_dtype) : isz;
    const auto out_osz = out ? dtype_size(out->offset_dtype) : osz;
    if (out) write_index(static_cast<std::byte*>(out->offsets), out->offset_dtype, 0, n);
    for (std::size_t m = 0; m < major.size; ++m) {
        const auto source_major = major.at(m);
        backend.read_numeric(offsets_path, {source_major}, {2}, pair.data());
        const auto begin = read_index(pair.data(), d.offsets.dtype, n);
        const auto end = read_index(pair.data() + osz, d.offsets.dtype, n);
        if (end < begin || end > nnz) fail(ErrorCode::invalid_sparse_structure, n, "sparse offsets are nonmonotonic or exceed nnz");
        if (n->session->options.validation == Validation::full &&
            ((source_major == 0 && begin != 0) || (source_major + 1 == d.source_shape[major_axis] && end != nnz)))
            fail(ErrorCode::invalid_sparse_structure, n, "sparse boundary offset differs from nnz");
        for (auto pos = begin; pos < end;) {
            const auto len = std::min(chunk, end - pos);
            index_chunk.resize(multiply(len, isz, n));
            backend.read_numeric(indices_path, {pos}, {len}, index_chunk.data());
            if (out) {
                value_chunk.resize(multiply(len, vsz, n));
                backend.read_numeric(data_path, {pos}, {len}, value_chunk.data());
            }
            visit_integer(d.indices.dtype, n, [&]<class I> { for (std::size_t j = 0; j < len; ++j) {
                const auto source_minor = unpack_index<I>(index_chunk.data() + j * sizeof(I), n);
                if (source_minor >= d.source_shape[minor_axis])
                    fail(ErrorCode::invalid_sparse_structure, n, "sparse index exceeds minor dimension");
                lookup.each(source_minor, [&](std::size_t selected_minor) {
                    if (emitted == std::numeric_limits<std::size_t>::max())
                        fail(ErrorCode::invalid_sparse_structure, n, "selected nnz overflows size_t");
                    if (out) {
                        if (emitted >= out->values_capacity || emitted >= out->indices_capacity)
                            fail(ErrorCode::shape_mismatch, n, "sparse output capacity became insufficient");
                        auto* values = static_cast<std::byte*>(out->values);
                        auto* indices = static_cast<std::byte*>(out->indices);
                        std::memcpy(values + emitted * vsz, value_chunk.data() + j * vsz, vsz);
                        if (d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
                            values[emitted] = values[emitted] == std::byte{0} ? std::byte{0} : std::byte{1};
                        write_index(indices + emitted * out_isz, out->index_dtype, selected_minor, n);
                    }
                    ++emitted;
                    if (require_original_widths) {
                        std::array<std::byte, 8> checked{};
                        pack_index<I>(checked.data(), selected_minor, n);
                    }
                });
            }});
            pos += len;
        }
        if (require_original_widths) {
            std::array<std::byte, 8> checked{};
            write_index(checked.data(), d.offsets.dtype, emitted, n);
        }
        if (out) write_index(static_cast<std::byte*>(out->offsets) + (m + 1) * out_osz, out->offset_dtype, emitted, n);
    }
    return emitted;
}
ReadRequirements requirements(const NodePtr& n, const Descriptor& d,
                              std::size_t scratch_limit = std::numeric_limits<std::size_t>::max(),
                              bool require_original_widths = false) {
    ReadRequirements r;
    r.layout = d.layout; r.dtype = d.data.dtype; r.shape = d.shape;
    if (d.layout != Layout::dense) {
        r.index_dtype = d.indices.dtype; r.offset_dtype = d.offsets.dtype;
        r.nnz = scan_sparse(n, d, nullptr, require_original_widths, scratch_limit);
    }
    return r;
}
void sparse_into(const NodePtr& n, const Descriptor& d, const ReadRequirements& r, const SparseBuffer& b) {
    if (d.layout == Layout::dense) fail(ErrorCode::unsupported_layout, n, "sparse destination supplied for dense array");
    if (b.dtype != r.dtype) fail(ErrorCode::dtype_mismatch, n, "sparse value dtype does not match stored dtype");
    if (!integer_type(b.index_dtype) || !integer_type(b.offset_dtype))
        fail(ErrorCode::unsupported_dtype, n, "sparse destination index and offset types must be integers");
    const auto major = r.shape[r.layout == Layout::csr ? 0 : 1];
    if (major == std::numeric_limits<std::size_t>::max() || b.values_capacity < r.nnz || b.indices_capacity < r.nnz || b.offsets_capacity < major + 1)
        fail(ErrorCode::shape_mismatch, n, "sparse destination capacities are insufficient");
    if (!b.offsets || (r.nnz && (!b.values || !b.indices)))
        fail(ErrorCode::invalid_argument, n, "sparse destination buffer is null");
    const auto value_bytes = multiply(r.nnz, dtype_size(b.dtype), n);
    const auto index_bytes = multiply(r.nnz, dtype_size(b.index_dtype), n);
    const auto offset_bytes = multiply(major + 1, dtype_size(b.offset_dtype), n);
    const auto max_addressable = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (value_bytes > max_addressable || index_bytes > max_addressable || offset_bytes > max_addressable)
        fail(ErrorCode::invalid_argument, n, "sparse output byte span exceeds ptrdiff_t");
    const auto actual = scan_sparse(n, d, &b);
    if (actual != r.nnz) fail(ErrorCode::invalid_sparse_structure, n, "sparse source changed after preparation");
}
// Compare the complete selected sparse structure before writing any value.
// Repeated selectors can address the same stored cell; writes follow selected
// CSR/CSC order, so the last supplied value for that cell wins.
void write_sparse(const NodePtr& n, const Descriptor& d, const ArrayView& source,
                  const SparseBytesView<const std::byte>& src) {
    const auto major_axis = d.layout == Layout::csr ? std::size_t{0} : std::size_t{1};
    const auto minor_axis = 1 - major_axis;
    const auto major = d.axes[major_axis].size;
    const auto isz = dtype_size(d.indices.dtype), osz = dtype_size(d.offsets.dtype);
    const auto src_isz = dtype_size(source.index_dtype()), src_osz = dtype_size(source.offset_dtype());
    const auto vsz = dtype_size(source.dtype());
    const auto indices_path = backed_detail::child(n->path, "indices");
    const auto offsets_path = backed_detail::child(n->path, "indptr");
    const auto values_path = backed_detail::child(n->path, "data");
    auto& backend = *n->session->backend;
    if (read_index(src.offsets(), source.offset_dtype(), n) != 0)
        fail(ErrorCode::invalid_sparse_structure, n, "source first sparse offset must be zero");
    auto process = [&](bool write) {
        const auto scratch = backed_detail::scratch_remaining(n);
        if (scratch / 2 < isz) fail(ErrorCode::invalid_argument, n, "remaining scratch budget is smaller than one sparse index");
        MinorLookup lookup(d.axes[minor_axis], scratch / 4);
        std::array<std::byte, 16> pair{};
        std::vector<std::byte> index_chunk;
        const auto chunk = (scratch / 2) / isz;
        std::size_t emitted = 0, pending_start = 0, pending_source = 0, pending_length = 0;
        auto flush = [&] {
            if (pending_length) backend.write_numeric(values_path, {pending_start}, {pending_length},
                                                      src.values() + pending_source * vsz);
            pending_length = 0;
        };
        for (std::size_t m = 0; m < major; ++m) {
            auto physical = d.axes[major_axis].at(m);
            backend.read_numeric(offsets_path, {physical}, {2}, pair.data());
            const auto begin = read_index(pair.data(), d.offsets.dtype, n);
            const auto end = read_index(pair.data() + osz, d.offsets.dtype, n);
            if (end < begin || end > d.data.shape[0])
                fail(ErrorCode::invalid_sparse_structure, n, "stored sparse offsets are invalid");
            if (n->session->options.validation == Validation::full &&
                ((physical == 0 && begin != 0) || (physical + 1 == d.source_shape[major_axis] && end != d.data.shape[0])))
                fail(ErrorCode::invalid_sparse_structure, n, "stored sparse boundary offset differs from nnz");
            for (auto pos = begin; pos < end;) {
                auto length = std::min(chunk, end - pos);
                index_chunk.resize(multiply(length, isz, n));
                backend.read_numeric(indices_path, {pos}, {length}, index_chunk.data());
                visit_integer(d.indices.dtype, n, [&]<class I> { for (std::size_t j = 0; j < length; ++j) {
                    const auto target_minor = unpack_index<I>(index_chunk.data() + j * sizeof(I), n);
                    if (target_minor >= d.source_shape[minor_axis])
                        fail(ErrorCode::invalid_sparse_structure, n, "stored sparse index exceeds minor dimension");
                    lookup.each(target_minor, [&](std::size_t selected_minor) {
                        if (emitted >= src.nnz() || read_index(src.indices() + emitted * src_isz, source.index_dtype(), n) != selected_minor)
                            fail(ErrorCode::invalid_sparse_structure, n, "sparse direct update would change the selected index structure");
                        if (write) {
                            const auto target = pos + j;
                            if (pending_length && (target != pending_start + pending_length || emitted != pending_source + pending_length)) flush();
                            if (!pending_length) { pending_start = target; pending_source = emitted; }
                            ++pending_length;
                        }
                        ++emitted;
                    });
                }});
                pos += length;
            }
            if (read_index(src.offsets() + (m + 1) * src_osz, source.offset_dtype(), n) != emitted)
                fail(ErrorCode::invalid_sparse_structure, n, "sparse direct update would change selected offsets");
        }
        if (emitted != src.nnz()) fail(ErrorCode::invalid_sparse_structure, n, "sparse direct update would change selected nnz");
        if (write) flush();
    };
    process(false);
    process(true);
}
// Copy the selected representation directly to a destination Backend. Tiles
// hold only bounded scratch; no selected Array or full-axis index is created.
void copy_dense(Backend& target, std::string_view path, const NodePtr& n,
                const Descriptor& d, std::size_t scratch_bytes) {
    const auto width = dtype_size(d.data.dtype);
    const auto element_count = elements(d.shape, n);
    if (element_count && scratch_bytes < width) fail(ErrorCode::invalid_argument, n, "scratch budget is smaller than one numeric element");
    target.create_numeric(path, d.data.dtype, d.shape, d.data.attrs, d.data.byte_order);
    if (!element_count) return;
    const auto rank = d.shape.size();
    auto& source = *n->session->backend;
    const auto target_info = target.inspect(path);
    const auto workspace = std::max(source.numeric_workspace_bytes(d.data, false),
                                    target.numeric_workspace_bytes(target_info, true));
    const auto stage_bytes = staging_budget(scratch_bytes, workspace, n);
    if (stage_bytes < width)
        fail(ErrorCode::invalid_argument, n, "remaining scratch budget is smaller than one numeric staging element");
    if (rank == 0) {
        std::array<std::byte, 8> scalar{};
        source.read_numeric(n->path, {}, {}, scalar.data());
        if (d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
            scalar[0] = scalar[0] == std::byte{0} ? std::byte{0} : std::byte{1};
        target.write_numeric(path, {}, {}, scalar.data());
        return;
    }
    if (forward_rectangle(d)) {
        // A backend accepts an arbitrary packed logical rectangle. Coalesce
        // adjacent rows/planes instead of decoding and encoding its chunks
        // again for each row. The same bounded buffer serves every block.
        const auto maximum_cells = stage_bytes / width;
        const auto block = dense_block_shape(d, maximum_cells, target_info.codec_tile_shape, n);
        const auto maximum_block_cells = elements(block, n);
        std::vector<std::byte> tile(multiply(maximum_block_cells, width, n));
        std::vector<std::size_t> src_start(rank), dst_start(rank, 0), count(rank);
        while (true) {
            std::size_t block_cells = 1;
            for (std::size_t axis = 0; axis < rank; ++axis) {
                src_start[axis] = d.axes[axis].at(dst_start[axis]);
                count[axis] = std::min(block[axis], d.shape[axis] - dst_start[axis]);
                block_cells = multiply(block_cells, count[axis], n);
            }
            source.read_numeric(n->path, src_start, count, tile.data());
            if (d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
                for (std::size_t i = 0; i < block_cells; ++i)
                    tile[i] = tile[i] == std::byte{0} ? std::byte{0} : std::byte{1};
            target.write_numeric(path, dst_start, count, tile.data());
            bool finished = true;
            for (std::size_t axis = rank; axis-- > 0;) {
                if (block[axis] < d.shape[axis] - dst_start[axis]) {
                    dst_start[axis] += block[axis]; finished = false; break;
                }
                dst_start[axis] = 0;
            }
            if (finished) return;
        }
    }
    std::size_t outer = 1;
    for (std::size_t i = 0; i + 1 < rank; ++i) outer = multiply(outer, d.shape[i], n);
    const auto last = rank - 1;
    const auto max_run = stage_bytes / width;
    std::vector<std::size_t> src_start(rank), dst_start(rank), count(rank, 1);
    std::vector<std::byte> tile;
    for (std::size_t row = 0; row < outer; ++row) {
        auto rem = row;
        for (std::size_t i = last; i-- > 0;) {
            dst_start[i] = rem % d.shape[i]; rem /= d.shape[i];
            src_start[i] = d.axes[i].at(dst_start[i]);
        }
        for (std::size_t col = 0; col < d.shape[last];) {
            src_start[last] = d.axes[last].at(col); dst_start[last] = col;
            const auto limit = std::min(max_run, d.shape[last] - col);
            std::size_t run = 1;
            while (run < limit && d.axes[last].at(col + run) == src_start[last] + run) ++run;
            count[last] = run;
            tile.resize(multiply(run, width, n));
            source.read_numeric(n->path, src_start, count, tile.data());
            if (d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
                for (std::size_t i = 0; i < run; ++i) tile[i] = tile[i] == std::byte{0} ? std::byte{0} : std::byte{1};
            target.write_numeric(path, dst_start, count, tile.data());
            col += run;
        }
    }
}
void copy_sparse(Backend& target, std::string_view path, const NodePtr& n,
                 const Descriptor& d, const ReadRequirements& r, std::size_t scratch_bytes) {
    const auto isz = dtype_size(d.indices.dtype), osz = dtype_size(d.offsets.dtype), vsz = dtype_size(d.data.dtype);
    const auto cell_bytes = 2 * (isz + vsz) + osz;
    if (scratch_bytes / 2 < cell_bytes)
        fail(ErrorCode::invalid_argument, n, "remaining scratch budget is smaller than one sparse transfer cell");
    const auto chunk = (scratch_bytes / 2) / cell_bytes;
    auto attrs = n->session->backend->inspect(n->path).attrs;
    std::vector<std::int64_t> shape;
    for (auto extent : d.shape) {
        if (extent > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
            fail(ErrorCode::invalid_argument, n, "sparse output shape exceeds encoding range");
        shape.push_back(static_cast<std::int64_t>(extent));
    }
    attrs["shape"] = std::move(shape);
    target.create_group(path, attrs);
    const auto out_data = backed_detail::child(path, "data");
    const auto out_indices = backed_detail::child(path, "indices");
    const auto out_offsets = backed_detail::child(path, "indptr");
    const auto major_axis = d.layout == Layout::csr ? std::size_t{0} : std::size_t{1};
    const auto minor_axis = 1 - major_axis;
    const auto& major = d.axes[major_axis];
    if (major.size == std::numeric_limits<std::size_t>::max()) fail(ErrorCode::invalid_argument, n, "sparse major extent overflows");
    target.create_numeric(out_data, d.data.dtype, {r.nnz}, d.data.attrs, d.data.byte_order);
    target.create_numeric(out_indices, d.indices.dtype, {r.nnz}, d.indices.attrs, d.indices.byte_order);
    target.create_numeric(out_offsets, d.offsets.dtype, {major.size + 1}, d.offsets.attrs, d.offsets.byte_order);
    auto& source = *n->session->backend;
    const auto in_data = backed_detail::child(n->path, "data");
    const auto in_indices = backed_detail::child(n->path, "indices");
    const auto in_offsets = backed_detail::child(n->path, "indptr");
    MinorLookup lookup(d.axes[minor_axis], scratch_bytes / 4);
    std::array<std::byte, 16> pair{};
    std::vector<std::byte> input_indices, input_values, values, indices, offsets;
    values.reserve(chunk * vsz); indices.reserve(chunk * isz); offsets.reserve(chunk * osz);
    std::size_t written = 0, emitted = 0, offset_written = 0;
    auto flush_values = [&] {
        const auto cells = values.size() / vsz;
        if (cells) {
            target.write_numeric(out_data, {written}, {cells}, values.data());
            target.write_numeric(out_indices, {written}, {cells}, indices.data());
            written += cells;
            values.clear(); indices.clear();
        }
    };
    auto flush_offsets = [&] {
        const auto cells = offsets.size() / osz;
        if (cells) {
            target.write_numeric(out_offsets, {offset_written}, {cells}, offsets.data());
            offset_written += cells; offsets.clear();
        }
    };
    auto add_offset = [&](std::size_t value) {
        const auto pos = offsets.size(); offsets.resize(pos + osz);
        write_index(offsets.data() + pos, d.offsets.dtype, value, n);
        if (offsets.size() / osz == chunk) flush_offsets();
    };
    add_offset(0);
    for (std::size_t m = 0; m < major.size; ++m) {
        const auto physical = major.at(m);
        source.read_numeric(in_offsets, {physical}, {2}, pair.data());
        const auto begin = read_index(pair.data(), d.offsets.dtype, n);
        const auto end = read_index(pair.data() + osz, d.offsets.dtype, n);
        if (end < begin || end > d.data.shape[0]) fail(ErrorCode::invalid_sparse_structure, n, "sparse offsets changed while copying");
        for (auto pos = begin; pos < end;) {
            const auto length = std::min(chunk, end - pos);
            input_indices.resize(multiply(length, isz, n));
            input_values.resize(multiply(length, vsz, n));
            source.read_numeric(in_indices, {pos}, {length}, input_indices.data());
            source.read_numeric(in_data, {pos}, {length}, input_values.data());
            visit_integer(d.indices.dtype, n, [&]<class I> { for (std::size_t j = 0; j < length; ++j) {
                const auto index = unpack_index<I>(input_indices.data() + j * sizeof(I), n);
                if (index >= d.source_shape[minor_axis]) fail(ErrorCode::invalid_sparse_structure, n, "sparse index exceeds minor dimension");
                lookup.each(index, [&](std::size_t selected) {
                    if (emitted >= r.nnz) fail(ErrorCode::invalid_sparse_structure, n, "sparse source changed while copying");
                    auto vo = values.size(); values.resize(vo + vsz);
                    std::memcpy(values.data() + vo, input_values.data() + j * vsz, vsz);
                    if (d.data.dtype == DType::boolean && n->session->options.validation == Validation::full)
                        values[vo] = values[vo] == std::byte{0} ? std::byte{0} : std::byte{1};
                    auto io = indices.size(); indices.resize(io + isz);
                    pack_index<I>(indices.data() + io, selected, n);
                    ++emitted;
                    if (values.size() / vsz == chunk) flush_values();
                });
            }});
            pos += length;
        }
        add_offset(emitted);
    }
    flush_values(); flush_offsets();
    if (emitted != r.nnz) fail(ErrorCode::invalid_sparse_structure, n, "sparse source changed while copying");
    if (&target == n->session->backend.get() || target.identity() == n->session->backend->identity()) {
        target.copy_attributes(in_data, out_data);
        target.copy_attributes(in_indices, out_indices);
        target.copy_attributes(in_offsets, out_offsets);
    }
}
} // namespace

namespace detail {
struct PreparedReadImpl {
    NodePtr node;
    ReadRequirements requirements;
};
}
namespace {
Array read_prepared(const NodePtr& n, const ReadRequirements& r, const Descriptor& d) {
    if (d.layout != r.layout || d.shape != r.shape || d.data.dtype != r.dtype)
        fail(ErrorCode::invalid_sparse_structure, n, "stored array changed after preparation");
    if (r.layout == Layout::dense) {
        auto result = allocate(r.dtype, r.shape);
        DenseBuffer b{result.edit().bytes().data(), r.dtype, r.shape, {}};
        dense_transfer(n, d, b, false);
        result.set_serialization_byte_order(d.data.byte_order);
        return result;
    }
    if (d.indices.dtype != r.index_dtype || d.offsets.dtype != r.offset_dtype)
        fail(ErrorCode::invalid_sparse_structure, n, "stored sparse dtypes changed after preparation");
    auto major = r.shape[r.layout == Layout::csr ? 0 : 1];
    auto values = allocate(r.dtype, {r.nnz});
    auto indices = allocate(r.index_dtype, {r.nnz});
    auto offsets = allocate(r.offset_dtype, {major + 1});
    SparseBuffer b{values.edit().bytes().data(), indices.edit().bytes().data(), offsets.edit().bytes().data(),
                   r.dtype, r.index_dtype, r.offset_dtype, r.nnz, r.nnz, major + 1};
    sparse_into(n, d, r, b);
    // sparse_into has already checked each source segment and index, generated
    // monotone offsets and checked both integer widths. A second complete core
    // validation would only rescan the freshly constructed buffers.
    auto result = Array::from_sparse_buffers(r.layout, std::move(values), std::move(indices), std::move(offsets),
                                              r.shape[0], r.shape[1], Validation::none);
    result.set_serialization_byte_order(d.data.byte_order);
    result.set_index_serialization_byte_order(d.indices.byte_order);
    result.set_offset_serialization_byte_order(d.offsets.byte_order);
    return result;
}
Array read_prepared(const NodePtr& n, const ReadRequirements& r) {
    return read_prepared(n, r, describe(n));
}
}
namespace backed_detail {
Array read_array(const NodePtr& n) {
    auto d = describe(n);
    return read_prepared(n, requirements(n, d), d);
}
void copy_array(Backend& destination, std::string_view dest_path, const NodePtr& n, std::size_t scratch_bytes) {
    auto d = describe(n);
    auto budget = std::min(scratch_bytes, backed_detail::scratch_remaining(n));
    if (n->color_override) { copy_promoted_color(destination, dest_path, n, budget); return; }
    if (n->code_recode) { copy_recoded_codes(destination, dest_path, n, budget, n->code_recode); return; }
    if (d.layout == Layout::dense) copy_dense(destination, dest_path, n, d, budget);
    else {
        const auto& source = *n->session->backend;
        const bool same_store = &destination == &source || destination.identity() == source.identity();
        if (!same_store) {
            const auto reject_opaque = [&](const NodeInfo& info, std::string_view name) {
                if (info.has_opaque_attrs)
                    throw Error(ErrorCode::unsupported_dtype, child(n->path, name),
                                "opaque backend attributes cannot be represented in a different store");
            };
            reject_opaque(d.data, "data");
            reject_opaque(d.indices, "indices");
            reject_opaque(d.offsets, "indptr");
        }
        auto r = requirements(n, d, budget, true);
        copy_sparse(destination, dest_path, n, d, r, budget);
    }
}
}
PreparedRead::PreparedRead() = default;
PreparedRead::PreparedRead(std::shared_ptr<detail::PreparedReadImpl> impl) : impl_(std::move(impl)) {}
ReadRequirements PreparedRead::requirements() const {
    if (!impl_) throw Error(ErrorCode::invalid_argument, "prepared read", "empty prepared read");
    backed_detail::check_node(impl_->node);
    return impl_->requirements;
}
Array PreparedRead::read() const {
    if (!impl_) throw Error(ErrorCode::invalid_argument, "prepared read", "empty prepared read");
    return read_prepared(impl_->node, impl_->requirements);
}
void PreparedRead::read_into(const DenseBuffer& b) const {
    if (!impl_) throw Error(ErrorCode::invalid_argument, "prepared read", "empty prepared read");
    auto d = describe(impl_->node);
    if (d.layout != impl_->requirements.layout || d.shape != impl_->requirements.shape || d.data.dtype != impl_->requirements.dtype)
        fail(ErrorCode::invalid_sparse_structure, impl_->node, "stored array changed after preparation");
    dense_transfer(impl_->node, d, b, false);
}
void PreparedRead::read_into(const SparseBuffer& b) const {
    if (!impl_) throw Error(ErrorCode::invalid_argument, "prepared read", "empty prepared read");
    auto d = describe(impl_->node);
    const auto& r = impl_->requirements;
    if (d.layout != r.layout || d.shape != r.shape || d.data.dtype != r.dtype || d.indices.dtype != r.index_dtype || d.offsets.dtype != r.offset_dtype)
        fail(ErrorCode::invalid_sparse_structure, impl_->node, "stored array changed after preparation");
    sparse_into(impl_->node, d, r, b);
}
BackedArray::BackedArray() = default;
BackedArray::BackedArray(std::shared_ptr<detail::BackedNode> node) : node_(std::move(node)) {}
Layout BackedArray::layout() const { return describe(node_).layout; }
DType BackedArray::dtype() const { return describe(node_).data.dtype; }
std::size_t BackedArray::rank() const { return describe(node_).shape.size(); }
std::size_t BackedArray::extent(std::size_t axis) const {
    auto d = describe(node_);
    if (axis >= d.shape.size()) fail(ErrorCode::out_of_bounds, node_, "axis is out of bounds");
    return d.shape[axis];
}
std::vector<std::size_t> BackedArray::shape() const { return describe(node_).shape; }
DType BackedArray::index_dtype() const {
    auto d = describe(node_);
    if (d.layout == Layout::dense) fail(ErrorCode::unsupported_layout, node_, "dense array has no index dtype");
    return d.indices.dtype;
}
DType BackedArray::offset_dtype() const {
    auto d = describe(node_);
    if (d.layout == Layout::dense) fail(ErrorCode::unsupported_layout, node_, "dense array has no offset dtype");
    return d.offsets.dtype;
}
BackedArray BackedArray::select(std::vector<Selection> selections) const {
    auto d = describe(node_);
    if (selections.size() != d.axes.size()) fail(ErrorCode::shape_mismatch, node_, "selection rank differs from array rank");
    for (std::size_t i = 0; i < d.axes.size(); ++i) d.axes[i] = backed_detail::select_axis(d.axes[i], selections[i]);
    if (node_->color_override) {
        auto selected = std::make_shared<detail::BackedNode>(*node_);
        selected->color_override = std::make_shared<backed_detail::ColorOverride>(*node_->color_override);
        selected->color_override->axis = d.axes.front();
        return BackedArray(std::move(selected));
    }
    auto selected = backed_detail::node(node_->session, node_->path, std::move(d.axes));
    selected->writable = node_->writable;
    selected->code_recode = node_->code_recode;
    selected->prune_categories = node_->prune_categories;
    return BackedArray(std::move(selected));
}
PreparedRead BackedArray::prepare_read() const {
    auto d = describe(node_);
    auto p = std::make_shared<detail::PreparedReadImpl>();
    p->node = node_; p->requirements = requirements(node_, d);
    return PreparedRead(std::move(p));
}
Array BackedArray::read() const { return backed_detail::read_array(node_); }
void BackedArray::read_into(const DenseBuffer& b) const { auto d = describe(node_); dense_transfer(node_, d, b, false); }
void BackedArray::read_into(const SparseBuffer& b) const {
    auto d = describe(node_);
    sparse_into(node_, d, requirements(node_, d), b);
}
MappedArray BackedArray::map_readonly() const {
    auto d = describe(node_);
    if (node_->code_recode || node_->color_override) fail(ErrorCode::unsupported_layout, node_, "transformed arrays cannot be memory mapped");
    if (d.layout != Layout::dense) fail(ErrorCode::unsupported_layout, node_, "sparse groups cannot be memory mapped");
    for (std::size_t i = 0; i < d.axes.size(); ++i)
        if (!d.axes[i].all(d.source_shape[i])) fail(ErrorCode::non_contiguous, node_, "only an unselected full dataset can be mapped");
    auto result = node_->session->backend->map_numeric(node_->path);
    if (!result) fail(ErrorCode::unsupported_layout, node_, "dataset cannot be mapped as native contiguous read-only bytes");
    return std::move(*result);
}
namespace {
backed_detail::NumericWrite selected_write(const Descriptor& d,const std::vector<std::size_t>& start,
                                          const std::vector<std::size_t>& count,std::size_t budget) {
    backed_detail::NumericWrite w;w.scratch_bytes=budget;w.axes.reserve(count.size());
    for(std::size_t i=0;i<count.size();++i) {
        const auto& a=d.axes[i];
        if(a.category_positions) throw Error(ErrorCode::unsupported_layout,"selected write","transformed axes are read-only");
        w.axes.push_back({count[i],a.positions?0:std::ptrdiff_t(a.at(start[i])),a.step,a.positions?a.positions->data()+start[i]:nullptr});
    }
    return w;
}
std::size_t packing_metadata(const Descriptor& d,const NodePtr& node) {
    std::size_t positions=0;
    for(std::size_t i=0;i<d.shape.size();++i) {
        const auto chunk=d.data.codec_tile_shape.empty()?d.shape[i]:d.data.codec_tile_shape[i];
        positions+=std::min(chunk,d.shape[i]);
    }
    return multiply(positions,6*sizeof(std::size_t),node)+multiply(d.shape.size(),4*sizeof(anndatapp::Axis)+1024,node)+2048;
}
template<class T> void pack_view(void* context,const std::vector<std::vector<std::size_t>>& positions,void* out) {
    const auto& source=*static_cast<const ArrayView*>(context);
    std::vector<anndatapp::Axis> axes;axes.reserve(positions.size());
    std::vector<std::size_t> shape;shape.reserve(positions.size());
    for(const auto& a:positions) {axes.push_back(anndatapp::Axis::gather(a.data(),a.size()));shape.push_back(a.size());}
    std::vector<std::ptrdiff_t> strides(shape.size());std::ptrdiff_t stride=1;
    for(std::size_t i=shape.size();i--;) {strides[i]=stride;stride*=std::ptrdiff_t(shape[i]);}
    auto destination=Array::borrow_dense(static_cast<T*>(out),std::move(shape),std::move(strides),Validation::none);
    source.select(axes.data(),axes.size()).copy_into(destination.edit(),Validation::none);
}
auto view_packer(DType type) -> void (*)(void*,const std::vector<std::vector<std::size_t>>&,void*) {
    switch(type) {
    case DType::boolean:return pack_view<bool>;
    case DType::int8:return pack_view<std::int8_t>;case DType::uint8:return pack_view<std::uint8_t>;
    case DType::int16:return pack_view<std::int16_t>;case DType::uint16:return pack_view<std::uint16_t>;
    case DType::int32:return pack_view<std::int32_t>;case DType::uint32:return pack_view<std::uint32_t>;
    case DType::int64:return pack_view<std::int64_t>;case DType::uint64:return pack_view<std::uint64_t>;
    case DType::float32:return pack_view<float>;case DType::float64:return pack_view<double>;
    }
    throw Error(ErrorCode::unsupported_dtype,"selected write","unsupported numerical dtype");
}
void write_gathered_view(const NodePtr& node,const Descriptor& d,const ArrayView& source) {
    if(node->session->options.validation==Validation::full) source.validate();
    const auto budget=backed_detail::scratch_remaining(node),rank=d.shape.size(),width=dtype_size(d.data.dtype);
    if(!elements(d.shape,node)) return;
    auto backend=node->session->backend->fork(budget);
    if(backend->identity()!=node->session->backend->identity()) fail(ErrorCode::invalid_argument,node,"store identity changed during write preparation");
    auto transfer=backend->prepare_numeric(node->path);
    auto input=selected_write(d,std::vector<std::size_t>(rank,0),d.shape,budget);
    input.context=const_cast<ArrayView*>(&source);input.pack=view_packer(d.data.dtype);input.packing_metadata_bytes=packing_metadata(d,node);
    if(d.data.sharded_payload && transfer->write_selection(input)) return;
    // Bounded gathering for the remaining numerical backends. There is no
    // owning copy of the complete source selection.
    const auto workspace=backend->numeric_workspace_bytes(d.data,true),available=staging_budget(budget,workspace,node);
    const auto metadata=multiply(rank,4*sizeof(anndatapp::Axis)+1024,node)+2048;
    if(available<=metadata || (available-metadata)/(width+6*sizeof(std::size_t))<1)
        fail(ErrorCode::invalid_argument,node,"gathered source packing exceeds scratch memory budget");
    if(forward_rectangle(d)) {
        auto maximum=std::min<std::size_t>(1024*1024,available-metadata)/width;
        std::vector<std::size_t> block;
        while(maximum) {
            block=dense_block_shape(d,maximum,d.data.codec_tile_shape,node);
            std::size_t axis_positions=0;for(auto e:block)axis_positions+=e;
            const auto needed=multiply(elements(block,node),width,node)+multiply(axis_positions,6*sizeof(std::size_t),node);
            if(needed<=available-metadata)break;
            maximum/=2;
        }
        if(!maximum)fail(ErrorCode::invalid_argument,node,"gathered rectangle cannot fit scratch memory budget");
        std::vector<std::byte> packed(multiply(elements(block,node),width,node));
        std::vector<std::vector<std::size_t>> positions(rank);
        std::vector<std::size_t> logical(rank,0),physical(rank),count(rank);
        while(true) {
            for(std::size_t i=0;i<rank;++i) {
                count[i]=std::min(block[i],d.shape[i]-logical[i]);physical[i]=d.axes[i].at(logical[i]);
                positions[i].resize(count[i]);for(std::size_t j=0;j<count[i];++j)positions[i][j]=logical[i]+j;
            }
            input.pack(input.context,positions,packed.data());transfer->write(physical,count,packed.data());
            bool finished=true;
            for(std::size_t i=rank;i--;) {if(block[i]<d.shape[i]-logical[i]){logical[i]+=block[i];finished=false;break;}logical[i]=0;}
            if(finished)return;
        }
    }
    const auto maximum=std::min<std::size_t>(1024*1024,available-metadata)/(width+6*sizeof(std::size_t));
    const auto columns=rank?d.shape.back():1;
    std::vector<std::byte> packed(multiply(std::min(columns,maximum),width,node));
    std::vector<std::vector<std::size_t>> positions(rank);
    std::vector<std::size_t> physical(rank,0),count(rank,1);
    const auto outer=elements(d.shape,node)/columns;
    for(std::size_t row=0;row<outer;++row) {
        auto k=row;
        for(std::size_t i=rank?rank-1:0;i--;) {const auto j=k%d.shape[i];k/=d.shape[i];positions[i]={j};physical[i]=d.axes[i].at(j);}
        for(std::size_t first=0;first<columns;) {
            const auto limit=std::min(maximum,columns-first);std::size_t run=1;
            if(rank) {physical.back()=d.axes.back().at(first);while(run<limit && d.axes.back().at(first+run)==physical.back()+run)++run;positions.back().resize(run);for(std::size_t j=0;j<run;++j)positions.back()[j]=first+j;count.back()=run;}
            input.pack(input.context,positions,packed.data());transfer->write(physical,count,packed.data());first+=run;
        }
    }
}
}
void BackedArray::write(const ArrayView& source) const {
    auto d = describe(node_);
    if (node_->code_recode || node_->color_override) fail(ErrorCode::invalid_argument, node_, "transformed arrays are read-only");
    node_->session->check_writable();
    if (!node_->writable) fail(ErrorCode::invalid_argument, node_, "backed view is read-only");
    if (source.layout() != d.layout) fail(ErrorCode::unsupported_layout, node_, "source layout differs from stored layout");
    if (source.dtype() != d.data.dtype) fail(ErrorCode::dtype_mismatch, node_, "source dtype differs from stored dtype");
    if (source.rank() != d.shape.size()) fail(ErrorCode::shape_mismatch, node_, "source rank differs from selected shape");
    for (std::size_t i = 0; i < d.shape.size(); ++i)
        if (source.extent(i) != d.shape[i]) fail(ErrorCode::shape_mismatch, node_, "source shape differs from selected shape");
    if (d.layout != Layout::dense) {
        // A gathered sparse view lacks a direct descriptor; materialize only that
        // explicit source selection. The destination remains a direct update.
        std::optional<Array> gathered;
        std::optional<SparseBytesView<const std::byte>> bytes;
        try { bytes.emplace(source.sparse_bytes()); }
        catch (const Error& e) {
            if (e.code() != ErrorCode::non_contiguous) throw;
            gathered.emplace(source.copy());
            bytes.emplace(gathered->view().sparse_bytes());
        }
        const auto value = gathered ? gathered->view() : source;
        if (node_->session->options.validation == Validation::full) value.validate();
        write_sparse(node_, d, value, *bytes);
        return;
    }
    std::optional<DenseBytesView<const std::byte>> bytes;
    try { bytes.emplace(source.bytes()); }
    catch (const Error& e) {
        if (e.code() != ErrorCode::non_contiguous) throw;
        write_gathered_view(node_,d,source); return;
    }
    std::vector<std::ptrdiff_t> strides;
    for (std::size_t i = 0; i < bytes->rank(); ++i) strides.push_back(bytes->stride_bytes(i));
    DenseBuffer b{const_cast<std::byte*>(bytes->data()), d.data.dtype, d.shape, std::move(strides)};
    if(d.data.sharded_payload) {auto transfer=prepare_transfer();transfer.write_from(std::vector<std::size_t>(d.shape.size(),0),d.shape,b);}
    else dense_transfer(node_, d, b, true);
}
namespace detail {
struct TransferContextImpl {
    NodePtr node;
    Descriptor descriptor;
    StorageGeometry geometry;
    std::unique_ptr<Backend> backend;
    std::unique_ptr<backed_detail::NumericTransfer> data, indices, offsets;
    std::vector<std::byte> tile;
    std::size_t budget = 0, tile_allocations = 0;
    bool sparse_prepared = false;
    std::vector<std::size_t> sparse_start, sparse_count;
    std::size_t sparse_begin = 0, sparse_end = 0;

    void check() const { backed_detail::check_node(node); }
    void block(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count) const {
        check();
        const auto& shape = descriptor.shape;
        if (start.size() != shape.size() || count.size() != shape.size())
            fail(ErrorCode::shape_mismatch, node, "transfer block rank differs from array rank");
        for (std::size_t i = 0; i < shape.size(); ++i)
            if (start[i] > shape[i] || count[i] > shape[i] - start[i])
                fail(ErrorCode::out_of_bounds, node, "transfer block exceeds selected array");
    }
    ReadRequirements requirements(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count) {
        block(start, count);
        const auto& d = descriptor;
        ReadRequirements r;
        r.layout = d.layout; r.dtype = d.data.dtype; r.shape = count;
        if (d.layout == Layout::dense) return r;
        const auto major = d.layout == Layout::csr ? 0u : 1u, minor = 1u - major;
        if (start[minor] != 0 || count[minor] != d.shape[minor])
            fail(ErrorCode::unsupported_layout, node, "sparse transfer requires the complete minor axis");
        r.index_dtype = d.indices.dtype; r.offset_dtype = d.offsets.dtype;
        if (sparse_prepared && start == sparse_start && count == sparse_count) {
            r.nnz = sparse_end - sparse_begin;
            return r;
        }
        sparse_prepared = false;
        std::array<std::byte, 16> pair{};
        const auto first = static_cast<std::size_t>(d.axes[major].start) + start[major];
        const auto osz = dtype_size(d.offsets.dtype);
        offsets->read({first}, {1}, pair.data());
        offsets->read({first + count[major]}, {1}, pair.data() + osz);
        const auto begin = read_index(pair.data(), d.offsets.dtype, node);
        const auto end = read_index(pair.data() + osz, d.offsets.dtype, node);
        if (end < begin || end > d.data.shape[0])
            fail(ErrorCode::invalid_sparse_structure, node, "sparse block offsets are invalid");
        if (node->session->options.validation == Validation::full &&
            ((first == 0 && begin != 0) || (first + count[major] == d.source_shape[major] && end != d.data.shape[0])))
            fail(ErrorCode::invalid_sparse_structure, node, "sparse boundary offset differs from nnz");
        sparse_start = start; sparse_count = count; sparse_begin = begin; sparse_end = end; sparse_prepared = true;
        r.nnz = end - begin;
        return r;
    }
    void dense(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count,
               const DenseBuffer& b, bool writing) {
        block(start, count);
        const auto& d = descriptor;
        if (d.layout != Layout::dense) fail(ErrorCode::unsupported_layout, node, "dense buffer supplied for sparse transfer");
        Descriptor selected; selected.layout = Layout::dense; selected.data.dtype = d.data.dtype; selected.shape = count;
        validate_buffer(selected, b, node);
        if (writing) {
            node->session->check_writable();
            if (!node->writable) fail(ErrorCode::invalid_argument, node, "backed view is read-only");
        }
        const auto n = elements(count, node), width = dtype_size(d.data.dtype), rank = count.size();
        if (!n) return;
        auto strides = b.strides.empty() ? contiguous_strides(count, width, node) : b.strides;
        auto* p = static_cast<std::byte*>(b.data);
        if(writing && d.data.sharded_payload) {
            auto selected=selected_write(d,start,count,budget-tile.capacity());selected.data=b.data;selected.strides=strides;
            if(data->write_selection(selected)) return;
        }
        if(!writing && d.data.sharded_payload && !forward_rectangle(d)) {
            const auto workspace=std::max(backend->numeric_workspace_bytes(d.data,false),
                                          geometry.writable?backend->numeric_workspace_bytes(d.data,true):std::size_t{0});
            const auto available=staging_budget(budget,workspace,node);
            if(grouped_shard_read(node,d,start,count,strides,p,*data,tile,available,tile_allocations))return;
        }
        const bool canonical = !writing && d.data.dtype == DType::boolean && node->session->options.validation == Validation::full;
        auto transfer = [&](const std::vector<std::size_t>& source, const std::vector<std::size_t>& ext, std::byte* q) {
            if (writing) data->write(source, ext, q); else data->read(source, ext, q);
            if (canonical) for (std::size_t j = 0, m = elements(ext, node); j < m; ++j)
                q[j] = q[j] == std::byte{0} ? std::byte{0} : std::byte{1};
        };
        if (!rank) { transfer({}, {}, p); return; }
        std::vector<std::size_t> source(rank), ext(rank, 1);
        if (forward_rectangle(d) && strides == contiguous_strides(count, width, node)) {
            for (std::size_t i = 0; i < rank; ++i) source[i] = d.axes[i].at(start[i]);
            transfer(source, count, p);
            return;
        }
        const auto last = rank - 1;
        const bool packed = strides[last] == static_cast<std::ptrdiff_t>(width);
        const auto maximum = packed ? std::numeric_limits<std::size_t>::max() : tile.size() / width;
        if (!maximum) fail(ErrorCode::invalid_argument, node, "scratch budget cannot stage one numerical value");
        const auto outer = n / count[last];
        for (std::size_t row = 0; row < outer; ++row) {
            auto rem = row;
            std::ptrdiff_t off = 0;
            for (std::size_t i = last; i-- > 0;) {
                const auto j = rem % count[i]; rem /= count[i];
                source[i] = d.axes[i].at(start[i] + j);
                off = add_stride(off, mul_stride(strides[i], j, node), node);
            }
            for (std::size_t col = 0; col < count[last];) {
                source[last] = d.axes[last].at(start[last] + col);
                std::size_t run = 1;
                const auto limit = std::min(maximum, count[last] - col);
                while (run < limit && d.axes[last].at(start[last] + col + run) == source[last] + run) ++run;
                ext[last] = run;
                auto* q = p + add_stride(off, mul_stride(strides[last], col, node), node);
                if (packed) transfer(source, ext, q);
                else {
                    if (writing) for (std::size_t j = 0; j < run; ++j)
                        std::memcpy(tile.data() + j*width, q + mul_stride(strides[last], j, node), width);
                    transfer(source, ext, tile.data());
                    if (!writing) for (std::size_t j = 0; j < run; ++j)
                        std::memcpy(q + mul_stride(strides[last], j, node), tile.data() + j*width, width);
                }
                col += run;
            }
        }
    }
};
}
TransferContext::TransferContext() = default;
TransferContext::~TransferContext() = default;
TransferContext::TransferContext(TransferContext&&) noexcept = default;
TransferContext& TransferContext::operator=(TransferContext&&) noexcept = default;
TransferContext::TransferContext(std::unique_ptr<detail::TransferContextImpl> p) : impl_(std::move(p)) {}
namespace {
detail::TransferContextImpl& transfer_context(const std::unique_ptr<detail::TransferContextImpl>& p) {
    if (!p) throw Error(ErrorCode::invalid_argument, "transfer context", "empty transfer context");
    p->check(); return *p;
}
}
StorageGeometry TransferContext::geometry() const { return transfer_context(impl_).geometry; }
std::size_t TransferContext::scratch_capacity() const { return transfer_context(impl_).budget; }
ReadRequirements TransferContext::requirements(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count) {
    return transfer_context(impl_).requirements(start, count);
}
void TransferContext::read_into(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, const DenseBuffer& b) {
    transfer_context(impl_).dense(start, count, b, false);
}
void TransferContext::write_from(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, const DenseBuffer& b) {
    transfer_context(impl_).dense(start, count, b, true);
}
void TransferContext::read_into(const std::vector<std::size_t>& start, const std::vector<std::size_t>& count, const SparseBuffer& b) {
    auto& p = transfer_context(impl_);
    const auto r = p.requirements(start, count);
    if (r.layout == Layout::dense) fail(ErrorCode::unsupported_layout, p.node, "sparse buffer supplied for dense transfer");
    if (b.dtype != r.dtype || b.index_dtype != r.index_dtype || b.offset_dtype != r.offset_dtype)
        fail(ErrorCode::dtype_mismatch, p.node, "sparse transfer requires the stored numerical widths");
    const auto major = r.layout == Layout::csr ? 0u : 1u, minor = 1u - major;
    const auto m = count[major], vsz = dtype_size(r.dtype), isz = dtype_size(r.index_dtype), osz = dtype_size(r.offset_dtype);
    if (m == std::numeric_limits<std::size_t>::max() || b.values_capacity < r.nnz || b.indices_capacity < r.nnz || b.offsets_capacity < m + 1)
        fail(ErrorCode::shape_mismatch, p.node, "sparse destination capacities are insufficient");
    if (!b.offsets || (r.nnz && (!b.values || !b.indices))) fail(ErrorCode::invalid_argument, p.node, "sparse destination buffer is null");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (multiply(r.nnz, vsz, p.node) > limit || multiply(r.nnz, isz, p.node) > limit || multiply(m + 1, osz, p.node) > limit)
        fail(ErrorCode::invalid_argument, p.node, "sparse destination byte span exceeds ptrdiff_t");
    const auto first = static_cast<std::size_t>(p.descriptor.axes[major].start) + start[major];
    auto* out = static_cast<std::byte*>(b.offsets);
    p.offsets->read({first}, {m + 1}, out);
    std::size_t previous = p.sparse_begin;
    for (std::size_t j = 0; j <= m; ++j) {
        const auto v = read_index(out + j*osz, r.offset_dtype, p.node);
        if (v < previous || v > p.sparse_end || (j == 0 && v != p.sparse_begin) || (j == m && v != p.sparse_end))
            fail(ErrorCode::invalid_sparse_structure, p.node, "sparse structure changed after block preparation");
        write_index(out + j*osz, r.offset_dtype, v - p.sparse_begin, p.node); previous = v;
    }
    if (!r.nnz) return;
    p.indices->read({p.sparse_begin}, {r.nnz}, b.indices);
    if (p.node->session->options.validation == Validation::full)
        visit_integer(r.index_dtype, p.node, [&]<class I> { for (std::size_t j = 0; j < r.nnz; ++j)
            if (unpack_index<I>(static_cast<const std::byte*>(b.indices) + j*sizeof(I), p.node) >= r.shape[minor])
                fail(ErrorCode::invalid_sparse_structure, p.node, "sparse index exceeds minor dimension");
        });
    p.data->read({p.sparse_begin}, {r.nnz}, b.values);
    if (r.dtype == DType::boolean && p.node->session->options.validation == Validation::full)
        for (std::size_t j = 0; j < r.nnz; ++j) {
            auto& v = static_cast<std::byte*>(b.values)[j]; v = v == std::byte{0} ? std::byte{0} : std::byte{1};
        }
}
TransferStats TransferContext::stats() const {
    const auto& p = transfer_context(impl_);
    TransferStats out;
    for (const auto* resource : {p.data.get(), p.indices.get(), p.offsets.get()}) if (resource) {
        const auto s = resource->stats();
        out.read_calls += s.read_calls; out.write_calls += s.write_calls;
        out.read_bytes += s.read_bytes; out.write_bytes += s.write_bytes;
        out.scratch_allocations += s.scratch_allocations; out.scratch_bytes += s.scratch_bytes;
    }
    out.scratch_allocations += p.backend->scratch_allocations() + p.tile_allocations;
    out.scratch_bytes += p.backend->retained_scratch_bytes() + p.tile.capacity();
    return out;
}
TransferContext BackedArray::prepare_transfer(std::size_t scratch_bytes) const {
    auto d = describe(node_);
    if (node_->code_recode || node_->color_override)
        fail(ErrorCode::unsupported_layout, node_, "transformed annotation buffers cannot prepare a native transfer");
    if (d.layout != Layout::dense) {
        const auto major = d.layout == Layout::csr ? 0u : 1u, minor = 1u - major;
        if (d.axes[major].has_positions() || d.axes[major].step != 1 || !d.axes[minor].all(d.source_shape[minor]))
            fail(ErrorCode::unsupported_layout, node_, "sparse transfers require consecutive native major-axis blocks and a complete minor axis");
    }
    auto p = std::make_unique<detail::TransferContextImpl>();
    p->node = node_; p->geometry = storage_geometry(); p->descriptor = std::move(d);
    p->budget = scratch_bytes ? scratch_bytes : backed_detail::scratch_remaining(node_);
    if (p->budget < 1024) fail(ErrorCode::invalid_argument, node_, "transfer scratch budget must be at least 1024 bytes");
    p->backend = node_->session->backend->fork(p->budget);
    if (p->backend->identity() != node_->session->backend->identity())
        fail(ErrorCode::invalid_argument, node_, "store identity changed before transfer preparation");
    const auto& desc = p->descriptor;
    const auto path = desc.layout == Layout::dense ? node_->path : backed_detail::child(node_->path, "data");
    p->data = p->backend->prepare_numeric(path);
    if (desc.layout != Layout::dense) {
        p->indices = p->backend->prepare_numeric(backed_detail::child(node_->path, "indices"));
        p->offsets = p->backend->prepare_numeric(backed_detail::child(node_->path, "indptr"));
    } else if (!desc.shape.empty()) {
        const auto workspace = std::max(p->backend->numeric_workspace_bytes(desc.data, false),
                                       p->geometry.writable ? p->backend->numeric_workspace_bytes(desc.data, true) : std::size_t{0});
        const auto available = staging_budget(p->budget, workspace, node_);
        const auto required = multiply(desc.shape.back(), dtype_size(desc.data.dtype), node_);
        p->tile.resize(std::min({available, required, std::size_t{1024*1024}}));
        p->tile_allocations += !p->tile.empty();
    }
    return TransferContext(std::move(p));
}
} // namespace anndatapp
