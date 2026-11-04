module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

module anndata.streaming;

namespace anndata {
namespace {
[[noreturn]] void fail(const char* s) { throw Error(ErrorCode::invalid_argument, "block stream", s); }
std::size_t mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b) fail("size overflows size_t");
    return a*b;
}
std::size_t product(const std::vector<std::size_t>& v) {
    if (std::find(v.begin(),v.end(),0)!=v.end()) return 0;
    std::size_t n=1; for (auto k:v) n=mul(n,k); return n;
}
std::size_t width(DType t) {
    switch(t) {
    case DType::boolean: case DType::int8: case DType::uint8: return 1;
    case DType::int16: case DType::uint16: return 2;
    case DType::int32: case DType::uint32: case DType::float32: return 4;
    default: return 8;
    }
}
std::size_t ceil_div(std::size_t n,std::size_t d) { return n/d + bool(n%d); }
void validate(const StreamOptions& o) {
    if (!o.min_block_bytes || o.min_block_bytes>o.target_block_bytes || o.target_block_bytes>o.max_block_bytes)
        fail("require 0 < min_block_bytes <= target_block_bytes <= max_block_bytes");
    if (!o.max_in_flight || !o.scratch_bytes_per_worker || !o.sparse_major_count) fail("worker, scratch and sparse-major limits must be positive");
}
struct Lane {
    bool live=false, pending=false;
    std::vector<std::size_t> start,count,cursor,tile,conflict;
    BlockRequest request;
    std::size_t sparse_limit=std::numeric_limits<std::size_t>::max();
};
}
namespace detail {
struct BlockStreamImpl {
    StorageGeometry g;
    StreamOptions options;
    StreamSettings settings;
    StreamStats stats;
    std::vector<Lane> lanes;
    std::vector<std::size_t> unit,grid,position,origin;
    bool source_grid=false, exhausted=false, transition=false;
    std::size_t pending=0, major=0, coalesce=0, item=0, epoch_count=0, trial_kind=0, attempts=0, shifted=0;
    std::size_t saved_block=0,saved_flight=0;
    std::uint64_t next_id=1;
    double epoch_bytes=0,epoch_seconds=0,reference=0,stable_rate=0,sparse_row_bytes=0;
    bool intervals=true,reference_intervals=true;
    std::vector<std::pair<double,double>> interval_samples;

    BlockStreamImpl(StorageGeometry geometry,StreamOptions o):g(std::move(geometry)),options(o) {
        validate(o); const auto rank=g.shape.size(); item=width(g.dtype);
        if (o.max_block_bytes<item) fail("output capacity is smaller than one numerical value");
        if (g.axes.size()!=rank || g.source_shape.size()!=rank) fail("geometry rank mismatch");
        product(g.shape);
        const bool writing=o.operation==StreamOperation::write;
        if (writing && !g.writable) fail("write stream requires writable storage");
        if (g.layout!=Layout::dense && (rank!=2 || writing)) fail("only dense writes and native-major sparse reads are supported");
        for(std::size_t i=0;i<rank;++i) {
            const auto& a=g.axes[i];
            if (a.affine && g.shape[i]) {
                if(a.start<0 || std::size_t(a.start)>=g.source_shape[i]) fail("affine source start is out of bounds");
                const auto step=a.step<0?std::size_t(-(a.step+1))+1:std::size_t(a.step);
                const auto distance=a.step<0?std::size_t(a.start):g.source_shape[i]-1-std::size_t(a.start);
                if(step && g.shape[i]-1>distance/step) fail("affine source selection is out of bounds");
            }
            if (writing && (!a.affine || a.step!=1)) fail("write planning requires positive unit-stride affine axes");
        }
        settings.reserved_workers=o.max_in_flight;
        settings.output_bytes_per_worker=o.max_block_bytes;
        settings.scratch_bytes_per_worker=o.scratch_bytes_per_worker;
        if (o.max_block_bytes>std::numeric_limits<std::size_t>::max()-o.scratch_bytes_per_worker) fail("worker capacity overflows size_t");
        settings.aggregate_reserved_bytes=mul(o.max_in_flight,o.max_block_bytes+o.scratch_bytes_per_worker);
        settings.backend_serialized=writing?!g.parallel_writes:!g.parallel_reads;
        lanes.resize(o.max_in_flight);
        if(o.max_in_flight>std::numeric_limits<std::size_t>::max()-64) fail("timing capacity overflows size_t");
        interval_samples.reserve(o.max_in_flight+64);
        settings.target_block_bytes=o.target_block_bytes; settings.in_flight_limit=1;
        if(o.mode==StreamMode::manual) settings.in_flight_limit=settings.backend_serialized?1:o.max_in_flight;
        settings.mode=o.mode; settings.phase=o.mode==StreamMode::manual?StreamPhase::manual:StreamPhase::baseline;
        settings.limit=settings.backend_serialized?StreamLimit::backend_serialized:o.mode==StreamMode::manual?StreamLimit::manual:StreamLimit::insufficient_samples;
        unit.assign(rank,1); origin.assign(rank,0); grid.assign(rank,1); position.assign(rank,0);
        source_grid=g.layout==Layout::dense;
        for(const auto& a:g.axes) source_grid=source_grid && a.affine && a.step==1;
        if(g.layout!=Layout::dense) {
            major=g.layout==Layout::csr?0:1;
            unit=g.shape; unit[major]=1;
            grid[major]=g.shape[major];
        } else {
            const auto& p=g.data;
            const auto& physical=writing?p.write_unit:!p.inner_chunks.empty()?p.inner_chunks:p.chunks;
            if(source_grid && physical.size()==rank) {
                unit=physical;
                for(std::size_t i=0;i<rank;++i) {
                    if(!unit[i]) fail("zero physical unit extent");
                    origin[i]=std::size_t(g.axes[i].start)/unit[i];
                    auto r=std::size_t(g.axes[i].start)%unit[i];
                    if(g.shape[i]>std::numeric_limits<std::size_t>::max()-r) fail("physical grid extent overflows size_t");
                    grid[i]=ceil_div(g.shape[i]+r,unit[i]);
                }
            } else {
                source_grid=false;
                unit.assign(rank,1);
                std::size_t capacity=writing?std::numeric_limits<std::size_t>::max():std::max<std::size_t>(1,o.min_block_bytes/item);
                for(std::size_t i=rank;i--;) {unit[i]=std::min(std::max<std::size_t>(1,g.shape[i]),capacity);capacity/=unit[i];if(!capacity)capacity=1;}
                for(std::size_t i=0;i<rank;++i) grid[i]=ceil_div(g.shape[i],unit[i]);
            }
        }
        for(std::size_t i=0;i<rank;++i) if(grid[i]>1) coalesce=i;
        exhausted=product(g.shape)==0;
    }
    void reset_epoch() { epoch_count=0;epoch_bytes=epoch_seconds=0;interval_samples.clear();intervals=true;transition=false; }
    void advance_grid(std::size_t n) {
        if(position.empty()) { exhausted=true; return; }
        const auto axis=g.layout==Layout::dense?coalesce:major;
        position[axis]+=n;
        if(g.layout!=Layout::dense) { exhausted=position[axis]>=grid[axis];return; }
        for(std::size_t i=axis+1;i--;) {
            if(position[i]<grid[i]) return;
            position[i]=0;
            if(!i) {exhausted=true;return;}
            ++position[i-1];
        }
    }
    void region(Lane& lane) {
        const auto rank=g.shape.size(); lane.start.assign(rank,0);lane.count=g.shape;lane.conflict.clear();
        std::size_t grouped=1;
        if(g.layout!=Layout::dense) {
            auto rows=options.sparse_major_count;
            if(options.mode==StreamMode::adaptive && !g.data.shape.empty() && g.shape[major]) {
                const auto average=g.data.shape[0]/g.shape[major];
                const auto per_row=average>(std::numeric_limits<std::size_t>::max()-8)/(item+8)?std::numeric_limits<std::size_t>::max():average*(item+8)+8;
                const auto measured=sparse_row_bytes>0?sparse_row_bytes:double(per_row);
                const auto available=double(settings.target_block_bytes)/measured;
                rows=available>=double(rows)?rows:std::max<std::size_t>(1,std::size_t(available));
            }
            grouped=std::min(rows,grid[major]-position[major]);
            lane.start[major]=position[major];lane.count[major]=grouped;
        } else {
            if(source_grid && options.operation==StreamOperation::write) lane.conflict=position;
            for(std::size_t i=0;i<rank;++i) {
                const auto k=mul(origin[i]+position[i],unit[i]),a=source_grid?std::size_t(g.axes[i].start):0;
                const auto lo=k<a?0:k-a;
                lane.start[i]=lo; lane.count[i]=std::min(unit[i]-(k<a?a-k:0),g.shape[i]-lo);
                if(!lane.conflict.empty()) lane.conflict[i]+=origin[i];
            }
            if(rank && options.operation==StreamOperation::read) {
                const auto bytes=mul(product(lane.count),item),available=grid[coalesce]-position[coalesce];
                grouped=bytes?std::min(available,std::max<std::size_t>(1,settings.target_block_bytes/bytes)):1;
                if(grouped>1) {
                    const auto tail=mul(grouped-1,unit[coalesce]);
                    const auto remaining=g.shape[coalesce]-lane.start[coalesce],first=lane.count[coalesce];
                    lane.count[coalesce]=tail>remaining-first?remaining:first+tail;
                }
            }
        }
        lane.cursor.assign(rank,0);lane.tile.assign(rank,1);
        if(g.layout!=Layout::dense) lane.tile=lane.count;
        else {
            std::size_t capacity=std::max<std::size_t>(1,settings.target_block_bytes/item);
            for(std::size_t i=rank;i--;) {lane.tile[i]=std::min(lane.count[i],capacity);capacity/=lane.tile[i];if(!capacity)capacity=1;}
        }
        lane.live=true;lane.pending=false;lane.sparse_limit=std::numeric_limits<std::size_t>::max();
        advance_grid(grouped);
    }
    void advance(Lane& lane) {
        if(lane.cursor.empty()) {lane.live=false;return;}
        for(std::size_t i=lane.cursor.size();i--;) {
            lane.cursor[i]+=lane.request.count[i];
            if(lane.cursor[i]<lane.count[i]) return;
            lane.cursor[i]=0;
        }
        lane.live=false;
    }
    Lane& find(const BlockRequest& r) {
        for(auto& lane:lanes) if(lane.pending && lane.request.id==r.id) {
            if(lane.request.start!=r.start || lane.request.count!=r.count || lane.request.conflict_start!=r.conflict_start || lane.request.conflict_count!=r.conflict_count)
                fail("request descriptor was modified");
            return lane;
        }
        fail("request is unknown or already completed");
    }
    bool candidate() {
        for(;trial_kind<4;++trial_kind) {
            const auto block=settings.target_block_bytes,flight=settings.in_flight_limit;
            auto b=block,f=flight;
            if(trial_kind==0) b=block>options.max_block_bytes/2?options.max_block_bytes:block*2;
            if(trial_kind==1) b=std::max(options.min_block_bytes,block/2);
            if(trial_kind==2 && settings.overlap_timing_available && !settings.backend_serialized) f=std::min(options.max_in_flight,flight>options.max_in_flight/2?options.max_in_flight:flight*2);
            if(trial_kind==3 && settings.overlap_timing_available) f=std::max<std::size_t>(1,flight/2);
            if(b==block && f==flight) continue;
            saved_block=block;saved_flight=flight;
            settings.target_block_bytes=b;settings.in_flight_limit=f;settings.phase=StreamPhase::trial;
            settings.limit=StreamLimit::none;++stats.trials;++attempts;++trial_kind;return true;
        }
        settings.phase=StreamPhase::stable;stable_rate=reference;
        settings.limit=settings.backend_serialized?StreamLimit::backend_serialized:!settings.overlap_timing_available?StreamLimit::no_overlap_evidence:StreamLimit::capacity;
        return false;
    }
    void tune() {
        if(!transition || pending) return;
        double seconds=epoch_seconds;
        if(intervals) {
            // The union measures I/O busy time, excluding caller computation
            // gaps while retaining actual worker overlap. Capacity was reserved
            // for the largest epoch plus all outstanding requests.
            std::sort(interval_samples.begin(),interval_samples.end());
            double first=interval_samples.front().first,last=interval_samples.front().second;
            seconds=0;
            for(const auto& [a,b]:interval_samples) {
                if(a>last) {seconds+=last-first;first=a;last=b;}
                else last=std::max(last,b);
            }
            seconds+=last-first;
        }
        const auto rate=seconds>0?epoch_bytes/seconds:0;
        stats.throughput_bytes_per_second=rate;
        settings.overlap_timing_available=intervals;
        if(settings.phase!=StreamPhase::baseline && intervals!=reference_intervals) {
            if(settings.phase==StreamPhase::trial) {settings.target_block_bytes=saved_block;settings.in_flight_limit=saved_flight;}
            reference=rate;reference_intervals=intervals;trial_kind=attempts=shifted=0;
            settings.phase=StreamPhase::baseline;candidate();reset_epoch();return;
        }
        if(settings.phase==StreamPhase::stable) {
            shifted=stable_rate>0 && std::abs(rate/stable_rate-1)>0.25?shifted+1:0;
            if(shifted>=2) {trial_kind=attempts=shifted=0;reference=rate;candidate();}
        } else if(settings.phase==StreamPhase::baseline) {reference=rate;reference_intervals=intervals;candidate();}
        else {
            if(rate>reference*1.05) {reference=rate;++stats.adaptations;}
            else {settings.target_block_bytes=saved_block;settings.in_flight_limit=saved_flight;if(rate<reference*0.8)++stats.rollbacks;}
            candidate();
        }
        reset_epoch();
    }
};
}
BlockStream::BlockStream(StorageGeometry g,StreamOptions o):impl_(std::make_unique<detail::BlockStreamImpl>(std::move(g),o)){}
BlockStream::~BlockStream()=default;
BlockStream::BlockStream(BlockStream&&) noexcept=default;
BlockStream& BlockStream::operator=(BlockStream&&) noexcept=default;
std::optional<BlockRequest> BlockStream::next() {
    if(!impl_) fail("empty stream"); auto& p=*impl_;
    p.tune();
    if(p.transition || p.pending>=p.settings.in_flight_limit) return {};
    Lane* lane=nullptr;
    for(auto& x:p.lanes) if(x.live && !x.pending) {lane=&x;break;}
    if(!lane && !p.exhausted) for(auto& x:p.lanes) if(!x.live) {p.region(x);lane=&x;break;}
    if(!lane) return {};
    auto& r=lane->request;r.id=p.next_id++;
    if(!r.id) fail("request identity overflow");
    const auto rank=p.g.shape.size();r.start.resize(rank);r.count.resize(rank);
    for(std::size_t i=0;i<rank;++i) {r.start[i]=lane->start[i]+lane->cursor[i];r.count[i]=std::min(lane->tile[i],lane->count[i]-lane->cursor[i]);}
    if(p.g.layout!=Layout::dense) r.count[p.major]=std::min(r.count[p.major],lane->sparse_limit);
    r.conflict_start=lane->conflict;r.conflict_count.assign(r.conflict_start.size(),1);
    r.estimated_bytes=p.g.layout==Layout::dense?mul(product(r.count),p.item):0;
    lane->pending=true;++p.pending;++p.stats.issued;
    return r;
}
void BlockStream::complete(const BlockRequest& r,const BlockCompletion& sample) {
    if(!impl_) fail("empty stream");auto& p=*impl_;auto& lane=p.find(r);
    const bool observe=p.options.mode==StreamMode::adaptive||p.options.statistics;
    if(observe && (!std::isfinite(sample.io_seconds)||sample.io_seconds<0)) fail("invalid transfer duration");
    if(p.options.mode==StreamMode::adaptive && sample.io_seconds<=0) fail("adaptive completion requires a positive transfer duration");
    if(p.stats.useful_bytes>std::numeric_limits<std::size_t>::max()-sample.useful_bytes) fail("completed byte count overflows size_t");
    ++p.stats.completed;p.stats.useful_bytes+=sample.useful_bytes;
    if(observe) p.stats.io_seconds+=sample.io_seconds;
    if(p.options.mode==StreamMode::adaptive) {
        const bool interval=std::isfinite(sample.started_seconds)&&std::isfinite(sample.finished_seconds)&&sample.finished_seconds>sample.started_seconds;
        p.intervals=p.intervals&&interval;
        if(interval) {
            p.interval_samples.emplace_back(sample.started_seconds,sample.finished_seconds);
        }
        if(p.g.layout!=Layout::dense && r.count[p.major] && sample.useful_bytes) {
            const auto observed=double(sample.useful_bytes)/double(r.count[p.major]);
            p.sparse_row_bytes=p.sparse_row_bytes>0?0.75*p.sparse_row_bytes+0.25*observed:observed;
        }
        p.epoch_bytes+=sample.useful_bytes;p.epoch_seconds+=sample.io_seconds;++p.epoch_count;
        const auto count=p.settings.phase==StreamPhase::stable?64u:16u;
        p.transition=p.transition||p.epoch_count>=count;
    }
    lane.pending=false;--p.pending;p.advance(lane);
}
void BlockStream::complete(const BlockRequest& r,std::size_t bytes,double seconds) {complete(r,BlockCompletion{bytes,seconds,0,0});}
void BlockStream::split(const BlockRequest& r,std::size_t max_major_count) {
    if(!impl_) fail("empty stream");auto& p=*impl_;auto& lane=p.find(r);
    if(p.g.layout==Layout::dense || !max_major_count || max_major_count>=r.count[p.major]) fail("split requires a smaller positive sparse major count");
    lane.sparse_limit=max_major_count;lane.pending=false;--p.pending;++p.stats.split_requests;
}
void BlockStream::configure(StreamOptions o) {
    if(!impl_) fail("empty stream");auto& p=*impl_;validate(o);
    if(p.pending) fail("drain outstanding requests before configuring");
    if(o.operation!=p.options.operation) fail("stream operation cannot change");
    if(o.max_in_flight>p.settings.reserved_workers || o.max_block_bytes>p.settings.output_bytes_per_worker || o.scratch_bytes_per_worker>p.settings.scratch_bytes_per_worker)
        fail("configuration exceeds initial reserved capacities");
    for(const auto& lane:p.lanes) if(lane.live && p.g.layout==Layout::dense && mul(product(lane.tile),p.item)>o.max_block_bytes)
        fail("finish the currently owned region before reducing output capacity");
    p.options=o;p.settings.mode=o.mode;p.settings.phase=o.mode==StreamMode::manual?StreamPhase::manual:StreamPhase::baseline;
    p.settings.limit=p.settings.backend_serialized?StreamLimit::backend_serialized:o.mode==StreamMode::manual?StreamLimit::manual:StreamLimit::insufficient_samples;
    p.settings.target_block_bytes=o.target_block_bytes;
    p.settings.in_flight_limit=o.mode==StreamMode::manual&&!p.settings.backend_serialized?o.max_in_flight:1;
    p.trial_kind=p.attempts=p.shifted=0;p.reference=p.stable_rate=0;p.settings.overlap_timing_available=false;p.reset_epoch();
    // Retile only the remaining suffix within each already-owned region.
    for(auto& lane:p.lanes) if(lane.live) {
        if(p.g.layout!=Layout::dense) lane.tile[p.major]=std::min(lane.tile[p.major],o.sparse_major_count);
        // Dense lane tiles stay fixed until their region completes, preserving
        // its multidimensional coverage across a runtime mode change.
    }
}
StreamSettings BlockStream::settings() const {if(!impl_)fail("empty stream");return impl_->settings;}
StreamStats BlockStream::stats() const {if(!impl_)fail("empty stream");return impl_->stats;}
bool BlockStream::finished() const {
    if(!impl_)fail("empty stream");const auto& p=*impl_;
    if(!p.exhausted||p.pending)return false;
    for(const auto& lane:p.lanes)if(lane.live)return false;
    return true;
}
}
