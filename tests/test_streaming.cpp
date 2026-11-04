#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

import anndata.streaming;
namespace {
using namespace anndata;
void check(bool p,std::string_view s) {if(!p)throw std::runtime_error(std::string(s));}
template<class F> void bad(F&& f) {try{f();}catch(const Error&){return;}throw std::runtime_error("expected error");}
StorageGeometry geometry(std::vector<std::size_t> shape,std::vector<std::size_t> chunks={},std::vector<std::size_t> shard={}) {
    StorageGeometry g;g.shape=shape;g.source_shape=shape;g.axes.resize(shape.size());g.dtype=DType::float32;
    g.parallel_reads=g.parallel_writes=g.writable=true;g.data.shape=shape;
    g.data.organization=shard.empty()?chunks.empty()?StorageOrganization::contiguous:StorageOrganization::chunked:StorageOrganization::sharded;
    g.data.chunks=shard.empty()?chunks:shard;g.data.inner_chunks=chunks;g.data.write_unit=shard.empty()?chunks:shard;
    return g;
}
std::size_t product(const std::vector<std::size_t>& shape) {std::size_t n=1;for(auto k:shape)n*=k;return n;}
void visit(const BlockRequest& r,const std::vector<std::size_t>& shape,std::vector<int>& seen) {
    const auto count=product(r.count);std::vector<std::size_t> p(r.count.size());
    for(std::size_t i=0;i<count;++i) {
        auto k=i;for(std::size_t d=p.size();d--;) {p[d]=k%r.count[d];k/=r.count[d];}
        std::size_t n=0;for(std::size_t d=0;d<p.size();++d) {check(r.start[d]+p[d]<shape[d],"request in bounds");n=n*shape[d]+r.start[d]+p[d];}
        check(++seen[n]==1,"coverage must be exactly once");
    }
}
StreamOptions manual(std::size_t bytes=64,std::size_t workers=4) {
    StreamOptions o;o.mode=StreamMode::manual;o.min_block_bytes=4;o.target_block_bytes=bytes;o.max_block_bytes=256;o.max_in_flight=workers;return o;
}
void coverage(StorageGeometry g,StreamOptions o,bool switch_mode=false) {
    BlockStream stream(g,o);std::vector<int> seen(product(g.shape));std::vector<BlockRequest> active;
    std::size_t completed=0;double t=1;
    while(!stream.finished()) {
        while(auto r=stream.next()) {
            check(r->estimated_bytes<=o.max_block_bytes,"bounded dense output");
            if(o.operation==StreamOperation::write) for(const auto& p:active) check(r->conflict_start!=p.conflict_start,"distinct write ownership");
            active.push_back(std::move(*r));
        }
        check(!active.empty(),"stream must make progress");
        auto r=std::move(active.back());active.pop_back();visit(r,g.shape,seen);
        stream.complete(r,BlockCompletion{r.estimated_bytes,0.01,t,t+0.01});t+=0.01;++completed;
        if(switch_mode && completed==12) {
            while(!active.empty()) {auto q=std::move(active.back());active.pop_back();visit(q,g.shape,seen);stream.complete(q,q.estimated_bytes,0.01);}
            o.mode=StreamMode::manual;stream.configure(o);
        }
    }
    for(auto n:seen)check(n==1,"all values covered");
    check(stream.stats().completed>0 || seen.empty(),"completion count");
}
void test_geometry_and_coverage() {
    coverage(geometry({23,31},{7,5}),manual(64));
    coverage(geometry({4,13,19},{2,5,7}),manual(120));
    coverage(geometry({32,1},{4,1}),manual(16));
    coverage(geometry({32,1},{4,1}),manual(240));
    coverage(geometry({5,7,1},{2,3,1}),manual(48));
    coverage(geometry({61,37}),manual(64));
    auto gathered=geometry({41,19},{7,5});gathered.axes[0].affine=false;
    coverage(gathered,manual(128));
    coverage(geometry({}),manual(16));
    coverage(geometry({0,12}),manual());
    auto selected=geometry({19,21},{4,5});selected.source_shape={31,42};selected.axes[0].start=3;selected.axes[1].start=7;
    coverage(selected,manual(64));
    auto write=manual(32);write.operation=StreamOperation::write;
    coverage(geometry({23,31},{3,4},{9,12}),write);
    selected.writable=true;coverage(selected,write);
    auto adaptive=manual(32);adaptive.mode=StreamMode::adaptive;adaptive.min_block_bytes=16;
    coverage(geometry({127,113},{7,9}),adaptive,true);
    coverage(geometry({117,123}),adaptive);
}
void test_parallel_and_contracts() {
    BlockStream stream(geometry({4096}),manual(64));std::vector<BlockRequest> active;
    for(int i=0;i<4;++i) {auto r=stream.next();check(bool(r),"contiguous reads admit four requests");active.push_back(*r);}
    check(!stream.next(),"inflight cap");bad([&]{stream.configure(manual());});
    for(auto& r:active)stream.complete(r,r.estimated_bytes);
    bad([&]{stream.complete(active[0],active[0].estimated_bytes);});
    auto o=manual();o.mode=StreamMode::adaptive;o.min_block_bytes=1;o.max_block_bytes=0;bad([&]{BlockStream x(geometry({2}),o);});
    auto g=geometry({2});g.writable=false;o=manual();o.operation=StreamOperation::write;bad([&]{BlockStream x(g,o);});
    g=geometry({6},{2});g.axes[0].step=-1;g.axes[0].start=5;bad([&]{BlockStream x(g,o);});
    g=geometry({6},{2});g.parallel_writes=false;BlockStream serial(g,o);check(serial.settings().in_flight_limit==1,"backend write clamp");
    check(serial.settings().limit==StreamLimit::backend_serialized,"explain backend clamp");
    auto r=serial.next();check(bool(r),"serial request");check(!serial.next(),"only one writer");
    auto forged=*r;forged.count[0]++;bad([&]{serial.complete(forged,0);});serial.complete(*r,0);
}
void test_sparse() {
    for(auto layout:{Layout::csr,Layout::csc}) {
        auto g=geometry({17,11});g.layout=layout;auto o=manual();o.sparse_major_count=7;
        BlockStream stream(g,o);std::vector<int> seen(product(g.shape));auto r=stream.next();check(bool(r),"sparse first request");
        const auto major=layout==Layout::csr?0u:1u;check(r->count[major]==7,"native-major request count");
        stream.split(*r,2);bad([&]{stream.complete(*r,0);});
        while(auto q=stream.next()) {check(q->count[major]<=7,"major bound");check(q->count[1-major]==g.shape[1-major],"minor axis retained");visit(*q,g.shape,seen);stream.complete(*q,0);}
        check(stream.finished(),"sparse stream completes");for(auto n:seen)check(n==1,"sparse split exact coverage");
        check(stream.stats().split_requests==1,"split recorded");
        o.operation=StreamOperation::write;bad([&]{BlockStream x(g,o);});
    }
}
void test_adaptation() {
    auto g=geometry({1u<<20},{16});auto o=manual(64,4);o.mode=StreamMode::adaptive;o.min_block_bytes=32;o.max_block_bytes=256;
    BlockStream stream(g,o);double t=1;std::size_t n=0;bool larger=false,parallel=false;
    while(n<180) {
        std::vector<BlockRequest> active;const auto settings=stream.settings();
        while(auto r=stream.next())active.push_back(*r);
        check(!active.empty(),"adaptive progress");
        const auto f=stream.settings().in_flight_limit;
        for(auto& r:active) {
            auto dt=r.estimated_bytes/10000.0;
            stream.complete(r,BlockCompletion{r.estimated_bytes,dt,t,t+dt});++n;
        }
        t+=0.1;larger=larger||stream.settings().target_block_bytes>64;parallel=parallel||f>1;
        if(stream.settings().phase==StreamPhase::stable && n>100)break;
    }
    check(stream.stats().trials>0 && larger,"block-size exploration");check(parallel,"overlap evidence enables concurrency trial");
    // Elapsed durations alone cannot establish parallel throughput.
    BlockStream durations(g,o);
    for(int i=0;i<110;++i) {auto r=durations.next();check(bool(r),"duration-only progress");durations.complete(*r,r->estimated_bytes,0.01);}
    check(durations.settings().in_flight_limit==1,"no concurrency inference without intervals");
    o.mode=StreamMode::manual;durations.configure(o);check(durations.settings().phase==StreamPhase::manual,"manual fallback");
    for(int i=0;i<40;++i){auto r=durations.next();check(bool(r),"manual progress");durations.complete(*r,r->estimated_bytes);}
    check(durations.settings().target_block_bytes==o.target_block_bytes,"manual target fixed");
    // A controlled rate change rolls back a bad trial, and two 64-sample
    // windows reopen exploration after the initial candidates stabilize.
    o.mode=StreamMode::adaptive;o.max_in_flight=1;BlockStream changing(g,o);double now=1;
    auto step=[&](double rate) {
        auto r=changing.next();check(bool(r),"rate-trace progress");const auto dt=r->estimated_bytes/rate;
        changing.complete(*r,BlockCompletion{r->estimated_bytes,dt,now,now+dt});now+=dt;
    };
    for(int i=0;i<16;++i)step(1000);
    for(int i=0;i<16;++i)step(100);
    step(1000);check(changing.stats().rollbacks==1,"bad trial rolls back");
    for(int i=0;i<80 && changing.settings().phase!=StreamPhase::stable;++i)step(1000);
    check(changing.settings().phase==StreamPhase::stable,"bounded initial exploration stabilizes");
    const auto trials=changing.stats().trials;
    for(int i=0;i<130;++i)step(2000);
    check(changing.stats().trials>trials,"two shifted stable windows reopen exploration");
    auto gap_trace=[&](double gap) {
        BlockStream x(g,o);double now=1;
        for(int i=0;i<100;++i) {
            auto r=x.next();check(bool(r),"gap-trace progress");const auto dt=r->estimated_bytes/1000.0;
            x.complete(*r,BlockCompletion{r->estimated_bytes,dt,now,now+dt});now+=dt+gap;
        }
        return std::pair{x.settings().target_block_bytes,x.stats().adaptations};
    };
    const auto compact=gap_trace(0),delayed=gap_trace(10);
    check(compact==delayed && !compact.second,"caller compute gaps cannot improve an I/O candidate");
    BlockStream mixed(g,o);double clock=1;
    for(int i=0;i<96;++i) {
        auto r=mixed.next();check(bool(r),"mixed-metric progress");const auto dt=r->estimated_bytes/1000.0;
        if((i/16)%2) mixed.complete(*r,r->estimated_bytes,dt);
        else mixed.complete(*r,BlockCompletion{r->estimated_bytes,dt,clock,clock+dt});
        clock+=dt;
    }
    check(!mixed.stats().adaptations,"incompatible timing metrics never promote a candidate");
}
}
int main() {try {test_geometry_and_coverage();test_parallel_and_contracts();test_sparse();test_adaptation();std::cout<<"streaming tests passed\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
