#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

import anndatapp.streaming;
namespace { bool measuring=false; std::size_t largest_allocation=0; }
// TSan owns new/delete interception; native and ASan builds check allocation size.
#if !__has_feature(thread_sanitizer)
void* operator new(std::size_t n) {
    if(measuring && n>largest_allocation) largest_allocation=n;
    if(auto* p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }
#endif
namespace {
using namespace anndatapp;
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
    adaptive.operation=StreamOperation::write;coverage(geometry({63,53},{7,9}),adaptive);
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
template<class Rate>
StreamSettings epoch(BlockStream& stream,double& now,Rate rate,bool intervals=true,double gap=0,bool reverse=false) {
    auto first=stream.next();check(bool(first),"adaptive epoch starts");
    const auto settings=stream.settings();
    const auto target=settings.phase==StreamPhase::stable?64u:16u;
    const auto before=stream.stats().completed;
    std::size_t done=0;
    std::vector<BlockRequest> active{*first};
    while(done<target) {
        while(auto r=stream.next()) active.push_back(std::move(*r));
        check(!active.empty(),"adaptive epoch makes progress");
        double bytes=0;for(const auto& r:active)bytes+=r.estimated_bytes;
        const auto rps=rate(settings);check(rps>0,"trace rate is positive");
        const auto span=bytes/rps;
        auto complete=[&](const BlockRequest& r) {
            const auto seconds=intervals?span:double(r.estimated_bytes)/rps;
            stream.complete(r,BlockCompletion{r.estimated_bytes,seconds,intervals?now:0,intervals?now+span:0});
        };
        if(reverse) for(std::size_t i=active.size();i--;) complete(active[i]);
        else for(const auto& r:active) complete(r);
        done+=active.size();now+=span+gap;active.clear();
    }
    check(stream.stats().completed-before>=target,"epoch has enough useful completions");
    return settings;
}
void test_adaptation() {
    auto g=geometry({1u<<20},{16});auto o=manual(32,8);o.mode=StreamMode::adaptive;o.min_block_bytes=16;o.max_block_bytes=256;
    auto favorable=[](StreamSettings s) {return 1000.0*double(s.target_block_bytes)/32.0*double(s.in_flight_limit);};
    BlockStream growth(g,o);double now=1;std::vector<StreamSettings> settings;
    for(int i=0;i<8;++i) settings.push_back(epoch(growth,now,favorable));
    const std::vector<std::size_t> blocks{32,64,64,128,128,256,256,256};
    const std::vector<std::size_t> flights{1,1,2,2,4,4,8,8};
    for(std::size_t i=0;i<settings.size();++i) {
        check(settings[i].target_block_bytes==blocks[i],"accepted block direction repeats to its bound");
        check(settings[i].in_flight_limit==flights[i],"in-flight doubling is interleaved");
        check(settings[i].in_flight_limit<=settings[i].reserved_workers,"workers remain within reservation");
    }
    check(growth.stats().trials==6 && growth.settings().phase==StreamPhase::stable,"search stops at both bounds");
    check(growth.settings().aggregate_reserved_bytes==o.max_in_flight*(o.max_block_bytes+o.scratch_bytes_per_worker),"adaptive trials keep initial aggregate reservation");

    // A rejected increase rolls back, then the decrease direction is tried once.
    o.max_block_bytes=64;o.max_in_flight=1;BlockStream degraded(g,o);now=1;
    epoch(degraded,now,[](StreamSettings){return 1000.0;});
    epoch(degraded,now,[](StreamSettings){return 700.0;});
    check(degraded.settings().target_block_bytes==64,"failed candidate remains installed until its epoch is judged");
    epoch(degraded,now,[](StreamSettings){return 1000.0;});
    auto settled=epoch(degraded,now,[](StreamSettings){return 1000.0;});
    check(settled.phase==StreamPhase::stable && settled.target_block_bytes==32,"degradation restores the best setting");
    check(degraded.stats().rollbacks==1 && degraded.stats().trials==2,"large degradation is counted and bounded");

    // A plateau at the upper bound does not reverse a successful direction.
    BlockStream bounded(g,o);now=1;
    epoch(bounded,now,[](StreamSettings s){return double(s.target_block_bytes);});
    auto upper=epoch(bounded,now,[](StreamSettings s){return double(s.target_block_bytes);});
    auto plateau=epoch(bounded,now,[](StreamSettings s){return double(s.target_block_bytes);});
    check(upper.target_block_bytes==64 && plateau.phase==StreamPhase::stable && plateau.target_block_bytes==64,"upper bound ends an accepted direction");
    check(bounded.stats().trials==1,"bound does not create a reverse trial");

    // Starting at the upper bound still probes smaller blocks when they can fit.
    auto upper_start=manual(64,1);upper_start.mode=StreamMode::adaptive;upper_start.min_block_bytes=16;upper_start.max_block_bytes=64;
    BlockStream shrinking(g,upper_start);now=1;
    auto original=epoch(shrinking,now,[](StreamSettings s){return 64000.0/double(s.target_block_bytes);});
    auto half=epoch(shrinking,now,[](StreamSettings s){return 64000.0/double(s.target_block_bytes);});
    auto quarter=epoch(shrinking,now,[](StreamSettings s){return 64000.0/double(s.target_block_bytes);});
    auto shrink_done=epoch(shrinking,now,[](StreamSettings s){return 64000.0/double(s.target_block_bytes);});
    check(original.target_block_bytes==64 && half.phase==StreamPhase::trial && half.target_block_bytes==32,"an initial upper bound permits a downward probe");
    check(quarter.phase==StreamPhase::trial && quarter.target_block_bytes==16,"accepted decreases repeat to the lower bound");
    check(shrink_done.phase==StreamPhase::stable && shrink_done.target_block_bytes==16,"lower bound ends a successful downward direction");
    check(shrinking.stats().trials==2 && shrinking.stats().adaptations==2,"downward bound probes retain the best rates");

    // An exploration has at most eight candidate epochs, each with 16 useful completions.
    o.max_block_bytes=1u<<20;o.max_in_flight=256;BlockStream capped(g,o);now=1;
    for(int i=0;i<10;++i) epoch(capped,now,favorable);
    check(capped.stats().trials==8 && capped.settings().phase==StreamPhase::stable,"exploration stops after eight candidate epochs");

    // Stable shifts reopen with a fresh reference at the current best setting.
    o.max_block_bytes=64;o.max_in_flight=1;BlockStream changing(g,o);now=1;
    epoch(changing,now,[](StreamSettings){return 1000.0;});
    epoch(changing,now,[](StreamSettings){return 700.0;});
    epoch(changing,now,[](StreamSettings){return 1000.0;});
    epoch(changing,now,[](StreamSettings){return 1000.0;});
    check(changing.settings().phase==StreamPhase::stable,"initial bounded search stabilizes");
    const auto old_trials=changing.stats().trials;
    epoch(changing,now,[](StreamSettings){return 2000.0;});
    epoch(changing,now,[](StreamSettings){return 2000.0;});
    auto reopened=epoch(changing,now,[](StreamSettings){return 2000.0;});
    check(reopened.phase==StreamPhase::trial && reopened.target_block_bytes==64,"two shifted windows reopen from a fresh baseline");
    check(changing.stats().trials==old_trials+1,"reopened exploration starts a fresh candidate budget");

    // Valid overlapping intervals are unioned even when completions arrive out of order.
    auto overlap=manual(64,2);overlap.mode=StreamMode::adaptive;overlap.min_block_bytes=overlap.max_block_bytes=64;
    BlockStream concurrent(geometry({4096},{16}),overlap);now=1;
    for(int i=0;i<16;++i) {
        auto r=concurrent.next();check(bool(r),"overlap baseline progresses");
        const auto dt=double(r->estimated_bytes)/1000.0;
        concurrent.complete(*r,BlockCompletion{r->estimated_bytes,dt,now,now+dt});now+=dt;
    }
    auto a=concurrent.next();check(bool(a),"concurrency candidate starts");
    auto b=concurrent.next();check(bool(b),"concurrency candidate admits two workers");
    check(concurrent.settings().in_flight_limit==2,"common interval evidence permits concurrency trial");
    for(int i=0;i<8;++i) {
        std::vector<BlockRequest> pair;
        if(i) {auto x=concurrent.next();check(bool(x),"overlap trial continues");pair.push_back(*x);auto y=concurrent.next();check(bool(y),"overlap trial fills both workers");pair.push_back(*y);}
        else {pair.push_back(*a);pair.push_back(*b);}
        const auto d=double(pair[0].estimated_bytes)/1000.0;
        concurrent.complete(pair[1],BlockCompletion{pair[1].estimated_bytes,d,now+d/2,now+1.5*d});
        concurrent.complete(pair[0],BlockCompletion{pair[0].estimated_bytes,d,now,now+d});
        now+=1.5*d;
    }
    auto next=concurrent.next();check(bool(next),"overlap trial is judged");
    check(concurrent.settings().phase==StreamPhase::stable && concurrent.settings().in_flight_limit==2,"interval union accepts overlapping throughput");
    check(concurrent.stats().adaptations==1,"out-of-order interval union clears the five-percent threshold");

    // Duration-only samples cannot justify a concurrency trial.
    auto no_overlap=manual(64,4);no_overlap.mode=StreamMode::adaptive;no_overlap.min_block_bytes=no_overlap.max_block_bytes=64;
    BlockStream durations(g,no_overlap);now=1;
    epoch(durations,now,[](StreamSettings){return 1000.0;},false);
    auto duration_request=durations.next();check(bool(duration_request),"duration-only baseline completes");
    check(durations.settings().phase==StreamPhase::stable && durations.settings().in_flight_limit==1,"duration-only evidence leaves concurrency disabled");
    check(durations.stats().trials==0 && durations.settings().limit==StreamLimit::no_overlap_evidence,"no concurrency trials without intervals");
    durations.complete(*duration_request,BlockCompletion{duration_request->estimated_bytes,0.01,0,0});
    no_overlap.mode=StreamMode::manual;durations.configure(no_overlap);
    check(durations.settings().phase==StreamPhase::manual,"manual fallback");
    const auto manual_stats=durations.stats();
    auto manual_request=durations.next();check(bool(manual_request),"manual request issues");
    durations.complete(*manual_request,BlockCompletion{manual_request->estimated_bytes,-1,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()});
    check(durations.stats().io_seconds==manual_stats.io_seconds && durations.stats().trials==manual_stats.trials,"manual completions do not read timing fields");

    // Switching timing kinds rolls back, then collects a fresh baseline before another trial.
    o=manual(32,1);o.mode=StreamMode::adaptive;o.min_block_bytes=16;o.max_block_bytes=64;
    BlockStream switched(g,o);now=1;
    epoch(switched,now,[](StreamSettings){return 1000.0;});
    epoch(switched,now,[](StreamSettings){return 4000.0;},false);
    auto fresh=epoch(switched,now,[](StreamSettings){return 2000.0;},false);
    check(fresh.phase==StreamPhase::baseline && fresh.target_block_bytes==32,"timing-kind change restores best setting for a new baseline");
    auto trial=epoch(switched,now,[](StreamSettings){return 2000.0;},false);
    check(trial.phase==StreamPhase::trial && trial.target_block_bytes==64,"fresh duration baseline precedes the next candidate");
    auto judge=switched.next();check(bool(judge),"freshly measured candidate can be judged");
    check(switched.stats().adaptations==0,"candidate is compared with its new timing baseline");

    // Application gaps are excluded by the interval union and cannot change acceptance.
    auto gap_trace=[&](double gap) {
        auto q=manual(32,1);q.mode=StreamMode::adaptive;q.min_block_bytes=32;q.max_block_bytes=64;
        BlockStream x(g,q);double t=1;
        epoch(x,t,[](StreamSettings){return 1000.0;},true,gap);
        epoch(x,t,[](StreamSettings){return 1060.0;},true,gap);
        auto judged=x.next();check(bool(judged),"gap trace candidate is judged");
        return std::pair{x.settings().target_block_bytes,x.stats().adaptations};
    };
    check(gap_trace(0)==gap_trace(10),"caller computation gaps do not alter candidate acceptance");

    // A baseline cannot finish on zero-byte completions; it needs 16 useful ones.
    BlockStream useful(g,o);now=1;
    for(int i=0;i<15;++i) {auto r=useful.next();check(bool(r),"useful baseline progresses");useful.complete(*r,BlockCompletion{r->estimated_bytes,0.01,now,now+0.01});now+=0.01;}
    for(int i=0;i<10;++i) {auto r=useful.next();check(bool(r),"zero-byte baseline progresses");useful.complete(*r,BlockCompletion{0,0.01,now,now+0.01});now+=0.01;}
    check(useful.settings().phase==StreamPhase::baseline && useful.stats().trials==0,"zero-byte completions do not count toward exploration");
    auto last=useful.next();check(bool(last),"sixteenth useful completion request");
    useful.complete(*last,BlockCompletion{last->estimated_bytes,0.01,now,now+0.01});
    auto candidate=useful.next();check(bool(candidate),"sixteenth useful completion starts exploration");
    check(useful.settings().phase==StreamPhase::trial && useful.stats().trials==1,"16 useful completions open a candidate epoch");
}

void test_adaptive_region_drain() {
    auto g=geometry({8192},{16},{4096});
    auto o=manual(64,2);o.operation=StreamOperation::write;o.mode=StreamMode::adaptive;
    o.min_block_bytes=32;o.max_block_bytes=128;
    BlockStream s(g,o);std::vector<int> seen(product(g.shape));double now=1;
    for(int i=0;i<256;++i) {
        auto r=s.next();check(bool(r),"old region drains without stalling");
        check(r->start[0]<4096 && r->count[0]==16,"old shard keeps its 64-byte tiles");
        check(s.settings().phase==StreamPhase::baseline && !s.stats().trials,"trial waits for the owned shard");
        visit(*r,g.shape,seen);
        const auto dt=i<16?0.001:5.0;
        s.complete(*r,BlockCompletion{r->estimated_bytes,dt,now,now+dt});now+=dt+1;
    }
    auto first=s.next();check(bool(first),"candidate starts after draining old shard");
    check(s.settings().phase==StreamPhase::trial && s.settings().target_block_bytes==128,"candidate target is installed");
    check(first->start[0]==4096 && first->count[0]==32,"candidate actually issues 128-byte tiles");
    check(std::abs(s.stats().throughput_bytes_per_second-64000.0)<1e-6,"drain work is excluded from the completed epoch");
    visit(*first,g.shape,seen);s.complete(*first,BlockCompletion{first->estimated_bytes,0.001,now,now+0.001});now+=1;
    while(!s.finished()) {
        auto r=s.next();check(bool(r),"candidate shard makes progress");
        check(r->count[0]==32,"candidate keeps its requested tile shape");
        visit(*r,g.shape,seen);s.complete(*r,BlockCompletion{r->estimated_bytes,0.001,now,now+0.001});now+=1;
    }
    for(auto n:seen)check(n==1,"retuning retains exact write coverage");
    check(s.settings().phase==StreamPhase::stable && s.settings().target_block_bytes==128 && s.stats().adaptations==1,
          "the final complete candidate is judged when the stream ends");

    // An incomplete final trial is not retained as if it had been measured.
    BlockStream short_trial(geometry({16*17},{16}),o);now=1;
    for(int i=0;i<17;++i) {
        auto r=short_trial.next();check(bool(r),"short stream makes progress");
        short_trial.complete(*r,BlockCompletion{r->estimated_bytes,1,now,now+1});now+=1;
    }
    check(short_trial.finished() && short_trial.settings().target_block_bytes==64 &&
          short_trial.settings().limit==StreamLimit::insufficient_samples,
          "a short final trial restores the best measured settings");

    // Reconfiguration during an owned region collects no baseline until its
    // old tiling has finished; subsequent requests use the new target.
    auto m=manual(32,1);BlockStream changed(geometry({8192},{4096}),m);
    auto r=changed.next();check(bool(r),"manual region starts");changed.complete(*r,r->estimated_bytes);
    m.mode=StreamMode::adaptive;m.min_block_bytes=32;m.target_block_bytes=64;m.max_block_bytes=128;
    changed.configure(m);now=1;
    for(int i=1;i<512;++i) {
        auto q=changed.next();check(bool(q) && q->start[0]<4096 && q->count[0]==8,"configured region retains old tiling");
        changed.complete(*q,BlockCompletion{q->estimated_bytes,0.01,now,now+0.01});now+=0.01;
    }
    auto q=changed.next();check(bool(q) && q->start[0]==4096 && q->count[0]==16,"new region adopts configured target");
    check(!changed.stats().trials && changed.settings().phase==StreamPhase::baseline,"old tiling did not enter new baseline");
    changed.complete(*q,BlockCompletion{q->estimated_bytes,0.01,now,now+0.01});
}

void test_sparse_observation_windows() {
    auto g=geometry({1u<<20},{16});auto o=manual(64,1);o.mode=StreamMode::adaptive;
    o.min_block_bytes=32;o.max_block_bytes=128;
    auto zero_trace=[&](double gap,bool missing_interval) {
        BlockStream s(g,o);double now=1;
        for(int i=0;i<116;++i) {
            auto r=s.next();check(bool(r),"mixed useful and empty requests progress");
            const bool useful=i<15 || i==115,missing=missing_interval && i==40;
            const double dt=useful?1:10;
            s.complete(*r,BlockCompletion{useful?r->estimated_bytes:0,dt,missing?0:now,missing?0:now+dt});
            now+=dt+gap;
        }
        auto r=s.next();check(bool(r),"mixed epoch reaches a candidate");
        return std::pair{s.stats().throughput_bytes_per_second,s.settings().overlap_timing_available};
    };
    const auto expected=16.0*64.0/(16.0+1000.0);
    const auto compact=zero_trace(0,false),gaps=zero_trace(10,false),missing=zero_trace(0,true);
    check(std::abs(compact.first-expected)<1e-10 && std::abs(gaps.first-expected)<1e-10,
          "empty operations contribute I/O time and exclude application gaps");
    check(compact.second && gaps.second && !missing.second && std::abs(missing.first-expected)<1e-10,
          "an empty operation with no interval changes the whole epoch timing kind");

    // Stable observations still need 64 useful completions, even when each
    // bounded timing window contains mostly empty operations.
    auto fixed=manual(64,1);fixed.mode=StreamMode::adaptive;
    fixed.min_block_bytes=fixed.target_block_bytes=fixed.max_block_bytes=64;
    BlockStream stable(g,fixed);double t=1;
    for(int i=0;i<16;++i) {
        auto r=stable.next();check(bool(r),"stable baseline progress");
        stable.complete(*r,BlockCompletion{r->estimated_bytes,1,t,t+1});t+=6;
    }
    for(int i=0;i<384;++i) {
        auto r=stable.next();check(bool(r),"stable mixed observations progress");
        const bool useful=i%6==5;const double dt=useful?1:10;
        stable.complete(*r,BlockCompletion{useful?r->estimated_bytes:0,dt,t,t+dt});t+=dt+5;
    }
    auto after=stable.next();check(bool(after),"stable mixed epoch is evaluated");
    check(stable.settings().phase==StreamPhase::stable && !stable.stats().trials &&
          std::abs(stable.stats().throughput_bytes_per_second-(64.0*64.0/(64.0+3200.0)))<1e-10,
          "stable 64-useful threshold includes all empty I/O windows");
    stable.complete(*after,BlockCompletion{after->estimated_bytes,1,t,t+1});

    // Keep one outstanding request when observation sample 64 arrives, then
    // complete it out of order. Both intervals belong to the first window.
    o.min_block_bytes=o.target_block_bytes=o.max_block_bytes=64;o.max_in_flight=2;
    BlockStream overlap(g,o);double now=1;
    for(int i=0;i<16;++i) {
        auto r=overlap.next();check(bool(r),"overlap baseline progress");
        overlap.complete(*r,BlockCompletion{r->estimated_bytes,1,now,now+1});now+=11;
    }
    auto pair=[&](bool useful,bool check_cap=false) {
        auto a=overlap.next(),b=overlap.next();check(bool(a)&&bool(b),"two workers are available");
        overlap.complete(*b,BlockCompletion{useful?b->estimated_bytes:0,2,now+1,now+3});
        if(check_cap) check(!overlap.next(),"full observation window drains outstanding requests");
        overlap.complete(*a,BlockCompletion{useful?a->estimated_bytes:0,2,now,now+2});now+=13;
    };
    for(int i=0;i<31;++i) pair(false);
    auto single=overlap.next();check(bool(single),"single empty request progresses");
    overlap.complete(*single,BlockCompletion{0,2,now,now+2});now+=12;
    pair(false,true);
    for(int i=0;i<7;++i) pair(false);
    single=overlap.next();check(bool(single),"last empty request progresses");
    overlap.complete(*single,BlockCompletion{0,2,now,now+2});now+=12;
    for(int i=0;i<8;++i) pair(true);
    auto next=overlap.next();check(bool(next),"overlap trial is evaluated");
    check(std::abs(overlap.stats().throughput_bytes_per_second-(16.0*64.0/145.0))<1e-10,
          "bounded windows retain out-of-order interval union and empty I/O time");
    check(overlap.settings().in_flight_limit==1 && overlap.stats().rollbacks==1,"slow empty I/O rolls back the trial");
    overlap.complete(*next,BlockCompletion{next->estimated_bytes,1,now,now+1});

    // An all-empty stream cannot grow timing history or invent a candidate.
    o.max_in_flight=1;BlockStream empty(geometry({16*10000},{16}),o);
    now=1;largest_allocation=0;measuring=true;
    while(!empty.finished()) {
        auto r=empty.next();check(bool(r),"empty stream continues after a full observation window");
        empty.complete(*r,BlockCompletion{0,0.01,now,now+0.01});now+=0.02;
    }
    measuring=false;
#if !__has_feature(thread_sanitizer)
    check(largest_allocation<1024,"empty completions caused unbounded timing allocation");
#endif
    check(empty.stats().completed==10000 && empty.stats().trials==0 &&
          empty.settings().phase==StreamPhase::baseline,"all-empty stream finishes without a false trial");
    check(std::abs(empty.stats().io_seconds-100.0)<1e-6,"empty I/O remains in ordinary statistics");
}
}
int main() {try {test_geometry_and_coverage();test_parallel_and_contracts();test_sparse();test_adaptation();test_adaptive_region_drain();test_sparse_observation_windows();std::cout<<"streaming tests passed\n";}catch(const std::exception& e){measuring=false;std::cerr<<e.what()<<'\n';return 1;}}
