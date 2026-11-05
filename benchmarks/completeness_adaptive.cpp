// A caller-owned worker pool executes real backed reads. The probe interposer
// supplies optional controlled per-pread latency; the library owns no threads.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <utility>
#include <vector>

import anndatapp;

extern "C" int anndatapp_completeness_probe_begin(std::uint64_t,std::uint64_t,const char*,std::size_t,int) __attribute__((weak));
extern "C" void anndatapp_completeness_probe_end() __attribute__((weak));
using namespace anndatapp;
using Clock = std::chrono::steady_clock;

static double stamp(Clock::time_point t) { return std::chrono::duration<double>(t.time_since_epoch()).count(); }
static float value(std::size_t r,std::size_t c) { return float((17*r+13*c+(r*c)%97)%1009)/1009.0f; }
struct Result {
    BlockRequest block;
    BlockCompletion completion;
    std::exception_ptr error;
};
struct Worker {
    BackedAnnData store;
    TransferContext transfer;
    Array buffer;
    float* p;
    explicit Worker(const std::string& path):store(open_zarr(path)),transfer(store.x().prepare_transfer(1024*1024)),
        buffer(Array::empty<float>({64,128})),p(buffer.edit().dense<float>().data()) {}
};
struct Snapshot { std::size_t completed, limit, target; StreamPhase phase; };

int main(int argc,char** argv) {
    if(argc!=3) { std::cerr << "usage: completeness_adaptive adaptive|1|2|4|8 STORE\n"; return 2; }
    try {
        const std::string mode=argv[1],path=argv[2],xdir=path+"/X";
        const bool adaptive=mode=="adaptive";
        const std::size_t limit=adaptive?8:std::stoull(mode);
        if(!limit || limit>8 || (!adaptive && mode!="1" && mode!="2" && mode!="4" && mode!="8"))
            throw std::runtime_error("invalid worker limit");
        if(!anndatapp_completeness_probe_begin || !anndatapp_completeness_probe_end)
            throw std::runtime_error("bounded probe interposer is required");
        if(!std::getenv("ANNDATAPP_COMPLETENESS_DELAY_FAST_US") ||
           !std::getenv("ANNDATAPP_COMPLETENESS_DELAY_SLOW_US") ||
           !std::getenv("ANNDATAPP_COMPLETENESS_DELAY_SLOW_FROM_SHARD_ROW"))
            throw std::runtime_error("controlled latency settings are required");
        auto setup_start=Clock::now();
        std::vector<std::unique_ptr<Worker>> workers;
        for(int i=0;i<8;++i) workers.push_back(std::make_unique<Worker>(path));
        auto g=workers[0]->transfer.geometry();
        if(g.shape!=std::vector<std::size_t>{4096,1024} ||
           g.data.inner_chunks!=std::vector<std::size_t>{64,128} ||
           g.data.write_unit!=std::vector<std::size_t>{256,512})
            throw std::runtime_error("adaptive fixture geometry differs");
        StreamOptions options;
        options.mode=adaptive?StreamMode::adaptive:StreamMode::manual;
        options.min_block_bytes=options.target_block_bytes=options.max_block_bytes=64*128*sizeof(float);
        options.max_in_flight=limit; options.scratch_bytes_per_worker=1024*1024; options.statistics=true;
        BlockStream stream(g,options);
        const auto setup=std::chrono::duration<double>(Clock::now()-setup_start).count();
        std::unique_ptr<std::atomic<unsigned>[]> seen(new std::atomic<unsigned>[512]{});
        std::mutex mutex;
        std::condition_variable work_ready,done_ready;
        std::deque<BlockRequest> queue;
        std::deque<Result> done;
        bool stop=false;
        std::vector<std::thread> threads;
        try { for(std::size_t w=0;w<workers.size();++w) threads.emplace_back([&,w] {
            auto& worker=*workers[w];
            for(;;) {
                std::optional<BlockRequest> block;
                { std::unique_lock<std::mutex> lock(mutex);work_ready.wait(lock,[&]{return stop||!queue.empty();});
                  if(stop && queue.empty()) return;block.emplace(std::move(queue.front()));queue.pop_front(); }
                Result result{std::move(*block),{},nullptr};
                try {
                    const auto& b=result.block;
                    if(b.start.size()!=2 || b.count!=std::vector<std::size_t>{64,128} || b.start[0]%64 || b.start[1]%128)
                        throw std::runtime_error("unexpected block grid");
                    const auto slot=b.start[0]/64*8+b.start[1]/128;
                    if(slot>=512 || seen[slot].fetch_add(1)!=0) throw std::runtime_error("block coverage repeated or out of range");
                    const auto first=Clock::now();
                    worker.transfer.read_into(b.start,b.count,DenseBuffer::from(worker.p,b.count));
                    const auto last=Clock::now();
                    result.completion={64*128*sizeof(float),std::chrono::duration<double>(last-first).count(),stamp(first),stamp(last)};
                    for(std::size_t i=0;i<64;++i) for(std::size_t j=0;j<128;++j)
                        if(worker.p[i*128+j]!=value(b.start[0]+i,b.start[1]+j))
                            throw std::runtime_error("independent dense formula differs");
                } catch(...) { result.error=std::current_exception(); }
                { std::lock_guard<std::mutex> lock(mutex);done.push_back(std::move(result)); }
                done_ready.notify_one();
            }
        }); } catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);stop=true;}
            work_ready.notify_all();for(auto& thread:threads) thread.join();throw;
        }
        std::exception_ptr failure;
        std::vector<Snapshot> trace;
        std::size_t active=0,max_active=0;
        double elapsed=0;
        bool armed=false;
        try {
            if(anndatapp_completeness_probe_begin(512ULL<<20,512ULL<<20,xdir.c_str(),260,0))
                throw std::runtime_error("cannot arm bounded probe");
            armed=true;
            auto first=Clock::now();
            while(!stream.finished()) {
                while(auto b=stream.next()) {
                    {std::lock_guard<std::mutex> lock(mutex);queue.push_back(std::move(*b));}
                    ++active;max_active=std::max(max_active,active);work_ready.notify_one();
                }
                if(!active) throw std::runtime_error("stream stalled without outstanding work");
                std::optional<Result> result;
                {std::unique_lock<std::mutex> lock(mutex);done_ready.wait(lock,[&]{return !done.empty();});
                 result.emplace(std::move(done.front()));done.pop_front();}
                --active;
                if(result->error) std::rethrow_exception(result->error);
                stream.complete(result->block,result->completion);
                const auto s=stream.settings();
                if(trace.empty() || s.in_flight_limit!=trace.back().limit || s.phase!=trace.back().phase || s.target_block_bytes!=trace.back().target)
                    trace.push_back({stream.stats().completed,s.in_flight_limit,s.target_block_bytes,s.phase});
            }
            elapsed=std::chrono::duration<double>(Clock::now()-first).count();
        } catch(...) { failure=std::current_exception(); }
        {std::lock_guard<std::mutex> lock(mutex);stop=true;if(failure) queue.clear();}
        work_ready.notify_all(); for(auto& thread:threads) thread.join();
        if(armed) anndatapp_completeness_probe_end();
        if(failure) std::rethrow_exception(failure);
        for(std::size_t i=0;i<512;++i) if(seen[i].load()!=1) throw std::runtime_error("block coverage incomplete");
        const auto stats=stream.stats(); const auto settings=stream.settings();
        std::size_t backend_calls=0,backend_bytes=0,scratch_allocations=0,scratch_bytes=0;
        for(auto& w:workers) {
            const auto s=w->transfer.stats();backend_calls+=s.read_calls;backend_bytes+=s.read_bytes;
            scratch_allocations+=s.scratch_allocations;scratch_bytes+=s.scratch_bytes;
        }
        rusage usage{};if(getrusage(RUSAGE_SELF,&usage))throw std::runtime_error("getrusage failed");
        std::cout << "{\"mode\":\"" << mode << "\",\"calls\":" << stats.completed
                  << ",\"payload_bytes\":" << stats.useful_bytes << ",\"setup_seconds\":" << setup
                  << ",\"seconds\":" << elapsed << ",\"max_outstanding\":" << max_active
                  << ",\"backend_calls\":" << backend_calls << ",\"backend_bytes\":" << backend_bytes
                  << ",\"scratch_allocations\":" << scratch_allocations << ",\"scratch_bytes\":" << scratch_bytes
                  << ",\"process_max_rss_kib\":" << usage.ru_maxrss
                  << ",\"trials\":" << stats.trials << ",\"adaptations\":" << stats.adaptations
                  << ",\"rollbacks\":" << stats.rollbacks << ",\"final_limit\":" << settings.in_flight_limit
                  << ",\"trace\":[";
        for(std::size_t i=0;i<trace.size();++i) {
            if(i) std::cout << ',';
            const auto& s=trace[i];
            std::cout << "{\"completed\":" << s.completed << ",\"in_flight\":" << s.limit
                      << ",\"target_bytes\":" << s.target << ",\"phase\":" << int(s.phase) << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
