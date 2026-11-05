#include "harness.hpp"
#include <algorithm>
#include <array>
#include <barrier>
#include <exception>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

import anndatapp;
using namespace anndatapp;
namespace v1 {
namespace {
float value(std::size_t r, std::size_t c) {
    return float(double((17*r + 13*c + (r*c)%97)%1009)/1009.0);
}
void check(const float* p, std::size_t r, std::size_t c, std::size_t m, std::size_t n) {
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j)
        require(p[i*n+j] == value(r+i,c+j), "dense fixture formula differs");
}
BackedAnnData open(const std::string& path, bool h5ad, bool writable = false) {
    BackedOptions o; o.mode = writable ? AccessMode::read_write : AccessMode::read_only;
    o.scratch_bytes = 1u << 20; o.decoded_cache_bytes = 0;
    return h5ad ? open_h5ad(path, o) : open_zarr(path, o);
}
void dense_transfer(const std::string& path, bool h5ad, const char* prefix) {
    auto store = open(path, h5ad, true);
    auto x = store.x(); auto context = x.prepare_transfer(1u << 20);
    constexpr std::size_t m = 64, n = 64, repeats = 4;
    std::vector<float> p(m*n), q(m*n, -0.25f);
    const std::size_t first = h5ad ? 96 : 32;
    const std::vector<std::size_t> start{first,first}, count{m,n};
    auto read = DenseBuffer::from(p.data(),count), write = DenseBuffer::from(q.data(),count);
    const auto before = context.stats();
    auto a = measure([&] { for (std::size_t i = 0; i < repeats; ++i) context.read_into(start,count,read); });
    const auto after = context.stats();
    check(p.data(),first,first,m,n);
    const std::string read_name = std::string(prefix) + "_dense_read";
    emit(read_name.c_str(),repeats,a,repeats*m*n*sizeof(float),
         after.read_calls-before.read_calls,after.read_bytes-before.read_bytes);
    auto b = measure([&] { for (std::size_t i = 0; i < repeats; ++i) context.write_from(start,count,write); });
    const auto written = context.stats();
    context.read_into(start,count,read);
    for (auto x : p) require(x == -0.25f, "dense write readback differs");
    const std::string write_name = std::string(prefix) + "_dense_write";
    emit(write_name.c_str(),repeats,b,repeats*m*n*sizeof(float),
         written.write_calls-after.write_calls,written.write_bytes-after.write_bytes);
    store.flush();
}
void sparse_transfer(const std::string& path, bool csr, bool native) {
    auto store = open(path,true);
    auto x = store.x().select(native
        ? csr ? std::vector<Selection>{range{32,64},all} : std::vector<Selection>{all,range{32,64}}
        : csr ? std::vector<Selection>{all,range{32,64}} : std::vector<Selection>{range{32,64},all});
    auto prepared = x.prepare_read();
    auto context = native ? x.prepare_transfer(1u << 20) : TransferContext{};
    const auto shape = x.shape();
    std::vector<std::size_t> start{0,0};
    const auto req = native ? context.requirements(start,shape) : prepared.requirements();
    require(req.nnz == 32*8,"sparse selected nnz differs");
    require(req.index_dtype == DType::int32 && req.offset_dtype == DType::int32,
            "sparse fixture requires int32 structure");
    std::vector<float> values(req.nnz);
    std::vector<std::int32_t> indices(req.nnz), offsets((csr ? shape[0] : shape[1])+1);
    SparseBuffer out{values.data(),indices.data(),offsets.data(),DType::float32,DType::int32,DType::int32,
                     values.size(),indices.size(),offsets.size()};
    constexpr std::size_t repeats = 4;
    const auto before = native ? context.stats() : TransferStats{};
    const auto a = measure([&] { for (std::size_t i = 0; i < repeats; ++i) {
        if (native) context.read_into(start,shape,out); else prepared.read_into(out);
    } });
    const auto after = native ? context.stats() : TransferStats{};
    // The independent fixture is a cyclic eight-diagonal matrix. Check every
    // selected stored entry against its physical row/column and formula.
    require(offsets.front() == 0 && std::size_t(offsets.back()) == req.nnz,"sparse offsets differ");
    for (std::size_t major = 0; major+1 < offsets.size(); ++major) {
        auto p = offsets[major];
        require(p >= 0 && offsets[major+1] >= p && std::size_t(offsets[major+1]) <= req.nnz,"sparse offset bounds differ");
        for (std::size_t minor = 0; minor < (csr ? shape[1] : shape[0]); ++minor) {
            const auto r = csr ? major+(native ? 32 : 0) : minor+(native ? 0 : 32);
            const auto c = csr ? minor+(native ? 0 : 32) : major+(native ? 32 : 0);
            bool present = false;
            for (std::size_t k = 0; k < 8; ++k) present |= c == (r+k*61)%512;
            if (!present) continue;
            require(p < offsets[major+1] && std::size_t(indices[p]) == minor,"sparse expected occurrence differs");
            require(values[p] == float(1 + (17*r + 13*c)%101),"sparse value formula differs"); ++p;
        }
        require(p == offsets[major+1],"sparse extra occurrence");
    }
    const std::string name = std::string("h5ad_") + (csr ? "csr_" : "csc_") + (native ? "native" : "opposite");
    emit(name.c_str(),repeats,a,repeats*(values.size()*4+indices.size()*4+offsets.size()*4),
         native ? std::int64_t(after.read_calls-before.read_calls) : -1,
         native ? std::int64_t(after.read_bytes-before.read_bytes) : -1,
         native ? "transfer_context" : "prepared_read");
}
void zarr3(const std::string& path) {
    auto store = open(path,false,true);
    const std::vector<std::ptrdiff_t> ri{0,63,64,127,128,255,128}, ci{255,128,63,0,64,127,64};
    auto selected = store.x().select({indices{ri},indices{ci}});
    auto transfer = selected.prepare_transfer(1u << 20);
    std::vector<float> p(ri.size()*ci.size());
    const std::vector<std::size_t> zero{0,0}, shape{ri.size(),ci.size()};
    const auto output = DenseBuffer::from(p.data(),shape);
    constexpr std::size_t repeats = 4;
    const auto before = transfer.stats();
    auto a = measure([&] { for (std::size_t k = 0; k < repeats; ++k) transfer.read_into(zero,shape,output); });
    const auto after = transfer.stats();
    for (std::size_t i = 0; i < ri.size(); ++i) for (std::size_t j = 0; j < ci.size(); ++j)
        require(p[i*ci.size()+j] == value(ri[i],ci[j]),"Zarr3 gathered formula differs");
    emit("zarr3_gather_read",repeats,a,repeats*p.size()*4,
         after.read_calls-before.read_calls,after.read_bytes-before.read_bytes);
    auto x = store.x(); auto context = x.prepare_transfer(1u << 20);
    require(context.geometry().data.organization == StorageOrganization::sharded,"Zarr3 fixture must be sharded");
    std::vector<float> q(64*64,-0.5f), readback(q.size());
    const std::vector<std::size_t> start{96,96}, count{64,64};
    const auto input = DenseBuffer::from(q.data(),count), result = DenseBuffer::from(readback.data(),count);
    const auto initial = context.stats();
    auto b = measure([&] { for (std::size_t k = 0; k < repeats; ++k) context.write_from(start,count,input); });
    const auto written = context.stats();
    context.read_into(start,count,result);
    require(q == readback,"Zarr3 partial shard write differs");
    emit("zarr3_partial_shard_write",repeats,b,repeats*q.size()*4,
         written.write_calls-initial.write_calls,written.write_bytes-initial.write_bytes);
    store.flush();
}
struct Worker {
    BackedAnnData store;
    TransferContext transfer;
    std::vector<float> data;
    DenseBuffer buffer;
    std::exception_ptr error;
    std::optional<BlockRequest> request;
    BlockCompletion completion;
    Worker(const std::string& path) : store(open(path,false,true)), transfer(store.x().prepare_transfer(1u << 20)),
        data(64*64), buffer(DenseBuffer::from(data.data(),{64,64})) {}
};
template<class F> Sample workers(std::array<std::unique_ptr<Worker>,4>& w, F f) {
    std::barrier gate(5);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < w.size(); ++i) threads.emplace_back([&,i] {
        gate.arrive_and_wait();
        try { f(*w[i],i); } catch (...) { w[i]->error = std::current_exception(); }
        gate.arrive_and_wait();
    });
    const auto a = measure([&] { gate.arrive_and_wait(); gate.arrive_and_wait(); });
    for (auto& t : threads) t.join();
    for (auto& x : w) if (x->error) std::rethrow_exception(x->error);
    return a;
}
void contexts(const std::string& path) {
    std::array<std::unique_ptr<Worker>,4> w;
    // Numerical payload edit deconsolidates shared metadata before any workers.
    { auto store = open(path,false,true); float p = value(0,0);
      store.x().select({range{0,1},range{0,1}}).write(Array::borrow_dense(&p,{1,1},{1,1}).view()); store.flush(); }
    for (auto& x : w) x = std::make_unique<Worker>(path);
    std::array<TransferStats,4> before;
    const std::array<std::vector<std::size_t>,4> starts{{{0,0},{0,64},{64,0},{64,64}}};
    const std::vector<std::size_t> count{64,64};
    for (std::size_t i = 0; i < 4; ++i) before[i] = w[i]->transfer.stats();
    constexpr std::size_t repeats = 4;
    auto a = workers(w,[&](Worker& x, std::size_t i) {
        for (std::size_t k = 0; k < repeats; ++k) x.transfer.read_into(starts[i],count,x.buffer);
    });
    std::size_t calls = 0, bytes = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        check(w[i]->data.data(),i/2*64,i%2*64,64,64);
        const auto s = w[i]->transfer.stats(); calls += s.read_calls-before[i].read_calls; bytes += s.read_bytes-before[i].read_bytes;
        before[i] = s; std::fill(w[i]->data.begin(),w[i]->data.end(),-float(i+1));
    }
    emit("reused_contexts_4worker_read",4*repeats,a,4*repeats*64*64*4,calls,bytes);
    auto b = workers(w,[&](Worker& x, std::size_t i) {
        for (std::size_t k = 0; k < repeats; ++k) x.transfer.write_from(starts[i],count,x.buffer);
    });
    calls = bytes = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const auto s = w[i]->transfer.stats(); calls += s.write_calls-before[i].write_calls; bytes += s.write_bytes-before[i].write_bytes;
        w[i]->transfer.read_into({i/2*64,i%2*64},{64,64},w[i]->buffer);
        for (auto p : w[i]->data) require(p == -float(i+1),"disjoint worker write differs");
        w[i]->store.flush();
    }
    emit("disjoint_chunk_4worker_write",4*repeats,b,4*repeats*64*64*4,calls,bytes);
    // The adaptive scheduler is tested with real overlapping reads, unchanged
    // latency, a fixed four-worker reservation and no synthetic delay.
    StreamOptions options; options.mode = StreamMode::adaptive; options.max_in_flight = 4;
    options.min_block_bytes = options.target_block_bytes = options.max_block_bytes = 64*64*4;
    options.scratch_bytes_per_worker = 1u << 20; options.statistics = true;
    BlockStream stream(w[0]->transfer.geometry(),options);
    require(w[0]->transfer.geometry().shape == std::vector<std::size_t>{2048,256},"context fixture geometry differs");
    std::vector<float> adaptive_output(2048*256,-9.0f);
    std::array<DenseBuffer,128> adaptive_buffers;
    std::array<unsigned,128> seen{};
    for (std::size_t i = 0; i < 128; ++i)
        adaptive_buffers[i] = DenseBuffer::from(adaptive_output.data()+i*64*64,{64,64});
    std::barrier gate(5); bool stop = false;
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < 4; ++i) threads.emplace_back([&,i] {
        auto& x = *w[i];
        for (;;) {
            gate.arrive_and_wait(); if (stop) break;
            if (x.request) try {
                const auto slot = x.request->start[0]/64*4+x.request->start[1]/64;
                const auto first = Clock::now();
                x.transfer.read_into(x.request->start,x.request->count,adaptive_buffers[slot]);
                const auto last = Clock::now();
                x.completion = {64*64*4,std::chrono::duration<double>(last-first).count(),
                    std::chrono::duration<double>(first.time_since_epoch()).count(),
                    std::chrono::duration<double>(last.time_since_epoch()).count()};
            } catch (...) { x.error = std::current_exception(); }
            gate.arrive_and_wait();
        }
    });
    for (std::size_t i = 0; i < 4; ++i) before[i] = w[i]->transfer.stats();
    auto c = measure([&] {
        while (!stream.finished()) {
            std::size_t active = 0;
            for (auto& x : w) { x->request = stream.next(); active += bool(x->request); }
            require(active != 0,"adaptive stream stalled");
            gate.arrive_and_wait(); gate.arrive_and_wait();
            for (auto& x : w) if (x->request) {
                if (x->error) std::rethrow_exception(x->error);
                ++seen[x->request->start[0]/64*4+x->request->start[1]/64];
                stream.complete(*x->request,x->completion);
            }
        }
    });
    stop = true; gate.arrive_and_wait(); for (auto& t : threads) t.join();
    const auto stats = stream.stats(); require(stats.completed == 128,"adaptive block coverage differs");
    require(stats.trials > 0,"adaptive trial window was not reached");
    c.stream_trials = stats.trials; c.stream_adaptations = stats.adaptations;
    c.stream_rollbacks = stats.rollbacks; c.stream_final_limit = stream.settings().in_flight_limit;
    for (auto n : seen) require(n == 1,"adaptive coverage repeated or missing");
    for (std::size_t r = 0; r < 2048; ++r) for (std::size_t col = 0; col < 256; ++col) {
        const auto expected = r < 128 && col < 128 ? -float((r/64)*2+col/64+1) : value(r,col);
        const auto slot = r/64*4+col/64, offset = (r%64)*64+col%64;
        require(adaptive_output[slot*64*64+offset] == expected,"adaptive transfer output differs");
    }
    calls = bytes = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const auto s = w[i]->transfer.stats(); calls += s.read_calls-before[i].read_calls; bytes += s.read_bytes-before[i].read_bytes;
    }
    emit("adaptive_4reserved_worker_read",stats.completed,c,stats.useful_bytes,calls,bytes);
}
}
void storage_case(const std::string& name, const std::string& fixtures) {
    if (name == "h5ad") {
        dense_transfer(fixtures+"/dense.h5ad",true,"h5ad");
        for (bool csr : {true,false}) for (bool native : {true,false})
            sparse_transfer(fixtures+(csr ? "/csr.h5ad" : "/csc.h5ad"),csr,native);
    } else if (name == "zarr2") dense_transfer(fixtures+"/z2.zarr",false,"zarr2_chunk");
    else if (name == "zarr3") zarr3(fixtures+"/z3.zarr");
    else if (name == "contexts") contexts(fixtures+"/contexts.zarr");
    else require(false,"unknown storage scenario");
}
}
