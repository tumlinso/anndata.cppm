// One public backed numerical write per process, bracketed by the probe.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

import anndatapp;

#ifdef __APPLE__
extern "C" int anndatapp_completeness_probe_begin(std::uint64_t, std::uint64_t,
                                                const char*, std::size_t, int) __attribute__((weak_import));
extern "C" void anndatapp_completeness_probe_end() __attribute__((weak_import));
#else
extern "C" int anndatapp_completeness_probe_begin(std::uint64_t, std::uint64_t,
                                                const char*, std::size_t, int) __attribute__((weak));
extern "C" void anndatapp_completeness_probe_end() __attribute__((weak));
#endif
using namespace anndatapp;
using Clock = std::chrono::steady_clock;

static float value(std::size_t i, std::size_t j) { return float(10000 + 131*i + j); }

int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "usage: completeness CASE STORE INDEX_BYTES INDEX_AT_START\n"; return 2; }
    try {
        const std::string test = argv[1], path = argv[2], xdir = path + "/X";
        const auto index_bytes = std::stoull(argv[3]);
        const int index_at_start = std::stoi(argv[4]);
        if (!anndatapp_completeness_probe_begin || !anndatapp_completeness_probe_end)
            throw std::runtime_error("bounded probe interposer is required");
        BackedOptions options; options.mode = AccessMode::read_write;
        auto store = open_zarr(path, options);
        auto root = store.x();
        auto geometry = root.storage_geometry();
        if (geometry.data.organization != StorageOrganization::sharded || geometry.data.write_unit.size() != 2)
            throw std::runtime_error("fixture X must be a two-dimensional sharded array");
        const auto sr = geometry.data.write_unit[0], sc = geometry.data.write_unit[1];
        std::size_t rows = 0, columns = 0, touched = 0;
        std::vector<std::ptrdiff_t> ri, ci;
        std::vector<std::size_t> start{0,0};
        if (test == "full") { rows = sr; columns = sc; touched = 1; }
        else if (test == "partial") {
            if (sr != 256 || sc != 512) throw std::runtime_error("partial case requires standard shard geometry");
            start = {150,300}; rows = 200; columns = 400; touched = 4;
        } else if (test == "gather_stride" || test == "gather_view") {
            ri = {270,64,270,9,256,511,257,3,95,10,448,1,384,65,258,0};
            ci = {520,4,520,129,513,255,127,6};
            rows = ri.size(); columns = ci.size(); touched = 4;
        } else if (test == "huge_gap") {
            if (geometry.shape != std::vector<std::size_t>{std::size_t{1}<<30,std::size_t{1}<<20})
                throw std::runtime_error("huge-gap fixture shape differs");
            ri = {0,(std::ptrdiff_t{1}<<30)-1}; ci = {0,(std::ptrdiff_t{1}<<20)-1};
            rows = ri.size(); columns = ci.size(); touched = 4;
        } else throw std::runtime_error("unknown case");
        auto x = ri.empty() ? root.select({all,all}) : root.select({indices{ri},indices{ci}});
        auto source = Array::empty<float>({rows,columns});
        auto p = source.edit().dense<float>().data();
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < columns; ++j) p[i*columns+j] = value(i,j);
        auto transfer = x.prepare_transfer();
        std::vector<std::size_t> count{rows,columns};
        std::optional<ArrayView> gathered;
        if (test == "gather_view") {
            std::vector<std::size_t> r(rows),c(columns);
            for (std::size_t i = 0; i < rows; ++i) r[i] = rows-1-i;
            for (std::size_t j = 0; j < columns; ++j) c[j] = (3*j)%columns;
            gathered.emplace(source.view().select({Axis::gather(r.data(),r.size()),Axis::gather(c.data(),c.size())}));
        }
        auto input = test == "gather_stride"
            ? DenseBuffer::from(p + rows*columns-1,count,{-std::ptrdiff_t(columns*sizeof(float)),-std::ptrdiff_t(sizeof(float))})
            : DenseBuffer::from(p,count);
        if (anndatapp_completeness_probe_begin(512ULL<<20,512ULL<<20,xdir.c_str(),index_bytes,index_at_start))
            throw std::runtime_error("cannot arm bounded probe");
        auto t = Clock::now();
        if (gathered) x.write(*gathered);
        else transfer.write_from(start,count,input);
        auto seconds = std::chrono::duration<double>(Clock::now()-t).count();
        anndatapp_completeness_probe_end();
        auto stats = transfer.stats();
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
        std::cout << "{\"case\":\"" << test << "\",\"calls\":1,\"rows\":" << rows
                  << ",\"columns\":" << columns << ",\"touched_shards\":" << touched
                  << ",\"payload_bytes\":" << rows*columns*sizeof(float)
                  << ",\"seconds\":" << seconds << ",\"process_max_rss_kib\":" << usage.ru_maxrss;
        if (test != "gather_view") std::cout << ",\"backend_calls\":" << stats.write_calls
                  << ",\"backend_bytes\":" << stats.write_bytes << ",\"scratch_allocations\":" << stats.scratch_allocations
                  << ",\"scratch_bytes\":" << stats.scratch_bytes;
        else std::cout << ",\"backend_calls\":null,\"backend_bytes\":null,\"scratch_allocations\":null,\"scratch_bytes\":null";
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& e) {
        if (anndatapp_completeness_probe_end) anndatapp_completeness_probe_end();
        std::cerr << e.what() << '\n'; return 1;
    }
}
