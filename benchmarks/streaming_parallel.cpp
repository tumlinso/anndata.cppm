// Caller-managed workers with one independent store and transfer per worker.
// Every write request owns a complete physical Zarr chunk or shard.
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

import anndata;
extern "C" int anndata_probe_begin(std::uint64_t, std::uint64_t) __attribute__((weak));
extern "C" void anndata_probe_end() __attribute__((weak));
using namespace anndata;
using Clock = std::chrono::steady_clock;

static float value(std::size_t r, std::size_t c) {
    return float((17*r + 13*c + (r*c)%97)%1009)/1009.0f;
}
static std::uint64_t hash_bytes(const float* p, std::size_t n) {
    auto q = reinterpret_cast<const unsigned char*>(p);
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n*sizeof(float); ++i) h = (h ^ q[i])*1099511628211ULL;
    return h;
}
static std::vector<Selection> select(std::size_t i, std::size_t j, std::size_t m, std::size_t n) {
    return {range{std::ptrdiff_t(i),std::ptrdiff_t(i+m),1},
            range{std::ptrdiff_t(j),std::ptrdiff_t(j+n),1}};
}
struct Worker {
    BackedAnnData store;
    BackedArray x;
#if ANNDATA_STREAM_CANDIDATE
    TransferContext transfer;
#endif
    Array data;
    float* p;
    double io_seconds = 0;
    std::exception_ptr error;
    Worker(const std::string& path, bool writing, std::size_t m, std::size_t n)
        : store(open_zarr(path, BackedOptions{writing ? AccessMode::read_write : AccessMode::read_only})),
          x(store.x())
#if ANNDATA_STREAM_CANDIDATE
        , transfer(x.prepare_transfer())
#endif
        , data(Array::empty<float>({m,n})), p(data.edit().dense<float>().data()) {}
};
int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: streaming_parallel read|write PATH ROW_BLOCK COLUMN_BLOCK WORKERS");
        const std::string action = argv[1], path = argv[2];
        const auto m = std::stoul(argv[3]), n = std::stoul(argv[4]), nw = std::stoul(argv[5]);
        const bool writing = action == "write";
        if ((!writing && action != "read") || !m || !n || !nw || nw > 16) throw std::runtime_error("invalid action, dimensions or worker count");
        // Serially deconsolidate shared metadata before writer sessions. This
        // idempotent write restores the original first value and is untimed.
        std::unique_ptr<BackedAnnData> preparation;
        if (writing) {
            preparation = std::make_unique<BackedAnnData>(open_zarr(path, BackedOptions{AccessMode::read_write}));
            auto initial = Array::zeros<float>({1,1});
            preparation->x().select(select(0,0,1,1)).write(initial.view());
        }
        std::vector<std::unique_ptr<Worker>> workers;
        for (std::size_t i = 0; i < nw; ++i) workers.push_back(std::make_unique<Worker>(path,writing,m,n));
        const auto shape = workers[0]->x.shape();
        if (shape.size() != 2 || shape[0]%m || shape[1]%n) throw std::runtime_error("fixture must be an exact block grid");
        const auto nr = shape[0]/m, nc = shape[1]/n, blocks = nr*nc;
        std::vector<std::uint64_t> hashes(blocks);
        std::barrier gate(std::ptrdiff_t(nw + 1));
        std::vector<std::thread> threads;
        for (std::size_t w = 0; w < nw; ++w) threads.emplace_back([&, w] {
            auto& worker = *workers[w];
            gate.arrive_and_wait();
            try {
                for (std::size_t k = w; k < blocks; k += nw) {
                    const auto r = k/nc*m, c = k%nc*n;
                    if (writing) for (std::size_t i=0; i<m; ++i) for (std::size_t j=0; j<n; ++j)
                        worker.p[i*n+j] = value(r+i,c+j) + 0.5f;
                    auto t = Clock::now();
#if ANNDATA_STREAM_CANDIDATE
                    DenseBuffer b{worker.p, DType::float32, {m,n}, {}};
                    if (writing) worker.transfer.write_from({r,c},{m,n},b);
                    else worker.transfer.read_into({r,c},{m,n},b);
#else
                    auto selected = worker.x.select(select(r,c,m,n));
                    if (writing) selected.write(worker.data.view());
                    else selected.prepare_read().read_into(DenseBuffer{worker.p,DType::float32,{m,n},{}});
#endif
                    worker.io_seconds += std::chrono::duration<double>(Clock::now()-t).count();
                    if (!writing) for (std::size_t i=0; i<m; ++i) for (std::size_t j=0; j<n; ++j)
                        if (worker.p[i*n+j] != value(r+i,c+j)) throw std::runtime_error("read values differ");
                    hashes[k] = hash_bytes(worker.p,m*n);
                }
            } catch (...) { worker.error = std::current_exception(); }
        });
        if (!anndata_probe_begin || !anndata_probe_end) throw std::runtime_error("bounded probe interposer required");
        if (anndata_probe_begin(512ULL*1024*1024,512ULL*1024*1024)) throw std::runtime_error("cannot arm bounded probe");
        const auto t = Clock::now();
        gate.arrive_and_wait();
        for (auto& thread:threads) thread.join();
        const auto elapsed = std::chrono::duration<double>(Clock::now()-t).count();
        anndata_probe_end();
        double io = 0; for (auto& worker:workers) { if (worker->error) std::rethrow_exception(worker->error); io += worker->io_seconds; }
        std::uint64_t hash = 1469598103934665603ULL;
        for (auto h:hashes) hash = (hash ^ h)*1099511628211ULL;
        std::cout << "{\"mode\":\"" << (ANNDATA_STREAM_CANDIDATE?"context":"select")
                  << "\",\"calls\":" << blocks << ",\"payload_bytes\":" << blocks*m*n*sizeof(float)
                  << ",\"total_seconds\":" << elapsed << ",\"summed_io_seconds\":" << io
                  << ",\"workers\":" << nw << ",\"hash\":" << hash;
#if ANNDATA_STREAM_CANDIDATE
        std::size_t backend = 0, bytes = 0, allocations = 0, scratch = 0;
        for (auto& worker:workers) { auto s = worker->transfer.stats(); backend += writing?s.write_calls:s.read_calls;
            bytes += writing?s.write_bytes:s.read_bytes; allocations += s.scratch_allocations; scratch += s.scratch_bytes; }
        std::cout << ",\"backend_calls\":" << backend << ",\"backend_bytes\":" << bytes
                  << ",\"scratch_allocations\":" << allocations << ",\"scratch_bytes\":" << scratch;
#endif
        std::cout << "}\n";
        if (preparation) preparation->flush();
        return 0;
    } catch (const std::exception& e) {
        if (anndata_probe_end) anndata_probe_end();
        std::cerr << e.what() << '\n';
        return 2;
    }
}
