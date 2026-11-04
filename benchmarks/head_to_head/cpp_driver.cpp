// One prepared, serial operation per process. The runner checkpoints its
// measurement before allowing full-output hashing to begin.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <openssl/evp.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import anndata;

extern "C" int anndata_completeness_probe_begin(std::uint64_t, std::uint64_t,
                                                const char*, std::size_t, int) __attribute__((weak));
extern "C" void anndata_completeness_probe_end() __attribute__((weak));

using namespace anndata;
using Clock = std::chrono::steady_clock;

static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
static double cpu() { timespec t{}; if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t)) throw std::runtime_error("process CPU clock failed"); return double(t.tv_sec)+double(t.tv_nsec)*1e-9; }

struct Memory { std::size_t rss=0, peak=0, threads=0; };
static Memory memory() {
    std::ifstream f("/proc/self/status");
    std::string key, unit; std::size_t value=0; Memory m;
    while (f >> key) {
        if (key == "VmRSS:" || key == "VmHWM:") {
            f >> value >> unit;
            (key == "VmRSS:" ? m.rss : m.peak) = value;
        } else if (key == "Threads:") {
            f >> m.threads;
        } else std::getline(f, unit);
    }
    return m;
}

class Digest {
    EVP_MD_CTX* ctx_ = EVP_MD_CTX_new();
public:
    Digest() { if (!ctx_ || EVP_DigestInit_ex(ctx_,EVP_sha256(),nullptr) != 1) throw std::runtime_error("SHA-256 init failed"); }
    ~Digest() { EVP_MD_CTX_free(ctx_); }
    void add(const void* p,std::size_t n) { if (n && EVP_DigestUpdate(ctx_,p,n) != 1) throw std::runtime_error("SHA-256 update failed"); }
    std::string hex() {
        unsigned char v[EVP_MAX_MD_SIZE]; unsigned int n=0;
        if (EVP_DigestFinal_ex(ctx_,v,&n) != 1) throw std::runtime_error("SHA-256 final failed");
        constexpr char h[]="0123456789abcdef"; std::string s(2*n,'0');
        for (unsigned i=0;i<n;++i) {s[2*i]=h[v[i]>>4];s[2*i+1]=h[v[i]&15];} return s;
    }
};
static std::string hash(const void* p,std::size_t n) { Digest d; d.add(p,n); return d.hex(); }
static std::size_t width(DType t) {
    switch (t) {
    case DType::int8: case DType::uint8: case DType::boolean: return 1;
    case DType::int16: case DType::uint16: return 2;
    case DType::int32: case DType::uint32: case DType::float32: return 4;
    case DType::int64: case DType::uint64: case DType::float64: return 8;
    }
    throw std::runtime_error("unsupported dtype");
}
static const char* dtype(DType t) {
    switch (t) {
    case DType::int32:return "int32"; case DType::uint32:return "uint32";
    case DType::int64:return "int64"; case DType::uint64:return "uint64";
    case DType::float32:return "float32"; default:return "other";
    }
}

static void begin_probe(const std::filesystem::path& path) {
    if (!std::getenv("HEAD_TO_HEAD_PROBE")) return;
    auto root = std::filesystem::is_directory(path/"X") ? path/"X" : path.parent_path();
    if (!anndata_completeness_probe_begin || !anndata_completeness_probe_end ||
        anndata_completeness_probe_begin(512ull<<20,512ull<<20,root.c_str(),0,0))
        throw std::runtime_error("could not arm bounded interposer");
}
static void end_probe() { if (std::getenv("HEAD_TO_HEAD_PROBE")) anndata_completeness_probe_end(); }

static void checkpoint(std::string_view test,std::size_t n,double prepare,double seconds,double cpu_seconds,Memory before,Memory after,
                       const TransferStats* stats=nullptr) {
    std::cout << std::setprecision(12) << "{\"phase\":\"measurement\",\"case\":\"" << test
              << "\",\"size\":" << n << ",\"prepare_seconds\":" << prepare << ",\"seconds\":" << seconds << ",\"cpu_seconds\":" << cpu_seconds
              << ",\"rss_before_kib\":" << before.rss << ",\"rss_after_kib\":" << after.rss
              << ",\"lifetime_hwm_kib\":" << after.peak << ",\"threads_before\":" << before.threads
              << ",\"threads_after\":" << after.threads;
    if (stats) std::cout << ",\"logical_read_calls\":" << stats->read_calls << ",\"logical_write_calls\":" << stats->write_calls
                         << ",\"logical_read_bytes\":" << stats->read_bytes << ",\"logical_write_bytes\":" << stats->write_bytes
                         << ",\"scratch_bytes\":" << stats->scratch_bytes << ",\"scratch_allocations\":" << stats->scratch_allocations;
    std::cout << "}\n" << std::flush;
    std::string response; if (!std::getline(std::cin,response) || response!="continue") throw std::runtime_error("checkpoint acknowledgement missing");
}
static void verified(std::string_view checksum,std::size_t elements) {
    std::cout << "{\"phase\":\"verification\",\"sha256\":\"" << checksum << "\",\"elements\":" << elements << "}\n" << std::flush;
}

static void eager(std::string_view test,const std::string& source,const std::string& dest,Clock::time_point program) {
    const bool read = test=="h5ad_eager_read_dense", fused=test=="numpy_fused_loop";
    auto data = read ? AnnData{} : read_h5ad(source);
    const std::size_t n = read ? 0 : data.n_obs();
    const double prepare=since(program); const Memory before=memory();
    begin_probe(source); const auto cpu_start=cpu(); auto start=Clock::now();
    if (read) data=read_h5ad(source);
    else if (fused) {
        auto x=data.edit_x().dense<float>(); float* p=x.data();
        for (int k=0;k<8;++k) for (std::size_t i=0;i<n*n;++i) p[i]=p[i]*0.5f+0.125f;
    } else write_h5ad(dest,data);
    const double seconds=since(start), cpu_seconds=cpu()-cpu_start; end_probe(); const Memory after=memory();
    const auto rows=data.n_obs(), cols=data.n_vars();
    checkpoint(test,rows,prepare,seconds,cpu_seconds,before,after);
    if (rows!=cols) throw std::runtime_error("expected square X");
    auto v=data.x().dense<float>();
    if (!v.contiguous()) throw std::runtime_error("expected contiguous X");
    verified(hash(v.data(),rows*cols*sizeof(float)),rows*cols);
}

struct SparseBlock {
    std::vector<std::size_t> start,count;
    ReadRequirements r;
    std::vector<std::byte> values,indices,offsets;
};

static void backed(std::string_view test,const std::string& input,const std::string& output,Clock::time_point program) {
    const bool zarr=test.starts_with("zarr3_"), writing=test=="zarr3_full_write" || test=="zarr3_partial_write";
    const bool sparse=test=="h5ad_sparse_rows_csr" || test=="h5ad_sparse_cols_csc";
    const bool csr=test=="h5ad_sparse_rows_csr", gather=test=="zarr3_gather_read";
    const auto& path=writing?output:input;
    BackedOptions options; options.mode=writing?AccessMode::read_write:AccessMode::read_only;
    auto store=zarr?open_zarr(path,options):open_h5ad(path,options);
    auto array=store.x(); const auto n=array.extent(0), m=array.extent(1);
    if (n!=m || n%256) throw std::runtime_error("expected square matrix with side divisible by 256");
    std::vector<std::ptrdiff_t> ri,ci;
    if (gather) {
        ri={0,3,63,64,127,128,223,224,255,256,511,512,767,768,std::ptrdiff_t(n-2),std::ptrdiff_t(n-1)};
        for (std::size_t i=0;i<64;++i) ci.push_back(std::ptrdiff_t((i*37)%n));
        array=array.select({indices{ri},indices{ci}});
    }
    auto ctx=array.prepare_transfer();
    std::size_t nr=gather?16:writing?(test=="zarr3_full_write"?256:96):n;
    std::size_t nc=gather?64:writing?(test=="zarr3_full_write"?512:128):n;
    std::unique_ptr<float[]> dense;
    std::vector<SparseBlock> blocks;
    if (!sparse) {
        dense.reset(new float[nr*nc]);
        if (writing) std::fill_n(dense.get(),nr*nc,7.5f);
    } else {
        blocks.reserve(n/256);
        for (std::size_t i=0;i<n;i+=256) {
            SparseBlock b;
            b.start=csr?std::vector<std::size_t>{i,0}:std::vector<std::size_t>{0,i};
            b.count=csr?std::vector<std::size_t>{256,n}:std::vector<std::size_t>{n,256};
            b.r=ctx.requirements(b.start,b.count);
            b.values.resize(b.r.nnz*width(b.r.dtype)); b.indices.resize(b.r.nnz*width(b.r.index_dtype));
            b.offsets.resize(257*width(b.r.offset_dtype)); blocks.push_back(std::move(b));
        }
    }
    const double prepare=since(program);const Memory before=memory();
    begin_probe(path); const auto cpu_start=cpu(); auto start=Clock::now();
    if (sparse) for (auto& b:blocks) {
        SparseBuffer out{b.values.data(),b.indices.data(),b.offsets.data(),b.r.dtype,b.r.index_dtype,b.r.offset_dtype,
                         b.r.nnz,b.r.nnz,257};
        ctx.read_into(b.start,b.count,out);
    } else if (writing) {
        const std::vector<std::size_t> first=test=="zarr3_full_write"?std::vector<std::size_t>{0,0}:std::vector<std::size_t>{224,448};
        ctx.write_from(first,{nr,nc},DenseBuffer::from(dense.get(),{nr,nc}));
    } else if (gather) {
        auto out=DenseBuffer::from(dense.get(),{nr,nc});
        for (int k=0;k<8;++k) ctx.read_into({0,0},{nr,nc},out);
    } else for (std::size_t i=0;i<n;i+=128) {
        ctx.read_into({i,0},{128,n},DenseBuffer::from(dense.get()+i*n,{128,n}));
    }
    const double seconds=since(start), cpu_seconds=cpu()-cpu_start;end_probe();const Memory after=memory();const auto stats=ctx.stats();
    checkpoint(test,n,prepare,seconds,cpu_seconds,before,after,&stats);
    if (!sparse) verified(hash(dense.get(),nr*nc*sizeof(float)),nr*nc);
    else {
        Digest d;std::size_t nnz=0;
        for (auto& b:blocks) {d.add(b.values.data(),b.values.size());d.add(b.indices.data(),b.indices.size());d.add(b.offsets.data(),b.offsets.size());nnz+=b.r.nnz;}
        std::cout << "{\"phase\":\"verification\",\"sha256\":\"" << d.hex() << "\",\"elements\":" << nnz
                  << ",\"blocks\":" << blocks.size() << ",\"value_dtype\":\"" << dtype(blocks.front().r.dtype)
                  << "\",\"index_dtype\":\"" << dtype(blocks.front().r.index_dtype)
                  << "\",\"offset_dtype\":\"" << dtype(blocks.front().r.offset_dtype) << "\"}\n" << std::flush;
    }
}

int main(int argc,char** argv) {
    auto start=Clock::now();
    if (argc!=4) {std::cerr << "usage: head_to_head CASE FIXTURE OUTPUT\n";return 2;}
    try {
        std::string_view test=argv[1];
        if (test=="h5ad_eager_read_dense" || test=="h5ad_eager_write_dense" || test=="numpy_fused_loop")
            eager(test,argv[2],argv[3],start);
        else backed(test,argv[2],argv[3],start);
    } catch (const std::exception& e) {if (std::getenv("HEAD_TO_HEAD_PROBE") && anndata_completeness_probe_end) anndata_completeness_probe_end();std::cerr << e.what() << '\n';return 1;}
}
