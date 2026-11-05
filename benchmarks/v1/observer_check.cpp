// Untimed observer preflight. A missing interposition mechanism fails the build
// receipt instead of silently turning unsupported observations into zeroes.
#include <hdf5.h>
#include <zstd.h>
#include <zlib.h>
#include <dlfcn.h>
#include <sys/uio.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <new>
#include <stdexcept>

namespace {
void* volatile escaped = nullptr;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
template<class F> F symbol(const char* name) {
    auto f = reinterpret_cast<F>(dlsym(RTLD_DEFAULT,name));
    require(f != nullptr,"missing injected observer entry point"); return f;
}
}
int main() {
    try {
        const auto alloc_begin = symbol<void(*)()>("v1_alloc_begin");
        const auto alloc_end = symbol<void(*)(std::uint64_t*)>("v1_alloc_end");
        const auto io_begin = symbol<void(*)()>("v1_io_begin");
        const auto io_end = symbol<void(*)(std::uint64_t*)>("v1_io_end");
        alloc_begin();
        void* a = ::operator new(17); escaped = a;
        void* b = ::operator new[](19); escaped = b;
        void* c = ::operator new(23,std::align_val_t(64)); escaped = c;
        void* d = ::operator new[](29,std::align_val_t(64)); escaped = d;
        std::uint64_t allocations[2]{}; alloc_end(allocations);
        ::operator delete(a); ::operator delete[](b);
        ::operator delete(c,std::align_val_t(64)); ::operator delete[](d,std::align_val_t(64));
        require(allocations[0] == 4 && allocations[1] == 88,"global new observer did not count all four allocation forms");
        auto* file = std::tmpfile(); require(file != nullptr,"observer temporary file failed");
        const int fd = fileno(file); char data[8]{}; iovec vector{data,sizeof(data)};
        io_begin();
        require(::write(fd,data,sizeof(data)) == sizeof(data),"observer write failed");
        require(::pwrite(fd,data,sizeof(data),0) == sizeof(data),"observer pwrite failed");
        require(::writev(fd,&vector,1) == sizeof(data),"observer writev failed");
        require(lseek(fd,0,SEEK_SET) == 0,"observer seek failed");
        require(::read(fd,data,sizeof(data)) == sizeof(data),"observer read failed");
        require(::pread(fd,data,sizeof(data),0) == sizeof(data),"observer pread failed");
        require(::readv(fd,&vector,1) == sizeof(data),"observer readv failed");
        std::uint64_t io[16]{}; io_end(io); std::fclose(file);
        require(io[0] == 3 && io[1] == 24 && io[2] == 3 && io[3] == 24,"libc observer did not count all six I/O forms");
        io_begin();
        auto* encoder = ZSTD_createCCtx(); auto* decoder = ZSTD_createDCtx();
        require(encoder && decoder,"observer codec context creation failed");
        char compressed[128]{};
        const auto size = ZSTD_compress2(encoder,compressed,sizeof(compressed),data,sizeof(data));
        require(!ZSTD_isError(size),"observer compression failed");
        ZSTD_inBuffer input{compressed,size,0}; ZSTD_outBuffer output{data,sizeof(data),0};
        require(!ZSTD_isError(ZSTD_decompressStream(decoder,&output,&input)),"observer decompression failed");
        require(!ZSTD_isError(ZSTD_freeCCtx(encoder)) && !ZSTD_isError(ZSTD_freeDCtx(decoder)),"observer codec release failed");
        z_stream deflater{}, inflater{};
        require(deflateInit(&deflater,1) == Z_OK && inflateInit(&inflater) == Z_OK,"observer zlib init failed");
        require(deflateEnd(&deflater) == Z_OK && inflateEnd(&inflater) == Z_OK,"observer zlib release failed");
        require(deflateInit2(&deflater,1,Z_DEFLATED,15,8,Z_DEFAULT_STRATEGY) == Z_OK &&
                inflateInit2(&inflater,15) == Z_OK,"observer zlib init2 failed");
        require(deflateEnd(&deflater) == Z_OK && inflateEnd(&inflater) == Z_OK,"observer zlib release2 failed");
        H5Eset_auto2(H5E_DEFAULT,nullptr,nullptr);
        require(H5Dread(-1,-1,-1,-1,-1,data) < 0 && H5Dwrite(-1,-1,-1,-1,-1,data) < 0,"invalid HDF5 requests unexpectedly succeeded");
        std::uint64_t codecs[16]{}; io_end(codecs);
        for (int i = 4; i < 16; ++i) require(codecs[i] > 0,"codec/HDF5 observer entry point was not intercepted");
        std::cout << "{\"observer_check\":true,\"new_calls\":" << allocations[0]
                  << ",\"new_requested_bytes\":" << allocations[1] << ",\"io\":[";
        for (int i = 0; i < 16; ++i) std::cout << (i ? "," : "") << io[i];
        std::cout << "],\"codec_calls\":[";
        for (int i = 4; i < 16; ++i) std::cout << (i > 4 ? "," : "") << codecs[i];
        std::cout << "]}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
