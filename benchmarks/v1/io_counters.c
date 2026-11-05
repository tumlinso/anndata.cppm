#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <errno.h>
#ifndef __APPLE__
#include <sys/syscall.h>
#endif
#include <unistd.h>
#include <time.h>
#include "dyld_interpose.h"

#ifdef __APPLE__
#define V1_SYSCALL(...) (errno = ENOSYS, (ssize_t)-1)
extern int H5Dread(int64_t,int64_t,int64_t,int64_t,int64_t,void*);
extern int H5Dwrite(int64_t,int64_t,int64_t,int64_t,int64_t,const void*);
extern void* ZSTD_createCCtx(void);
extern void* ZSTD_createDCtx(void);
extern size_t ZSTD_freeCCtx(void*);
extern size_t ZSTD_freeDCtx(void*);
extern size_t ZSTD_compress2(void*,void*,size_t,const void*,size_t);
extern size_t ZSTD_decompressStream(void*,void*,void*);
extern int deflateInit_(void*,int,const char*,int);
extern int deflateInit2_(void*,int,int,int,int,int,const char*,int);
extern int inflateInit_(void*,const char*,int);
extern int inflateInit2_(void*,int,const char*,int);
extern int deflateEnd(void*);
extern int inflateEnd(void*);
#else
#define V1_SYSCALL(...) syscall(__VA_ARGS__)
#endif

#define Z_STREAM_ERROR (-2)

/*
 * Scoped LD_PRELOAD counters for the V1 harness. Call v1_io_begin() before a
 * timed operation and v1_io_end(out) immediately after it. Begin resolves all
 * forwarding symbols before enabling counters. The 16 output slots are:
 *   0 read calls, 1 read bytes, 2 write calls, 3 write bytes
 *   4 ZSTD_createCCtx, 5 ZSTD_createDCtx, 6 ZSTD_freeCCtx, 7 ZSTD_freeDCtx
 *   8 ZSTD_compress2, 9 ZSTD_decompressStream
 *  10 deflateInit family (_ and _2_), 11 inflateInit family (_ and _2_)
 *  12 deflateEnd, 13 inflateEnd
 *  14 H5Dread, 15 H5Dwrite
 * libc totals include read/pread/readv and write/pwrite/writev respectively.
 * Codec and HDF5 slots count intercepted API calls, not bytes or chunks.
 */

enum { slot_count = 16 };
enum { read_calls = 0, read_bytes = 1, write_calls = 2, write_bytes = 3,
       zstd_create_cctx = 4, zstd_create_dctx = 5,
       zstd_free_cctx = 6, zstd_free_dctx = 7,
       zstd_compress2 = 8, zstd_decompress_stream = 9,
       zlib_deflate_init = 10, zlib_inflate_init = 11,
       zlib_deflate_end = 12, zlib_inflate_end = 13,
       h5d_read = 14, h5d_write = 15 };

typedef ssize_t (*read_fn)(int, void *, size_t);
typedef ssize_t (*pread_fn)(int, void *, size_t, off_t);
typedef ssize_t (*readv_fn)(int, const struct iovec *, int);
typedef ssize_t (*write_fn)(int, const void *, size_t);
typedef ssize_t (*pwrite_fn)(int, const void *, size_t, off_t);
typedef ssize_t (*writev_fn)(int, const struct iovec *, int);
typedef int (*h5d_read_fn)(int64_t, int64_t, int64_t, int64_t, int64_t, void *);
typedef int (*h5d_write_fn)(int64_t, int64_t, int64_t, int64_t, int64_t, const void *);
typedef void *(*zstd_create_fn)(void);
typedef size_t (*zstd_free_fn)(void *);
typedef size_t (*zstd_compress2_fn)(void *, void *, size_t, const void *, size_t);
typedef size_t (*zstd_decompress_stream_fn)(void *, void *, void *);
typedef int (*deflate_init_fn)(void *, int, const char *, int);
typedef int (*deflate_init2_fn)(void *, int, int, int, int, int, const char *, int);
typedef int (*inflate_init_fn)(void *, const char *, int);
typedef int (*inflate_init2_fn)(void *, int, const char *, int);
typedef int (*zlib_end_fn)(void *);

static _Atomic uint64_t counters[slot_count];
static _Atomic uint64_t codec_cpu[2];
static _Atomic int enabled;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;
static __thread int resolving;
static _Atomic int resolved;

static read_fn next_read;
static pread_fn next_pread;
static readv_fn next_readv;
static write_fn next_write;
static pwrite_fn next_pwrite;
static writev_fn next_writev;
static h5d_read_fn next_h5d_read;
static h5d_write_fn next_h5d_write;
static zstd_create_fn next_zstd_create_cctx;
static zstd_create_fn next_zstd_create_dctx;
static zstd_free_fn next_zstd_free_cctx;
static zstd_free_fn next_zstd_free_dctx;
static zstd_compress2_fn next_zstd_compress2;
static zstd_decompress_stream_fn next_zstd_decompress_stream;
static deflate_init_fn next_deflate_init;
static deflate_init2_fn next_deflate_init2;
static inflate_init_fn next_inflate_init;
static inflate_init2_fn next_inflate_init2;
static zlib_end_fn next_deflate_end;
static zlib_end_fn next_inflate_end;

static void resolve_symbols(void) {
    resolving = 1;
#ifdef __APPLE__
    // dyld excludes the replacing image's own references from interposition.
    // RTLD_NEXT can return interposed addresses and recurse on this platform.
    next_read = &read; next_pread = &pread; next_readv = &readv;
    next_write = &write; next_pwrite = &pwrite; next_writev = &writev;
    next_h5d_read = &H5Dread; next_h5d_write = &H5Dwrite;
    next_zstd_create_cctx = &ZSTD_createCCtx; next_zstd_create_dctx = &ZSTD_createDCtx;
    next_zstd_free_cctx = &ZSTD_freeCCtx; next_zstd_free_dctx = &ZSTD_freeDCtx;
    next_zstd_compress2 = &ZSTD_compress2; next_zstd_decompress_stream = &ZSTD_decompressStream;
    next_deflate_init = &deflateInit_; next_deflate_init2 = &deflateInit2_;
    next_inflate_init = &inflateInit_; next_inflate_init2 = &inflateInit2_;
    next_deflate_end = &deflateEnd; next_inflate_end = &inflateEnd;
#else
    next_read = (read_fn)dlsym(RTLD_NEXT, "read");
    next_pread = (pread_fn)dlsym(RTLD_NEXT, "pread");
    next_readv = (readv_fn)dlsym(RTLD_NEXT, "readv");
    next_write = (write_fn)dlsym(RTLD_NEXT, "write");
    next_pwrite = (pwrite_fn)dlsym(RTLD_NEXT, "pwrite");
    next_writev = (writev_fn)dlsym(RTLD_NEXT, "writev");
    next_h5d_read = (h5d_read_fn)dlsym(RTLD_NEXT, "H5Dread");
    next_h5d_write = (h5d_write_fn)dlsym(RTLD_NEXT, "H5Dwrite");
    next_zstd_create_cctx = (zstd_create_fn)dlsym(RTLD_NEXT, "ZSTD_createCCtx");
    next_zstd_create_dctx = (zstd_create_fn)dlsym(RTLD_NEXT, "ZSTD_createDCtx");
    next_zstd_free_cctx = (zstd_free_fn)dlsym(RTLD_NEXT, "ZSTD_freeCCtx");
    next_zstd_free_dctx = (zstd_free_fn)dlsym(RTLD_NEXT, "ZSTD_freeDCtx");
    next_zstd_compress2 = (zstd_compress2_fn)dlsym(RTLD_NEXT, "ZSTD_compress2");
    next_zstd_decompress_stream = (zstd_decompress_stream_fn)dlsym(RTLD_NEXT, "ZSTD_decompressStream");
    next_deflate_init = (deflate_init_fn)dlsym(RTLD_NEXT, "deflateInit_");
    next_deflate_init2 = (deflate_init2_fn)dlsym(RTLD_NEXT, "deflateInit2_");
    next_inflate_init = (inflate_init_fn)dlsym(RTLD_NEXT, "inflateInit_");
    next_inflate_init2 = (inflate_init2_fn)dlsym(RTLD_NEXT, "inflateInit2_");
    next_deflate_end = (zlib_end_fn)dlsym(RTLD_NEXT, "deflateEnd");
    next_inflate_end = (zlib_end_fn)dlsym(RTLD_NEXT, "inflateEnd");
#endif
    resolving = 0;
    atomic_store_explicit(&resolved, 1, memory_order_release);
}

static void ensure_symbols(void) {
    if (!atomic_load_explicit(&resolved, memory_order_acquire) && !resolving)
        pthread_once(&resolve_once, resolve_symbols);
}

static int count_now(void) {
    return atomic_load_explicit(&enabled, memory_order_relaxed) != 0;
}

static void add_slot(unsigned slot) {
    atomic_fetch_add_explicit(&counters[slot], 1, memory_order_relaxed);
}
static uint64_t cpu_stamp(void) {
    struct timespec t;
    return clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t) ? 0 : (uint64_t)t.tv_sec*1000000000ULL+(uint64_t)t.tv_nsec;
}
static void cpu_add(unsigned slot, uint64_t first) {
    const uint64_t last = cpu_stamp();
    if (first && last >= first) atomic_fetch_add_explicit(&codec_cpu[slot],last-first,memory_order_relaxed);
}

static void record_io(int writing, int counted, ssize_t result) {
    if (!counted) return;
    if (writing) {
        add_slot(write_calls);
        if (result > 0) atomic_fetch_add_explicit(&counters[write_bytes], (uint64_t)result, memory_order_relaxed);
    } else {
        add_slot(read_calls);
        if (result > 0) atomic_fetch_add_explicit(&counters[read_bytes], (uint64_t)result, memory_order_relaxed);
    }
}

__attribute__((visibility("default"))) void v1_io_begin(void) {
    ensure_symbols();
    for (unsigned i = 0; i < slot_count; ++i)
        atomic_store_explicit(&counters[i], 0, memory_order_relaxed);
    for (unsigned i = 0; i < 2; ++i) atomic_store_explicit(&codec_cpu[i],0,memory_order_relaxed);
    atomic_store_explicit(&enabled, 1, memory_order_release);
}
__attribute__((visibility("default"))) void v1_codec_cpu_end(uint64_t *out) {
    for (unsigned i = 0; i < 2; ++i) out[i] = atomic_load_explicit(&codec_cpu[i],memory_order_relaxed);
}

__attribute__((visibility("default"))) void v1_io_end(uint64_t *out) {
    atomic_store_explicit(&enabled, 0, memory_order_release);
    if (out)
        for (unsigned i = 0; i < slot_count; ++i)
            out[i] = atomic_load_explicit(&counters[i], memory_order_relaxed);
}

#ifdef __APPLE__
#define read v1_observe_read
#define pread v1_observe_pread
#define readv v1_observe_readv
#define write v1_observe_write
#define pwrite v1_observe_pwrite
#define writev v1_observe_writev
#define H5Dread v1_observe_H5Dread
#define H5Dwrite v1_observe_H5Dwrite
#define ZSTD_createCCtx v1_observe_ZSTD_createCCtx
#define ZSTD_createDCtx v1_observe_ZSTD_createDCtx
#define ZSTD_freeCCtx v1_observe_ZSTD_freeCCtx
#define ZSTD_freeDCtx v1_observe_ZSTD_freeDCtx
#define ZSTD_compress2 v1_observe_ZSTD_compress2
#define ZSTD_decompressStream v1_observe_ZSTD_decompressStream
#define deflateInit_ v1_observe_deflateInit_
#define deflateInit2_ v1_observe_deflateInit2_
#define inflateInit_ v1_observe_inflateInit_
#define inflateInit2_ v1_observe_inflateInit2_
#define deflateEnd v1_observe_deflateEnd
#define inflateEnd v1_observe_inflateEnd
#endif

ssize_t read(int fd, void *buf, size_t n) {
    ensure_symbols();
    const int counted = count_now();
    ssize_t result = next_read ? next_read(fd, buf, n) : V1_SYSCALL(SYS_read, fd, buf, n);
    record_io(0, counted, result);
    return result;
}

ssize_t pread(int fd, void *buf, size_t n, off_t offset) {
    ensure_symbols();
    const int counted = count_now();
    ssize_t result = next_pread ? next_pread(fd, buf, n, offset) : V1_SYSCALL(SYS_pread64, fd, buf, n, offset);
    record_io(0, counted, result);
    return result;
}

ssize_t readv(int fd, const struct iovec *iov, int count) {
    ensure_symbols();
    const int counted = count_now();
    ssize_t result = next_readv ? next_readv(fd, iov, count) : V1_SYSCALL(SYS_readv, fd, iov, count);
    record_io(0, counted, result);
    return result;
}

ssize_t write(int fd, const void *buf, size_t n) {
    ensure_symbols();
    const int counted = count_now();
    ssize_t result = next_write ? next_write(fd, buf, n) : V1_SYSCALL(SYS_write, fd, buf, n);
    record_io(1, counted, result);
    return result;
}

ssize_t pwrite(int fd, const void *buf, size_t n, off_t offset) {
    ensure_symbols();
    const int counted = count_now();
    ssize_t result = next_pwrite ? next_pwrite(fd, buf, n, offset) : V1_SYSCALL(SYS_pwrite64, fd, buf, n, offset);
    record_io(1, counted, result);
    return result;
}

ssize_t writev(int fd, const struct iovec *iov, int count) {
    ensure_symbols();
    const int counted = count_now();
    ssize_t result = next_writev ? next_writev(fd, iov, count) : V1_SYSCALL(SYS_writev, fd, iov, count);
    record_io(1, counted, result);
    return result;
}

int H5Dread(int64_t dset, int64_t type, int64_t mem_space, int64_t file_space,
            int64_t dxpl, void *buffer) {
    ensure_symbols();
    const int counted = count_now();
    if (!next_h5d_read) return -1;
    int result = next_h5d_read(dset, type, mem_space, file_space, dxpl, buffer);
    if (counted) add_slot(h5d_read);
    return result;
}

int H5Dwrite(int64_t dset, int64_t type, int64_t mem_space, int64_t file_space,
             int64_t dxpl, const void *buffer) {
    ensure_symbols();
    const int counted = count_now();
    if (!next_h5d_write) return -1;
    int result = next_h5d_write(dset, type, mem_space, file_space, dxpl, buffer);
    if (counted) add_slot(h5d_write);
    return result;
}

void *ZSTD_createCCtx(void) {
    ensure_symbols();
    const int counted = count_now();
    const uint64_t first = counted ? cpu_stamp() : 0;
    void *result = next_zstd_create_cctx ? next_zstd_create_cctx() : NULL;
    if (counted) cpu_add(0,first);
    if (counted) add_slot(zstd_create_cctx);
    return result;
}

void *ZSTD_createDCtx(void) {
    ensure_symbols();
    const int counted = count_now();
    const uint64_t first = counted ? cpu_stamp() : 0;
    void *result = next_zstd_create_dctx ? next_zstd_create_dctx() : NULL;
    if (counted) cpu_add(0,first);
    if (counted) add_slot(zstd_create_dctx);
    return result;
}

size_t ZSTD_freeCCtx(void *ctx) {
    ensure_symbols();
    const int counted = count_now();
    const uint64_t first = counted ? cpu_stamp() : 0;
    size_t result = next_zstd_free_cctx ? next_zstd_free_cctx(ctx) : SIZE_MAX;
    if (counted) cpu_add(0,first);
    if (counted) add_slot(zstd_free_cctx);
    return result;
}

size_t ZSTD_freeDCtx(void *ctx) {
    ensure_symbols();
    const int counted = count_now();
    const uint64_t first = counted ? cpu_stamp() : 0;
    size_t result = next_zstd_free_dctx ? next_zstd_free_dctx(ctx) : SIZE_MAX;
    if (counted) cpu_add(0,first);
    if (counted) add_slot(zstd_free_dctx);
    return result;
}

size_t ZSTD_compress2(void *ctx, void *dst, size_t dst_capacity,
                      const void *src, size_t src_size) {
    ensure_symbols();
    const int counted = count_now();
    const uint64_t first = counted ? cpu_stamp() : 0;
    size_t result = next_zstd_compress2 ? next_zstd_compress2(ctx, dst, dst_capacity, src, src_size) : SIZE_MAX;
    if (counted) cpu_add(1,first);
    if (counted) add_slot(zstd_compress2);
    return result;
}

size_t ZSTD_decompressStream(void *ctx, void *out, void *in) {
    ensure_symbols();
    const int counted = count_now();
    const uint64_t first = counted ? cpu_stamp() : 0;
    size_t result = next_zstd_decompress_stream ? next_zstd_decompress_stream(ctx, out, in) : SIZE_MAX;
    if (counted) cpu_add(1,first);
    if (counted) add_slot(zstd_decompress_stream);
    return result;
}

int deflateInit_(void *stream, int level, const char *version, int stream_size) {
    ensure_symbols();
    const int counted = count_now();
    int result = next_deflate_init ? next_deflate_init(stream, level, version, stream_size) : Z_STREAM_ERROR;
    if (counted) add_slot(zlib_deflate_init);
    return result;
}

int deflateInit2_(void *stream, int level, int method, int window_bits,
                  int memory_level, int strategy, const char *version, int stream_size) {
    ensure_symbols();
    const int counted = count_now();
    int result = next_deflate_init2 ?
        next_deflate_init2(stream, level, method, window_bits, memory_level, strategy,
                           version, stream_size) : Z_STREAM_ERROR;
    if (counted) add_slot(zlib_deflate_init);
    return result;
}

int inflateInit_(void *stream, const char *version, int stream_size) {
    ensure_symbols();
    const int counted = count_now();
    int result = next_inflate_init ? next_inflate_init(stream, version, stream_size) : Z_STREAM_ERROR;
    if (counted) add_slot(zlib_inflate_init);
    return result;
}

int inflateInit2_(void *stream, int window_bits, const char *version, int stream_size) {
    ensure_symbols();
    const int counted = count_now();
    int result = next_inflate_init2 ?
        next_inflate_init2(stream, window_bits, version, stream_size) : Z_STREAM_ERROR;
    if (counted) add_slot(zlib_inflate_init);
    return result;
}

int deflateEnd(void *stream) {
    ensure_symbols();
    const int counted = count_now();
    int result = next_deflate_end ? next_deflate_end(stream) : Z_STREAM_ERROR;
    if (counted) add_slot(zlib_deflate_end);
    return result;
}

int inflateEnd(void *stream) {
    ensure_symbols();
    const int counted = count_now();
    int result = next_inflate_end ? next_inflate_end(stream) : Z_STREAM_ERROR;
    if (counted) add_slot(zlib_inflate_end);
    return result;
}

#ifdef __APPLE__
#undef read
#undef pread
#undef readv
#undef write
#undef pwrite
#undef writev
#undef H5Dread
#undef H5Dwrite
#undef ZSTD_createCCtx
#undef ZSTD_createDCtx
#undef ZSTD_freeCCtx
#undef ZSTD_freeDCtx
#undef ZSTD_compress2
#undef ZSTD_decompressStream
#undef deflateInit_
#undef deflateInit2_
#undef inflateInit_
#undef inflateInit2_
#undef deflateEnd
#undef inflateEnd
V1_DYLD_INTERPOSE(read, &v1_observe_read, &read);
V1_DYLD_INTERPOSE(pread, &v1_observe_pread, &pread);
V1_DYLD_INTERPOSE(readv, &v1_observe_readv, &readv);
V1_DYLD_INTERPOSE(write, &v1_observe_write, &write);
V1_DYLD_INTERPOSE(pwrite, &v1_observe_pwrite, &pwrite);
V1_DYLD_INTERPOSE(writev, &v1_observe_writev, &writev);
V1_DYLD_INTERPOSE(H5Dread, &v1_observe_H5Dread, &H5Dread);
V1_DYLD_INTERPOSE(H5Dwrite, &v1_observe_H5Dwrite, &H5Dwrite);
V1_DYLD_INTERPOSE(ZSTD_createCCtx, &v1_observe_ZSTD_createCCtx, &ZSTD_createCCtx);
V1_DYLD_INTERPOSE(ZSTD_createDCtx, &v1_observe_ZSTD_createDCtx, &ZSTD_createDCtx);
V1_DYLD_INTERPOSE(ZSTD_freeCCtx, &v1_observe_ZSTD_freeCCtx, &ZSTD_freeCCtx);
V1_DYLD_INTERPOSE(ZSTD_freeDCtx, &v1_observe_ZSTD_freeDCtx, &ZSTD_freeDCtx);
V1_DYLD_INTERPOSE(ZSTD_compress2, &v1_observe_ZSTD_compress2, &ZSTD_compress2);
V1_DYLD_INTERPOSE(ZSTD_decompressStream, &v1_observe_ZSTD_decompressStream, &ZSTD_decompressStream);
V1_DYLD_INTERPOSE(deflateInit_, &v1_observe_deflateInit_, &deflateInit_);
V1_DYLD_INTERPOSE(deflateInit2_, &v1_observe_deflateInit2_, &deflateInit2_);
V1_DYLD_INTERPOSE(inflateInit_, &v1_observe_inflateInit_, &inflateInit_);
V1_DYLD_INTERPOSE(inflateInit2_, &v1_observe_inflateInit2_, &inflateInit2_);
V1_DYLD_INTERPOSE(deflateEnd, &v1_observe_deflateEnd, &deflateEnd);
V1_DYLD_INTERPOSE(inflateEnd, &v1_observe_inflateEnd, &inflateEnd);
#endif
