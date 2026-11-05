#define _GNU_SOURCE
#include <blosc.h>
#include <dlfcn.h>
#include <errno.h>
#include <hdf5.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>
#include <zstd.h>

/*
 * glibc/Linux LD_PRELOAD probe. The driver calls begin(read_cap, write_cap)
 * immediately before one operation and end() immediately after it. Each
 * direction is capped independently at no more than 512 MiB. Ordinary C++
 * operator new is counted once through malloc in this toolchain; no separate
 * new wrapper is installed. HDF5 byte counts are logical selected-buffer bytes.
 * libc interposition misses direct syscalls, mmap I/O, and hidden libc/library
 * calls; codec counts cover only the APIs wrapped below.
 *
 * Build with the project HDF5 include directory and -lblosc -lzstd -lz -ldl
 * -pthread. HDF5 function calls are resolved dynamically, so HDF5 need not be
 * linked into this interposer.
 */

#define IO_MAX_BYTES (512ULL * 1024ULL * 1024ULL)
#define H5_PATHS 32
#define H5_PATH_BYTES 256
#define LOAD(p) __atomic_load_n(&(p), __ATOMIC_RELAXED)
#define STORE(p, v) __atomic_store_n(&(p), (v), __ATOMIC_RELAXED)

typedef struct { uint64_t calls, bytes; } IoCount;
typedef struct { uint64_t calls, in_bytes, out_bytes; } CodecCount;
typedef struct { uint64_t calls, successes, requested_bytes, successful_bytes; } AllocCount;

static volatile int active;
static uint64_t read_cap, write_cap, read_budget, write_budget;
static uint64_t read_rejections, write_rejections;
static IoCount io_read, io_pread, io_readv, io_preadv;
static IoCount io_write, io_pwrite, io_writev, io_pwritev;
static CodecCount blosc_decode, blosc_encode, zstd_decode, zstd_encode;
static CodecCount zlib_decode, zlib_encode;
static AllocCount alloc_malloc, alloc_calloc, alloc_realloc;
static AllocCount alloc_posix_memalign, alloc_aligned_alloc, alloc_memalign;
static AllocCount alloc_valloc, alloc_pvalloc;
static uint64_t calloc_overflows;
static __thread unsigned io_depth, codec_depth, h5_depth, internal_depth;

static void add(uint64_t *counter, uint64_t value) {
    __atomic_fetch_add(counter, value, __ATOMIC_RELAXED);
}
static int measuring(void) { return LOAD(active) == 1; }
static int power_of_two(size_t n) { return n && !(n & (n - 1)); }

static void clear_io(IoCount *c) { STORE(c->calls, 0); STORE(c->bytes, 0); }
static void clear_codec(CodecCount *c) {
    STORE(c->calls, 0); STORE(c->in_bytes, 0); STORE(c->out_bytes, 0);
}
static void clear_alloc(AllocCount *c) {
    STORE(c->calls, 0); STORE(c->successes, 0);
    STORE(c->requested_bytes, 0); STORE(c->successful_bytes, 0);
}

static int reserve_io(int writing, size_t request) {
    if (!measuring()) return 1;
    uint64_t *used = writing ? &write_budget : &read_budget;
    const uint64_t cap = writing ? LOAD(write_cap) : LOAD(read_cap);
    uint64_t old = LOAD(*used);
    for (;;) {
        if (old > cap || (uint64_t)request > cap - old) {
            add(writing ? &write_rejections : &read_rejections, 1);
            errno = EFBIG;
            return 0;
        }
        if (__atomic_compare_exchange_n(used, &old, old + request, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return 1;
    }
}
static void settle_io(int writing, size_t request, ssize_t result) {
    uint64_t *used = writing ? &write_budget : &read_budget;
    const size_t transferred = result > 0 ? (size_t)result : 0;
    if (transferred < request)
        __atomic_fetch_sub(used, (uint64_t)(request - transferred), __ATOMIC_RELAXED);
}
static size_t iov_bytes(const struct iovec *iov, int count) {
    size_t total = 0;
    for (int i = 0; i < count; ++i) {
        if (iov[i].iov_len > SIZE_MAX - total) return SIZE_MAX;
        total += iov[i].iov_len;
    }
    return total;
}

#define DEFINE_IO(name, params, args, requested, writing, ctr)                   \
    typedef ssize_t (*name##_fn) params;                                        \
    static name##_fn real_##name;                                               \
    static name##_fn next_##name(void) {                                        \
        name##_fn fn = LOAD(real_##name);                                       \
        if (!fn) {                                                              \
            ++internal_depth;                                                   \
            fn = (name##_fn)dlsym(RTLD_NEXT, #name);                            \
            --internal_depth;                                                   \
            if (fn) STORE(real_##name, fn);                                     \
        }                                                                       \
        return fn;                                                              \
    }                                                                           \
    ssize_t name params {                                                       \
        name##_fn fn = next_##name();                                           \
        if (!fn) { errno = ENOSYS; return -1; }                                 \
        const int sample = measuring() && !internal_depth && !io_depth;        \
        if (!sample) { ++io_depth; ssize_t r = fn args; --io_depth; return r; } \
        const size_t request = (requested);                                     \
        add(&(ctr).calls, 1);                                                   \
        if (!reserve_io((writing), request)) return -1;                         \
        ++io_depth; ssize_t r = fn args; --io_depth;                            \
        settle_io((writing), request, r);                                       \
        if (r > 0) add(&(ctr).bytes, (uint64_t)r);                              \
        return r;                                                               \
    }

DEFINE_IO(read, (int fd, void *buffer, size_t count), (fd, buffer, count), count, 0, io_read)
DEFINE_IO(pread, (int fd, void *buffer, size_t count, off_t offset),
         (fd, buffer, count, offset), count, 0, io_pread)
DEFINE_IO(pread64, (int fd, void *buffer, size_t count, off64_t offset),
         (fd, buffer, count, offset), count, 0, io_pread)
DEFINE_IO(readv, (int fd, const struct iovec *iov, int count),
         (fd, iov, count), iov_bytes(iov, count), 0, io_readv)
DEFINE_IO(preadv, (int fd, const struct iovec *iov, int count, off_t offset),
         (fd, iov, count, offset), iov_bytes(iov, count), 0, io_preadv)
DEFINE_IO(preadv64, (int fd, const struct iovec *iov, int count, off64_t offset),
         (fd, iov, count, offset), iov_bytes(iov, count), 0, io_preadv)
DEFINE_IO(write, (int fd, const void *buffer, size_t count), (fd, buffer, count), count, 1, io_write)
DEFINE_IO(pwrite, (int fd, const void *buffer, size_t count, off_t offset),
         (fd, buffer, count, offset), count, 1, io_pwrite)
DEFINE_IO(pwrite64, (int fd, const void *buffer, size_t count, off64_t offset),
         (fd, buffer, count, offset), count, 1, io_pwrite)
DEFINE_IO(writev, (int fd, const struct iovec *iov, int count),
         (fd, iov, count), iov_bytes(iov, count), 1, io_writev)
DEFINE_IO(pwritev, (int fd, const struct iovec *iov, int count, off_t offset),
         (fd, iov, count, offset), iov_bytes(iov, count), 1, io_pwritev)
DEFINE_IO(pwritev64, (int fd, const struct iovec *iov, int count, off64_t offset),
         (fd, iov, count, offset), iov_bytes(iov, count), 1, io_pwritev)
#undef DEFINE_IO

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);

static void record_alloc(AllocCount *c, size_t bytes, int ok) {
    if (!measuring() || internal_depth) return;
    add(&c->calls, 1); add(&c->requested_bytes, (uint64_t)bytes);
    if (ok) { add(&c->successes, 1); add(&c->successful_bytes, (uint64_t)bytes); }
}
void *malloc(size_t size) {
    void *p = __libc_malloc(size); record_alloc(&alloc_malloc, size, p != NULL); return p;
}
void *calloc(size_t count, size_t size) {
    size_t bytes = 0;
    const int overflow = __builtin_mul_overflow(count, size, &bytes);
    void *p = __libc_calloc(count, size);
    if (overflow) { if (measuring() && !internal_depth) add(&calloc_overflows, 1); bytes = SIZE_MAX; }
    record_alloc(&alloc_calloc, bytes, p != NULL); return p;
}
void *realloc(void *ptr, size_t size) {
    void *p = __libc_realloc(ptr, size); record_alloc(&alloc_realloc, size, p != NULL); return p;
}
int posix_memalign(void **ptr, size_t alignment, size_t size) {
    if (!ptr || alignment % sizeof(void *) || !power_of_two(alignment)) return EINVAL;
    void *p = __libc_memalign(alignment, size);
    if (!p) { record_alloc(&alloc_posix_memalign, size, 0); return ENOMEM; }
    *ptr = p; record_alloc(&alloc_posix_memalign, size, 1); return 0;
}
void *aligned_alloc(size_t alignment, size_t size) {
    if (!power_of_two(alignment) || size % alignment) {
        errno = EINVAL; record_alloc(&alloc_aligned_alloc, size, 0); return NULL;
    }
    void *p = __libc_memalign(alignment, size);
    record_alloc(&alloc_aligned_alloc, size, p != NULL); return p;
}
void *memalign(size_t alignment, size_t size) {
    void *p = __libc_memalign(alignment, size);
    record_alloc(&alloc_memalign, size, p != NULL); return p;
}
void *valloc(size_t size) {
    void *p = __libc_memalign((size_t)getpagesize(), size);
    record_alloc(&alloc_valloc, size, p != NULL); return p;
}
void *pvalloc(size_t size) {
    const size_t page = (size_t)getpagesize();
    if (size > SIZE_MAX - (page - 1)) {
        errno = ENOMEM; record_alloc(&alloc_pvalloc, size, 0); return NULL;
    }
    const size_t rounded = (size + page - 1) & ~(page - 1);
    void *p = __libc_memalign(page, rounded);
    record_alloc(&alloc_pvalloc, size, p != NULL); return p;
}

static void record_codec(CodecCount *c, uint64_t in, uint64_t out) {
    if (!measuring() || internal_depth || codec_depth) return;
    add(&c->calls, 1); add(&c->in_bytes, in); add(&c->out_bytes, out);
}

/* Blosc/Zstd nesting is suppressed: one outer codec operation counts once. */
int blosc_decompress(const void *src, void *dest, size_t destsize) {
    typedef int (*fn_t)(const void *, void *, size_t); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "blosc_decompress"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return -1;
    size_t out = 0, in = 0, block = 0; blosc_cbuffer_sizes(src, &out, &in, &block);
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(src, dest, destsize); --codec_depth;
    if (sample && result >= 0) record_codec(&blosc_decode, in, (uint64_t)result);
    return result;
}
int blosc_decompress_ctx(const void *src, void *dest, size_t destsize, int nthreads) {
    typedef int (*fn_t)(const void *, void *, size_t, int); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "blosc_decompress_ctx"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return -1;
    size_t out = 0, in = 0, block = 0; blosc_cbuffer_sizes(src, &out, &in, &block);
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(src, dest, destsize, nthreads); --codec_depth;
    if (sample && result >= 0) record_codec(&blosc_decode, in, (uint64_t)result);
    return result;
}
int blosc_compress(int level, int shuffle, size_t typesize, size_t nbytes,
                   const void *src, void *dest, size_t destsize) {
    typedef int (*fn_t)(int, int, size_t, size_t, const void *, void *, size_t); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "blosc_compress"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return -1;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(level, shuffle, typesize, nbytes, src, dest, destsize); --codec_depth;
    if (sample && result >= 0) record_codec(&blosc_encode, nbytes, (uint64_t)result);
    return result;
}
int blosc_compress_ctx(int level, int shuffle, size_t typesize, size_t nbytes,
                       const void *src, void *dest, size_t destsize,
                       const char *compressor, size_t blocksize, int nthreads) {
    typedef int (*fn_t)(int, int, size_t, size_t, const void *, void *, size_t,
                        const char *, size_t, int); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "blosc_compress_ctx"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return -1;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(level, shuffle, typesize, nbytes, src, dest, destsize,
                                   compressor, blocksize, nthreads); --codec_depth;
    if (sample && result >= 0) record_codec(&blosc_encode, nbytes, (uint64_t)result);
    return result;
}

/* Full-buffer Zstd wrappers use a custom body to retain byte sizes. */
size_t ZSTD_decompress(void *dst, size_t dstcap, const void *src, size_t srcsize) {
    typedef size_t (*fn_t)(void *, size_t, const void *, size_t); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "ZSTD_decompress"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(dst, dstcap, src, srcsize); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_decode, srcsize, result);
    return result;
}
size_t ZSTD_decompressDCtx(ZSTD_DCtx *ctx, void *dst, size_t dstcap,
                           const void *src, size_t srcsize) {
    typedef size_t (*fn_t)(ZSTD_DCtx *, void *, size_t, const void *, size_t); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "ZSTD_decompressDCtx"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(ctx, dst, dstcap, src, srcsize); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_decode, srcsize, result);
    return result;
}
size_t ZSTD_decompressStream(ZSTD_DCtx *ctx, ZSTD_outBuffer *out, ZSTD_inBuffer *in) {
    typedef size_t (*fn_t)(ZSTD_DCtx *, ZSTD_outBuffer *, ZSTD_inBuffer *); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "ZSTD_decompressStream"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const size_t in0 = in->pos, out0 = out->pos;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(ctx, out, in); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_decode, in->pos - in0, out->pos - out0);
    return result;
}
size_t ZSTD_compress(void *dst, size_t capacity, const void *src, size_t size, int level) {
    typedef size_t (*fn_t)(void *, size_t, const void *, size_t, int); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "ZSTD_compress"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(dst, capacity, src, size, level); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_encode, size, result);
    return result;
}
size_t ZSTD_compressCCtx(ZSTD_CCtx *ctx, void *dst, size_t capacity,
                         const void *src, size_t size, int level) {
    typedef size_t (*fn_t)(ZSTD_CCtx *, void *, size_t, const void *, size_t, int); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "ZSTD_compressCCtx"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(ctx, dst, capacity, src, size, level); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_encode, size, result);
    return result;
}
size_t ZSTD_compress2(ZSTD_CCtx *ctx, void *dst, size_t capacity,
                      const void *src, size_t size) {
    typedef size_t (*fn_t)(ZSTD_CCtx *, void *, size_t, const void *, size_t); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "ZSTD_compress2"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(ctx, dst, capacity, src, size); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_encode, size, result);
    return result;
}
size_t ZSTD_compressStream2(ZSTD_CCtx *ctx, ZSTD_outBuffer *out,
                            ZSTD_inBuffer *in, ZSTD_EndDirective endop) {
    typedef size_t (*fn_t)(ZSTD_CCtx *, ZSTD_outBuffer *, ZSTD_inBuffer *, ZSTD_EndDirective); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "ZSTD_compressStream2"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const size_t in0 = in->pos, out0 = out->pos;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(ctx, out, in, endop); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_encode, in->pos - in0, out->pos - out0);
    return result;
}

int inflate(z_streamp stream, int flush) {
    typedef int (*fn_t)(z_streamp, int); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "inflate"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return Z_STREAM_ERROR;
    const uInt in0 = stream->avail_in, out0 = stream->avail_out;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(stream, flush); --codec_depth;
    if (sample) record_codec(&zlib_decode, in0 - stream->avail_in, out0 - stream->avail_out);
    return result;
}
int deflate(z_streamp stream, int flush) {
    typedef int (*fn_t)(z_streamp, int); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "deflate"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return Z_STREAM_ERROR;
    const uInt in0 = stream->avail_in, out0 = stream->avail_out;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(stream, flush); --codec_depth;
    if (sample) record_codec(&zlib_encode, in0 - stream->avail_in, out0 - stream->avail_out);
    return result;
}

/* HDF5 counts are selected memory-buffer bytes, separate from libc transfer bytes. */
typedef struct {
    char path[H5_PATH_BYTES];
    uint64_t reads, read_bytes, writes, write_bytes;
} H5DatasetCount;
static H5DatasetCount h5_datasets[H5_PATHS];
static size_t h5_dataset_count;
static uint64_t h5_reads, h5_read_bytes, h5_writes, h5_write_bytes;
static uint64_t h5_projection_errors, h5_path_overflow;
static pthread_mutex_t h5_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t h5_once = PTHREAD_ONCE_INIT;
typedef herr_t (*h5_read_fn)(hid_t, hid_t, hid_t, hid_t, hid_t, void *);
typedef herr_t (*h5_write_fn)(hid_t, hid_t, hid_t, hid_t, hid_t, const void *);
typedef size_t (*h5_size_fn)(hid_t);
typedef hid_t (*h5_space_fn)(hid_t);
typedef hssize_t (*h5_points_fn)(hid_t);
typedef herr_t (*h5_close_fn)(hid_t);
typedef ssize_t (*h5_name_fn)(hid_t, char *, size_t);
static h5_read_fn next_h5read;
static h5_write_fn next_h5write;
static h5_size_fn next_h5size;
static h5_space_fn next_h5getspace;
static h5_points_fn next_h5points;
static h5_close_fn next_h5close;
static h5_name_fn next_h5name;

static void resolve_hdf5(void) {
    ++internal_depth;
    next_h5read = (h5_read_fn)dlsym(RTLD_NEXT, "H5Dread");
    next_h5write = (h5_write_fn)dlsym(RTLD_NEXT, "H5Dwrite");
    next_h5size = (h5_size_fn)dlsym(RTLD_NEXT, "H5Tget_size");
    next_h5getspace = (h5_space_fn)dlsym(RTLD_NEXT, "H5Dget_space");
    next_h5points = (h5_points_fn)dlsym(RTLD_NEXT, "H5Sget_select_npoints");
    next_h5close = (h5_close_fn)dlsym(RTLD_NEXT, "H5Sclose");
    next_h5name = (h5_name_fn)dlsym(RTLD_NEXT, "H5Iget_name");
    --internal_depth;
}
static uint64_t h5_selected_bytes(hid_t dset, hid_t type, hid_t memspace, hid_t filespace) {
    const size_t width = next_h5size ? next_h5size(type) : 0;
    hid_t owned = -1, space = memspace;
    if (space == H5S_ALL) space = filespace;
    if (space == H5S_ALL && next_h5getspace) { owned = next_h5getspace(dset); space = owned; }
    const hssize_t points = space >= 0 && next_h5points ? next_h5points(space) : -1;
    if (owned >= 0 && next_h5close) next_h5close(owned);
    if (!width || points < 0) { add(&h5_projection_errors, 1); return 0; }
    return (uint64_t)points > UINT64_MAX / width ? UINT64_MAX : (uint64_t)points * width;
}
static void h5_path(hid_t dset, char out[H5_PATH_BYTES]) {
    if (!next_h5name || next_h5name(dset, out, H5_PATH_BYTES) <= 0)
        snprintf(out, H5_PATH_BYTES, "<unnamed>");
    out[H5_PATH_BYTES - 1] = '\0';
}
static void h5_record(const char *path, uint64_t bytes, int writing) {
    pthread_mutex_lock(&h5_lock);
    if (writing) { ++h5_writes; h5_write_bytes += bytes; }
    else { ++h5_reads; h5_read_bytes += bytes; }
    size_t i = 0;
    while (i < h5_dataset_count && strcmp(h5_datasets[i].path, path)) ++i;
    if (i == h5_dataset_count && h5_dataset_count < H5_PATHS) {
        snprintf(h5_datasets[i].path, H5_PATH_BYTES, "%s", path);
        ++h5_dataset_count;
    }
    if (i < h5_dataset_count) {
        if (writing) { ++h5_datasets[i].writes; h5_datasets[i].write_bytes += bytes; }
        else { ++h5_datasets[i].reads; h5_datasets[i].read_bytes += bytes; }
    } else ++h5_path_overflow;
    pthread_mutex_unlock(&h5_lock);
}
herr_t H5Dread(hid_t dset, hid_t type, hid_t memspace, hid_t filespace,
               hid_t dxpl, void *buffer) {
    pthread_once(&h5_once, resolve_hdf5);
    if (!next_h5read) return -1;
    const int sample = measuring() && !internal_depth && !h5_depth;
    if (!sample) { ++h5_depth; herr_t r = next_h5read(dset, type, memspace, filespace, dxpl, buffer); --h5_depth; return r; }
    char path[H5_PATH_BYTES];
    ++internal_depth;
    const uint64_t bytes = h5_selected_bytes(dset, type, memspace, filespace);
    h5_path(dset, path);
    --internal_depth;
    ++h5_depth; herr_t r = next_h5read(dset, type, memspace, filespace, dxpl, buffer); --h5_depth;
    h5_record(path, bytes, 0); return r;
}
herr_t H5Dwrite(hid_t dset, hid_t type, hid_t memspace, hid_t filespace,
                hid_t dxpl, const void *buffer) {
    pthread_once(&h5_once, resolve_hdf5);
    if (!next_h5write) return -1;
    const int sample = measuring() && !internal_depth && !h5_depth;
    if (!sample) { ++h5_depth; herr_t r = next_h5write(dset, type, memspace, filespace, dxpl, buffer); --h5_depth; return r; }
    char path[H5_PATH_BYTES];
    ++internal_depth;
    const uint64_t bytes = h5_selected_bytes(dset, type, memspace, filespace);
    h5_path(dset, path);
    --internal_depth;
    ++h5_depth; herr_t r = next_h5write(dset, type, memspace, filespace, dxpl, buffer); --h5_depth;
    h5_record(path, bytes, 1); return r;
}

int anndatapp_probe_begin(uint64_t read_cap_bytes, uint64_t write_cap_bytes) {
    if (read_cap_bytes > IO_MAX_BYTES || write_cap_bytes > IO_MAX_BYTES) { errno = EFBIG; return -1; }
    int expected = 0;
    if (!__atomic_compare_exchange_n(&active, &expected, 2, 0,
                                     __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { errno = EBUSY; return -1; }
    STORE(read_cap, read_cap_bytes); STORE(write_cap, write_cap_bytes);
    STORE(read_budget, 0); STORE(write_budget, 0);
    STORE(read_rejections, 0); STORE(write_rejections, 0);
    clear_io(&io_read); clear_io(&io_pread); clear_io(&io_readv); clear_io(&io_preadv);
    clear_io(&io_write); clear_io(&io_pwrite); clear_io(&io_writev); clear_io(&io_pwritev);
    clear_codec(&blosc_decode); clear_codec(&blosc_encode);
    clear_codec(&zstd_decode); clear_codec(&zstd_encode);
    clear_codec(&zlib_decode); clear_codec(&zlib_encode);
    clear_alloc(&alloc_malloc); clear_alloc(&alloc_calloc); clear_alloc(&alloc_realloc);
    clear_alloc(&alloc_posix_memalign); clear_alloc(&alloc_aligned_alloc);
    clear_alloc(&alloc_memalign); clear_alloc(&alloc_valloc); clear_alloc(&alloc_pvalloc);
    STORE(calloc_overflows, 0);
    pthread_mutex_lock(&h5_lock);
    memset(h5_datasets, 0, sizeof h5_datasets); h5_dataset_count = 0;
    h5_reads = h5_read_bytes = h5_writes = h5_write_bytes = 0;
    h5_projection_errors = h5_path_overflow = 0;
    pthread_mutex_unlock(&h5_lock);
    STORE(active, 1);
    return 0;
}

static void json_path(const char *path) {
    fputc('"', stderr);
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        if (*p == '"' || *p == '\\') { fputc('\\', stderr); fputc(*p, stderr); }
        else if (*p < 0x20) fprintf(stderr, "\\u%04x", (unsigned)*p);
        else fputc(*p, stderr);
    }
    fputc('"', stderr);
}
static void json_io(const char *name, const IoCount *c, int *first) {
    fprintf(stderr, "%s\"%s\":{\"calls\":%llu,\"bytes\":%llu}", *first ? "" : ",", name,
            (unsigned long long)LOAD(c->calls), (unsigned long long)LOAD(c->bytes));
    *first = 0;
}
static void json_codec(const char *name, const CodecCount *dec, const CodecCount *enc) {
    fprintf(stderr,
        "\"%s\":{\"decode_calls\":%llu,\"decode_bytes_in\":%llu,\"decode_bytes_out\":%llu,"
        "\"encode_calls\":%llu,\"encode_bytes_in\":%llu,\"encode_bytes_out\":%llu}", name,
        (unsigned long long)LOAD(dec->calls), (unsigned long long)LOAD(dec->in_bytes),
        (unsigned long long)LOAD(dec->out_bytes), (unsigned long long)LOAD(enc->calls),
        (unsigned long long)LOAD(enc->in_bytes), (unsigned long long)LOAD(enc->out_bytes));
}
static void json_alloc(const char *name, const AllocCount *c, int *first) {
    fprintf(stderr, "%s\"%s\":{\"calls\":%llu,\"successes\":%llu,\"requested_bytes\":%llu,\"successful_bytes\":%llu}",
            *first ? "" : ",", name, (unsigned long long)LOAD(c->calls),
            (unsigned long long)LOAD(c->successes), (unsigned long long)LOAD(c->requested_bytes),
            (unsigned long long)LOAD(c->successful_bytes));
    *first = 0;
}
static void report_probe(void) {
    fprintf(stderr,
        "ANNDATAPP_PROBE {\"schema\":\"anndatapp.streaming_probe.counters.v1\","
        "\"read_cap_bytes\":%llu,\"write_cap_bytes\":%llu,"
        "\"read_budget_used_bytes\":%llu,\"write_budget_used_bytes\":%llu,"
        "\"read_rejections\":%llu,\"write_rejections\":%llu,\"io\":{",
        (unsigned long long)LOAD(read_cap), (unsigned long long)LOAD(write_cap),
        (unsigned long long)LOAD(read_budget), (unsigned long long)LOAD(write_budget),
        (unsigned long long)LOAD(read_rejections), (unsigned long long)LOAD(write_rejections));
    int first = 1;
    json_io("read", &io_read, &first); json_io("pread", &io_pread, &first);
    json_io("readv", &io_readv, &first); json_io("preadv", &io_preadv, &first);
    json_io("write", &io_write, &first); json_io("pwrite", &io_pwrite, &first);
    json_io("writev", &io_writev, &first); json_io("pwritev", &io_pwritev, &first);
    fprintf(stderr, "},\"codec\":{");
    json_codec("blosc", &blosc_decode, &blosc_encode); fputc(',', stderr);
    json_codec("zstd", &zstd_decode, &zstd_encode); fputc(',', stderr);
    json_codec("zlib", &zlib_decode, &zlib_encode);
    fprintf(stderr, "},\"alloc\":{"); first = 1;
    json_alloc("malloc", &alloc_malloc, &first); json_alloc("calloc", &alloc_calloc, &first);
    json_alloc("realloc", &alloc_realloc, &first); json_alloc("posix_memalign", &alloc_posix_memalign, &first);
    json_alloc("aligned_alloc", &alloc_aligned_alloc, &first); json_alloc("memalign", &alloc_memalign, &first);
    json_alloc("valloc", &alloc_valloc, &first); json_alloc("pvalloc", &alloc_pvalloc, &first);
    fprintf(stderr, "},\"calloc_overflows\":%llu,\"new_counting\":\"ordinary operator new is included through malloc; no separate new wrapper\",\"hdf5\":{",
            (unsigned long long)LOAD(calloc_overflows));
    pthread_mutex_lock(&h5_lock);
    fprintf(stderr, "\"read_calls\":%llu,\"read_native_logical_bytes\":%llu,\"write_calls\":%llu,\"write_native_logical_bytes\":%llu,\"projection_errors\":%llu,\"dataset_path_overflow\":%llu,\"datasets\":[",
            (unsigned long long)h5_reads, (unsigned long long)h5_read_bytes,
            (unsigned long long)h5_writes, (unsigned long long)h5_write_bytes,
            (unsigned long long)h5_projection_errors, (unsigned long long)h5_path_overflow);
    for (size_t i = 0; i < h5_dataset_count; ++i) {
        if (i) fputc(',', stderr);
        fprintf(stderr, "{\"path\":"); json_path(h5_datasets[i].path);
        fprintf(stderr, ",\"read_calls\":%llu,\"read_native_logical_bytes\":%llu,\"write_calls\":%llu,\"write_native_logical_bytes\":%llu}",
                (unsigned long long)h5_datasets[i].reads, (unsigned long long)h5_datasets[i].read_bytes,
                (unsigned long long)h5_datasets[i].writes, (unsigned long long)h5_datasets[i].write_bytes);
    }
    pthread_mutex_unlock(&h5_lock);
    fputs("]}}\n", stderr);
}
void anndatapp_probe_end(void) {
    if (__atomic_exchange_n(&active, 0, __ATOMIC_RELAXED) == 1) report_probe();
}
__attribute__((destructor)) static void report_if_armed(void) {
    if (LOAD(active) == 1) anndatapp_probe_end();
}
