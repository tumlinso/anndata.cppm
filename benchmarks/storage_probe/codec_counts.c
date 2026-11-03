#define _GNU_SOURCE
#include <blosc.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <zstd.h>

static volatile int active;
static uint64_t blosc_dec_calls, blosc_dec_in, blosc_dec_out, blosc_dec_cpu_ns;
static uint64_t blosc_enc_calls, blosc_enc_in, blosc_enc_out, blosc_enc_cpu_ns;
static uint64_t zstd_dec_calls, zstd_dec_in, zstd_dec_out, zstd_dec_cpu_ns;
static uint64_t zstd_enc_calls, zstd_enc_in, zstd_enc_out, zstd_enc_cpu_ns;
static uint64_t read_calls, read_bytes, pread_calls, pread_bytes;
static uint64_t write_calls, write_bytes, pwrite_calls, pwrite_bytes;
static uint64_t readv_calls, readv_bytes, preadv_calls, preadv_bytes;
static uint64_t writev_calls, writev_bytes, pwritev_calls, pwritev_bytes;

void bulk_probe_begin(void) { __atomic_store_n(&active, 1, __ATOMIC_RELAXED); }
void bulk_probe_end(void) { __atomic_store_n(&active, 0, __ATOMIC_RELAXED); }

static uint64_t thread_cpu_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
static void add(uint64_t *counter, uint64_t value) {
    __atomic_fetch_add(counter, value, __ATOMIC_RELAXED);
}
static int measuring(void) { return __atomic_load_n(&active, __ATOMIC_RELAXED); }

#define WRAP_IO(name, signature, callargs, calls_counter, bytes_counter) \
    ssize_t name signature { \
        static ssize_t (*real_fn) signature; \
        if (!real_fn) real_fn = dlsym(RTLD_NEXT, #name); \
        const ssize_t result = real_fn callargs; \
        if (measuring() && result > 0) { add(&calls_counter, 1); add(&bytes_counter, (uint64_t)result); } \
        return result; \
    }
WRAP_IO(read, (int fd, void *buffer, size_t count), (fd, buffer, count), read_calls, read_bytes)
WRAP_IO(pread, (int fd, void *buffer, size_t count, off_t offset), (fd, buffer, count, offset), pread_calls, pread_bytes)
WRAP_IO(pread64, (int fd, void *buffer, size_t count, off64_t offset), (fd, buffer, count, offset), pread_calls, pread_bytes)
WRAP_IO(write, (int fd, const void *buffer, size_t count), (fd, buffer, count), write_calls, write_bytes)
WRAP_IO(pwrite, (int fd, const void *buffer, size_t count, off_t offset), (fd, buffer, count, offset), pwrite_calls, pwrite_bytes)
WRAP_IO(pwrite64, (int fd, const void *buffer, size_t count, off64_t offset), (fd, buffer, count, offset), pwrite_calls, pwrite_bytes)
WRAP_IO(readv, (int fd, const struct iovec *iov, int count), (fd, iov, count), readv_calls, readv_bytes)
WRAP_IO(preadv, (int fd, const struct iovec *iov, int count, off_t offset), (fd, iov, count, offset), preadv_calls, preadv_bytes)
WRAP_IO(writev, (int fd, const struct iovec *iov, int count), (fd, iov, count), writev_calls, writev_bytes)
WRAP_IO(pwritev, (int fd, const struct iovec *iov, int count, off_t offset), (fd, iov, count, offset), pwritev_calls, pwritev_bytes)
#undef WRAP_IO

int blosc_decompress_ctx(const void *src, void *dest, size_t destsize, int nthreads) {
    static int (*real_fn)(const void *, void *, size_t, int);
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "blosc_decompress_ctx");
    size_t nbytes = 0, cbytes = 0, blocksize = 0;
    blosc_cbuffer_sizes(src, &nbytes, &cbytes, &blocksize);
    const uint64_t start = measuring() ? thread_cpu_ns() : 0;
    const int result = real_fn(src, dest, destsize, nthreads);
    if (measuring()) {
        add(&blosc_dec_calls, 1);
        add(&blosc_dec_cpu_ns, thread_cpu_ns() - start);
        if (result >= 0) { add(&blosc_dec_in, cbytes); add(&blosc_dec_out, (uint64_t)result); }
    }
    return result;
}

int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize, size_t nbytes,
                       const void *src, void *dest, size_t destsize, const char *compressor,
                       size_t blocksize, int nthreads) {
    static int (*real_fn)(int, int, size_t, size_t, const void *, void *, size_t, const char *, size_t, int);
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "blosc_compress_ctx");
    const uint64_t start = measuring() ? thread_cpu_ns() : 0;
    const int result = real_fn(clevel, doshuffle, typesize, nbytes, src, dest, destsize,
                               compressor, blocksize, nthreads);
    if (measuring()) {
        add(&blosc_enc_calls, 1);
        add(&blosc_enc_cpu_ns, thread_cpu_ns() - start);
        if (result >= 0) { add(&blosc_enc_in, nbytes); add(&blosc_enc_out, (uint64_t)result); }
    }
    return result;
}

size_t ZSTD_decompressStream(ZSTD_DCtx *ctx, ZSTD_outBuffer *output, ZSTD_inBuffer *input) {
    static size_t (*real_fn)(ZSTD_DCtx *, ZSTD_outBuffer *, ZSTD_inBuffer *);
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ZSTD_decompressStream");
    const size_t input_before = input->pos;
    const size_t output_before = output->pos;
    const uint64_t start = measuring() ? thread_cpu_ns() : 0;
    const size_t result = real_fn(ctx, output, input);
    if (measuring()) {
        add(&zstd_dec_calls, 1);
        add(&zstd_dec_cpu_ns, thread_cpu_ns() - start);
        add(&zstd_dec_in, input->pos - input_before);
        add(&zstd_dec_out, output->pos - output_before);
    }
    return result;
}

size_t ZSTD_compress2(ZSTD_CCtx *ctx, void *dest, size_t capacity, const void *src, size_t size) {
    static size_t (*real_fn)(ZSTD_CCtx *, void *, size_t, const void *, size_t);
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ZSTD_compress2");
    const uint64_t start = measuring() ? thread_cpu_ns() : 0;
    const size_t result = real_fn(ctx, dest, capacity, src, size);
    if (measuring()) {
        add(&zstd_enc_calls, 1);
        add(&zstd_enc_cpu_ns, thread_cpu_ns() - start);
        if (!ZSTD_isError(result)) { add(&zstd_enc_in, size); add(&zstd_enc_out, result); }
    }
    return result;
}

__attribute__((destructor)) static void report_counts(void) {
    fprintf(stderr,
        "PROBE_COUNTS read=%llu/%llu pread=%llu/%llu readv=%llu/%llu preadv=%llu/%llu write=%llu/%llu pwrite=%llu/%llu writev=%llu/%llu pwritev=%llu/%llu bytes/calls\n"
        "CODEC_COUNTS blosc_decode=%llu/%llu/%llu bytes_in/out_calls cpu_ns=%llu blosc_encode=%llu/%llu/%llu bytes_in/out_calls cpu_ns=%llu zstd_decode=%llu/%llu/%llu bytes_in/out_calls cpu_ns=%llu zstd_encode=%llu/%llu/%llu bytes_in/out_calls cpu_ns=%llu\n",
        (unsigned long long)read_bytes, (unsigned long long)read_calls,
        (unsigned long long)pread_bytes, (unsigned long long)pread_calls,
        (unsigned long long)readv_bytes, (unsigned long long)readv_calls,
        (unsigned long long)preadv_bytes, (unsigned long long)preadv_calls,
        (unsigned long long)write_bytes, (unsigned long long)write_calls,
        (unsigned long long)pwrite_bytes, (unsigned long long)pwrite_calls,
        (unsigned long long)writev_bytes, (unsigned long long)writev_calls,
        (unsigned long long)pwritev_bytes, (unsigned long long)pwritev_calls,
        (unsigned long long)blosc_dec_in, (unsigned long long)blosc_dec_out,
        (unsigned long long)blosc_dec_calls, (unsigned long long)blosc_dec_cpu_ns,
        (unsigned long long)blosc_enc_in, (unsigned long long)blosc_enc_out,
        (unsigned long long)blosc_enc_calls, (unsigned long long)blosc_enc_cpu_ns,
        (unsigned long long)zstd_dec_in, (unsigned long long)zstd_dec_out,
        (unsigned long long)zstd_dec_calls, (unsigned long long)zstd_dec_cpu_ns,
        (unsigned long long)zstd_enc_in, (unsigned long long)zstd_enc_out,
        (unsigned long long)zstd_enc_calls, (unsigned long long)zstd_enc_cpu_ns);
}
