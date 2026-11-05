#define _GNU_SOURCE
#include <blosc.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>
#include <zstd.h>

/*
 * Linux LD_PRELOAD instrumentation for one armed X write/read operation.
 * The caller supplies the existing absolute X-array directory and the
 * configured encoded shard-index size/location. Read/write caps apply to
 * requested bytes across intercepted libc read/write calls; short transfers
 * refund unused budget. Direct syscalls, mmap, and hidden library I/O bypass
 * this accounting. Codec counters are wrapped C API stream invocations, not
 * chunk decodes. Index counters classify intercepted positional byte ranges;
 * end-index writes are reported as EOF/size-matched candidates.
 */

#define IO_MAX_BYTES (512ULL * 1024ULL * 1024ULL)
#define MAX_DELAY_US 1000000ULL
#define MAX_SHARDS 256
#define LOAD(p) __atomic_load_n(&(p), __ATOMIC_RELAXED)
#define STORE(p, v) __atomic_store_n(&(p), (v), __ATOMIC_RELAXED)

typedef struct { uint64_t calls, bytes; } IoCount;
typedef struct { uint64_t calls, in_bytes, out_bytes; } CodecCount;
typedef struct { uint64_t calls, successes, requested_bytes, successful_bytes; } AllocCount;
typedef struct {
    char path[PATH_MAX];
    uint64_t read_calls, read_bytes, write_calls, write_bytes;
    uint64_t pread_calls, pread_bytes, pwrite_calls, pwrite_bytes;
    uint64_t index_read_calls, index_read_bytes;
    uint64_t index_write_calls, index_write_bytes;
    uint64_t index_end_write_candidates, index_end_write_candidate_bytes;
    uint64_t rename_calls, rename_successes, replacements, new_shards, rename_failures;
} ShardCount;
typedef struct {
    char key[PATH_MAX];
    uint64_t file_size;
    int shard, file_size_valid;
} IoTarget;

static volatile int active;
static char x_root[PATH_MAX];
static size_t x_root_len, index_bytes;
static int index_at_start;
static uint64_t read_cap, write_cap, read_budget, write_budget;
static uint64_t read_rejections, write_rejections;
static uint64_t x_pread_active, max_concurrent_x_pread;
static uint64_t delay_fast_us, delay_slow_us, delay_slow_from_row;
static int delay_slow_row_set;
static IoCount io_read, io_pread, io_readv, io_preadv;
static IoCount io_write, io_pwrite, io_writev, io_pwritev;
static CodecCount blosc_decode, blosc_encode, zstd_decode, zstd_encode;
static CodecCount zlib_decode, zlib_encode;
static AllocCount alloc_malloc, alloc_calloc, alloc_realloc;
static AllocCount alloc_posix_memalign, alloc_aligned_alloc, alloc_memalign;
static AllocCount alloc_valloc, alloc_pvalloc;
static uint64_t calloc_overflows;
static ShardCount shards[MAX_SHARDS];
static size_t shard_count;
static uint64_t shard_path_overflow;
static pthread_mutex_t shard_lock = PTHREAD_MUTEX_INITIALIZER;
static __thread unsigned io_depth, codec_depth, rename_depth, internal_depth;

static void add(uint64_t *p, uint64_t n) { __atomic_fetch_add(p, n, __ATOMIC_RELAXED); }
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

static int parse_env_u64(const char *name, uint64_t *value, uint64_t max, int *set) {
    const char *s = getenv(name);
    if (!s) { *set = 0; *value = 0; return 0; }
    if (*s < '0' || *s > '9') return -1;
    char *end = NULL;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || end == s || *end || n > max) return -1;
    *set = 1; *value = (uint64_t)n;
    return 0;
}
static int capture_delay(void) {
    int fast_set = 0, slow_set = 0;
    if (parse_env_u64("ANNDATAPP_COMPLETENESS_DELAY_FAST_US", &delay_fast_us, MAX_DELAY_US, &fast_set) ||
        parse_env_u64("ANNDATAPP_COMPLETENESS_DELAY_SLOW_US", &delay_slow_us, MAX_DELAY_US, &slow_set) ||
        parse_env_u64("ANNDATAPP_COMPLETENESS_DELAY_SLOW_FROM_SHARD_ROW",
                      &delay_slow_from_row, UINT64_MAX, &delay_slow_row_set)) return -1;
    (void)fast_set; (void)slow_set;
    return 0;
}

static int metadata_name(const char *rel) {
    const char *base = strrchr(rel, '/');
    base = base ? base + 1 : rel;
    return !strcmp(base, ".zarray") || !strcmp(base, ".zattrs") ||
           !strcmp(base, ".zgroup") || !strcmp(base, ".zmetadata") ||
           !strcmp(base, "zarr.json");
}
static int shard_key(const char *full, char *key) {
    if (!x_root_len || strncmp(full, x_root, x_root_len) || full[x_root_len] != '/') return 0;
    const char *rel = full + x_root_len + 1;
    size_t n = strlen(rel);
    if (!n || n >= PATH_MAX) return 0;
    memcpy(key, rel, n + 1);
    char *deleted = strstr(key, " (deleted)");
    if (deleted && !deleted[10]) *deleted = 0;
    char *tmp = strstr(key, ".anndatapp-tmp");
    if (tmp && !tmp[12]) *tmp = 0;
    tmp = strstr(key, ".tmp-");
    if (tmp) *tmp = 0;
    if (!key[0] || metadata_name(key)) return 0;
    for (char *p = key; *p;) {
        char *end = strchr(p, '/');
        const size_t part = end ? (size_t)(end - p) : strlen(p);
        if ((part == 1 && p[0] == '.') || (part == 2 && p[0] == '.' && p[1] == '.')) return 0;
        if (!end) break;
        p = end + 1;
    }
    if (index_bytes && strncmp(key, "c/", 2)) return 0;
    return 1;
}
static int read_fd_path(int fd, char *path) {
    char proc[64];
    int n = snprintf(proc, sizeof proc, "/proc/self/fd/%d", fd);
    if (n < 0 || (size_t)n >= sizeof proc) return 0;
    ssize_t len = readlink(proc, path, PATH_MAX - 1);
    if (len <= 0 || len >= PATH_MAX - 1) return 0;
    path[len] = 0;
    return 1;
}
static int target_fd(int fd, int need_size, IoTarget *target) {
    char full[PATH_MAX];
    memset(target, 0, sizeof *target);
    if (!read_fd_path(fd, full) || !shard_key(full, target->key)) return 0;
    target->shard = 1;
    if (need_size) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size >= 0) {
            target->file_size = (uint64_t)st.st_size;
            target->file_size_valid = 1;
        }
    }
    return 1;
}
static ShardCount *shard_get(const char *key) {
    for (size_t i = 0; i < shard_count; ++i)
        if (!strcmp(shards[i].path, key)) return &shards[i];
    if (shard_count == MAX_SHARDS) { ++shard_path_overflow; return NULL; }
    ShardCount *c = &shards[shard_count++];
    memset(c, 0, sizeof *c);
    size_t n = strlen(key);
    if (n >= sizeof c->path) { --shard_count; ++shard_path_overflow; return NULL; }
    memcpy(c->path, key, n + 1);
    return c;
}
static uint64_t overlap(uint64_t a, uint64_t n, uint64_t b, uint64_t m) {
    if (!n || !m || a > UINT64_MAX - n || b > UINT64_MAX - m) return 0;
    const uint64_t lo = a > b ? a : b, hi = a + n < b + m ? a + n : b + m;
    return hi > lo ? hi - lo : 0;
}
static void record_path_io(const IoTarget *t, int writing, int positional, int64_t offset,
                           size_t request, ssize_t result) {
    if (!t->shard) return;
    ++internal_depth;
    pthread_mutex_lock(&shard_lock);
    ShardCount *c = shard_get(t->key);
    if (c) {
        if (writing) {
            ++c->write_calls; if (result > 0) c->write_bytes += (uint64_t)result;
            if (positional) {
                ++c->pwrite_calls; if (result > 0) c->pwrite_bytes += (uint64_t)result;
            }
        } else {
            ++c->read_calls; if (result > 0) c->read_bytes += (uint64_t)result;
            if (positional) {
                ++c->pread_calls; if (result > 0) c->pread_bytes += (uint64_t)result;
            }
        }
        if (positional && result > 0 && index_bytes) {
            if (writing && !index_at_start) {
                if (t->file_size_valid && offset >= 0 && (uint64_t)offset == t->file_size &&
                    request == index_bytes) {
                    ++c->index_end_write_candidates;
                    c->index_end_write_candidate_bytes += (uint64_t)result;
                }
            } else {
                const uint64_t start = writing || index_at_start ? 0 :
                    t->file_size_valid && t->file_size >= index_bytes ? t->file_size - index_bytes : UINT64_MAX;
                const uint64_t bytes = start == UINT64_MAX ? 0 :
                    overlap((uint64_t)offset, (uint64_t)result, start, index_bytes);
                if (bytes) {
                    if (writing) { ++c->index_write_calls; c->index_write_bytes += bytes; }
                    else { ++c->index_read_calls; c->index_read_bytes += bytes; }
                }
            }
        }
    }
    pthread_mutex_unlock(&shard_lock);
    --internal_depth;
}
static int absolute_path_at(int dirfd, const char *path, char *out) {
    if (!path || !*path) return 0;
    if (path[0] == '/') {
        size_t n = strlen(path);
        if (n >= PATH_MAX) return 0;
        memcpy(out, path, n + 1);
        return 1;
    }
    char base[PATH_MAX];
    if (dirfd == AT_FDCWD) {
        if (!getcwd(base, sizeof base)) return 0;
    } else if (!read_fd_path(dirfd, base)) return 0;
    size_t a = strlen(base), b = strlen(path);
    if (a + 1 + b >= PATH_MAX) return 0;
    memcpy(out, base, a);
    out[a] = '/';
    memcpy(out + a + 1, path, b + 1);
    return 1;
}
static int target_path_at(int dirfd, const char *path, char *key) {
    char full[PATH_MAX];
    return absolute_path_at(dirfd, path, full) && shard_key(full, key);
}
static int path_exists_at(int dirfd, const char *path) {
    struct stat st;
    return fstatat(dirfd, path, &st, AT_SYMLINK_NOFOLLOW) == 0;
}
static void record_rename(const char *key, int exists, int result) {
    ++internal_depth;
    pthread_mutex_lock(&shard_lock);
    ShardCount *c = shard_get(key);
    if (c) {
        ++c->rename_calls;
        if (!result) {
            ++c->rename_successes;
            if (exists) ++c->replacements; else ++c->new_shards;
        } else ++c->rename_failures;
    }
    pthread_mutex_unlock(&shard_lock);
    --internal_depth;
}

static uint64_t shard_row(const char *key) {
    if (strncmp(key, "c/", 2)) return UINT64_MAX;
    const char *p = key + 2;
    if (*p < '0' || *p > '9') return UINT64_MAX;
    uint64_t row = 0;
    while (*p >= '0' && *p <= '9') {
        unsigned d = (unsigned)(*p++ - '0');
        if (row > (UINT64_MAX - d) / 10) return UINT64_MAX;
        row = row * 10 + d;
    }
    if (*p++ != '/' || *p < '0' || *p > '9') return UINT64_MAX;
    while (*p >= '0' && *p <= '9') ++p;
    return *p == 0 || *p == '/' ? row : UINT64_MAX;
}
static void delay_pread(const IoTarget *t) {
    if (!t->shard || (!delay_fast_us && !delay_slow_us)) return;
    const uint64_t row = shard_row(t->key);
    if (row == UINT64_MAX) return;
    const uint64_t us = delay_slow_row_set && row >= delay_slow_from_row ?
        delay_slow_us : delay_fast_us;
    if (!us) return;
    struct timespec rem = {(time_t)(us / 1000000), (long)(us % 1000000) * 1000};
    ++internal_depth;
    while (nanosleep(&rem, &rem) && errno == EINTR) {}
    --internal_depth;
}
static void enter_x_pread(void) {
    uint64_t n = __atomic_add_fetch(&x_pread_active, 1, __ATOMIC_RELAXED);
    uint64_t max = LOAD(max_concurrent_x_pread);
    while (n > max && !__atomic_compare_exchange_n(&max_concurrent_x_pread, &max, n, 0,
                                                   __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}
static void leave_x_pread(void) { __atomic_fetch_sub(&x_pread_active, 1, __ATOMIC_RELAXED); }

#define DEFINE_IO(name, params, args, fdarg, requested, offset, positional, writing, predelay, ctr) \
    typedef ssize_t (*name##_fn) params;                                                            \
    static name##_fn real_##name;                                                                   \
    static name##_fn next_##name(void) {                                                            \
        name##_fn fn = LOAD(real_##name);                                                           \
        if (!fn) {                                                                                  \
            ++internal_depth; fn = (name##_fn)dlsym(RTLD_NEXT, #name); --internal_depth;            \
            if (fn) STORE(real_##name, fn);                                                         \
        }                                                                                           \
        return fn;                                                                                  \
    }                                                                                               \
    ssize_t name params {                                                                           \
        name##_fn fn = next_##name();                                                               \
        if (!fn) { errno = ENOSYS; return -1; }                                                     \
        const int sample = measuring() && !internal_depth && !io_depth;                             \
        if (!sample) { ++io_depth; ssize_t r = fn args; --io_depth; return r; }                      \
        const size_t request = (requested);                                                         \
        if (!reserve_io((writing), request)) return -1;                                             \
        IoTarget target;                                                                            \
        ++internal_depth; (void)target_fd((fdarg), (positional), &target); --internal_depth;         \
        const int x_pread = (predelay) && target.shard;                                              \
        if (x_pread) enter_x_pread();                                                                \
        if (predelay) delay_pread(&target);                                                         \
        ++io_depth; ssize_t r = fn args; --io_depth;                                                \
        if (x_pread) leave_x_pread();                                                                \
        settle_io((writing), request, r);                                                           \
        add(&(ctr).calls, 1); if (r > 0) add(&(ctr).bytes, (uint64_t)r);                             \
        record_path_io(&target, (writing), (positional), (int64_t)(offset), request, r);             \
        return r;                                                                                   \
    }

DEFINE_IO(read, (int fd, void *buffer, size_t count), (fd, buffer, count),
          fd, count, 0, 0, 0, 0, io_read)
DEFINE_IO(pread, (int fd, void *buffer, size_t count, off_t offset), (fd, buffer, count, offset),
          fd, count, offset, 1, 0, 1, io_pread)
DEFINE_IO(pread64, (int fd, void *buffer, size_t count, off64_t offset), (fd, buffer, count, offset),
          fd, count, offset, 1, 0, 1, io_pread)
DEFINE_IO(readv, (int fd, const struct iovec *iov, int count), (fd, iov, count),
          fd, iov_bytes(iov, count), 0, 0, 0, 0, io_readv)
DEFINE_IO(preadv, (int fd, const struct iovec *iov, int count, off_t offset), (fd, iov, count, offset),
          fd, iov_bytes(iov, count), offset, 1, 0, 0, io_preadv)
DEFINE_IO(preadv64, (int fd, const struct iovec *iov, int count, off64_t offset), (fd, iov, count, offset),
          fd, iov_bytes(iov, count), offset, 1, 0, 0, io_preadv)
DEFINE_IO(write, (int fd, const void *buffer, size_t count), (fd, buffer, count),
          fd, count, 0, 0, 1, 0, io_write)
DEFINE_IO(pwrite, (int fd, const void *buffer, size_t count, off_t offset), (fd, buffer, count, offset),
          fd, count, offset, 1, 1, 0, io_pwrite)
DEFINE_IO(pwrite64, (int fd, const void *buffer, size_t count, off64_t offset), (fd, buffer, count, offset),
          fd, count, offset, 1, 1, 0, io_pwrite)
DEFINE_IO(writev, (int fd, const struct iovec *iov, int count), (fd, iov, count),
          fd, iov_bytes(iov, count), 0, 0, 1, 0, io_writev)
DEFINE_IO(pwritev, (int fd, const struct iovec *iov, int count, off_t offset), (fd, iov, count, offset),
          fd, iov_bytes(iov, count), offset, 1, 1, 0, io_pwritev)
DEFINE_IO(pwritev64, (int fd, const struct iovec *iov, int count, off64_t offset), (fd, iov, count, offset),
          fd, iov_bytes(iov, count), offset, 1, 1, 0, io_pwritev)
#undef DEFINE_IO

int rename(const char *oldpath, const char *newpath) {
    typedef int (*fn_t)(const char *, const char *); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "rename"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) { errno = ENOSYS; return -1; }
    const int sample = measuring() && !internal_depth && !rename_depth;
    char key[PATH_MAX];
    if (!sample) { ++rename_depth; int r = fn(oldpath, newpath); --rename_depth; return r; }
    ++internal_depth; int matched = target_path_at(AT_FDCWD, newpath, key);
    int exists = path_exists_at(AT_FDCWD, newpath); --internal_depth;
    ++rename_depth; int r = fn(oldpath, newpath); --rename_depth;
    if (matched) record_rename(key, exists, r);
    return r;
}
int renameat(int olddirfd, const char *oldpath, int newdirfd, const char *newpath) {
    typedef int (*fn_t)(int, const char *, int, const char *); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "renameat"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) { errno = ENOSYS; return -1; }
    const int sample = measuring() && !internal_depth && !rename_depth;
    char key[PATH_MAX];
    if (!sample) { ++rename_depth; int r = fn(olddirfd, oldpath, newdirfd, newpath); --rename_depth; return r; }
    ++internal_depth; int matched = target_path_at(newdirfd, newpath, key);
    int exists = path_exists_at(newdirfd, newpath); --internal_depth;
    ++rename_depth; int r = fn(olddirfd, oldpath, newdirfd, newpath); --rename_depth;
    if (matched) record_rename(key, exists, r);
    return r;
}
int renameat2(int olddirfd, const char *oldpath, int newdirfd, const char *newpath, unsigned int flags) {
    typedef int (*fn_t)(int, const char *, int, const char *, unsigned int); static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "renameat2"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) { errno = ENOSYS; return -1; }
    const int sample = measuring() && !internal_depth && !rename_depth;
    char key[PATH_MAX];
    if (!sample) { ++rename_depth; int r = fn(olddirfd, oldpath, newdirfd, newpath, flags); --rename_depth; return r; }
    ++internal_depth; int matched = target_path_at(newdirfd, newpath, key);
    int exists = path_exists_at(newdirfd, newpath); --internal_depth;
    ++rename_depth; int r = fn(olddirfd, oldpath, newdirfd, newpath, flags); --rename_depth;
    if (matched) record_rename(key, exists, r);
    return r;
}

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
    if (alignment % sizeof(void *) || !power_of_two(alignment)) return EINVAL;
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

typedef size_t (*zstd_compress2_fn)(ZSTD_CCtx *, void *, size_t, const void *, size_t);
typedef size_t (*zstd_decompress_stream_fn)(ZSTD_DCtx *, ZSTD_outBuffer *, ZSTD_inBuffer *);
typedef int (*zlib_stream_fn)(z_streamp, int);

#define NEXT_CODEC(name, type)                                                   \
    static type real_##name;                                                    \
    static type next_##name(void) {                                             \
        type fn = LOAD(real_##name);                                            \
        if (!fn) {                                                              \
            ++internal_depth; fn = (type)dlsym(RTLD_NEXT, #name); --internal_depth; \
            if (fn) STORE(real_##name, fn);                                     \
        }                                                                       \
        return fn;                                                              \
    }

int blosc_compress_ctx(int level, int shuffle, size_t typesize, size_t nbytes,
                       const void *src, void *dest, size_t destsize,
                       const char *compressor, size_t blocksize, int nthreads) {
    typedef int (*fn_t)(int, int, size_t, size_t, const void *, void *, size_t, const char *, size_t, int);
    static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "blosc_compress_ctx"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return -1;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(level, shuffle, typesize, nbytes, src, dest, destsize, compressor, blocksize, nthreads); --codec_depth;
    if (sample && result >= 0) record_codec(&blosc_encode, nbytes, (uint64_t)result);
    return result;
}
int blosc_decompress_ctx(const void *src, void *dest, size_t destsize, int nthreads) {
    typedef int (*fn_t)(const void *, void *, size_t, int);
    static fn_t real_fn;
    fn_t fn = LOAD(real_fn);
    if (!fn) { ++internal_depth; fn = (fn_t)dlsym(RTLD_NEXT, "blosc_decompress_ctx"); --internal_depth; if (fn) STORE(real_fn, fn); }
    if (!fn) return -1;
    size_t out = 0, in = 0, block = 0; blosc_cbuffer_sizes(src, &out, &in, &block);
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(src, dest, destsize, nthreads); --codec_depth;
    if (sample && result >= 0) record_codec(&blosc_decode, in, (uint64_t)result);
    return result;
}
NEXT_CODEC(ZSTD_compress2, zstd_compress2_fn)
size_t ZSTD_compress2(ZSTD_CCtx *ctx, void *dst, size_t capacity, const void *src, size_t size) {
    typedef size_t (*fn_t)(ZSTD_CCtx *, void *, size_t, const void *, size_t);
    fn_t fn = next_ZSTD_compress2();
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(ctx, dst, capacity, src, size); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_encode, size, result);
    return result;
}
NEXT_CODEC(ZSTD_decompressStream, zstd_decompress_stream_fn)
size_t ZSTD_decompressStream(ZSTD_DCtx *ctx, ZSTD_outBuffer *out, ZSTD_inBuffer *in) {
    typedef size_t (*fn_t)(ZSTD_DCtx *, ZSTD_outBuffer *, ZSTD_inBuffer *);
    fn_t fn = next_ZSTD_decompressStream();
    if (!fn) return ZSTD_CONTENTSIZE_ERROR;
    const size_t in0 = in->pos, out0 = out->pos;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; size_t result = fn(ctx, out, in); --codec_depth;
    if (sample && !ZSTD_isError(result)) record_codec(&zstd_decode, in->pos - in0, out->pos - out0);
    return result;
}
NEXT_CODEC(inflate, zlib_stream_fn)
int inflate(z_streamp stream, int flush) {
    typedef int (*fn_t)(z_streamp, int);
    fn_t fn = next_inflate();
    if (!fn) return Z_STREAM_ERROR;
    const uInt in0 = stream->avail_in, out0 = stream->avail_out;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(stream, flush); --codec_depth;
    if (sample) record_codec(&zlib_decode, in0 - stream->avail_in, out0 - stream->avail_out);
    return result;
}
NEXT_CODEC(deflate, zlib_stream_fn)
int deflate(z_streamp stream, int flush) {
    typedef int (*fn_t)(z_streamp, int);
    fn_t fn = next_deflate();
    if (!fn) return Z_STREAM_ERROR;
    const uInt in0 = stream->avail_in, out0 = stream->avail_out;
    const int sample = measuring() && !internal_depth && !codec_depth;
    ++codec_depth; int result = fn(stream, flush); --codec_depth;
    if (sample) record_codec(&zlib_encode, in0 - stream->avail_in, out0 - stream->avail_out);
    return result;
}
#undef NEXT_CODEC

static int begin_error(int error) { errno = error; return -1; }
static int begin_reset_error(int error) { STORE(active, 0); errno = error; return -1; }
int anndatapp_completeness_probe_begin(uint64_t read_cap_bytes, uint64_t write_cap_bytes,
                                     const char *x_dir, size_t encoded_index_bytes,
                                     int index_location_start) {
    if (!x_dir || !*x_dir || read_cap_bytes > IO_MAX_BYTES || write_cap_bytes > IO_MAX_BYTES ||
        (index_location_start != 0 && index_location_start != 1)) return begin_error(EINVAL);
    int expected = 0;
    if (!__atomic_compare_exchange_n(&active, &expected, 2, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
        return begin_error(EBUSY);
    if (!realpath(x_dir, x_root)) return begin_reset_error(errno);
    x_root_len = strlen(x_root);
    if (!x_root_len || x_root_len >= PATH_MAX - 1) return begin_reset_error(ENAMETOOLONG);
    if (capture_delay()) return begin_reset_error(EINVAL);
    read_cap = read_cap_bytes; write_cap = write_cap_bytes;
    index_bytes = encoded_index_bytes; index_at_start = index_location_start;
    read_budget = write_budget = read_rejections = write_rejections = calloc_overflows = 0;
    STORE(x_pread_active, 0); STORE(max_concurrent_x_pread, 0);
    clear_io(&io_read); clear_io(&io_pread); clear_io(&io_readv); clear_io(&io_preadv);
    clear_io(&io_write); clear_io(&io_pwrite); clear_io(&io_writev); clear_io(&io_pwritev);
    clear_codec(&blosc_decode); clear_codec(&blosc_encode);
    clear_codec(&zstd_decode); clear_codec(&zstd_encode);
    clear_codec(&zlib_decode); clear_codec(&zlib_encode);
    clear_alloc(&alloc_malloc); clear_alloc(&alloc_calloc); clear_alloc(&alloc_realloc);
    clear_alloc(&alloc_posix_memalign); clear_alloc(&alloc_aligned_alloc);
    clear_alloc(&alloc_memalign); clear_alloc(&alloc_valloc); clear_alloc(&alloc_pvalloc);
    pthread_mutex_lock(&shard_lock);
    memset(shards, 0, sizeof shards); shard_count = 0; shard_path_overflow = 0;
    pthread_mutex_unlock(&shard_lock);
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
    uint64_t read_bytes = 0, write_bytes = 0;
    uint64_t path_overflow;
#define SUM_IO(c, r) do { (r) += LOAD((c).bytes); } while (0)
    SUM_IO(io_read, read_bytes); SUM_IO(io_pread, read_bytes);
    SUM_IO(io_readv, read_bytes); SUM_IO(io_preadv, read_bytes);
    SUM_IO(io_write, write_bytes); SUM_IO(io_pwrite, write_bytes);
    SUM_IO(io_writev, write_bytes); SUM_IO(io_pwritev, write_bytes);
#undef SUM_IO
    pthread_mutex_lock(&shard_lock);
    path_overflow = shard_path_overflow;
    pthread_mutex_unlock(&shard_lock);
    fprintf(stderr,
        "ANNDATAPP_COMPLETENESS_PROBE {\"schema\":\"anndatapp.completeness_probe.counters.v1\","
        "\"x_dir\":"); json_path(x_root);
    fprintf(stderr,
        ",\"index_bytes\":%llu,\"index_at_start\":%d,"
        "\"read_cap_bytes\":%llu,\"write_cap_bytes\":%llu,"
        "\"read_budget_used_bytes\":%llu,\"write_budget_used_bytes\":%llu,"
        "\"read_rejections\":%llu,\"write_rejections\":%llu,"
        "\"max_concurrent_x_pread\":%llu,\"x_pread_active_at_report\":%llu,"
        "\"intercepted_read_bytes\":%llu,\"intercepted_write_bytes\":%llu,\"io\":{",
        (unsigned long long)index_bytes, index_at_start,
        (unsigned long long)LOAD(read_cap), (unsigned long long)LOAD(write_cap),
        (unsigned long long)LOAD(read_budget), (unsigned long long)LOAD(write_budget),
        (unsigned long long)LOAD(read_rejections), (unsigned long long)LOAD(write_rejections),
        (unsigned long long)LOAD(max_concurrent_x_pread), (unsigned long long)LOAD(x_pread_active),
        (unsigned long long)read_bytes, (unsigned long long)write_bytes);
    int first = 1;
    json_io("read", &io_read, &first); json_io("pread", &io_pread, &first);
    json_io("readv", &io_readv, &first); json_io("preadv", &io_preadv, &first);
    json_io("write", &io_write, &first); json_io("pwrite", &io_pwrite, &first);
    json_io("writev", &io_writev, &first); json_io("pwritev", &io_pwritev, &first);
    fprintf(stderr, "},\"codec_api_stream_calls\":{");
    json_codec("blosc", &blosc_decode, &blosc_encode); fputc(',', stderr);
    json_codec("zstd", &zstd_decode, &zstd_encode); fputc(',', stderr);
    json_codec("zlib", &zlib_decode, &zlib_encode);
    fprintf(stderr, "},\"alloc\":{"); first = 1;
    json_alloc("malloc", &alloc_malloc, &first); json_alloc("calloc", &alloc_calloc, &first);
    json_alloc("realloc", &alloc_realloc, &first); json_alloc("posix_memalign", &alloc_posix_memalign, &first);
    json_alloc("aligned_alloc", &alloc_aligned_alloc, &first); json_alloc("memalign", &alloc_memalign, &first);
    json_alloc("valloc", &alloc_valloc, &first); json_alloc("pvalloc", &alloc_pvalloc, &first);
    fprintf(stderr,
        "},\"calloc_overflows\":%llu,\"new_counting\":\"ordinary operator new is included through malloc; no separate new wrapper\","
        "\"index_io_classification\":\"start/end reads and start writes count actual positional byte overlap; end writes count EOF and exact-size candidates\","
        "\"delay\":{\"fast_us\":%llu,\"slow_us\":%llu,\"slow_from_shard_row_set\":%d,\"slow_from_shard_row\":%llu,\"max_us\":%llu},"
        "\"delay_scope\":\"armed pread calls on X shard paths c/<row>/<col>; no other I/O is delayed\","
        "\"shard_path_limit\":%u,\"shard_path_overflow\":%llu,\"shards\":[",
        (unsigned long long)LOAD(calloc_overflows),
        (unsigned long long)delay_fast_us, (unsigned long long)delay_slow_us, delay_slow_row_set,
        (unsigned long long)delay_slow_from_row, (unsigned long long)MAX_DELAY_US,
        (unsigned)MAX_SHARDS,
        (unsigned long long)path_overflow);
    pthread_mutex_lock(&shard_lock);
    for (size_t i = 0; i < shard_count; ++i) {
        const ShardCount *c = &shards[i];
        if (i) fputc(',', stderr);
        fputs("{\"path\":", stderr); json_path(c->path);
        fprintf(stderr,
            ",\"read_calls\":%llu,\"read_bytes\":%llu,\"write_calls\":%llu,\"write_bytes\":%llu,"
            "\"pread_calls\":%llu,\"pread_bytes\":%llu,\"pwrite_calls\":%llu,\"pwrite_bytes\":%llu,"
            "\"index_read_calls\":%llu,\"index_read_bytes\":%llu,"
            "\"index_write_calls\":%llu,\"index_write_bytes\":%llu,"
            "\"index_end_write_candidates\":%llu,\"index_end_write_candidate_bytes\":%llu,"
            "\"rename_calls\":%llu,\"rename_successes\":%llu,\"replacements\":%llu,\"new_shards\":%llu,\"rename_failures\":%llu}",
            (unsigned long long)c->read_calls, (unsigned long long)c->read_bytes,
            (unsigned long long)c->write_calls, (unsigned long long)c->write_bytes,
            (unsigned long long)c->pread_calls, (unsigned long long)c->pread_bytes,
            (unsigned long long)c->pwrite_calls, (unsigned long long)c->pwrite_bytes,
            (unsigned long long)c->index_read_calls, (unsigned long long)c->index_read_bytes,
            (unsigned long long)c->index_write_calls, (unsigned long long)c->index_write_bytes,
            (unsigned long long)c->index_end_write_candidates,
            (unsigned long long)c->index_end_write_candidate_bytes,
            (unsigned long long)c->rename_calls, (unsigned long long)c->rename_successes,
            (unsigned long long)c->replacements, (unsigned long long)c->new_shards,
            (unsigned long long)c->rename_failures);
    }
    pthread_mutex_unlock(&shard_lock);
    fputs("]}\n", stderr);
}
void anndatapp_completeness_probe_end(void) {
    if (__atomic_exchange_n(&active, 0, __ATOMIC_RELAXED) == 1) report_probe();
}
__attribute__((destructor)) static void report_if_armed(void) {
    if (LOAD(active) == 1) anndatapp_completeness_probe_end();
}
