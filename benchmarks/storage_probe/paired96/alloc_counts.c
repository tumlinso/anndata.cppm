#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
 * ROI-only glibc allocator API counts. Normal libstdc++ operator new calls
 * malloc in this environment, so they are included once in malloc counts.
 * Aligned allocation entrypoints are counted separately and are not added to
 * malloc/calloc/realloc totals. This records requested sizes, not live peak.
 */
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);

typedef struct {
    uint64_t calls;
    uint64_t successes;
    uint64_t requested_bytes;
    uint64_t successful_bytes;
} AllocationCounts;

static volatile int active;
static AllocationCounts malloc_counts, calloc_counts, realloc_counts;
static AllocationCounts posix_memalign_counts, aligned_alloc_counts;
static AllocationCounts memalign_counts, valloc_counts, pvalloc_counts;
static uint64_t calloc_overflows;

static void add_u64(uint64_t *counter, uint64_t value) {
    __atomic_fetch_add(counter, value, __ATOMIC_RELAXED);
}
static int measuring(void) {
    return __atomic_load_n(&active, __ATOMIC_RELAXED);
}
static void record(AllocationCounts *counts, size_t requested, int succeeded) {
    if (!measuring()) return;
    add_u64(&counts->calls, 1);
    add_u64(&counts->requested_bytes, (uint64_t)requested);
    if (succeeded) {
        add_u64(&counts->successes, 1);
        add_u64(&counts->successful_bytes, (uint64_t)requested);
    }
}
static void clear_counts(AllocationCounts *counts) {
    __atomic_store_n(&counts->calls, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&counts->successes, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&counts->requested_bytes, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&counts->successful_bytes, 0, __ATOMIC_RELAXED);
}
static int is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}
static void *aligned_alloc_impl(size_t alignment, size_t size) {
    return __libc_memalign(alignment, size);
}

void bulk_alloc_probe_begin(void) {
    __atomic_store_n(&active, 0, __ATOMIC_RELAXED);
    clear_counts(&malloc_counts);
    clear_counts(&calloc_counts);
    clear_counts(&realloc_counts);
    clear_counts(&posix_memalign_counts);
    clear_counts(&aligned_alloc_counts);
    clear_counts(&memalign_counts);
    clear_counts(&valloc_counts);
    clear_counts(&pvalloc_counts);
    __atomic_store_n(&calloc_overflows, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&active, 1, __ATOMIC_RELAXED);
}
void bulk_alloc_probe_end(void) { __atomic_store_n(&active, 0, __ATOMIC_RELAXED); }

void *malloc(size_t size) {
    void *result = __libc_malloc(size);
    record(&malloc_counts, size, result != NULL);
    return result;
}
void *calloc(size_t count, size_t size) {
    size_t requested = 0;
    const int overflow = __builtin_mul_overflow(count, size, &requested);
    void *result = __libc_calloc(count, size);
    if (overflow) {
        if (measuring()) add_u64(&calloc_overflows, 1);
        requested = SIZE_MAX;
    }
    record(&calloc_counts, requested, result != NULL);
    return result;
}
void *realloc(void *pointer, size_t size) {
    void *result = __libc_realloc(pointer, size);
    record(&realloc_counts, size, result != NULL);
    return result;
}

int posix_memalign(void **pointer, size_t alignment, size_t size) {
    if (!pointer || alignment % sizeof(void *) != 0 || !is_power_of_two(alignment))
        return EINVAL;
    void *result = aligned_alloc_impl(alignment, size);
    if (!result) {
        record(&posix_memalign_counts, size, 0);
        return ENOMEM;
    }
    *pointer = result;
    record(&posix_memalign_counts, size, 1);
    return 0;
}
void *aligned_alloc(size_t alignment, size_t size) {
    if (!is_power_of_two(alignment) || size % alignment != 0) {
        errno = EINVAL;
        record(&aligned_alloc_counts, size, 0);
        return NULL;
    }
    void *result = aligned_alloc_impl(alignment, size);
    record(&aligned_alloc_counts, size, result != NULL);
    return result;
}
void *memalign(size_t alignment, size_t size) {
    if (alignment == 0) {
        errno = EINVAL;
        record(&memalign_counts, size, 0);
        return NULL;
    }
    void *result = aligned_alloc_impl(alignment, size);
    record(&memalign_counts, size, result != NULL);
    return result;
}
void *valloc(size_t size) {
    const size_t alignment = (size_t)getpagesize();
    void *result = aligned_alloc_impl(alignment, size);
    record(&valloc_counts, size, result != NULL);
    return result;
}
void *pvalloc(size_t size) {
    const size_t page = (size_t)getpagesize();
    if (size > SIZE_MAX - (page - 1)) {
        errno = ENOMEM;
        record(&pvalloc_counts, size, 0);
        return NULL;
    }
    const size_t rounded = (size + page - 1) & ~(page - 1);
    void *result = aligned_alloc_impl(page, rounded);
    record(&pvalloc_counts, size, result != NULL);
    return result;
}

static void report_one(const char *name, const AllocationCounts *counts) {
    fprintf(stderr, "%s calls=%llu successes=%llu requested_bytes=%llu successful_bytes=%llu\n",
            name,
            (unsigned long long)__atomic_load_n(&counts->calls, __ATOMIC_RELAXED),
            (unsigned long long)__atomic_load_n(&counts->successes, __ATOMIC_RELAXED),
            (unsigned long long)__atomic_load_n(&counts->requested_bytes, __ATOMIC_RELAXED),
            (unsigned long long)__atomic_load_n(&counts->successful_bytes, __ATOMIC_RELAXED));
}
__attribute__((destructor)) static void report_allocator_counts(void) {
    fputs("ALLOCATOR_SCOPE=interposed glibc public allocation entrypoints during write_zarr; regular C++ new is included via malloc; aligned APIs are separate; requested bytes are cumulative; no live peak; direct __libc internal allocations are not intercepted\n", stderr);
    report_one("MALLOC", &malloc_counts);
    report_one("CALLOC", &calloc_counts);
    report_one("REALLOC", &realloc_counts);
    report_one("POSIX_MEMALIGN", &posix_memalign_counts);
    report_one("ALIGNED_ALLOC", &aligned_alloc_counts);
    report_one("MEMALIGN", &memalign_counts);
    report_one("VALLOC", &valloc_counts);
    report_one("PVALLOC", &pvalloc_counts);
    fprintf(stderr, "CALLOC_OVERFLOW_CALLS=%llu\n",
            (unsigned long long)__atomic_load_n(&calloc_overflows, __ATOMIC_RELAXED));
}
