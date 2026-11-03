#define _GNU_SOURCE
#include <hdf5.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "probe.h"

typedef herr_t (*read_fn)(hid_t, hid_t, hid_t, hid_t, hid_t, void *);
typedef herr_t (*write_fn)(hid_t, hid_t, hid_t, hid_t, hid_t, const void *);

enum { dataset_capacity = 128, path_capacity = 512 };

typedef struct {
    char path[path_capacity];
    uint64_t reads, read_bytes, writes, write_bytes;
} dataset_counts;

static read_fn next_read;
static write_fn next_write;
static pthread_once_t next_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t counts_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int enabled;
static dataset_counts datasets[dataset_capacity];
static size_t dataset_count;
static uint64_t read_calls, read_bytes, write_calls, write_bytes, measure_errors;

static void resolve_next(void) {
    next_read = (read_fn)dlsym(RTLD_NEXT, "H5Dread");
    next_write = (write_fn)dlsym(RTLD_NEXT, "H5Dwrite");
}

static hid_t selected_space(hid_t dset, hid_t mem_space, hid_t file_space, hid_t *owned) {
    *owned = -1;
    if (mem_space != H5S_ALL) return mem_space;
    if (file_space != H5S_ALL) return file_space;
    *owned = H5Dget_space(dset);
    return *owned;
}

static uint64_t projected_bytes(hid_t dset, hid_t type, hid_t mem_space, hid_t file_space) {
    const size_t width = H5Tget_size(type);
    hid_t owned, space = selected_space(dset, mem_space, file_space, &owned);
    const hssize_t points = space >= 0 ? H5Sget_select_npoints(space) : -1;
    if (owned >= 0) H5Sclose(owned);
    if (!width || points < 0) {
        pthread_mutex_lock(&counts_lock);
        ++measure_errors;
        pthread_mutex_unlock(&counts_lock);
        return 0;
    }
    return (uint64_t)points > UINT64_MAX / width ? UINT64_MAX : (uint64_t)points * width;
}

static void dataset_path(hid_t dset, char path[path_capacity]) {
    const ssize_t n = H5Iget_name(dset, path, path_capacity);
    if (n <= 0 || n >= path_capacity) snprintf(path, path_capacity, "<unnamed>");
}

static void account(hid_t dset, uint64_t bytes, int write) {
    char path[path_capacity];
    dataset_path(dset, path);
    pthread_mutex_lock(&counts_lock);
    if (write) { ++write_calls; write_bytes += bytes; }
    else { ++read_calls; read_bytes += bytes; }
    size_t i = 0;
    while (i < dataset_count && strcmp(datasets[i].path, path)) ++i;
    if (i == dataset_count && dataset_count < dataset_capacity) {
        snprintf(datasets[i].path, path_capacity, "%s", path);
        ++dataset_count;
    }
    if (i < dataset_count) {
        if (write) { ++datasets[i].writes; datasets[i].write_bytes += bytes; }
        else { ++datasets[i].reads; datasets[i].read_bytes += bytes; }
    }
    pthread_mutex_unlock(&counts_lock);
}

herr_t H5Dread(hid_t dset, hid_t type, hid_t mem_space, hid_t file_space,
               hid_t dxpl, void *buffer) {
    pthread_once(&next_once, resolve_next);
    if (!next_read) return -1;
    const int count = atomic_load(&enabled);
    const uint64_t bytes = count ? projected_bytes(dset, type, mem_space, file_space) : 0;
    const herr_t result = next_read(dset, type, mem_space, file_space, dxpl, buffer);
    if (count) account(dset, bytes, 0);
    return result;
}

herr_t H5Dwrite(hid_t dset, hid_t type, hid_t mem_space, hid_t file_space,
                hid_t dxpl, const void *buffer) {
    pthread_once(&next_once, resolve_next);
    if (!next_write) return -1;
    const int count = atomic_load(&enabled);
    const uint64_t bytes = count ? projected_bytes(dset, type, mem_space, file_space) : 0;
    const herr_t result = next_write(dset, type, mem_space, file_space, dxpl, buffer);
    if (count) account(dset, bytes, 1);
    return result;
}

void anndata_record_probe_enable(int value) { atomic_store(&enabled, !!value); }

void anndata_record_probe_reset(void) {
    atomic_store(&enabled, 0);
    pthread_mutex_lock(&counts_lock);
    memset(datasets, 0, sizeof datasets);
    dataset_count = 0;
    read_calls = read_bytes = write_calls = write_bytes = measure_errors = 0;
    pthread_mutex_unlock(&counts_lock);
}

static int append(char *out, size_t cap, size_t *used, const char *format, ...) {
    if (*used >= cap) return -1;
    va_list args;
    va_start(args, format);
    const int n = vsnprintf(out + *used, cap - *used, format, args);
    va_end(args);
    if (n < 0 || (size_t)n >= cap - *used) { *used = cap; return -1; }
    *used += (size_t)n;
    return 0;
}

static int append_path(char *out, size_t cap, size_t *used, const char *path) {
    if (append(out, cap, used, "\"") < 0) return -1;
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        if (*p == '"' || *p == '\\') {
            char escaped[3] = {'\\', (char)*p, '\0'};
            if (append(out, cap, used, "%s", escaped) < 0) return -1;
        } else if (*p < 0x20) {
            if (append(out, cap, used, "\\u%04x", (unsigned)*p) < 0) return -1;
        } else {
            char c[2] = {(char)*p, '\0'};
            if (append(out, cap, used, "%s", c) < 0) return -1;
        }
    }
    return append(out, cap, used, "\"");
}

size_t anndata_record_probe_json(char *out, size_t cap) {
    if (!out || !cap) return 0;
    size_t used = 0;
    pthread_mutex_lock(&counts_lock);
    int failed = append(out, cap, &used,
        "{\"read_calls\":%" PRIu64 ",\"read_native_logical_bytes\":%" PRIu64
        ",\"write_calls\":%" PRIu64 ",\"write_native_logical_bytes\":%" PRIu64
        ",\"projection_errors\":%" PRIu64 ",\"datasets\":[",
        read_calls, read_bytes, write_calls, write_bytes, measure_errors) < 0;
    for (size_t i = 0; !failed && i < dataset_count; ++i) {
        if (i) failed = append(out, cap, &used, ",") < 0;
        if (!failed) failed = append(out, cap, &used, "{\"path\":") < 0;
        if (!failed) failed = append_path(out, cap, &used, datasets[i].path) < 0;
        if (!failed) failed = append(out, cap, &used,
            ",\"read_calls\":%" PRIu64 ",\"read_native_logical_bytes\":%" PRIu64
            ",\"write_calls\":%" PRIu64 ",\"write_native_logical_bytes\":%" PRIu64 "}",
            datasets[i].reads, datasets[i].read_bytes, datasets[i].writes, datasets[i].write_bytes) < 0;
    }
    if (!failed) failed = append(out, cap, &used, "]}") < 0;
    pthread_mutex_unlock(&counts_lock);
    if (failed) { out[cap - 1] = '\0'; return 0; }
    return used;
}
