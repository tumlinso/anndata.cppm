#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <linux/perf_event.h>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

import anndata;

extern "C" void bulk_probe_begin() __attribute__((weak));
extern "C" void bulk_probe_end() __attribute__((weak));

namespace allocation_probe {
std::atomic<bool> active{false};
std::atomic<unsigned long long> calls{0};
std::atomic<unsigned long long> requested_bytes{0};

void* allocate(std::size_t size, std::size_t alignment) {
    const auto requested = size;
    if (size == 0) size = 1;
    void* pointer = nullptr;
    if (alignment > alignof(std::max_align_t)) {
        if (posix_memalign(&pointer, alignment, size) != 0) pointer = nullptr;
    } else {
        pointer = std::malloc(size);
    }
    if (!pointer) throw std::bad_alloc();
    if (active.load(std::memory_order_relaxed)) {
        calls.fetch_add(1, std::memory_order_relaxed);
        requested_bytes.fetch_add(requested, std::memory_order_relaxed);
    }
    return pointer;
}
}

void* operator new(std::size_t size) { return allocation_probe::allocate(size, alignof(std::max_align_t)); }
void* operator new[](std::size_t size) { return allocation_probe::allocate(size, alignof(std::max_align_t)); }
void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocation_probe::allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return allocation_probe::allocate(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }

namespace {
using Clock = std::chrono::steady_clock;
constexpr int kPerfCount = 2;
const std::uint64_t kPerfConfigs[kPerfCount] = {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS};

struct PerfGroup {
    int fds[kPerfCount]{-1, -1};
    int leader = -1;
    int error = 0;
    std::uint64_t values[6]{};
    std::uint64_t time_enabled = 0;
    std::uint64_t time_running = 0;
    bool active = false;

    bool start() {
        for (int i = 0; i < kPerfCount; ++i) {
            perf_event_attr attr{};
            attr.size = sizeof(attr);
            attr.type = PERF_TYPE_HARDWARE;
            attr.config = kPerfConfigs[i];
            attr.disabled = i == 0;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                               PERF_FORMAT_TOTAL_TIME_RUNNING;
            fds[i] = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, leader, 0));
            if (fds[i] < 0) {
                error = errno;
                close_all();
                return false;
            }
            if (leader < 0) leader = fds[i];
        }
        if (ioctl(leader, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
            ioctl(leader, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
            error = errno;
            close_all();
            return false;
        }
        active = true;
        return true;
    }

    void stop() {
        if (!active) return;
        if (ioctl(leader, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) error = errno;
        std::uint64_t data[3 + kPerfCount]{};
        const auto count = ::read(leader, data, sizeof(data));
        if (count == static_cast<ssize_t>(sizeof(data)) && data[0] == kPerfCount) {
            time_enabled = data[1];
            time_running = data[2];
            for (int i = 0; i < kPerfCount; ++i) values[i] = data[3 + i];
        } else if (count < 0) {
            error = errno;
        } else {
            error = EIO;
        }
        active = false;
        close_all();
    }

    void close_all() {
        for (int i = 0; i < kPerfCount; ++i) {
            if (fds[i] >= 0) close(fds[i]);
            fds[i] = -1;
        }
        leader = -1;
        active = false;
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::uint64_t timeval_ns(const timeval& value) {
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_usec) * 1000ULL;
}
std::uint64_t thread_cpu_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return 0;
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<std::uint64_t>(value.tv_nsec);
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: bulk_export FORMAT SOURCE DESTINATION");
        const std::string_view format(argv[1]);
        const std::string source(argv[2]);
        const std::string destination(argv[3]);
        require(format == "zarr2" || format == "zarr3", "format must be zarr2 or zarr3");
        require(!std::filesystem::exists(destination), "destination already exists");

        anndata::BackedOptions open_options;
        open_options.mode = anndata::AccessMode::read_only;
        open_options.scratch_bytes = 64u * 1024u * 1024u;
        open_options.decoded_cache_bytes = 0;
        auto backed = anndata::open_zarr(source, open_options);
        require(backed.n_obs() == 4096 && backed.n_vars() == 4096, "source dimensions differ");
        auto selected = backed.slice(anndata::range{1792, 2176, 1}, anndata::all);
        require(selected.n_obs() == 384 && selected.n_vars() == 4096, "selected dimensions differ");
        require(selected.x().layout() == anndata::Layout::dense, "source X is not dense");

        anndata::ZarrWriteOptions write_options;
        write_options.format = format == "zarr2" ? anndata::ZarrFormat::v2 : anndata::ZarrFormat::v3;
        write_options.scratch_bytes = 64u * 1024u * 1024u;
        PerfGroup counters;
        const bool perf_available = counters.start();
        if (bulk_probe_begin) bulk_probe_begin();
        const auto start = Clock::now();
        const auto thread_cpu_start = thread_cpu_ns();
        rusage usage_start{}, usage_stop{};
        getrusage(RUSAGE_SELF, &usage_start);
        allocation_probe::calls.store(0, std::memory_order_relaxed);
        allocation_probe::requested_bytes.store(0, std::memory_order_relaxed);
        allocation_probe::active.store(true, std::memory_order_relaxed);
        bool write_succeeded = true;
        int write_failure_errno = 0;
        try {
            anndata::write_zarr(destination, selected, write_options);
        } catch (...) {
            allocation_probe::active.store(false, std::memory_order_relaxed);
            write_succeeded = false;
            write_failure_errno = errno;
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        const auto thread_cpu_ns = ::thread_cpu_ns() - thread_cpu_start;
        getrusage(RUSAGE_SELF, &usage_stop);
        const auto process_user_cpu_ns = timeval_ns(usage_stop.ru_utime) - timeval_ns(usage_start.ru_utime);
        const auto process_system_cpu_ns = timeval_ns(usage_stop.ru_stime) - timeval_ns(usage_start.ru_stime);
        allocation_probe::active.store(false, std::memory_order_relaxed);
        if (bulk_probe_end) bulk_probe_end();
        counters.stop();
        std::printf("{\"status\":\"%s\",\"write_failure_errno\":%d,\"format\":\"%.*s\",\"selected_rows\":384,\"row_start\":1792,\"row_stop\":2176,\"columns\":4096,\"payload_bytes\":%u,\"write_seconds\":%.9f,\"thread_cpu_ns\":%llu,\"process_user_cpu_ns\":%llu,\"process_system_cpu_ns\":%llu,\"global_new_calls_during_write_zarr\":%llu,\"global_new_requested_bytes_during_write_zarr\":%llu,\"perf_available\":%s,\"perf_error_errno\":%d,\"perf_time_enabled_ns\":%llu,\"perf_time_running_ns\":%llu,\"perf_cycles_user\":%llu,\"perf_instructions_user\":%llu}\n",
            write_succeeded ? "ok" : "partial", write_failure_errno,
            static_cast<int>(format.size()), format.data(), 384u * 4096u * 4u, elapsed,
            static_cast<unsigned long long>(thread_cpu_ns),
            static_cast<unsigned long long>(process_user_cpu_ns),
            static_cast<unsigned long long>(process_system_cpu_ns),
            allocation_probe::calls.load(std::memory_order_relaxed),
            allocation_probe::requested_bytes.load(std::memory_order_relaxed),
            perf_available ? "true" : "false", counters.error,
            static_cast<unsigned long long>(counters.time_enabled),
            static_cast<unsigned long long>(counters.time_running),
            static_cast<unsigned long long>(counters.values[0]),
            static_cast<unsigned long long>(counters.values[1]));
        return write_succeeded ? 0 : 2;
    } catch (const std::exception& error) {
        allocation_probe::active.store(false, std::memory_order_relaxed);
        if (bulk_probe_end) bulk_probe_end();
        std::fprintf(stderr, "bulk_export: %s\n", error.what());
        return 1;
    }
}
