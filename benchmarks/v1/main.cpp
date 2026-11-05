#include "harness.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <time.h>
#include <dlfcn.h>

namespace {
using Begin = void (*)();
using End = void (*)(std::uint64_t*);
Begin v1_io_begin = nullptr, v1_alloc_begin = nullptr;
End v1_io_end = nullptr, v1_alloc_end = nullptr, v1_codec_cpu_end = nullptr;
std::uint64_t peak_rss(const rusage& usage) {
#ifdef __APPLE__
    return (std::uint64_t(usage.ru_maxrss)+1023)/1024;
#else
    return usage.ru_maxrss;
#endif
}
}

namespace v1 {
volatile std::uint64_t sink = 0;
bool validation_only = false;
bool diagnostic = false;
std::uint64_t cpu_first = 0;
std::uint64_t cpu_stamp() {
    timespec t{}; return clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t) ? 0 : std::uint64_t(t.tv_sec)*1000000000ULL+std::uint64_t(t.tv_nsec);
}
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void begin() {
    if (v1_io_begin) v1_io_begin();
    if (v1_alloc_begin) v1_alloc_begin();
    cpu_first = cpu_stamp();
}
Sample end(Clock::time_point first) {
    const auto last = Clock::now();
    if (!diagnostic) return {std::chrono::duration<double>(last-first).count(),{}};
    const auto cpu_last = cpu_stamp();
    std::uint64_t allocation[2]{};
    if (v1_alloc_end) v1_alloc_end(allocation);
    std::uint64_t io[16]{};
    if (v1_io_end) v1_io_end(io);
    Sample s;
    s.seconds = 0;
    s.counters.allocations = allocation[0]; s.counters.requested_bytes = allocation[1];
    s.operation_cpu_ns = cpu_last-cpu_first;
    std::uint64_t codec_cpu[2]{};
    if (v1_codec_cpu_end) v1_codec_cpu_end(codec_cpu);
    s.codec_context_cpu_ns = codec_cpu[0]; s.codec_work_cpu_ns = codec_cpu[1];
    s.counters.read_calls = io[0]; s.counters.read_bytes = io[1];
    s.counters.write_calls = io[2]; s.counters.write_bytes = io[3];
    for (int i = 0; i < 12; ++i) s.counters.codec_calls[i] = io[i + 4];
    return s;
}
void emit(const char* operation, std::size_t iterations, const Sample& s,
          std::size_t payload, std::int64_t backend_calls,
          std::int64_t backend_bytes, const char* acquisition) {
    rusage usage{}; require(!getrusage(RUSAGE_SELF, &usage), "getrusage failed");
    std::cout << std::setprecision(12) << "{\"operation\":\"" << operation
              << "\",\"iterations\":" << iterations << ",\"seconds\":" << s.seconds
              << ",\"new_calls\":" << s.counters.allocations
              << ",\"new_requested_bytes\":" << s.counters.requested_bytes
              << ",\"diagnostic_operation_cpu_ns\":" << s.operation_cpu_ns
              << ",\"diagnostic_zstd_context_cpu_ns\":" << s.codec_context_cpu_ns
              << ",\"diagnostic_zstd_work_cpu_ns\":" << s.codec_work_cpu_ns
              << ",\"stream_trials\":" << s.stream_trials
              << ",\"stream_adaptations\":" << s.stream_adaptations
              << ",\"stream_rollbacks\":" << s.stream_rollbacks
              << ",\"stream_final_limit\":" << s.stream_final_limit
              << ",\"process_peak_rss_kib\":" << peak_rss(usage)
              << ",\"payload_bytes\":" << payload << ",\"acquisition\":\"" << acquisition
              << "\",\"backend_calls\":";
    if (backend_calls < 0) std::cout << "null"; else std::cout << backend_calls;
    std::cout << ",\"backend_bytes\":";
    if (backend_bytes < 0) std::cout << "null"; else std::cout << backend_bytes;
    std::cout << ",\"libc_read_calls\":" << s.counters.read_calls
              << ",\"libc_read_bytes\":" << s.counters.read_bytes
              << ",\"libc_write_calls\":" << s.counters.write_calls
              << ",\"libc_write_bytes\":" << s.counters.write_bytes << ",\"codec_calls\":[";
    for (int i = 0; i < 12; ++i) std::cout << (i ? "," : "") << s.counters.codec_calls[i];
    std::cout << "],\"validated\":true}\n";
}
}
int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) { std::cerr << "usage: v1_benchmark SCENARIO FIXTURE_DIR [--validate-only|--diagnostic]\n"; return 2; }
    try {
        v1::validation_only = argc == 4 && std::string(argv[3]) == "--validate-only";
        v1::diagnostic = argc == 4 && std::string(argv[3]) == "--diagnostic";
        if (argc == 4 && !v1::validation_only && !v1::diagnostic) throw std::runtime_error("unknown option");
        if (v1::diagnostic) {
            v1_io_begin = reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"v1_io_begin"));
            v1_alloc_begin = reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"v1_alloc_begin"));
            v1_io_end = reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"v1_io_end"));
            v1_alloc_end = reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"v1_alloc_end"));
            v1_codec_cpu_end = reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"v1_codec_cpu_end"));
            v1::require(v1_io_begin && v1_alloc_begin && v1_io_end && v1_alloc_end && v1_codec_cpu_end,
                        "diagnostics require both injected observers");
        }
        const std::string name = argv[1];
        if (name == "h5ad" || name == "zarr2" || name == "zarr3" || name == "contexts")
            v1::storage_case(name, argv[2]);
        else v1::core_case(name);
        rusage usage{}; v1::require(!getrusage(RUSAGE_SELF,&usage),"getrusage failed");
        std::cout << "{\"process_summary\":true,\"process_peak_rss_kib\":" << peak_rss(usage) << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
