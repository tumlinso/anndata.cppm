#pragma once
#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace v1 {
using Clock = std::chrono::steady_clock;
struct Counters {
    std::uint64_t allocations = 0, requested_bytes = 0;
    std::uint64_t read_calls = 0, read_bytes = 0, write_calls = 0, write_bytes = 0;
    std::array<std::uint64_t, 12> codec_calls{};
};
struct Sample {
    double seconds = 0;
    Counters counters;
    std::uint64_t operation_cpu_ns = 0, codec_context_cpu_ns = 0, codec_work_cpu_ns = 0;
    std::int64_t stream_trials = -1, stream_adaptations = -1, stream_rollbacks = -1, stream_final_limit = -1;
};
// Setup, initialization and correctness checks live outside measure(). The
// operation includes the lifetime/destruction of any temporary it constructs.
void begin();
Sample end(Clock::time_point);
void emit(const char* operation, std::size_t iterations, const Sample&,
          std::size_t payload_bytes = 0, std::int64_t backend_calls = -1,
          std::int64_t backend_bytes = -1, const char* acquisition = "same_api");
void require(bool, const char*);
extern volatile std::uint64_t sink;
extern bool validation_only;
extern bool diagnostic;
template<class F> Sample measure(F f) {
    if (validation_only) { f(); return {}; }
    if (diagnostic) begin();
    const auto first = Clock::now();
    f();
    return end(first);
}
void core_case(const std::string&);
void storage_case(const std::string&, const std::string&);
}
