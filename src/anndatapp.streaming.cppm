module;
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

export module anndatapp.streaming;
export import anndatapp.backed;

export namespace anndatapp {
enum class StreamMode { adaptive, manual };
enum class StreamOperation { read, write };
enum class StreamPhase { baseline, trial, stable, manual };
enum class StreamLimit { none, manual, backend_serialized, insufficient_samples, no_overlap_evidence, capacity };
struct StreamOptions {
    StreamMode mode = StreamMode::adaptive;
    StreamOperation operation = StreamOperation::read;
    std::size_t min_block_bytes = 64u * 1024u;
    std::size_t target_block_bytes = 1024u * 1024u;
    std::size_t max_block_bytes = 8u * 1024u * 1024u;
    std::size_t max_in_flight = 1;
    std::size_t scratch_bytes_per_worker = 64u * 1024u * 1024u;
    // Sparse output size needs TransferContext::requirements(). A rejected,
    // unexecuted request can be split explicitly without advancing coverage.
    std::size_t sparse_major_count = 1;
    bool statistics = false;
};
struct BlockRequest {
    std::uint64_t id = 0;
    std::vector<std::size_t> start, count;
    // Source physical unit coordinates; empty for a nonphysical read plan.
    std::vector<std::size_t> conflict_start, conflict_count;
    std::size_t estimated_bytes = 0;
};
struct BlockCompletion {
    std::size_t useful_bytes = 0;
    double io_seconds = 0;
    // Optional common monotonic timebase. Supply both for overlap-aware
    // concurrency tuning; elapsed durations alone never establish overlap.
    double started_seconds = 0, finished_seconds = 0;
};
struct StreamSettings {
    StreamMode mode = StreamMode::adaptive;
    StreamPhase phase = StreamPhase::baseline;
    StreamLimit limit = StreamLimit::insufficient_samples;
    std::size_t target_block_bytes = 0, in_flight_limit = 1;
    std::size_t reserved_workers = 1, output_bytes_per_worker = 0;
    std::size_t scratch_bytes_per_worker = 0, aggregate_reserved_bytes = 0;
    bool backend_serialized = false, overlap_timing_available = false;
};
struct StreamStats {
    std::size_t issued = 0, completed = 0, useful_bytes = 0;
    std::size_t trials = 0, rollbacks = 0, adaptations = 0, split_requests = 0;
    double io_seconds = 0, throughput_bytes_per_second = 0;
};
namespace detail { struct BlockStreamImpl; }
// One caller serializes issuance/completion. Worker execution remains external.
// Geometry is an immutable snapshot; recreate the stream after structural edits.
class BlockStream {
public:
    explicit BlockStream(StorageGeometry, StreamOptions = {});
    ~BlockStream();
    BlockStream(BlockStream&&) noexcept;
    BlockStream& operator=(BlockStream&&) noexcept;
    BlockStream(const BlockStream&) = delete;
    BlockStream& operator=(const BlockStream&) = delete;
    std::optional<BlockRequest> next();
    void complete(const BlockRequest&, const BlockCompletion&);
    void complete(const BlockRequest&, std::size_t useful_bytes, double io_seconds = 0);
    // Only before executing a sparse request. Its next issuance has fewer
    // major-axis rows/columns; no payload is consumed or silently completed.
    void split(const BlockRequest&, std::size_t max_major_count);
    // Drain outstanding requests first. Initial reserved capacities cannot grow.
    void configure(StreamOptions);
    StreamSettings settings() const;
    StreamStats stats() const;
    bool finished() const;
private:
    std::unique_ptr<detail::BlockStreamImpl> impl_;
};
}
