#pragma once

// Private, non-module byte storage. No AnnData or public module types cross this
// boundary; the backed adapter translates the small native scalar vocabulary.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace anndatapp_zarr {
using Json = nlohmann::json;
using Shape = std::vector<std::size_t>;
using Bytes = std::vector<std::byte>;
enum class Type { boolean, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64, string };
enum class Failure { io, encoding, version, dtype, codec, argument };
struct Error : std::runtime_error {
    Failure failure;
    std::string path;
    Error(Failure f, std::string p, std::string message) : std::runtime_error(std::move(message)), failure(f), path(std::move(p)) {}
};
struct ArrayMetadata {
    int format = 3;
    Type type = Type::float64;
    std::size_t item_size = 8;
    char endian = '=';
    Shape shape, chunks;
    Json codecs = Json::array();
    Json fill = 0;
    Json original = Json::object();
    char order = 'C';
    std::string separator = "/";
    bool default_chunk_keys = true;
};
struct Node {
    bool exists = false, group = false;
    Json attributes = Json::object();
    ArrayMetadata array;
};
struct WriteAxis {
    std::size_t count = 0;
    std::ptrdiff_t start = 0, step = 1;
    const std::size_t* positions = nullptr;
};
struct WritePoint { std::size_t physical = 0, logical = 0; };
struct NumericWrite {
    std::vector<WriteAxis> axes;
    const void* data = nullptr;
    std::vector<std::ptrdiff_t> strides;
    void* context = nullptr;
    void (*pack)(void*, const std::vector<std::vector<std::size_t>>&, void*) = nullptr;
    std::size_t scratch_bytes = 0, packing_metadata_bytes = 0;
};
struct Chunk {
    Bytes numeric;
    std::vector<std::string> strings;
};
struct PayloadMeta {
    const ArrayMetadata& array;
    const Json& codecs;
    std::size_t scratch_bytes;
};
struct NumericCodec {
    enum class Kind { none, gzip, zlib, zstd, blosc } kind = Kind::none;
    int level = 0;
    bool checksum = false;
    std::size_t item_size = 1;
    Json configuration = Json::object();
};
std::size_t checked_product(const Shape& shape);
std::size_t type_size(Type type);
// The payload codec excludes sharding; shape describes the complete logical
// inner chunk, including padding beyond an array edge. Returned data are C
// contiguous and native endian.
Chunk decode_payload(const PayloadMeta&, const Shape&, std::span<const std::byte>);
Bytes encode_payload(const PayloadMeta&, const Shape&, const Chunk&);
// Reuses caller-owned buffers for a single supported numerical byte codec.
NumericCodec prepare_numeric_codec(const PayloadMeta&, const Shape&);
std::size_t reusable_encoded_bound(const NumericCodec&, std::size_t decoded_size, std::size_t limit);
bool reusable_needs_block(const NumericCodec&, bool writing);
void decode_numeric_reusable(const NumericCodec&, std::size_t expected, std::size_t limit, std::span<const std::byte>, Bytes&, Bytes&);
void encode_numeric_reusable(const NumericCodec&, std::size_t limit, std::span<const std::byte>, Bytes&, Bytes&);

struct ShardSpec {
    Shape shard_shape, chunk_shape;
    Json codecs, index_codecs;
    std::string index_location = "end";
    std::size_t scratch_bytes = 64 * 1024 * 1024;
};
// Empty means absent. Reads use ranges; replacing one inner chunk copies the
// remaining encoded payloads through a bounded scratch buffer.
Bytes read_shard_chunk(const std::filesystem::path&, const ShardSpec&, const Shape& local_chunk);
void write_shard_chunk(const std::filesystem::path&, const ShardSpec&, const Shape& local_chunk, std::span<const std::byte>);

// Admission for the fixed-size index, excluding caller-owned payload and copy
// buffers. Call once before writing any shards. The immutable spec and one
// exclusively used workspace must outlive the writer. Each local chunk may be
// replaced only once; the original shard changes only at commit().
std::size_t shard_index_workspace(const ShardSpec&, const std::filesystem::path& context = {});
void validate_index_pipeline(const PayloadMeta&, const Shape&);
std::uint32_t shard_crc32c(std::span<const std::byte>);
struct ShardWorkspace {
    Bytes index, encoded;
    std::size_t allocations = 0;
    std::size_t capacity() const noexcept { return index.capacity() + encoded.capacity(); }
};
class ShardWriter {
public:
    ShardWriter(const std::filesystem::path&, const ShardSpec&, ShardWorkspace&);
    ShardWriter(const std::filesystem::path&, ShardSpec&&, ShardWorkspace&) = delete;
    ~ShardWriter();
    ShardWriter(const ShardWriter&) = delete;
    ShardWriter& operator=(const ShardWriter&) = delete;
    bool read(const Shape& local, Bytes& encoded, std::size_t max_bytes);
    void replace(const Shape& local, std::span<const std::byte> encoded);
    void commit(Bytes& copy_buffer);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct CreateOptions {
    int format = 3;
    std::size_t target_chunk_bytes = 1024 * 1024;
    std::string compression = "zstd";
    int compression_level = 3;
};
class Store {
public:
    Store(std::filesystem::path root, bool writable, std::size_t scratch_bytes = 64 * 1024 * 1024);
    const std::filesystem::path& root() const noexcept { return root_; }
    Node inspect(const std::string&) const;
    // Prepared metadata remains valid while callers exclude structural updates.
    ArrayMetadata prepare_numeric(const std::string&);
    NumericCodec numeric_codec(const ArrayMetadata&) const;
    std::size_t retained_scratch_bytes() const noexcept { return numeric_scratch_.capacity()+encoded_scratch_.capacity()+codec_scratch_.capacity()+packed_scratch_.capacity()+shard_scratch_.capacity()+selection_scratch_.capacity()*sizeof(WritePoint); }
    std::size_t scratch_allocations() const noexcept { return numeric_scratch_allocations_+shard_scratch_.allocations; }
    void read_numeric(const std::string&, const ArrayMetadata&, const NumericCodec&, const Shape&, const Shape&, void*) const;
    void write_numeric(const std::string&, const ArrayMetadata&, const NumericCodec&, const Shape&, const Shape&, const void*);
    bool write_selection(const std::string&, const ArrayMetadata&, const NumericWrite&);
    std::vector<std::string> children(const std::string&) const;
    void read_numeric(const std::string&, const Shape& start, const Shape& count, void*) const;
    std::vector<std::string> read_strings(const std::string&, const Shape& start, const Shape& count) const;
    std::vector<std::string> read_strings_bounded(const std::string&, const Shape& start,
                                                const Shape& count, std::size_t byte_budget) const;
    void write_numeric(const std::string&, const Shape& start, const Shape& count, const void*);
    void write_strings(const std::string&, const Shape& start, const Shape& count, const std::vector<std::string>&);
    void create_group(const std::string&, const Json& attributes, int format);
    void create_array(const std::string&, Type, const Shape&, const Json& attributes, char endian, const CreateOptions&);
    void attributes(const std::string&, const Json&);
    void resize(const std::string&, const Shape&);
    void erase(const std::string&);
    void rename(const std::string& source, const std::string& target);
    void flush();
    // Direct native, uncompressed, one-chunk arrays can be mapped by the adapter.
    std::filesystem::path mapping_file(const std::string&) const;
private:
    std::filesystem::path root_;
    bool writable_;
    std::size_t scratch_bytes_;
    // This stores only reusable capacity, never a cached chunk identity.
    mutable Bytes numeric_scratch_;
    mutable Bytes encoded_scratch_;
    mutable Bytes codec_scratch_;
    mutable Bytes packed_scratch_;
    mutable ShardWorkspace shard_scratch_;
    mutable std::vector<WritePoint> selection_scratch_;
    mutable std::size_t numeric_scratch_allocations_ = 0;
    mutable bool consolidation_v2_ = false, consolidation_v3_ = false;
    bool metadata_dirty_ = false;
    std::filesystem::path node_path(const std::string&) const;
    void mutation();
    Chunk chunk(const std::string&, const ArrayMetadata&, const Shape&) const;
    void chunk(const std::string&, const ArrayMetadata&, const Shape&, const Chunk&);
    void transfer(const std::string&, const Shape&, const Shape&, void*, std::vector<std::string>*, bool) const;
    void transfer(const std::string&, const ArrayMetadata&, const Shape&, const Shape&, void*, std::vector<std::string>*, bool, const NumericCodec* codec = nullptr) const;
};
} // namespace anndatapp_zarr
