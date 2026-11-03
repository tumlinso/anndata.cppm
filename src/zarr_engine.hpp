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

namespace anndata_zarr {
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
    std::size_t scratch_bytes = 64 * 1024 * 1024;
    char endian = '=';
    Shape shape, chunks;
    Json codecs = Json::array();
    Json fill = 0;
    Json attributes = Json::object();
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
struct Chunk {
    Bytes numeric;
    std::vector<std::string> strings;
};
std::size_t checked_product(const Shape& shape);
std::size_t type_size(Type type);
// The payload codec excludes sharding; shape describes the complete logical
// inner chunk, including padding beyond an array edge. Returned data are C
// contiguous and native endian.
Chunk decode_payload(const ArrayMetadata&, const Shape&, std::span<const std::byte>);
Bytes encode_payload(const ArrayMetadata&, const Shape&, const Chunk&);

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
    mutable bool consolidation_v2_ = false, consolidation_v3_ = false;
    std::filesystem::path node_path(const std::string&) const;
    void mutation();
    Chunk chunk(const std::string&, const ArrayMetadata&, const Shape&) const;
    void chunk(const std::string&, const ArrayMetadata&, const Shape&, const Chunk&);
    void transfer(const std::string&, const Shape&, const Shape&, void*, std::vector<std::string>*, bool) const;
};
} // namespace anndata_zarr
