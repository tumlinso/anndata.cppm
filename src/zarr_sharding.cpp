#include "zarr_engine.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace anndata_zarr {
namespace {

constexpr auto missing = std::numeric_limits<std::uint64_t>::max();

[[noreturn]] void fail(Failure failure, const std::filesystem::path& path, const std::string& message) {
    throw Error(failure, path.string(), message);
}

[[noreturn]] void syserr(const std::filesystem::path& path, const std::string& operation) {
    fail(Failure::io, path, operation + ": " + std::strerror(errno));
}

struct File {
    int fd = -1;
    File() = default;
    explicit File(int value) : fd(value) {}
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { if (fd >= 0) ::close(fd); }
};

struct TempFile {
    File file;
    std::filesystem::path path;
    bool committed = false;
    ~TempFile() { if (!committed && !path.empty()) ::unlink(path.c_str()); }
};

std::uint64_t file_size(int fd, const std::filesystem::path& path) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) syserr(path, "stat shard");
    if (!S_ISREG(st.st_mode) || st.st_size < 0) fail(Failure::io, path, "shard is not a regular file");
    return static_cast<std::uint64_t>(st.st_size);
}

void read_at(int fd, std::uint64_t offset, std::span<std::byte> bytes, const std::filesystem::path& path) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        bytes.size() > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) - offset)
        fail(Failure::encoding, path, "shard range exceeds supported file offsets");
    std::size_t done = 0;
    while (done < bytes.size()) {
        auto n = ::pread(fd, bytes.data() + done, std::min(bytes.size() - done, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())), static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) syserr(path, "read shard");
        if (!n) fail(Failure::io, path, "truncated shard");
        done += static_cast<std::size_t>(n);
    }
}

void write_at(int fd, std::uint64_t offset, std::span<const std::byte> bytes, const std::filesystem::path& path) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        bytes.size() > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) - offset)
        fail(Failure::io, path, "shard exceeds supported file offsets");
    std::size_t done = 0;
    while (done < bytes.size()) {
        auto n = ::pwrite(fd, bytes.data() + done, std::min(bytes.size() - done, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())), static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) syserr(path, "write shard");
        if (!n) fail(Failure::io, path, "short shard write");
        done += static_cast<std::size_t>(n);
    }
}

struct IndexLayout {
    Shape shape;
    std::size_t count = 1;
    std::size_t encoded_size = 0;
    std::size_t selected = 0;
    ArrayMetadata metadata;
};

IndexLayout layout(const ShardSpec& spec, const Shape& local, const std::filesystem::path& path) {
    if (spec.shard_shape.size() != spec.chunk_shape.size() || local.size() != spec.shard_shape.size())
        fail(Failure::encoding, path, "shard and inner chunk ranks do not match");
    if (spec.index_location != "start" && spec.index_location != "end")
        fail(Failure::encoding, path, "invalid shard index location");
    IndexLayout result;
    for (std::size_t d = 0; d < local.size(); ++d) {
        if (!spec.chunk_shape[d] || !spec.shard_shape[d] || spec.shard_shape[d] % spec.chunk_shape[d])
            fail(Failure::encoding, path, "shard shape must be divisible by inner chunk shape");
        auto size = spec.shard_shape[d] / spec.chunk_shape[d];
        if (local[d] >= size) fail(Failure::argument, path, "inner chunk coordinate outside shard");
        if (result.count > std::numeric_limits<std::size_t>::max() / size)
            fail(Failure::encoding, path, "shard index dimensions overflow");
        result.count *= size;
        result.selected = result.selected * size + local[d];
        result.shape.push_back(size);
    }
    result.shape.push_back(2);
    if (result.count > (std::numeric_limits<std::size_t>::max() - 4) / 16)
        fail(Failure::encoding, path, "shard index size overflow");
    result.encoded_size = result.count * 16;
    // The fixed-size index pipeline is bounded without guessing compression
    // output sizes. Transpose is fixed size, bytes is required, and crc32c adds
    // four bytes per occurrence.
    if (!spec.index_codecs.is_array()) fail(Failure::encoding, path, "shard index codecs must be an array");
    bool has_bytes = false;
    for (const auto& codec : spec.index_codecs) {
        if (!codec.is_object() || !codec.contains("name") || !codec["name"].is_string())
            fail(Failure::encoding, path, "invalid shard index codec");
        const auto name = codec["name"].get<std::string>();
        if (name == "transpose" && !has_bytes) continue;
        if (name == "bytes" && !has_bytes) { has_bytes = true; continue; }
        if (name == "crc32c" && has_bytes) {
            if (result.encoded_size > std::numeric_limits<std::size_t>::max() - 4)
                fail(Failure::encoding, path, "shard index size overflow");
            result.encoded_size += 4;
            continue;
        }
        fail(Failure::codec, path, "unsupported or misplaced shard index codec: " + name);
    }
    if (!has_bytes) fail(Failure::encoding, path, "shard index requires a bytes codec");
    // The payload codec may hold both encoded and decoded buffers as well as
    // temporary conversion buffers. Bound index work before allocation.
    const auto index_bytes = result.count * 16;
    if (result.encoded_size > spec.scratch_bytes || index_bytes > (spec.scratch_bytes - result.encoded_size) / 3)
        fail(Failure::argument, path, "shard index exceeds scratch memory budget");
    result.metadata.format = 3;
    result.metadata.type = Type::uint64;
    result.metadata.item_size = 8;
    return result;
}

std::uint64_t get(const Chunk& index, std::size_t entry, std::size_t field) {
    std::uint64_t value;
    std::memcpy(&value, index.numeric.data() + (entry * 2 + field) * sizeof(value), sizeof(value));
    return value;
}
void set(Chunk& index, std::size_t entry, std::uint64_t offset, std::uint64_t size) {
    std::memcpy(index.numeric.data() + (entry * 2) * sizeof(offset), &offset, sizeof(offset));
    std::memcpy(index.numeric.data() + (entry * 2 + 1) * sizeof(size), &size, sizeof(size));
}

Chunk read_index(int fd, std::uint64_t size, const IndexLayout& layout, const ShardSpec& spec, const std::filesystem::path& path) {
    if (size < layout.encoded_size) fail(Failure::encoding, path, "shard is shorter than its index");
    Bytes encoded(layout.encoded_size);
    auto position = spec.index_location == "start" ? 0 : size - layout.encoded_size;
    read_at(fd, position, encoded, path);
    Chunk index;
    try { index = decode_payload({layout.metadata, spec.index_codecs, spec.scratch_bytes}, layout.shape, encoded); }
    catch (const Error& e) { throw Error(e.failure, path.string(), e.what()); }
    if (index.numeric.size() != layout.count * 16) fail(Failure::encoding, path, "incorrect shard index length");
    const auto data_start = spec.index_location == "start" ? layout.encoded_size : 0;
    const auto data_end = spec.index_location == "end" ? size - layout.encoded_size : size;
    for (std::size_t i = 0; i < layout.count; ++i) {
        auto offset = get(index, i, 0), length = get(index, i, 1);
        if (offset == missing && length == missing) continue;
        if (offset == missing || length == missing || offset < data_start || offset > data_end || length > data_end - offset)
            fail(Failure::encoding, path, "invalid shard index entry");
    }
    return index;
}

void open_existing(File& file, const std::filesystem::path& path) {
    file.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file.fd < 0 && errno != ENOENT) syserr(path, "open shard");
}

} // namespace

Bytes read_shard_chunk(const std::filesystem::path& path, const ShardSpec& spec, const Shape& local_chunk) {
    auto index_layout = layout(spec, local_chunk, path);
    File file;
    open_existing(file, path);
    if (file.fd < 0) return {};
    auto index = read_index(file.fd, file_size(file.fd, path), index_layout, spec, path);
    auto offset = get(index, index_layout.selected, 0), length = get(index, index_layout.selected, 1);
    if (offset == missing) return {};
    if (!length) fail(Failure::encoding, path, "empty encoded inner chunk");
    if (length > std::numeric_limits<std::size_t>::max()) fail(Failure::encoding, path, "chunk exceeds addressable memory");
    if (length > spec.scratch_bytes / 2) fail(Failure::argument, path, "encoded inner chunk exceeds scratch memory budget");
    Bytes result(static_cast<std::size_t>(length));
    read_at(file.fd, offset, result, path);
    return result;
}

void write_shard_chunk(const std::filesystem::path& path, const ShardSpec& spec, const Shape& local_chunk, std::span<const std::byte> payload) {
    auto index_layout = layout(spec, local_chunk, path);
    File old;
    open_existing(old, path);
    Chunk index;
    if (old.fd >= 0) index = read_index(old.fd, file_size(old.fd, path), index_layout, spec, path);
    else index.numeric.assign(index_layout.count * 16, std::byte{0xff});

    std::error_code ec;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) fail(Failure::io, path, "create shard directory: " + ec.message());
    auto pattern = path.string() + ".tmp-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    TempFile temp;
    temp.file.fd = ::mkstemp(name.data());
    if (temp.file.fd < 0) syserr(path, "create temporary shard");
    temp.path = name.data();
    // Replacement must not silently remove existing group or other read
    // permissions just because mkstemp creates a private temporary file.
    if (old.fd >= 0) {
        struct stat previous{};
        if (::fstat(old.fd, &previous) != 0) syserr(path, "stat existing shard");
        if (::fchmod(temp.file.fd, previous.st_mode & 07777) != 0) syserr(path, "preserve shard permissions");
    }

    // Leave space for a leading index; pwrite writes it once after payloads.
    std::uint64_t output_offset = spec.index_location == "start" ? index_layout.encoded_size : 0;
    const auto index_reserved = index_layout.encoded_size + index_layout.count * 16 * 3;
    const auto copy_size = std::min<std::size_t>(1024 * 1024, spec.scratch_bytes - index_reserved);
    if (!copy_size) fail(Failure::argument, path, "scratch memory budget leaves no shard copy buffer");
    Bytes buffer(copy_size);
    for (std::size_t i = 0; i < index_layout.count; ++i) {
        auto offset = get(index, i, 0), length = get(index, i, 1);
        if (i == index_layout.selected) {
            if (payload.empty()) { set(index, i, missing, missing); continue; }
            if (payload.size() > std::numeric_limits<std::uint64_t>::max() - output_offset) fail(Failure::io, path, "shard size overflow");
            write_at(temp.file.fd, output_offset, payload, temp.path);
            set(index, i, output_offset, payload.size());
            output_offset += payload.size();
            continue;
        }
        if (offset == missing) continue;
        if (length > std::numeric_limits<std::uint64_t>::max() - output_offset) fail(Failure::io, path, "shard size overflow");
        const auto new_offset = output_offset;
        for (std::uint64_t done = 0; done < length;) {
            auto amount = static_cast<std::size_t>(std::min<std::uint64_t>(length - done, buffer.size()));
            read_at(old.fd, offset + done, std::span<std::byte>(buffer.data(), amount), path);
            write_at(temp.file.fd, output_offset, std::span<const std::byte>(buffer.data(), amount), temp.path);
            output_offset += amount;
            done += amount;
        }
        set(index, i, new_offset, length);
    }
    buffer.clear();
    buffer.shrink_to_fit();
    Bytes encoded;
    try { encoded = encode_payload({index_layout.metadata, spec.index_codecs, spec.scratch_bytes}, index_layout.shape, index); }
    catch (const Error& e) { throw Error(e.failure, path.string(), e.what()); }
    if (encoded.size() != index_layout.encoded_size) fail(Failure::encoding, path, "incorrect encoded shard index length");
    write_at(temp.file.fd, spec.index_location == "start" ? 0 : output_offset, encoded, temp.path);
    if (::rename(temp.path.c_str(), path.c_str()) != 0) syserr(path, "replace shard");
    temp.committed = true;
}

} // namespace anndata_zarr
