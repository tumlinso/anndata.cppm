#include "zarr_engine.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace anndatapp_zarr {
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

void check_index(const Chunk& index, std::uint64_t size, const IndexLayout& layout,
                 const ShardSpec& spec, const std::filesystem::path& path) {
    if (index.numeric.size() != layout.count * 16) fail(Failure::encoding, path, "incorrect shard index length");
    const auto data_start = spec.index_location == "start" ? layout.encoded_size : 0;
    const auto data_end = spec.index_location == "end" ? size - layout.encoded_size : size;
    for (std::size_t i = 0; i < layout.count; ++i) {
        auto offset = get(index, i, 0), length = get(index, i, 1);
        if (offset == missing && length == missing) continue;
        if (offset == missing || length == missing || offset < data_start || offset > data_end || length > data_end - offset)
            fail(Failure::encoding, path, "invalid shard index entry");
    }
}
Chunk read_index(int fd, std::uint64_t size, const IndexLayout& layout, const ShardSpec& spec, const std::filesystem::path& path) {
    if (size < layout.encoded_size) fail(Failure::encoding, path, "shard is shorter than its index");
    const auto bytes = std::max(layout.encoded_size, layout.count * std::size_t{16});
    if (bytes > std::numeric_limits<std::size_t>::max() / 4 || bytes * 4 > spec.scratch_bytes)
        fail(Failure::argument, path, "shard index exceeds scratch memory budget");
    Bytes encoded(layout.encoded_size);
    auto position = spec.index_location == "start" ? 0 : size - layout.encoded_size;
    read_at(fd, position, encoded, path);
    Chunk index;
    try { index = decode_payload({layout.metadata, spec.index_codecs, spec.scratch_bytes}, layout.shape, encoded); }
    catch (const Error& e) { throw Error(e.failure, path.string(), e.what()); }
    check_index(index, size, layout, spec, path);
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

std::size_t shard_index_workspace(const ShardSpec& spec, const std::filesystem::path& context) {
    auto info = layout(spec, Shape(spec.shard_shape.size(), 0), context);
    try { validate_index_pipeline({info.metadata, spec.index_codecs, spec.scratch_bytes}, info.shape); }
    catch (const Error& e) { throw Error(e.failure, context.string(), e.what()); }
    const auto bytes = std::max(info.encoded_size, info.count * std::size_t{16});
    const bool simple = spec.index_codecs.size() <= 2 && spec.index_codecs.front().value("name",std::string()) == "bytes";
    const std::size_t copies = simple ? 1 : 4;
    if (bytes > std::numeric_limits<std::size_t>::max() / copies || bytes * copies > spec.scratch_bytes)
        fail(Failure::argument, context, "shard index exceeds scratch memory budget");
    // The in-place bytes/CRC path uses one buffer; the general codec has a
    // bounded input, working copy and up to two simultaneous temporary copies.
    return bytes * copies;
}

namespace {
constexpr std::uint64_t replaced = std::uint64_t{1} << 63;
struct OutputFile {
    File file;
    std::string name;
    bool committed = false;
    ~OutputFile() { if (file.fd >= 0 && !committed) ::unlink(name.c_str()); }
};
}

struct ShardWriter::Impl {
    std::filesystem::path path;
    const ShardSpec& spec;
    ShardWorkspace& workspace;
    IndexLayout info;
    File old;
    OutputFile output;
    Chunk index;
    std::uint64_t next = 0;
    bool simple = false, crc = false, swap = false;

    Impl(const std::filesystem::path& p, const ShardSpec& s, ShardWorkspace& work)
        : path(p), spec(s), workspace(work), info(layout(s, Shape(s.shard_shape.size(), 0), p)) {
        simple = spec.index_codecs.size() <= 2 && spec.index_codecs.front().value("name",std::string()) == "bytes";
        crc = simple && spec.index_codecs.size() == 2;
        const auto bytes = std::max(info.encoded_size, info.count * std::size_t{16});
        const std::size_t copies = simple ? 1 : 4;
        if (bytes > std::numeric_limits<std::size_t>::max()/copies || bytes*copies > spec.scratch_bytes)
            fail(Failure::argument, path, "shard index exceeds scratch memory budget");
        if (simple) {
            const auto& codec = spec.index_codecs.front();
            const auto& cfg = codec.contains("configuration") ? codec.at("configuration") : codec;
            const auto endian = cfg.at("endian").get<std::string>();
            swap = (endian == "little") != (std::endian::native == std::endian::little);
        }
        index.numeric = std::move(workspace.index);
        open_existing(old, path);
        if (old.fd >= 0) {
            const auto size = file_size(old.fd, path);
            if (size < info.encoded_size) fail(Failure::encoding, path, "shard is shorter than its index");
            if (simple) {
                resize(bytes);
                read_at(old.fd, spec.index_location == "start" ? 0 : size - info.encoded_size, index.numeric, path);
                if (crc) {
                    const auto n = index.numeric.size()-4;
                    const auto* q = index.numeric.data()+n;
                    const auto stored = std::uint32_t(std::to_integer<unsigned char>(q[0])) |
                        (std::uint32_t(std::to_integer<unsigned char>(q[1])) << 8) |
                        (std::uint32_t(std::to_integer<unsigned char>(q[2])) << 16) |
                        (std::uint32_t(std::to_integer<unsigned char>(q[3])) << 24);
                    if (shard_crc32c(std::span(index.numeric.data(),n)) != stored)
                        fail(Failure::encoding, path, "CRC32C checksum mismatch");
                    index.numeric.resize(n);
                }
                if (swap) byte_swap();
                check_index(index, size, info, spec, path);
            } else {
                Bytes().swap(index.numeric);
                index = read_index(old.fd, size, info, spec, path);
                if (!index.numeric.empty()) ++workspace.allocations;
            }
        } else {
            resize(bytes);
            index.numeric.resize(info.count*16);
            std::fill(index.numeric.begin(),index.numeric.end(),std::byte{0xff});
        }
        std::error_code ec;
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) fail(Failure::io, path, "create shard directory: " + ec.message());
        output.name = path.string() + ".tmp-XXXXXX";
        output.file.fd = ::mkstemp(output.name.data());
        if (output.file.fd < 0) syserr(path, "create temporary shard");
        if (old.fd >= 0) {
            struct stat previous{};
            if (::fstat(old.fd, &previous) != 0) syserr(path, "stat existing shard");
            if (::fchmod(output.file.fd, previous.st_mode & 07777) != 0)
                syserr(path, "preserve shard permissions");
        }
        next = spec.index_location == "start" ? info.encoded_size : 0;
    }
    ~Impl() { workspace.index = std::move(index.numeric); }
    void resize(std::size_t bytes) {
        if (index.numeric.capacity() > bytes || index.numeric.capacity() < bytes) {
            Bytes().swap(index.numeric);
            index.numeric.resize(bytes);
            ++workspace.allocations;
        } else index.numeric.resize(bytes);
        if (index.numeric.capacity() > bytes) {
            Bytes().swap(index.numeric);
            fail(Failure::argument, path, "shard index exceeds scratch memory budget");
        }
    }
    void byte_swap() {
        for (std::size_t i = 0; i < index.numeric.size(); i += 8)
            std::reverse(index.numeric.begin()+i, index.numeric.begin()+i+8);
    }
    std::size_t entry(const Shape& local) const {
        if (local.size() + 1 != info.shape.size()) fail(Failure::argument, path, "inner chunk rank mismatch");
        std::size_t n = 0;
        for (std::size_t i = 0; i < local.size(); ++i) {
            if (local[i] >= info.shape[i]) fail(Failure::argument, path, "inner chunk coordinate outside shard");
            n = n * info.shape[i] + local[i];
        }
        return n;
    }
    void check() const {
        if (output.committed) fail(Failure::argument, path, "shard writer already committed");
    }
};

ShardWriter::ShardWriter(const std::filesystem::path& path, const ShardSpec& spec, ShardWorkspace& workspace)
    : impl_(std::make_unique<Impl>(path, spec, workspace)) {}
ShardWriter::~ShardWriter() = default;

bool ShardWriter::read(const Shape& local, Bytes& encoded, std::size_t max_bytes) {
    auto& w = *impl_; w.check();
    const auto i = w.entry(local), offset = get(w.index, i, 0), length = get(w.index, i, 1);
    if (offset == missing) { encoded.clear(); return false; }
    if (offset & replaced) fail(Failure::argument, w.path, "inner chunk has already been replaced");
    if (!length) fail(Failure::encoding, w.path, "empty encoded inner chunk");
    if (length > std::numeric_limits<std::size_t>::max() || length > max_bytes)
        fail(Failure::argument, w.path, "encoded inner chunk exceeds scratch memory budget");
    const auto bytes = static_cast<std::size_t>(length);
    if (encoded.capacity() > max_bytes || encoded.capacity() < bytes) Bytes().swap(encoded);
    encoded.resize(bytes);
    if (encoded.capacity() > max_bytes) {
        Bytes().swap(encoded);
        fail(Failure::argument, w.path, "encoded inner chunk exceeds scratch memory budget");
    }
    read_at(w.old.fd, offset, encoded, w.path);
    return true;
}

void ShardWriter::replace(const Shape& local, std::span<const std::byte> encoded) {
    auto& w = *impl_; w.check();
    const auto i = w.entry(local), previous = get(w.index, i, 0);
    if (previous != missing && (previous & replaced))
        fail(Failure::argument, w.path, "inner chunk has already been replaced");
    if (encoded.empty()) { set(w.index, i, replaced, missing); return; }
    write_at(w.output.file.fd, w.next, encoded, w.path);
    set(w.index, i, w.next | replaced, encoded.size());
    w.next += encoded.size();
}

void ShardWriter::commit(Bytes& copy_buffer) {
    auto& w = *impl_; w.check();
    if (copy_buffer.empty()) fail(Failure::argument, w.path, "shard copy buffer is empty");
    for (std::size_t i = 0; i < w.info.count; ++i) {
        auto offset = get(w.index, i, 0), length = get(w.index, i, 1);
        if (offset == missing) continue;
        if (offset & replaced) {
            set(w.index, i, length == missing ? missing : offset & ~replaced, length);
            continue;
        }
        if (length > std::uint64_t(std::numeric_limits<off_t>::max()) - w.next)
            fail(Failure::io, w.path, "shard size overflow");
        const auto output_offset = w.next;
        for (std::uint64_t done = 0; done < length;) {
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(length - done, copy_buffer.size()));
            auto bytes = std::span(copy_buffer.data(), n);
            read_at(w.old.fd, offset + done, bytes, w.path);
            write_at(w.output.file.fd, w.next, bytes, w.path);
            w.next += n; done += n;
        }
        set(w.index, i, output_offset, length);
    }
    if (w.simple) {
        if (w.swap) w.byte_swap();
        if (w.crc) {
            const auto n = w.index.numeric.size();
            const auto value = shard_crc32c(w.index.numeric);
            w.index.numeric.resize(n+4);
            for (int i = 0; i < 4; ++i) w.index.numeric[n+i] = std::byte((value >> (8*i)) & 255);
        }
        if (w.index.numeric.size() != w.info.encoded_size) fail(Failure::encoding, w.path, "incorrect encoded shard index length");
        write_at(w.output.file.fd, w.spec.index_location == "start" ? 0 : w.next, w.index.numeric, w.path);
    } else {
        Bytes encoded;
        try { encoded = encode_payload({w.info.metadata, w.spec.index_codecs, w.spec.scratch_bytes}, w.info.shape, w.index); }
        catch (const Error& e) { throw Error(e.failure, w.path.string(), e.what()); }
        if (encoded.size() != w.info.encoded_size) fail(Failure::encoding, w.path, "incorrect encoded shard index length");
        write_at(w.output.file.fd, w.spec.index_location == "start" ? 0 : w.next, encoded, w.path);
    }
    if (::rename(w.output.name.c_str(), w.path.c_str()) != 0) syserr(w.path, "replace shard");
    w.output.committed = true;
}

void write_shard_chunk(const std::filesystem::path& path, const ShardSpec& spec,
                       const Shape& local, std::span<const std::byte> payload) {
    const auto index = shard_index_workspace(spec, path);
    if (index >= spec.scratch_bytes) fail(Failure::argument, path, "scratch memory budget leaves no shard copy buffer");
    Bytes copy(std::min<std::size_t>(1024 * 1024, spec.scratch_bytes - index));
    ShardWorkspace workspace;
    ShardWriter writer(path, spec, workspace);
    writer.replace(local, payload);
    writer.commit(copy);
}

} // namespace anndatapp_zarr
