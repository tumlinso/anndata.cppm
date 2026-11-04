#include "zarr_engine.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <numeric>
#include <string_view>
#include <utility>

#include <blosc.h>
#include <zlib.h>
#include <zstd.h>

namespace anndata_zarr {
namespace {
[[noreturn]] void fail(const std::string& msg) {
    auto category=Failure::encoding;
    if(msg.find("unsupported")!=std::string::npos || msg.starts_with("unknown ")) category=Failure::codec;
    if(msg.find("working-memory")!=std::string::npos || msg.find("budget")!=std::string::npos) category=Failure::argument;
    throw Error(category,"",msg);
}
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) fail("chunk size overflow");
    return a * b;
}
void bounded(std::size_t n, std::size_t limit) {
    if (n > limit) fail("decoded or encoded chunk exceeds the codec working-memory limit");
}
void append(Bytes& out, const std::byte* p, std::size_t n, std::size_t limit) {
    if (n > limit - out.size()) fail("decoded or encoded chunk exceeds the codec working-memory limit");
    const auto size = out.size() + n;
    if (size > out.capacity()) {
        auto growth = out.capacity() > limit / 2 ? limit : out.capacity() * 2;
        out.reserve(std::max(size, growth));
    }
    out.insert(out.end(), p, p + n);
}
std::uint32_t get32(const std::byte* p) {
    return std::uint32_t(std::to_integer<unsigned char>(p[0])) |
        (std::uint32_t(std::to_integer<unsigned char>(p[1])) << 8) |
        (std::uint32_t(std::to_integer<unsigned char>(p[2])) << 16) |
        (std::uint32_t(std::to_integer<unsigned char>(p[3])) << 24);
}
void put32(Bytes& out, std::uint32_t n) {
    for (int i = 0; i < 4; ++i) out.push_back(std::byte((n >> (8 * i)) & 255));
}
std::uint32_t crc32c(std::span<const std::byte> input) {
    std::uint32_t v = ~std::uint32_t(0);
    for (auto b : input) {
        v ^= std::to_integer<unsigned char>(b);
        for (int k = 0; k < 8; ++k) v = (v >> 1) ^ ((0u - (v & 1u)) & 0x82f63b78u);
    }
    return ~v;
}

Bytes zlib_transform(std::span<const std::byte> input, bool encode, bool gzip, int level, std::size_t limit, Bytes result = {}, Bytes* reusable_block = nullptr) {
    if (input.size() > std::numeric_limits<uInt>::max()) fail("zlib chunk exceeds supported size");
    Bytes block=reusable_block?std::move(*reusable_block):Bytes{};
    block.resize(std::min<std::size_t>(limit, 65536));
    z_stream s{};
    int status = encode ? deflateInit2(&s, level, Z_DEFLATED, gzip ? 31 : 15, 8, Z_DEFAULT_STRATEGY)
                        : inflateInit2(&s, 47);
    if (status != Z_OK) fail("cannot initialize zlib codec");
    s.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(input.data()));
    s.avail_in = static_cast<uInt>(input.size());
    result.clear();
    do {
        s.next_out = reinterpret_cast<Bytef*>(block.data());
        s.avail_out = static_cast<uInt>(block.size());
        status = encode ? deflate(&s, Z_FINISH) : inflate(&s, Z_NO_FLUSH);
        if (status != Z_OK && status != Z_STREAM_END) {
            if (encode) deflateEnd(&s); else inflateEnd(&s);
            fail("invalid or truncated zlib/gzip payload");
        }
        try { append(result, block.data(), block.size() - s.avail_out, limit); }
        catch (...) { if (encode) deflateEnd(&s); else inflateEnd(&s); throw; }
    } while (status != Z_STREAM_END);
    const bool trailing = s.avail_in != 0;
    if (encode) deflateEnd(&s); else inflateEnd(&s);
    if (trailing) fail("trailing data in zlib/gzip payload");
    if (reusable_block) *reusable_block=std::move(block);
    return result;
}
Bytes zstd_transform(std::span<const std::byte> input, bool encode, int level, bool checksum, std::size_t limit, Bytes out = {}, Bytes* reusable_block = nullptr) {
    if (encode) {
        std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> ctx(ZSTD_createCCtx(), &ZSTD_freeCCtx);
        if (!ctx) fail("cannot initialize Zstd codec");
        auto p = ZSTD_CCtx_setParameter(ctx.get(), ZSTD_c_compressionLevel, level);
        if (!ZSTD_isError(p)) p = ZSTD_CCtx_setParameter(ctx.get(), ZSTD_c_checksumFlag, checksum ? 1 : 0);
        out.resize(std::min(limit, ZSTD_compressBound(input.size())));
        if (!ZSTD_isError(p)) p = ZSTD_compress2(ctx.get(), out.data(), out.size(), input.data(), input.size());
        if (ZSTD_isError(p)) fail(std::string("Zstd compression failed: ") + ZSTD_getErrorName(p));
        out.resize(p);
        return out;
    }
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> ctx(ZSTD_createDCtx(), &ZSTD_freeDCtx);
    if (!ctx) fail("cannot initialize Zstd codec");
    ZSTD_inBuffer in{input.data(), input.size(), 0};
    out.clear();
    Bytes block=reusable_block?std::move(*reusable_block):Bytes{};
    block.resize(std::min<std::size_t>(limit, 65536));
    std::size_t remaining = 1;
    while (in.pos < in.size || remaining != 0) {
        ZSTD_outBuffer ob{block.data(), block.size(), 0};
        const auto before = in.pos;
        remaining = ZSTD_decompressStream(ctx.get(), &ob, &in);
        if (ZSTD_isError(remaining) || (in.pos == before && ob.pos == 0 && remaining != 0)) {
            fail("invalid or truncated Zstd payload");
        }
        append(out, block.data(), ob.pos, limit);
        if (remaining == 0 && in.pos != in.size) fail("trailing data in Zstd payload");
    }
    if (reusable_block) *reusable_block=std::move(block);
    return out;
}
Bytes blosc_transform(std::span<const std::byte> input, const Json& cfg, bool encode, std::size_t item_size, std::size_t limit, Bytes out = {}) {
    if (encode) {
        if (input.size() > BLOSC_MAX_BUFFERSIZE) fail("Blosc chunk exceeds supported size");
        int shuffle = 1;
        if (cfg.contains("shuffle")) {
            if (cfg["shuffle"].is_string()) {
                const auto v = cfg["shuffle"].get<std::string>();
                if (v == "noshuffle") shuffle = 0;
                else if (v == "shuffle") shuffle = 1;
                else if (v == "bitshuffle") shuffle = 2;
                else fail("unknown Blosc shuffle mode: " + v);
            } else shuffle = cfg["shuffle"].get<int>();
        }
        if (shuffle < 0 || shuffle > 2) fail("invalid Blosc shuffle mode");
        const int clevel = cfg.value("clevel", 5);
        const auto cname = cfg.value("cname", std::string("lz4"));
        auto typesize = cfg.value("typesize", std::max<std::size_t>(item_size, 1));
        if (typesize == 0 || typesize > 255) fail("invalid Blosc type size");
        out.resize(std::min(limit, input.size() + BLOSC_MAX_OVERHEAD));
        int n = blosc_compress_ctx(clevel, shuffle, typesize, input.size(), input.data(), out.data(), out.size(), cname.c_str(), cfg.value("blocksize", 0), 1);
        if (n <= 0) fail("Blosc compression failed (unsupported compressor or invalid configuration)");
        out.resize(static_cast<std::size_t>(n));
        return out;
    }
    if (input.size() < BLOSC_MIN_HEADER_LENGTH) fail("truncated Blosc header");
    std::size_t nbytes = 0;
    if (blosc_cbuffer_validate(input.data(), input.size(), &nbytes) != 0 || nbytes > BLOSC_MAX_BUFFERSIZE)
        fail("invalid Blosc payload size");
    bounded(nbytes, limit);
    out.resize(nbytes);
    const int n = blosc_decompress_ctx(input.data(), out.data(), out.size(), 1);
    if (n < 0 || static_cast<std::size_t>(n) != nbytes) fail("Blosc decompression failed");
    return out;
}

struct Pipeline {
    struct Transpose { Shape input, order; };
    std::vector<Transpose> transposes;
    std::vector<Json> byte_codecs;
    bool strings = false;
    char endian = '=';
};
std::string name(const Json& codec) {
    if (!codec.is_object()) fail("codec must be a JSON object");
    if (codec.contains("name") && codec["name"].is_string()) return codec["name"].get<std::string>();
    if (codec.contains("id") && codec["id"].is_string()) return codec["id"].get<std::string>();
    fail("codec is missing its name");
}
Json config(const Json& codec) {
    if (codec.contains("configuration")) {
        if (!codec["configuration"].is_object()) fail("invalid codec configuration");
        return codec["configuration"];
    }
    return codec;
}
Pipeline pipeline(const PayloadMeta& meta, const Shape& shape) {
    const auto& m = meta.array;
    Pipeline p; p.endian = m.endian;
    if (m.type != Type::string && m.item_size != type_size(m.type)) fail("invalid scalar width");
    Shape current = shape;
    bool serialized = false;
    if (m.format == 2 && m.order == 'F') {
        Shape perm(shape.size());
        std::iota(perm.rbegin(), perm.rend(), std::size_t{0});
        p.transposes.push_back({current, perm});
        std::reverse(current.begin(), current.end());
    } else if (m.format == 2 && m.order != 'C') fail("invalid Zarr v2 array order");
    if (!meta.codecs.is_array()) fail("codec pipeline must be an array");
    for (const auto& codec : meta.codecs) {
        const auto n = name(codec);
        auto c = config(codec);
        if (n == "transpose") {
            if (serialized || m.format == 2) fail("invalid transpose codec position");
            if (!c.contains("order") || !c["order"].is_array()) fail("missing transpose order");
            auto perm = c["order"].get<Shape>();
            if (perm.size() != current.size()) fail("transpose rank mismatch");
            auto sorted = perm; std::sort(sorted.begin(), sorted.end());
            for (std::size_t i = 0; i < sorted.size(); ++i) if (sorted[i] != i) fail("invalid transpose permutation");
            p.transposes.push_back({current, perm});
            Shape next(current.size());
            for (std::size_t i = 0; i < current.size(); ++i) next[i] = current[perm[i]];
            current = std::move(next);
        } else if (n == "bytes" || n == "vlen-utf8") {
            if (serialized) fail("multiple array-to-bytes codecs");
            serialized = true;
            p.strings = n == "vlen-utf8";
            if (n == "bytes" && c.contains("endian")) {
                if (!c["endian"].is_string()) fail("invalid bytes endian");
                auto v = c["endian"].get<std::string>();
                if (v == "little") p.endian = '<';
                else if (v == "big") p.endian = '>';
                else fail("invalid bytes endian");
            } else if (n == "bytes" && m.format == 3 && m.item_size > 1) fail("missing bytes endian");
        } else if (n == "gzip" || n == "zlib" || n == "zstd" || n == "blosc" || n == "crc32c") {
            if (!serialized) {
                if (m.format != 2 || m.type == Type::string) fail("byte codec precedes serializer");
                serialized = true;
            }
            p.byte_codecs.push_back(codec);
        } else fail("unsupported Zarr codec: " + n);
    }
    if (!serialized && m.format == 2 && m.type != Type::string) serialized = true;
    if (!serialized) fail("missing array-to-bytes codec");
    if (p.strings != (m.type == Type::string)) fail("serializer does not match array data type");
    return p;
}

// Reorder a row-major input by new_axis[i] = old_axis[order[i]].
template<class Copy> void for_permuted(const Shape& input, const Shape& order, Copy copy) {
    Shape output(input.size()), strides(input.size(), 1);
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[order[i]];
    for (std::size_t i = input.size(); i > 1; --i) strides[i-2] = multiply(strides[i-1], input[i-1]);
    const auto n = checked_product(input);
    for (std::size_t target = 0; target < n; ++target) {
        auto k = target; std::size_t source = 0;
        for (std::size_t i = output.size(); i > 0; --i) {
            const auto axis = i - 1;
            auto coordinate = k % output[axis]; k /= output[axis];
            source += coordinate * strides[order[axis]];
        }
        copy(target, source);
    }
}
Chunk transpose(Chunk in, const Shape& shape, const Shape& order, bool strings, std::size_t width) {
    Chunk out;
    if (strings) {
        out.strings.resize(in.strings.size());
        for_permuted(shape, order, [&](auto dst, auto src) { out.strings[dst] = std::move(in.strings[src]); });
    } else {
        out.numeric.resize(in.numeric.size());
        for_permuted(shape, order, [&](auto dst, auto src) { std::memcpy(out.numeric.data() + dst * width, in.numeric.data() + src * width, width); });
    }
    return out;
}
void endian_convert(Bytes& b, std::size_t width, char endian) {
    if (width <= 1 || endian == '=' || endian == '|') return;
    if (endian != '<' && endian != '>') fail("invalid byte order");
    const bool file_little = endian == '<';
    if (file_little == (std::endian::native == std::endian::little)) return;
    for (std::size_t i = 0; i < b.size(); i += width) std::reverse(b.begin() + i, b.begin() + i + width);
}
bool valid_utf8(std::string_view text) {
    for (std::size_t i = 0; i < text.size();) {
        auto b = static_cast<unsigned char>(text[i++]);
        if (b < 0x80) continue;
        unsigned length = 0; std::uint32_t value = 0, minimum = 0;
        if (b >= 0xc2 && b <= 0xdf) { length = 1; value = b & 0x1f; minimum = 0x80; }
        else if (b >= 0xe0 && b <= 0xef) { length = 2; value = b & 0x0f; minimum = 0x800; }
        else if (b >= 0xf0 && b <= 0xf4) { length = 3; value = b & 0x07; minimum = 0x10000; }
        else return false;
        if (text.size() - i < length) return false;
        for (unsigned j = 0; j < length; ++j) {
            b = static_cast<unsigned char>(text[i++]);
            if ((b & 0xc0) != 0x80) return false;
            value = (value << 6) | (b & 0x3f);
        }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
    }
    return true;
}
Bytes serialise_strings(const std::vector<std::string>& strings, std::size_t limit) {
    if (strings.size() > std::numeric_limits<std::uint32_t>::max()) fail("too many strings in chunk");
    if (limit < 4 || strings.size() > (limit - 4) / 4) fail("string chunk exceeds the codec working-memory limit");
    Bytes out; put32(out, static_cast<std::uint32_t>(strings.size()));
    for (const auto& s : strings) {
        if (!valid_utf8(s)) fail("invalid UTF-8 string");
        if (s.size() > std::numeric_limits<std::uint32_t>::max()) fail("string too large");
        if (out.size() > limit - 4 || s.size() > limit - out.size() - 4) fail("string chunk exceeds the codec working-memory limit");
        put32(out, static_cast<std::uint32_t>(s.size()));
        const auto* p = reinterpret_cast<const std::byte*>(s.data());
        append(out, p, s.size(), limit);
    }
    return out;
}
std::vector<std::string> deserialise_strings(std::span<const std::byte> input, std::size_t expected, std::size_t limit) {
    if (input.size() < 4 || get32(input.data()) != expected) fail("invalid vlen UTF-8 element count");
    if (expected > (input.size() - 4) / 4) fail("truncated vlen UTF-8 lengths");
    if (expected > limit / sizeof(std::string)) fail("string chunk exceeds the codec working-memory limit");
    const std::size_t header_bytes = expected * sizeof(std::string);
    std::size_t pos = 4; std::vector<std::string> out; out.reserve(expected);
    std::size_t text_bytes = 0;
    for (std::size_t i = 0; i < expected; ++i) {
        if (input.size() - pos < 4) fail("truncated vlen UTF-8 length");
        auto n = get32(input.data() + pos); pos += 4;
        if (n > input.size() - pos) fail("truncated vlen UTF-8 content");
        if (n > limit - header_bytes - text_bytes) fail("string chunk exceeds the codec working-memory limit");
        text_bytes += n;
        out.emplace_back(reinterpret_cast<const char*>(input.data() + pos), n); pos += n;
        if (!valid_utf8(out.back())) fail("invalid UTF-8 string");
    }
    if (pos != input.size()) fail("trailing bytes in vlen UTF-8 payload");
    return out;
}
Bytes apply(Bytes bytes, const Json& codec, bool encode, std::size_t item_size, std::size_t limit) {
    const auto n = name(codec); const auto c = config(codec);
    if (n == "gzip" || n == "zlib") return zlib_transform(bytes, encode, n == "gzip", c.value("level", n == "gzip" ? 5 : 1), limit);
    if (n == "zstd") return zstd_transform(bytes, encode, c.value("level", 0), c.value("checksum", false), limit);
    if (n == "blosc") return blosc_transform(bytes, c, encode, item_size, limit);
    if (n == "crc32c") {
        if (encode) { if (limit < 4 || bytes.size() > limit - 4) fail("checksum chunk exceeds the codec working-memory limit"); put32(bytes, crc32c(bytes)); return bytes; }
        if (bytes.size() < 4 || crc32c(std::span<const std::byte>(bytes.data(), bytes.size()-4)) != get32(bytes.data()+bytes.size()-4)) fail("CRC32C checksum mismatch");
        bytes.resize(bytes.size()-4); return bytes;
    }
    fail("unsupported byte codec: " + n);
}
} // namespace

std::size_t checked_product(const Shape& shape) {
    std::size_t size = 1;
    for (auto n : shape) size = multiply(size, n);
    return size;
}
std::size_t type_size(Type type) {
    switch (type) {
    case Type::boolean: case Type::int8: case Type::uint8: return 1;
    case Type::int16: case Type::uint16: return 2;
    case Type::int32: case Type::uint32: case Type::float32: return 4;
    case Type::int64: case Type::uint64: case Type::float64: return 8;
    case Type::string: return 0;
    }
    fail("unknown scalar type");
}
NumericCodec prepare_numeric_codec(const PayloadMeta& meta,const Shape& shape) {
    NumericCodec result;
    if(meta.array.type==Type::string) return result;
    const auto p=pipeline(meta,shape);
    if(!p.transposes.empty() || p.byte_codecs.size()!=1) return result;
    if(meta.array.item_size>1 && p.endian!='=' && p.endian!='|' &&
       p.endian!=(std::endian::native==std::endian::little?'<':'>')) return result;
    const auto n=name(p.byte_codecs[0]);
    using Kind=NumericCodec::Kind;
    result.kind=n=="gzip"?Kind::gzip:n=="zlib"?Kind::zlib:n=="zstd"?Kind::zstd:n=="blosc"?Kind::blosc:Kind::none;
    if(result.kind==Kind::none) return result;
    result.configuration=config(p.byte_codecs[0]);
    result.item_size=meta.array.item_size;
    if(result.kind==Kind::gzip || result.kind==Kind::zlib) result.level=result.configuration.value("level",n=="gzip"?5:1);
    if(result.kind==Kind::zstd) { result.level=result.configuration.value("level",0); result.checksum=result.configuration.value("checksum",false); }
    return result;
}
std::size_t reusable_encoded_bound(const NumericCodec& codec,std::size_t size,std::size_t limit) {
    if(codec.kind==NumericCodec::Kind::zstd) return std::min(limit,ZSTD_compressBound(size));
    if(codec.kind==NumericCodec::Kind::blosc) return std::min(limit,size>std::numeric_limits<std::size_t>::max()-BLOSC_MAX_OVERHEAD?limit:size+BLOSC_MAX_OVERHEAD);
    if(size>std::numeric_limits<uLong>::max()) fail("zlib chunk exceeds supported size");
    const auto bound=compressBound(static_cast<uLong>(size));
    return bound>std::numeric_limits<std::size_t>::max()-64?limit:std::min(limit,std::size_t(bound)+64);
}
bool reusable_needs_block(const NumericCodec& codec,bool writing) {
    return codec.kind==NumericCodec::Kind::gzip || codec.kind==NumericCodec::Kind::zlib || (codec.kind==NumericCodec::Kind::zstd && !writing);
}
Bytes reusable_transform(const NumericCodec& codec,std::span<const std::byte> input,bool encode,std::size_t item_size,std::size_t limit,Bytes out,Bytes& block) {
    using Kind=NumericCodec::Kind;
    if(codec.kind==Kind::gzip || codec.kind==Kind::zlib) return zlib_transform(input,encode,codec.kind==Kind::gzip,codec.level,limit,std::move(out),&block);
    if(codec.kind==Kind::zstd) return zstd_transform(input,encode,codec.level,codec.checksum,limit,std::move(out),&block);
    if(codec.kind==Kind::blosc) return blosc_transform(input,codec.configuration,encode,item_size,limit,std::move(out));
    fail("unsupported reusable numeric codec");
}
void decode_numeric_reusable(const NumericCodec& codec,std::size_t expected,std::size_t limit,std::span<const std::byte> input,Bytes& output,Bytes& block) {
    bounded(input.size(),limit); bounded(expected,limit);
    output=reusable_transform(codec,input,false,0,expected,std::move(output),block);
    if(output.size()!=expected) fail("decoded chunk has wrong byte size");
}
void encode_numeric_reusable(const NumericCodec& codec,std::size_t limit,std::span<const std::byte> input,Bytes& output,Bytes& block) {
    bounded(input.size(),limit);
    output=reusable_transform(codec,input,true,codec.item_size,limit,std::move(output),block);
}
Chunk decode_payload(const PayloadMeta& meta, const Shape& shape, std::span<const std::byte> encoded) {
    const auto& m = meta.array;
    auto p = pipeline(meta, shape);
    const auto limit = meta.scratch_bytes / 4;
    if (limit < 4) fail("codec working-memory budget is too small");
    bounded(encoded.size(), limit);
    Bytes bytes(encoded.begin(), encoded.end());
    for (auto i = p.byte_codecs.rbegin(); i != p.byte_codecs.rend(); ++i) bytes = apply(std::move(bytes), *i, false, m.item_size, limit);
    Chunk result; const auto n = checked_product(shape);
    if (p.strings) result.strings = deserialise_strings(bytes, n, limit);
    else {
        bounded(multiply(n, m.item_size), limit);
        if (bytes.size() != multiply(n, m.item_size)) fail("decoded chunk has wrong byte size");
        endian_convert(bytes, m.item_size, p.endian); result.numeric = std::move(bytes);
    }
    for (auto i = p.transposes.rbegin(); i != p.transposes.rend(); ++i) {
        Shape transposed(i->order.size()), inverse(i->order.size());
        for (std::size_t j = 0; j < i->order.size(); ++j) { transposed[j] = i->input[i->order[j]]; inverse[i->order[j]] = j; }
        result = transpose(std::move(result), transposed, inverse, p.strings, m.item_size);
    }
    return result;
}
Bytes encode_payload(const PayloadMeta& meta, const Shape& shape, const Chunk& chunk) {
    const auto& m = meta.array;
    auto p = pipeline(meta, shape); auto n = checked_product(shape);
    const auto limit = meta.scratch_bytes / 4;
    if (limit < 4) fail("codec working-memory budget is too small");
    if ((p.strings && chunk.strings.size() != n) || (!p.strings && chunk.numeric.size() != multiply(n, m.item_size))) fail("chunk size does not match shape");
    if (p.strings) {
        if (n > limit / sizeof(std::string)) fail("string chunk exceeds the codec working-memory limit");
        auto total = n * sizeof(std::string);
        for (const auto& s : chunk.strings) {
            if (s.size() > limit - total) fail("string chunk exceeds the codec working-memory limit");
            total += s.size();
        }
    } else bounded(chunk.numeric.size(), limit);
    Chunk data = chunk;
    for (const auto& t : p.transposes) data = transpose(std::move(data), t.input, t.order, p.strings, m.item_size);
    Bytes bytes;
    if (p.strings) bytes = serialise_strings(data.strings, limit);
    else { bytes = std::move(data.numeric); endian_convert(bytes, m.item_size, p.endian); }
    for (const auto& codec : p.byte_codecs) bytes = apply(std::move(bytes), codec, true, m.item_size, limit);
    return bytes;
}
} // namespace anndata_zarr
