#include "zarr_engine.hpp"
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>
#if defined(__linux__) && defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8
#define ANNDATAPP_TEST_FAULT_IO 1
#include <sys/syscall.h>
#include <unistd.h>
// Test-only syscall interposition covers both static and shared libraries. The
// selected worker thread receives EIO after the requested number of operations.
static thread_local int fail_read_after = -1, fail_write_after = -1;
static thread_local bool partial_write = false;
extern "C" ssize_t pread(int fd, void* data, size_t n, off_t offset) {
    if (!fail_read_after) { fail_read_after = -1; errno = EIO; return -1; }
    if (fail_read_after > 0) --fail_read_after;
    return static_cast<ssize_t>(::syscall(SYS_pread64, fd, data, n, offset));
}
extern "C" ssize_t pwrite(int fd, const void* data, size_t n, off_t offset) {
    if (!fail_write_after) { fail_write_after = -1; errno = EIO; return -1; }
    if (fail_write_after > 0) --fail_write_after;
    if (partial_write && n > 1) { n /= 2; partial_write = false; }
    return static_cast<ssize_t>(::syscall(SYS_pwrite64, fd, data, n, offset));
}
#endif

using namespace anndatapp_zarr;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
Bytes bytes(std::initializer_list<unsigned> values) {
    Bytes result; for (auto value : values) result.push_back(std::byte(value)); return result;
}
Bytes file_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open shard");
    auto size = std::filesystem::file_size(path);
    Bytes result(size);
    if (size && !file.read(reinterpret_cast<char*>(result.data()), std::streamsize(size)))
        throw std::runtime_error("cannot read shard");
    return result;
}
template<class F> void error(F&& f, Failure expected) {
    try { f(); } catch (const Error& e) { check(e.failure == expected, "wrong error category"); return; }
    throw std::runtime_error("expected error");
}
void no_temps(const std::filesystem::path& root) {
    for (auto& p : std::filesystem::directory_iterator(root))
        check(p.path().filename().string().find(".tmp-") == std::string::npos, "temporary file survived");
}
void run(const std::filesystem::path& path, const std::string& index_location, bool transpose, int crc_count, const char* endian) {
    ShardSpec spec;
    spec.shard_shape = {2,3}; spec.chunk_shape = {1,1}; spec.index_location = index_location;
    spec.scratch_bytes = 128*1024;
    spec.index_codecs = Json::array();
    if (transpose) spec.index_codecs.push_back({{"name","transpose"},{"configuration",{{"order",{1,0,2}}}}});
    spec.index_codecs.push_back({{"name","bytes"},{"configuration",{{"endian",endian}}}});
    for (int i = 0; i < crc_count; ++i) spec.index_codecs.push_back({{"name","crc32c"}});
    const auto index_bytes = shard_index_workspace(spec);
    const auto encoded_size = 6*16 + 4*crc_count;
    const bool simple = !transpose && crc_count <= 1;
    check(index_bytes == encoded_size*(simple ? 1 : 4) && index_bytes < spec.scratch_bytes, "index reservation");
    Bytes copy(7), encoded;
    ShardWorkspace workspace;
    const auto a = bytes({1,2,3}), b = bytes({7,8,9,10}), c = bytes({0,255}), d = bytes({19});
    {
        ShardWriter writer(path,spec,workspace);
        check(!writer.read({0,0},encoded,4096), "missing shard must have missing chunks");
        writer.replace({0,0},a); writer.replace({0,2},b); writer.replace({1,1},c);
        error([&]{ writer.replace({0,0},a); }, Failure::argument);
        writer.commit(copy);
        error([&]{ writer.commit(copy); }, Failure::argument);
    }
    const auto* saved_index = workspace.index.data();
    const auto allocations = workspace.allocations;
    check(read_shard_chunk(path,spec,{0,0}) == a, "first chunk");
    check(read_shard_chunk(path,spec,{0,2}) == b, "second chunk");
    check(read_shard_chunk(path,spec,{1,1}) == c, "third chunk");
    check(read_shard_chunk(path,spec,{1,2}).empty(), "missing chunk");
    check(::chmod(path.c_str(),0640) == 0, "chmod fixture");
    auto original = file_bytes(path);
    {
        ShardWriter writer(path,spec,workspace);
        check(writer.read({0,0},encoded,4096) && encoded == a, "old chunk read");
        auto* p = encoded.data();
        check(writer.read({1,1},encoded,4096) && encoded == c && encoded.data() == p, "old payload buffer reused");
        writer.replace({0,0},d);
        writer.replace({1,1},{});
        writer.replace({1,2},a);
        check(writer.read({0,2},encoded,4096) && encoded == b, "unchanged old payload readable after replacement");
        check(file_bytes(path) == original, "original changed before commit");
        writer.commit(copy);
    }
    if (simple) check(workspace.index.data() == saved_index && workspace.allocations == allocations, "index workspace reused");
    struct stat st{}; check(::stat(path.c_str(),&st)==0 && (st.st_mode&0777)==0640, "permissions retained");
    check(read_shard_chunk(path,spec,{0,0}) == d, "replacement chunk");
    check(read_shard_chunk(path,spec,{0,2}) == b, "unchanged encoded chunk");
    check(read_shard_chunk(path,spec,{1,1}).empty(), "deleted chunk");
    check(read_shard_chunk(path,spec,{1,2}) == a, "new chunk");
    original = file_bytes(path);
    {
        ShardWriter writer(path,spec,workspace);
        error([&]{ writer.read({0,2},encoded,1); }, Failure::argument);
        writer.replace({0,0},b);
    }
    check(file_bytes(path) == original, "uncommitted or failed writer changed shard");
    no_temps(path.parent_path());
#if defined(ANNDATAPP_TEST_FAULT_IO)
    // Index read fails before creating a temporary file.
    fail_read_after = 0;
    error([&]{ ShardWriter writer(path,spec,workspace); }, Failure::io);
    check(file_bytes(path) == original, "failed index read changed shard");
    no_temps(path.parent_path());
    // Old payload read fails after the index was read and temporary file made.
    {
        ShardWriter writer(path,spec,workspace);
        fail_read_after = 0;
        error([&]{ writer.read({0,2},encoded,4096); }, Failure::io);
    }
    check(file_bytes(path) == original, "failed payload read changed shard");
    no_temps(path.parent_path());
    // Failure partway through a replacement leaves only private temp bytes.
    {
        ShardWriter writer(path,spec,workspace);
        writer.replace({0,0},b);
        partial_write = true; fail_write_after = 1;
        error([&]{ writer.replace({1,2},a); }, Failure::io);
    }
    check(file_bytes(path) == original, "partial payload write changed shard");
    no_temps(path.parent_path());
    // Fail the final index write after copying both untouched old payloads.
    {
        ShardWriter writer(path,spec,workspace);
        writer.replace({0,0},b);
        fail_write_after = 2;
        error([&]{ writer.commit(copy); }, Failure::io);
    }
    check(file_bytes(path) == original, "failed index write changed shard");
    no_temps(path.parent_path());
#endif
    auto invalid = spec;
    const std::size_t bytes_codec = transpose ? 1 : 0;
    invalid.index_codecs[bytes_codec]["configuration"]["endian"] = "invalid";
    error([&]{ shard_index_workspace(invalid); }, Failure::encoding);
    auto bad = spec; bad.scratch_bytes = index_bytes-1;
    error([&]{ shard_index_workspace(bad); }, Failure::argument);
    error([&]{ ShardWriter writer(path,bad,workspace); }, Failure::argument);
    check(file_bytes(path) == original, "insufficient budget changed shard");
    no_temps(path.parent_path());
    // Corrupt the independently stored index checksum, preserving the original.
    auto damaged = path.parent_path()/(path.filename().string()+"-bad");
    auto corrupt = original;
    const auto index_start = index_location == "start" ? 0 : corrupt.size()-encoded_size;
    if (crc_count) corrupt[index_start+encoded_size-1] ^= std::byte{1};
    else for (std::size_t i = 0; i < 8; ++i) corrupt[index_start+i] = std::byte{0xff};
    { std::ofstream out(damaged,std::ios::binary); out.write(reinterpret_cast<const char*>(corrupt.data()),std::streamsize(corrupt.size())); }
    error([&]{ ShardWriter writer(damaged,spec,workspace); }, Failure::encoding);
    check(file_bytes(damaged) == corrupt, "corrupt shard changed on rejected index");
    no_temps(path.parent_path());
}
int main(int argc,char** argv) {
    try {
        auto root = std::filesystem::path(argc > 1 ? argv[1] : "build/zarr-sharding-test");
        std::filesystem::create_directories(root);
        auto pattern = (root/"trial-XXXXXX").string();
        if (!::mkdtemp(pattern.data())) throw std::runtime_error("cannot create fixture directory");
        root = pattern;
        for (const auto& location : {"start","end"}) for (bool transpose : {false,true})
            for (int crc : {0,1,3}) for (const auto& endian : {"little","big"})
                run(root/(std::string(location)+(transpose?"-transpose":"-plain")+"-crc"+std::to_string(crc)+endian),location,transpose,crc,endian);
        std::cout << "shard writer tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
