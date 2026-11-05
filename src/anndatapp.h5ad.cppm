module;
#include <string_view>

export module anndatapp.h5ad;
export import anndatapp.backed;

export namespace anndatapp {

enum class WriteMode { truncate, exclusive };
enum class Compression { none, gzip };
enum class FileByteOrder { preserve, native, little, big };

struct ReadOptions {
    Validation validation = Validation::full;
};

struct WriteOptions {
    Validation validation = Validation::full;
    WriteMode mode = WriteMode::truncate;
    Compression compression = Compression::none;
    int gzip_level = 4;
    FileByteOrder byte_order = FileByteOrder::preserve;
};

// Eager loading closes the file before returning; all acquired numerical
// descriptors have the ordinary in-memory ownership and synchronization rules.
[[nodiscard]] AnnData read_h5ad(std::string_view path, ReadOptions options = {});
// Opening retains a file session and reads metadata without numerical payloads.
[[nodiscard]] BackedAnnData open_h5ad(std::string_view path, BackedOptions options = {});
[[nodiscard]] bool hdf5_library_threadsafe();
// Truncate writes directly to the destination. A failed write may leave a
// partial file. Exclusive creation fails if the destination already exists.
void write_h5ad(std::string_view path, const AnnData& data, WriteOptions options = {});
void write_h5ad(std::string_view path, const AnnDataView& data, WriteOptions options = {});
void write_h5ad(std::string_view path, const BackedAnnData& data, WriteOptions options = {});
void write_h5ad(std::string_view path, const BackedAnnDataView& data, WriteOptions options = {});

} // namespace anndatapp
