module;
#include <cstddef>
#include <string_view>

export module anndata.zarr;
export import anndata.backed;

export namespace anndata {
enum class ZarrFormat { v2 = 2, v3 = 3 };
enum class ZarrCompression { none, gzip, zstd, blosc };
enum class ZarrWriteMode { truncate, exclusive };
struct ZarrReadOptions {
    Validation validation = Validation::full;
    std::size_t scratch_bytes = 64u * 1024u * 1024u;
};
struct ZarrWriteOptions {
    Validation validation = Validation::full;
    ZarrFormat format = ZarrFormat::v3;
    ZarrWriteMode mode = ZarrWriteMode::truncate;
    ZarrCompression compression = ZarrCompression::zstd;
    int compression_level = 3;
    std::size_t scratch_bytes = 64u * 1024u * 1024u;
};
[[nodiscard]] BackedAnnData open_zarr(std::string_view path,BackedOptions options = {});
[[nodiscard]] AnnData read_zarr(std::string_view path,ZarrReadOptions options = {});
void write_zarr(std::string_view path,const AnnDataView& data,ZarrWriteOptions options = {});
void write_zarr(std::string_view path,const AnnData& data,ZarrWriteOptions options = {});
void write_zarr(std::string_view path,const BackedAnnDataView& data,ZarrWriteOptions options = {});
void write_zarr(std::string_view path,const BackedAnnData& data,ZarrWriteOptions options = {});
} // namespace anndata
