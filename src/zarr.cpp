module;
#include <string_view>

module anndata.zarr;

namespace anndata {
namespace {
std::string_view compression_name(ZarrCompression compression) {
    switch(compression) {
        case ZarrCompression::none: return "none";
        case ZarrCompression::gzip: return "gzip";
        case ZarrCompression::zstd: return "zstd";
        case ZarrCompression::blosc: return "blosc";
    }
    return "unknown";
}
}
BackedAnnData open_zarr(std::string_view path,BackedOptions options) {
    return detail::open_zarr_backed(path,options);
}
AnnData read_zarr(std::string_view path,ZarrReadOptions options) {
    BackedOptions backed; backed.validation=options.validation; backed.scratch_bytes=options.scratch_bytes;
    return open_zarr(path,backed).read();
}
void write_zarr(std::string_view path,const AnnDataView& data,ZarrWriteOptions options) {
    detail::write_zarr_eager(path,data,static_cast<int>(options.format),options.mode==ZarrWriteMode::exclusive,
                            options.scratch_bytes,compression_name(options.compression),options.compression_level,options.validation);
}
void write_zarr(std::string_view path,const AnnData& data,ZarrWriteOptions options) { write_zarr(path,data.view(),options); }
void write_zarr(std::string_view path,const BackedAnnDataView& data,ZarrWriteOptions options) {
    detail::write_zarr_backed(path,data,static_cast<int>(options.format),options.mode==ZarrWriteMode::exclusive,
                             options.scratch_bytes,compression_name(options.compression),options.compression_level,options.validation);
}
void write_zarr(std::string_view path,const BackedAnnData& data,ZarrWriteOptions options) { write_zarr(path,data.view(),options); }
} // namespace anndata
