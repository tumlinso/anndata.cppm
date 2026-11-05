// Reuse the existing bounded libc I/O/allocation instrumentation without
// interposing codec entry points. Python wheels may bundle independent codec
// versions; forwarding their calls through RTLD_NEXT can corrupt that ABI.
#define blosc_compress_ctx anndatapp_unused_blosc_compress_ctx
#define blosc_decompress_ctx anndatapp_unused_blosc_decompress_ctx
#define ZSTD_compress2 anndatapp_unused_ZSTD_compress2
#define ZSTD_decompressStream anndatapp_unused_ZSTD_decompressStream
#define inflate anndatapp_unused_inflate
#define deflate anndatapp_unused_deflate
#include "../completeness_probe/counters.c"
