#pragma once
#ifdef __APPLE__
#define V1_DYLD_INTERPOSE(tag, replacement, original) \
    __attribute__((used,section("__DATA,__interpose"))) \
    static const struct { const void* replace; const void* replacee; } v1_interpose_##tag = \
        { (const void*)(replacement), (const void*)(original) }
#endif
