// Diagnostic-only LD_PRELOAD observer. Timed runs do not load this library.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include "dyld_interpose.h"
namespace {
std::atomic<bool> counting{false};
std::atomic<std::uint64_t> calls{0}, bytes{0};
void* allocate(std::size_t n, std::size_t alignment = 0) {
    void* p = nullptr;
    if (alignment) { if (posix_memalign(&p,alignment,n ? n : 1)) throw std::bad_alloc(); }
    else if (!(p = std::malloc(n ? n : 1))) throw std::bad_alloc();
    if (counting.load(std::memory_order_relaxed)) {
        calls.fetch_add(1,std::memory_order_relaxed); bytes.fetch_add(n,std::memory_order_relaxed);
    }
    return p;
}
}
extern "C" void v1_alloc_begin() { calls = 0; bytes = 0; counting.store(true,std::memory_order_relaxed); }
extern "C" void v1_alloc_end(std::uint64_t* out) {
    counting.store(false,std::memory_order_relaxed); out[0] = calls; out[1] = bytes;
}
#ifdef __APPLE__
// The system delete functions release malloc/posix_memalign allocations.
static void* v1_new(std::size_t n) { return allocate(n); }
static void* v1_new_array(std::size_t n) { return allocate(n); }
static void* v1_new_aligned(std::size_t n, std::align_val_t a) { return allocate(n,std::size_t(a)); }
static void* v1_new_array_aligned(std::size_t n, std::align_val_t a) { return allocate(n,std::size_t(a)); }
V1_DYLD_INTERPOSE(new_, &v1_new, (static_cast<void*(*)(std::size_t)>(&::operator new)));
V1_DYLD_INTERPOSE(new_array, &v1_new_array, (static_cast<void*(*)(std::size_t)>(&::operator new[])));
V1_DYLD_INTERPOSE(new_aligned, &v1_new_aligned, (static_cast<void*(*)(std::size_t,std::align_val_t)>(&::operator new)));
V1_DYLD_INTERPOSE(new_array_aligned, &v1_new_array_aligned, (static_cast<void*(*)(std::size_t,std::align_val_t)>(&::operator new[])));
#else
void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocate(n,std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate(n,std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
#endif
