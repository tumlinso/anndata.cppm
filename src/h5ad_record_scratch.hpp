#pragma once
#include <cstddef>
#include <cstdlib>
#include <memory>

namespace anndata_h5ad_detail {
inline constexpr std::size_t record_target = 1024u*1024u;
inline constexpr std::size_t record_cap = 64u*1024u*1024u;

// Only record-read staging belongs here. Packed output buffers are final data.
// Each requested allocation includes its tracking header in the byte limit.
struct RecordVlenAllocator {
    struct Header { std::size_t bytes; Header* previous; Header* next; };
    std::size_t limit, used = 0, peak = 0;
    Header* first = nullptr;
    bool exceeded = false;
    ~RecordVlenAllocator() {
        while (first) { auto* next = first->next; std::free(first); first = next; }
    }
    static void* allocate(std::size_t n, void* context) noexcept {
        auto& a = *static_cast<RecordVlenAllocator*>(context);
        if (!n) n = 1;
        if (a.used > a.limit || sizeof(Header) > a.limit-a.used || n > a.limit-a.used-sizeof(Header)) {
            a.exceeded = true; return nullptr;
        }
        auto* p = static_cast<Header*>(std::malloc(sizeof(Header)+n));
        if (!p) return nullptr;
        *p = {n,nullptr,a.first};
        if (a.first) a.first->previous = p;
        a.first = p; a.used += sizeof(Header)+n;
        if (a.used > a.peak) a.peak = a.used;
        return p+1;
    }
    static void release(void* pointer, void* context) noexcept {
        if (!pointer) return;
        auto& a = *static_cast<RecordVlenAllocator*>(context);
        auto* p = static_cast<Header*>(pointer)-1;
        if (p->previous) p->previous->next = p->next;
        else a.first = p->next;
        if (p->next) p->next->previous = p->previous;
        a.used -= sizeof(Header)+p->bytes; std::free(p);
    }
    static std::size_t size(const char* p) noexcept { return (reinterpret_cast<const Header*>(p)-1)->bytes; }
};

inline std::size_t record_pointer_limit(std::size_t n, std::size_t budget) noexcept {
    constexpr auto base = sizeof(RecordVlenAllocator)+sizeof(std::unique_ptr<char*[]>);
    return budget < base || n > (budget-base)/sizeof(char*) ? 0 : budget-base-n*sizeof(char*);
}
}
