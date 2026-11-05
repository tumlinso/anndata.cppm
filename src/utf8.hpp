#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace anndatapp::detail {

enum class Utf8Error { none, invalid_lead, truncated, invalid_continuation, invalid_scalar };

inline Utf8Error utf8_error(std::string_view text) noexcept {
    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t i = 0;
    while (i < text.size()) {
        const auto first = bytes[i++];
        if (first < 0x80) continue;
        unsigned remaining;
        std::uint32_t code, minimum;
        if (first >= 0xc2 && first <= 0xdf) { remaining = 1; code = first & 0x1f; minimum = 0x80; }
        else if (first >= 0xe0 && first <= 0xef) { remaining = 2; code = first & 0x0f; minimum = 0x800; }
        else if (first >= 0xf0 && first <= 0xf4) { remaining = 3; code = first & 0x07; minimum = 0x10000; }
        else return Utf8Error::invalid_lead;
        if (remaining > text.size() - i) return Utf8Error::truncated;
        while (remaining--) {
            const auto next = bytes[i++];
            if ((next & 0xc0) != 0x80) return Utf8Error::invalid_continuation;
            code = (code << 6) | (next & 0x3f);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return Utf8Error::invalid_scalar;
    }
    return Utf8Error::none;
}

} // namespace anndatapp::detail
